# python train_recall.py --num_epochs 400 --optimizer sgd --pretrained 0 --dataset femnist --meta_batch_size 2 --support_size 50 --sampling_type meta_batch_groups --uniform_over_groups 1 --n_test_per_dist 2000 --seed $SEED --experiment_name femnist_recall_$SEED --log_wandb 1 --step_size 0.4



import os
from datetime import datetime
import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange, tqdm
import wandb
from sklearn import metrics
from dro_loss import LossComputer

import data
import utils

from maml import MAML

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# Arguments
parser = argparse.ArgumentParser()

# Training / Optimization args
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--drop_last', type=int, default=0)

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--use_lr_schedule', type=int, default=0)
parser.add_argument('--pret_add_channels', type=int, default=1, help="relevant when using context and pretrained resnet as prediction net")
parser.add_argument('--context_net', type=str, default='convnet')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['sgd', 'adam'])

# DRO
parser.add_argument('--use_robust_loss', type=int, default=0,
                    help='Use robust loss algo from DRNN paper')
parser.add_argument('--robust_step_size', type=float, default=0.01,
                    help='When using robust loss algo from DRNN paper')

# Model args
parser.add_argument('--model', type=str, default='ContextualConvNet', choices=['ContextualMLP', 'ContextualConvNet'])

parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
# If model is Convnet
parser.add_argument('--prediction_net', type=str, default='convnet',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'])

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
parser.add_argument('--use_context', type=int, default=0, help='Whether or not to condition the model.')

parser.add_argument('--bn', type=int, default=0, help='Whether or not to adapt batchnorm statistics.')


# Data args
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'femnist', 'celeba'])
parser.add_argument('--data_dir', type=str, default='../data/')

# Data sampling
parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of classes')
parser.add_argument('--support_size', type=int, default=50, help='Support size: same as what we call batch size in the appendix')
parser.add_argument('--shuffle_train', type=int, default=1,
                    help='Only relevant when no clustered sampling = 0 \
                    and --uniform_over_groups 0')
parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                    help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

# meta batch sampling
parser.add_argument('--sampling_type', type=str, default='regular',
        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                    help='Sampling type')
parser.add_argument('--uniform_over_groups', type=int, default=1,
                    help='Sample groups uniformly. This is relevant when sampling_type == meta_batch_groups')
parser.add_argument('--eval_corners_only', type=int, default=1,
                    help='Are evaluating mixtures or corners?')

# Evalaution
parser.add_argument('--n_test_dists', type=int, default=30,
                    help='Number of test distributions to evaluate on. These are sampled uniformly.')
parser.add_argument('--n_test_per_dist', type=int, default=1000,
                    help='Number of examples to evaluate on per test distribution')
parser.add_argument('--crop_type', type=float, default=0)
# parser.add_argument('--crop_size_factor', type=float, default=1)

parser.add_argument('--target_resolution', type=int, default=224,
                    help='Resize image to this size before feeding in to model')
parser.add_argument('--target_name', type=str, nargs='+', default=['Blond_Hair'],
                    help='The y value we are trying to predict')
parser.add_argument('--confounder_names', type=str, nargs='+',
                    default=['Male'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')

# Logging
parser.add_argument('--seed', type=int, default=None, help='Seed')
parser.add_argument('--plot', type=int, default=0, help='Plot or not')
parser.add_argument('--experiment_name', type=str, default='debug')
parser.add_argument('--epochs_per_eval', type=int, default=1)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--log_wandb', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                    Best practice is to use this')

parser.add_argument('--step_size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
parser.add_argument('--num_steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')

args = parser.parse_args()

if 'group' in args.sampling_type and not args.eval_corners_only:
    raise ValueError

tags = ['supervised', f'{args.dataset}', f'use_context_{args.use_context}']

# Save folder
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_dir = Path('output') / 'checkpoints' / f'{args.experiment_name}_{args.seed}_{datetime_now}'
args.ckpt_dir = ckpt_dir

if args.debug:
    tags.append('debug')

if args.log_wandb:
    wandb.init(name=args.experiment_name,
               project=f"arm_{args.dataset}",
               tags=tags)
    wandb.config.update(args)

# For reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]

def main():

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # Make as reproducible as possible.
    # Please note that pytorch does not let us make things completely reproducible across machines.
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if args.seed is not None:
        print('setting seed', args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Get data
    train_loader, train_eval_loader, val_loader, _ = data.get_loaders(args)
    z_loader = data.get_z_loader(args)
    args.n_groups = train_loader.dataset.n_groups

    # Get model
    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset in 'celeba':
        num_classes = 4
    elif args.dataset == 'femnist':
        num_classes = 62
    model = utils.MetaConvModel(train_loader.dataset.image_shape[0], num_classes, hidden_size=128, feature_size=128)
    z_model = utils.MetaConvModel(train_loader.dataset.image_shape[0], train_loader.dataset.n_groups, hidden_size=128, feature_size=128)
#     model = utils.get_model(args, image_shape=train_loader.dataset.image_shape)

    # Loss Fn
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'adam': # This is used for MNIST.
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.learning_rate)
        
        z_optimizer = torch.optim.Adam(z_model.parameters(),
                                    lr=1e-3)
    elif args.optimizer == 'sgd':

        # From DRNN paper
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay)
        
        z_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, z_model.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay)

#     import ipdb; ipdb.set_trace()
    maml_model = MAML(model, z_model, z_loader, optimizer, z_optimizer, num_adaptation_steps=args.num_steps, step_size=args.step_size, loss_function=loss_fn, device=args.device)
        
    z_epochs = 50
    for epoch in trange(z_epochs):
            loss, accuracy = maml_model.train_z_iter(train_loader)
            print(epoch, 'epoch ,', loss, 'z_loss ,', accuracy, 'z_accuracy')
    
    # Train loop
    best_worst_case_acc = 0
    best_worst_case_acc_epoch = 0
    avg_val_acc = 0
    empirical_val_acc = 0

    for epoch in trange(args.num_epochs):
        
        train_results = maml_model.train(train_loader, 
                                         verbose=True, 
                                         desc='Training', 
                                         leave=False)


        # Decay learning rate after one epoch
        if args.use_lr_schedule:
            if (args.dataset == 'celeba' and epoch == 0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
                    
#         import ipdb; ipdb.set_trace()
        if epoch % args.epochs_per_eval == 0:

            # validation
            worst_case_acc, stats = maml_model.evaluate(val_loader, 
                                                        epoch=epoch,
                                                        log_wandb=args.log_wandb,
                                                        n_samples_per_dist=args.n_test_per_dist, 
                                                        split='val') 
            
            # Track early stopping values with respect to worst case.
            if worst_case_acc > best_worst_case_acc:
                best_worst_case_acc = worst_case_acc

                save_model(model, ckpt_dir, epoch, args.device)

            # Log early stopping values
            if args.log_wandb:
                wandb.log({"Train Loss": train_results['mean_outer_loss'],
                            "Best Worst Case Val Acc": best_worst_case_acc,
                           "Train Accuracy": train_results['accuracy_after'], "epoch": epoch})

            print(f"Epoch: ", epoch, "Worst Case Acc: ", worst_case_acc)

def save_model(model ,ckpt_dir, epoch, device):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{epoch}_weights.pkl'
    model_state = model.to('cpu').state_dict(),

    # Overwrite best_weights.pkl with the latest.
    torch.save(model_state, ckpt_path)
    ckpt_path = ckpt_dir / f'best_weights.pkl'
    torch.save(model_state, ckpt_path)
    model.to(device)


if __name__ == "__main__":
    main()