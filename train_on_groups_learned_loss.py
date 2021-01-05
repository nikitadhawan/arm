import os
from datetime import datetime
import argparse
from pathlib import Path
import random

import higher
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange, tqdm
import wandb
from sklearn import metrics
from dro_loss import LossComputer
import models

import data
import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# Arguments
parser = argparse.ArgumentParser()

# Training / Optimization args
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--drop_last', type=int, default=0)

parser.add_argument('--clustering_filename', type=str, default='')
parser.add_argument('--learned_groups', type=int, default=0)

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--n_inner_iter', type=int, default=1)
parser.add_argument('--inner_lr', type=float, default=0.1)
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
parser.add_argument('--bn', type=int, default=0, help='Whether or not to adapt batchnorm statistics.')
parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
# If model is Convnet
parser.add_argument('--prediction_net', type=str, default='convnet',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'])

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
parser.add_argument('--use_context', type=int, default=0, help='Whether or not to condition the model.')


# Data args
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'femnist', 'cifar', 'tinyimagenet'])
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


args.learned_loss = True

# For reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]

def count_parameters(model): # copied from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    args.n_groups = train_loader.dataset.n_groups

    # Get model
    model = utils.get_model(args, image_shape=train_loader.dataset.image_shape)
    model = model.to(args.device)

    print(" Model, n parameters: ", count_parameters(model))

    # Learned loss fn

    if args.dataset in ('mnist', 'cifar'):
        num_classes = 10
    elif args.dataset in ('tinyimagenet'):
        num_classes = 200
    elif args.dataset in ('celeba'):
        num_classes = 2
    elif args.dataset == 'femnist':
        num_classes = 62

    learned_loss = models.LearnedLoss(in_size=num_classes).to(args.device)
    print(" Learned loss, n parameters: ", count_parameters(learned_loss))

    # Loss Fn
    if args.use_robust_loss:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_computer = LossComputer(loss_fn, is_robust=True,
                                     dataset=train_loader.dataset,
                                     step_size=args.robust_step_size,
                                     device=args.device,
                                     args=args)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    params = list(model.parameters()) + list(learned_loss.parameters())
    if args.optimizer == 'adam': # This is used for MNIST
        optimizer = torch.optim.Adam(params,
                                    lr=args.learning_rate)
    elif args.optimizer == 'sgd':

        # From DRNN paper
        optimizer = torch.optim.SGD(
            #filter(lambda p: p.requires_grad, model.parameters()),
            params,
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay)

    inner_opt = torch.optim.SGD(model.parameters(),
                            lr=args.inner_lr)

    # Train loop
    best_worst_case_acc = 0
    best_worst_case_acc_epoch = 0
    avg_val_acc = 0
    empirical_val_acc = 0

    for epoch in trange(args.num_epochs):
        total_loss = 0
        total_accuracy = 0
        total_examples = 0

        model.train()

        for batch_id, (images, labels, group_ids) in enumerate(tqdm(train_loader, desc='train loop')):

#             import ipdb; ipdb.set_trace()
            # Put on GPU
            images = images.to(args.device)
            labels = labels.to(args.device)

            n_tasks = int(len(images) / args.support_size)

            for task_id in range(n_tasks):

                # Select input and output for task
                start = task_id*args.support_size
                end = start + args.support_size
                task_images = images[start:end]
                task_labels = labels[start:end]

                with higher.innerloop_ctx(
                    model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):

                    # Inner loop adaptation
                    for _ in range(args.n_inner_iter):
                        spt_logits = fnet(task_images)
                        spt_loss = learned_loss(spt_logits)
                        diffopt.step(spt_loss)

                    # Forward
                    logits = fnet(task_images)

                    # Outer loop loss fn
                    if args.use_robust_loss:
                        group_ids = group_ids.to(args.device)
                        loss = loss_computer.loss(logits, task_labels, group_ids,
                                                  is_training=True)

                    else:
                        loss = loss_fn(logits, task_labels)
                    loss.backward()

                    # Evaluate
                    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                    accuracy = np.mean(preds == task_labels.detach().cpu().numpy().reshape(-1))
                    total_accuracy += accuracy * task_labels.shape[0]
                    total_loss += loss.item() * task_labels.shape[0]
                    total_examples += task_labels.shape[0]

            # Backprop
            optimizer.step()
            optimizer.zero_grad()


        if epoch % args.epochs_per_eval == 0:

            worst_case_acc, stats = evaluate_groups_with_learned_loss(args, model, val_loader, epoch, learned_loss, inner_opt, split='val') # validation

            # Track early stopping values with respect to worst case.
            if worst_case_acc > best_worst_case_acc:
                best_worst_case_acc = worst_case_acc

            save_model(model, learned_loss, ckpt_dir, epoch, args.device)

            # Log early stopping values
            if args.log_wandb:
                wandb.log({"Train Loss": total_loss / total_examples,
                            "Best Worst Case Val Acc": best_worst_case_acc,
                           "Train Accuracy": total_accuracy / total_examples, "epoch": epoch})

            print(f"Epoch: ", epoch, "Worst Case Acc: ", worst_case_acc)
        save_model(model, learned_loss, ckpt_dir, epoch, args.device)


def evaluate_groups_with_learned_loss(args, model, loader, epoch=None, learned_loss=None, inner_opt=None, split='val',  n_samples_per_dist=None):
    """ Test model on groups and log to wandb
        Separate script for femnist for speed."""

    groups = []
    num_examples = []
    accuracies = np.zeros(len(loader.dataset.groups))

    model.train()

    if n_samples_per_dist is None:
        n_samples_per_dist = args.n_test_per_dist

    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
        dist_id = group

        preds_all = []
        labels_all = []

        example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
        example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

        # Create batches
        batches = []
        X, Y, G = [], [], []
        counter = 0
        for i, idx in enumerate(example_ids):
            x, y, g = loader.dataset[idx]
            X.append(x); Y.append(y); G.append(g)
            if (i + 1) % args.support_size == 0:
                X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
                batches.append((X, Y, G))
                X, Y, G = [], [], []

            if i == (n_samples_per_dist - 1):
                break
        if X:
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))

        counter = 0
        for images, labels, group_id in batches:

            labels = labels.detach().numpy()
            images = images.to(args.device)

            task_labels = labels
            task_images = images

            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=True) as (fnet, diffopt):

                # Inner loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(task_images)
                    spt_loss = learned_loss(spt_logits)
                    diffopt.step(spt_loss)

                logits = fnet(task_images)
                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)

            preds_all.append(preds)
            labels_all.append(task_labels)
            counter += len(task_images)

            if counter >= n_samples_per_dist:
                break

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)

        # Evaluate
        accuracy = np.mean(preds_all == labels_all)

        num_examples.append(len(preds_all))
        accuracies[dist_id] = accuracy
        groups.append(dist_id)

        if args.log_wandb:
            if epoch is None:
                wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})
            else:
                wandb.log({f"{split}_acc_e{epoch}": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})

    # Log worst, average and empirical accuracy
    worst_case_acc = np.amin(accuracies)
    worst_case_group_size = num_examples[np.argmin(accuracies)]

    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()
    empirical_case_acc = accuracies.dot(props)
    average_case_acc = np.mean(accuracies)

    total_size = num_examples.sum()

    stats = {f'worst_case_{split}_acc': worst_case_acc,
            f'worst_case_group_size_{split}': worst_case_group_size,
            f'average_{split}_acc': average_case_acc,
            f'total_size_{split}': total_size,
            f'empirical_{split}_acc': empirical_case_acc}

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return worst_case_acc, stats

def save_model(model, learned_loss, ckpt_dir, epoch, device):

    # Learned loss
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{epoch}_weights.pkl'
    model_state = model.to('cpu').state_dict(),
    # torch.save(model_state, ckpt_path)

    # Overwriste best_weights.pkl with the latest.
    ckpt_path = ckpt_dir / f'best_weights.pkl'
    torch.save(model_state, ckpt_path)
    model.to(device)

    ## Learned loss
    ckpt_path = ckpt_dir / f'{epoch}_learned_loss_weights.pkl'
    learned_loss_state = learned_loss.to('cpu').state_dict(),
    # torch.save(learned_loss_state, ckpt_path)

    # Overwriste learned_best_weights.pkl with the latest.
    ckpt_path = ckpt_dir / f'best_learned_loss_weights.pkl'
    torch.save(learned_loss_state, ckpt_path)
    learned_loss.to(device)


if __name__ == "__main__":
    main()
