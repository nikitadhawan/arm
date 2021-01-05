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
import models
import higher

import data
import utils


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
                if len(logits.shape) == 1:
                    logits = logits.unsqueeze(0)
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


def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]

def test(args, eval_on):

    if args.log_wandb:
        wandb.init(name=args.experiment_name,
                   project=f"{args.dataset}_test",
                   reinit=True
                   )
        wandb.config.update(args, allow_val_change=True)

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # Make reproducible
    if args.seed is not None:
        print('setting seed', args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Get data
    train_loader, train_eval_loader, val_loader, test_loader = data.get_loaders(args)
    args.n_groups = train_loader.dataset.n_groups

    # Get model
    model = utils.get_model(args, image_shape=train_loader.dataset.image_shape)
    state_dict = torch.load(args.ckpt_path)[0]
    model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.train()

    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset in ('celeba'):
        num_classes = 2
    elif args.dataset == 'femnist':
        num_classes = 62
    elif args.dataset == 'cifar':
        num_classes = 10
    elif args.dataset == 'tinyimagenet':
        num_classes = 200

    # Get learned loss
    learned_loss = models.LearnedLoss(in_size=num_classes).to(args.device)
    state_dict = torch.load(args.ckpt_path_learned_loss)[0]
    learned_loss.load_state_dict(state_dict)
    learned_loss.to(args.device)


    inner_opt = torch.optim.SGD(model.parameters(),
                            lr=1e-1) # TODO Undo hardcoding

    if eval_on == 'train':
        worst_case_acc, stats = evaluate_groups_with_learned_loss(args, model, train_eval_loader,
                learned_loss=learned_loss, inner_opt=inner_opt, split='train')
    elif eval_on == 'val':
        worst_case_acc, stats = evaluate_groups_with_learned_loss(args, model, val_loader,
                learned_loss=learned_loss, inner_opt=inner_opt, split='val')
    elif eval_on == 'test':
        worst_case_acc, stats = evaluate_groups_with_learned_loss(args, model, test_loader,
                learned_loss=learned_loss, inner_opt=inner_opt, split='test')

    return worst_case_acc, stats

#################
### Arguments ###
#################

parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--model', type=str, default='ContextualConvNet')

parser.add_argument('--drop_last', type=int, default=0)
parser.add_argument('--ckpt_path', type=str, default=None)

parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
# If model is Convnet
parser.add_argument('--prediction_net', type=str, default='convnet',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'])

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
parser.add_argument('--use_context', type=int, default=0)
parser.add_argument('--bn', type=int, default=0, help='Whether or not to adapt batchnorm statistics.')

# Data args
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist','celeba','femnist', 'cifar', 'tinyimagenet'])
parser.add_argument('--data_dir', type=str, default='../data/')

# CelebA Data
parser.add_argument('--target_resolution', type=int, default=224,
                    help='Resize image to this size before feeding in to model')
parser.add_argument('--target_name', type=str, default='Blond_Hair',
                    help='The y value we are trying to predict')
parser.add_argument('--confounder_names', type=str, nargs='+',
                    default=['Male'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')
parser.add_argument('--eval_on', type=str, nargs='+',
                    default=['test'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')

# Data sampling
parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of classes')
parser.add_argument('--support_size', type=int, default=50, help='Support size. What we call batch size in the appendix.')
parser.add_argument('--shuffle_train', type=int, default=1,
                    help='Only relevant when do_clustered_sampling = 0 \
                    and --uniform_over_groups 0')

# Clustered sampling
parser.add_argument('--sampling_type', type=str, default='regular',
        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                    help='Sampling type')
parser.add_argument('--eval_corners_only', type=int, default=1,
                    help='Are evaluating mixtures or corners?')

parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                    help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

# Evalaution
parser.add_argument('--n_test_dists', type=int, default=100,
                    help='Number of test distributions to evaluate on. These are sampled uniformly.')
parser.add_argument('--n_test_per_dist', type=int, default=3000,
                    help='Number of examples to evaluate on per test distribution')

# Logging
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--log_wandb', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                    Best practice is to use this')

parser.add_argument('--crop_type', type=int, default=0)
parser.add_argument('--crop_size_factor', type=float, default=1)

parser.add_argument('--ckpt_folders', type=str, nargs='+')

parser.add_argument('--context_net', type=str, default='convnet')

parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--n_inner_iter', type=int, default=1)


args = parser.parse_args()


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.learned_loss = True


if __name__ == "__main__":

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = ckpt_folder + f'best_weights.pkl'
        state_dict = torch.load(ckpt_path)

        ckpt_path_learned_loss = ckpt_folder + f'best_learned_loss_weights.pkl'
        state_dict = torch.load(ckpt_path_learned_loss)

        print("Found: ", ckpt_path)

    all_test_stats = [] # Store all test results in list.

    for i, ckpt_folder in enumerate(args.ckpt_folders):

        args.seed = i + 10 # Mainly to make sure seed for training and testing is not the same. Not critical.
        args.ckpt_path = ckpt_folder + f'best_weights.pkl'
        args.ckpt_path_learned_loss = ckpt_folder + f'best_learned_loss_weights.pkl'
        args.experiment_name += f'_{args.seed}'

        if args.log_wandb and i != 0:
            wandb.join() # Initializes new wandb run

        train_stats, val_stats, test_stats = None, None, None

        if 'train' in args.eval_on:
            _, train_stats = test(args, eval_on='train')
        if 'val' in args.eval_on:
            _, val_stats = test(args, eval_on='val')
        if 'test' in args.eval_on:
            _, test_stats = test(args, eval_on='test')
            all_test_stats.append((i, test_stats))

        print("----SEED used when evaluating -----: ", args.seed)
        print("----CKPT FOLDER -----: ", ckpt_folder)
        print("TRAIN STATS:\n ", train_stats)
        print('-----------')

        print("VAL STATS: \n ", val_stats)
        print('-----------')

        print("TEST STATS: \n ", test_stats)

    print("All test stats: ", all_test_stats)
