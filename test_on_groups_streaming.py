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
import data
import utils
import scipy as sp

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
#         img = img.cpu().detach().numpy()
#         img = np.array(sp.ndimage.rotate(img, label*45, reshape=False, order=0))
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
#         img = torch.tensor(img).to(args.device)
        images.append(img.unsqueeze(0))
    return torch.cat(images)

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
                            lr=1e-4)

    if eval_on == 'train':
        worst_case_acc, stats = evaluate_groups(args, model, train_eval_loader, learned_loss=learned_loss, inner_opt=inner_opt, split='train')
    elif eval_on == 'val':
        worst_case_acc, stats = evaluate_groups(args, model, val_loader, learned_loss=learned_loss, inner_opt=inner_opt, split='val')
    elif eval_on == 'test':
        worst_case_acc, stats = evaluate_groups(args, model, test_loader, learned_loss=learned_loss, inner_opt=inner_opt, split='test')

    return worst_case_acc, stats



def evaluate_groups(args, model, loader, epoch=None, learned_loss=None, inner_opt=None, split='val', n_samples_per_dist=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    num_examples = []
    accuracies = np.zeros(len(loader.dataset.groups))

    model.train()

    if n_samples_per_dist is None:
        n_samples_per_dist = args.n_test_per_dist



    N = args.N

    n_groups = len(loader.dataset.groups)
    correct = np.zeros((n_groups, N, args.n_test_per_dist))

    # scheduler = torch.optim.lr_scheduler.StepLR(inner_opt, step_size=0.01)

    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):



        for j in range(N):
            args.seed = j
            # torch.manual_seed(args.seed)
            # if args.cuda:
            #     torch.cuda.manual_seed(args.seed)
#             np.random.seed(j)
            # random.seed(args.seed)


#             args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl' # final_weights.pkl
#             args.ckpt_path_learned_loss = Path('output') / 'checkpoints' / ckpt_folder / f'best_learned_loss_weights.pkl'
#             model = utils.get_model(args, image_shape=loader.dataset.image_shape)
#             state_dict = torch.load(args.ckpt_path)[0]
#             model.load_state_dict(state_dict)
#             model = model.to(args.device)
# #             model.train()
# #             learned_loss = models.LearnedLoss(in_size=10).to(args.device)
# #             state_dict = torch.load(args.ckpt_path_learned_loss)[0]
# #             learned_loss.load_state_dict(state_dict)
# #             learned_loss.to(args.device)


#             inner_opt = torch.optim.SGD(model.parameters(),
#                                     lr=1e-1)

            # model.reset()
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
            acc_images = []

            for images, labels, group_id in batches:
#                 acc_images.append(images)
#                 r_images = torch.cat(acc_images)
                model.train()
#                 r_images = rotate_batch_with_labels(images.repeat((10, 1, 1, 1)), torch.randint(2, (10,), dtype=torch.long))
#                 r_images = r_images.to(args.device)
                images = images.to(args.device)
#                 r_images = r_images.to(args.device)
                labels = labels.detach().numpy()
                for _ in range(1):
                    inner_opt.zero_grad()
#                     r_images = rotate_batch_with_labels(images.repeat((10, 1, 1, 1)), torch.randint(1, (10,), dtype=torch.long))
                    spt_logits = model(images)
                    spt_loss = learned_loss(spt_logits)
                    spt_loss.backward()
                    inner_opt.step()
                    # scheduler.step()

                model.eval()
                logits = model(images)
                if len(logits.shape) == 1:
                    logits = logits.unsqueeze(0)

                logits = logits.detach().cpu().numpy()
                preds = np.argmax(logits, axis=1)

                preds_all.append(preds)
                labels_all.append(labels)
                counter += len(images)

                if counter >= n_samples_per_dist:
                    break
#                 if counter%100 == 0:
#                     acc_images = []
              
                   
            # import ipdb; ipdb.set_trace()
            preds_all = np.concatenate(preds_all)
            labels_all = np.concatenate(labels_all)

            # j refers to seed
            is_correct = preds_all == labels_all
            correct[group, j, :] = is_correct

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

    with open(f'{args.matrix_save_name}.npy', 'wb') as f:
        np.save(f, correct)


    # group, seed, test_point
    print("worst case acc:",  np.min(np.mean(correct, axis=(1,2))))
    print(np.argmin(np.mean(correct, axis=(1,2))))
    print("avg case acc:",  np.mean(np.mean(correct, axis=(1,2))))

    #worst_case_group_size = num_examples[np.argmin(accuracies)]

    #num_examples = np.array(num_examples)
    #props = num_examples / num_examples.sum()
    #empirical_case_acc = accuracies.dot(props)
    #average_case_acc = np.mean(accuracies)

    #total_size = num_examples.sum()

    stats = {}
    #stats = {f'worst_case_{split}_acc': worst_case_acc,
    #        f'worst_case_group_size_{split}': worst_case_group_size,
    #        f'average_{split}_acc': average_case_acc,
    #        f'total_size_{split}': total_size,
    #        f'empirical_{split}_acc': empirical_case_acc}

    #if epoch is not None:
    #    stats['epoch'] = epoch

    #if args.log_wandb:
    #    wandb.log(stats)

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
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'mnist_unknown', 'celeba','femnist', 'tinyimagenet'])
parser.add_argument('--mnist_type', type=str, default='rotation')

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
parser.add_argument('--meta_batch_size', type=int, default=1, help='Number of classes')
parser.add_argument('--support_size', type=int, default=1, help='Support size. What we call batch size in the appendix.')
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
parser.add_argument('--n_test_per_dist', type=int, default=1000,
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
parser.add_argument('--streaming', type=int, default=1)

parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--matrix_save_name', type=str, default='streaming')


args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl'
        state_dict = torch.load(ckpt_path)
        print("Found: ", ckpt_path)

    all_test_stats = [] # Store all test results in list.

    if args.use_context:
        args.small_model = True
    else:
        args.small_model = False

    worst_case_accs = []
    avg_case_accs = []
    for i, ckpt_folder in enumerate(args.ckpt_folders):

        args.seed = i + 10 # Mainly to make sure seed for training and testing is not the same. Not critical.
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl' # final_weights.pkl
        args.ckpt_path_learned_loss = Path('output') / 'checkpoints' / ckpt_folder / f'best_learned_loss_weights.pkl'
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


        worst_case_accs.append(test_stats['worst_case_test_acc'])
        avg_case_accs.append(test_stats['average_test_acc'])

        print("----SEED used when evaluating -----: ", args.seed)
        print("----CKPT FOLDER -----: ", ckpt_folder)
        print("TRAIN STATS:\n ", train_stats)
        print('-----------')

        print("VAL STATS: \n ", val_stats)
        print('-----------')

        print("TEST STATS: \n ", test_stats)

    print("All test stats: ", all_test_stats)

    print("------- \n \n ----")

    print(f"Worst case: {np.mean(worst_case_accs)}, std: {np.std(worst_case_accs) / np.sqrt(2)}")
    print(f"Avg case: {np.mean(avg_case_accs)}, std: {np.std(avg_case_accs) / np.sqrt(2)}")



