from tqdm import tqdm
import numpy as np
import wandb
import torch


##########################
### Evaluate on groups ###
##########################

def evaluate_groups(args, model, loader, epoch=None, split='val', n_samples_per_dist=None, streaming=False):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    num_examples = []
    accuracies = np.zeros(len(loader.dataset.groups))

    if args.bn:
        model.train()
    else:
        model.eval()

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
#         all_images, all_labels = [], []
        for images, labels, group_id in batches:

            if streaming:
                all_images.append(images)
                all_labels.append(labels)
                images = torch.cat(all_images)
                labels = torch.cat(all_labels)
            
            labels = labels.detach().numpy()
            images = images.to(args.device)

            if args.bn:
                logits = model(images)
            else:
                logits = model(images)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

#             if counter == 0:
#                 import matplotlib
#                 matplotlib.use("Agg")
#                 import matplotlib.pyplot as plt
#                 pl = activations.detach().cpu().numpy()[1, :, 0]
#                 pts = np.random.normal(0,1,size=len(pl))
#                 plt.figure()
# #                 plt.ylim(top=100)
#                 plt.hist(pts, 100, alpha=0.5)
#                 plt.hist(pl, 100, alpha=0.5)
#                 plt.savefig('two.png')
#                 plt.close()
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            preds_all.append(preds)
            labels_all.append(labels)
            counter += len(images)

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

    return average_case_acc, stats


##########################################
### Evalauate on mixture distributions ###
##########################################


def eval_dist(dist, model, loader, args, n_samples_per_dist):
    """ Evaluates model on a specific sub distribution """
    preds_all = []
    labels_all = []

    # Set the correct test distribution
    loader.sampler.set_sub_dist(dist)

    counter = 0
    for images, labels, group_id in loader:

        labels = labels.detach().numpy()
        images = images.to(args.device)

        logits = model(images).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        preds_all.append(preds)
        labels_all.append(labels)
        counter += len(images)

        if counter >= n_samples_per_dist:
            break

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    return preds_all, labels_all


def eval_dists_fn(split, args, model, loader, dists, n_samples_per_dist=1500):
    """ Evaluates modl on a set of mixture distributions """
    accuracies = np.zeros(len(dists))
    num_examples = []
    counter = 0
    for dist_id, dist in enumerate(tqdm(dists, f"{split}_evaluation")):

        preds_all, labels_all = eval_dist(dist, model, loader, args, n_samples_per_dist)
        num_examples.append(len(preds_all))
        accuracies[dist_id] = np.mean(preds_all == labels_all)
        counter += len(labels_all)
        
    for a in zip(accuracies, num_examples):
        print(a)

    return accuracies, num_examples[np.argmin(accuracies)]


def evaluate_each_corner(args, model, loader, split):
    """
        Assumes that loader returns deterministically all examples in the split in order.
        Note: There is no binning here.

    """

    preds_all = []
    labels_all = []
    group_ids_all = []


    counter = 0
    for images, labels, group_id in loader:

        labels = labels.detach().numpy()
        images = images.to(args.device)

        logits = model(images).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        preds_all.append(preds)
        labels_all.append(labels)
        group_ids_all.append(group_id.numpy())
        counter += len(images)

    # Empirical
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    group_ids_all = np.concatenate(group_ids_all)

    is_correct = preds_all == labels_all

    empirical_acc = np.mean(is_correct)

    group1_acc = np.mean(is_correct[group_ids_all == 0])
    group2_acc = np.mean(is_correct[group_ids_all == 1])
    group3_acc = np.mean(is_correct[group_ids_all == 2])
    group4_acc = np.mean(is_correct[group_ids_all == 3])

    worst_case_acc = np.amin([group1_acc, group2_acc, group3_acc, group4_acc])
    avg_case_acc = np.mean([group1_acc, group2_acc, group3_acc, group4_acc])

    stats = {f"worst_case_{split}_all_acc": worst_case_acc,
               f"avg_acc_{split}": avg_case_acc,
               f"empirical_{split}_acc": empirical_acc}

    ### Log wandb ###
    if args.log_wandb:
        for dist_id, accuracy in enumerate([group1_acc, group2_acc, group3_acc, group4_acc]):
            wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                       "dist_id": dist_id})

        wandb.log(stats)

    return stats

def evaluate_mixtures(args, model, loader, dists, epoch=None, split='val'):
    """ Test model on a set of sub distributions (dists) and log to WandB"""

    groups = []
    dfs = []
    num_examples = []

    model.eval()

    n_samples_per_dist = args.n_test_per_dist


    if args.binning:

        corner_dists, groups, empirical_dist = dists
        accuracies_corners = eval_dists_fn(split, args, model, loader, corner_dists, n_samples_per_dist=n_samples_per_dist)

        group_avg_accuracies = np.zeros(len(groups))
        group_avg_ses = np.zeros(len(groups))
        for group in groups:

            group_dists = groups[group] # Sub dists within group
            accs_for_group = eval_dists_fn(split, args, model, loader, group_dists, n_samples_per_dist=n_samples_per_dist)

            avg_group_acc = np.mean(accs_for_group)
            group_acc_se = np.std(accs_for_group) / np.sqrt(len(accs_for_group))

            group_avg_accuracies[group] = avg_group_acc
            group_avg_ses[group] = group_acc_se


        preds_all_emp, labels_all_emp = eval_dist(empirical_dist, model, loader, args, n_samples_per_dist)
        empirical_case_acc = np.mean(preds_all_emp == labels_all_emp)

        accuracies = np.concatenate([accuracies_corners, group_avg_accuracies])

        ### Log wandb ###
        if args.log_wandb:
            for dist_id, accuracy in enumerate(accuracies):
                if epoch is None:
                    wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                               "dist_id": dist_id})
                else:
                    wandb.log({f"{split}_acc_e_{epoch}": accuracy, # Gives us Acc vs Group Id
                               "dist_id": dist_id})


            # Track the standard deviations also for the estimate of the avg acc for the group.
            for i, std in enumerate(group_avg_ses):
                if epoch is None:
                    wandb.log({f'{split}_std': std,
                                "dist_id_of_group": i})
                else:
                    wandb.log({f'{split}_std_e_{epoch}': std,
                                "dist_id_of_group": i})

        # Log worst, average and empirical accuracy
        worst_case_acc = np.amin(accuracies)
        worst_case_subset = np.amin(group_avg_accuracies)
        avg_acc_including_corners = np.mean(accuracies)
        avg_acc_excluding_corners = np.mean(group_avg_accuracies) # Avg acc across bins.

        # To calculate the average acc, we need to weight the groups according to their size in the probability simplex.
        weighting = np.array([1/8, 1/8, 1/8, 1/8, 1/2])
        avg_acc = np.sum(weighting * group_avg_accuracies)

        stats = {f"worst_case_{split}_all_acc": worst_case_acc,
                   f"avg_acc_including_corners_{split}": avg_acc_including_corners, # This is treating all bins equally
                   f"avg_acc_excluding_corners_{split}": avg_acc_excluding_corners, # this is treating all bins equally
                   f"worst_case_{split}_sub_acc": worst_case_subset,
                   f"empirical_{split}_acc": empirical_case_acc,
                   f"avg_{split}_acc": avg_acc} # This is the avg acc in the paper. The bins are weighted according to size.

        if epoch is None:
            stats['epoch'] = epoch

    else:
        accuracies, worst_case_group_size = eval_dists_fn(split, args, model, loader, dists, n_samples_per_dist=n_samples_per_dist)


        ### Log wandb ###
        if args.log_wandb:
            for dist_id, accuracy in enumerate(accuracies):
                if epoch is not None:
                    wandb.log({f"{split}_acc_e_{epoch}": accuracy, # Gives us Acc vs Group Id
                               "dist_id": dist_id})
                else:
                    wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                               "dist_id": dist_id})

        # Log worst, average and empirical accuracy
        worst_case_acc = np.amin(accuracies)
        if args.n_test_dists == 0: # This is for when running ERM and DRO (no context).
            avg_acc = np.mean(accuracies[:-1]) # Skip the empirical.
        else:
            avg_acc = np.mean(accuracies[args.n_groups:-1]) # Skip the corner dists and empirical
        empirical_case_acc = accuracies[-1]
        stats = {f"worst_case_{split}_all_acc": worst_case_acc,
                   f"avg_{split}_acc": avg_acc,
                   f'worst_case_group_size_{split}': worst_case_group_size,
                   f"empirical_{split}_acc": empirical_case_acc}

        if args.n_test_dists > 0:
            worst_case_subset = np.min(accuracies[args.n_groups:-1]) # Skip the corner dists and empirical
            stats[f'worst_case_{split}_sub_acc'] =  worst_case_subset

        if epoch is not None:
            stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return worst_case_acc, avg_acc, empirical_case_acc, stats



def evaluate_groups_large_dataset(args, model, loader, epoch=None, split='val', n_samples_per_dist=None):
    """ Test model on groups and log to wandb

        Separate script for celeba and larger datasets for speed."""

    groups = []
    num_examples = []
    accuracies = np.zeros(len(loader.dataset.groups))

    if args.bn:
        model.train()
    else:
        model.eval()

    if n_samples_per_dist is None:
        n_samples_per_dist = args.n_test_per_dist

    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
        dist_id = group

        preds_all = []
        labels_all = []

        counter = 0
        if loader.dataset.group_counts[group] == 0:
            num_examples.append(0)
            accuracies[dist_id] = 0
            groups.append(group)
            continue;

        loader.batch_sampler.set_group(group)
        for images, labels, group_id in loader:

            labels = labels.detach().numpy()
            images = images.to(args.device)

            logits = model(images).detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            preds_all.append(preds)
            labels_all.append(labels)
            counter += len(images)

            if counter >= n_samples_per_dist:
                break

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)

        # Evaluate
        accuracy = np.mean(preds_all == labels_all)

        num_examples.append(len(preds_all))
        accuracies[dist_id] = accuracy
        groups.append(group)

        if args.log_wandb:
            if epoch is None:
                wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                           "dist_id": group})
            else:
                wandb.log({f"{split}_acc_e{epoch}": accuracy, # Gives us Acc vs Group Id
                           "dist_id": group})

#     import ipdb; ipdb.set_trace()
#     accuracies = accuracies[accuracies!=0]
#     num_examples = np.array(num_examples)
#     num_examples = num_examples[num_examples!=0]
    # Log worst, average and empirical accuracy
    worst_case_acc = np.amin(accuracies)
    worst_case_group_size = num_examples[np.argmin(accuracies)]

    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()
    empirical_case_acc = accuracies.dot(props)
    average_case_acc = np.mean(accuracies)

    total_size = num_examples.sum()
    
    for a in zip(accuracies, num_examples):
        print(a)

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
