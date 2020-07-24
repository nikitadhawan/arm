import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()
      
    
        
class MAML(object):
    def __init__(self, model, z_model, z_loader, optimizer=None, 
                 z_optimizer=None, support_size=50, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        self.model = model.to(device=device)
        self.z_model = z_model.to(device=device)
        self.optimizer = optimizer
        self.z_optimizer = z_optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.support_size = support_size
        self.z_loader = z_loader
        

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])
                
    
    def train_z_iter(self, dataloader):
        if self.z_optimizer is None:
            raise RuntimeError('Trying to call `train_z`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        self.z_model.train()
        total_loss = 0
        total_accuracy = 0
        total_examples = 0
        for (images, labels, group_ids) in dataloader:

            self.z_optimizer.zero_grad()

            images = images.to(self.device)
            group_ids = group_ids.to(self.device)
            z_logits = self.z_model(images)
            z_loss = F.cross_entropy(z_logits, group_ids)
            
            z_preds = np.argmax(z_logits.detach().cpu().numpy(), axis=1)
            accuracy = np.sum(z_preds == group_ids.detach().cpu().numpy().reshape(-1))
            total_accuracy += accuracy * labels.shape[0]
            total_loss += z_loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

            z_loss.backward()
            self.z_optimizer.step()
        
        return total_loss / total_examples, total_accuracy / total_examples
            
    
    def get_z_dist(self, pred_z):
        group_count = []
        for group_id in self.z_loader.dataset.groups:
            ids = np.nonzero(pred_z == group_id)[0]
            group_count.append(len(ids))
        group_count = np.array(group_count)
        return group_count / np.sum(group_count)
    
    def get_outer_loss(self, images, labels, group_ids):
        batch_size, c, h, w = images.shape
        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        
        images = images.reshape((meta_batch_size, support_size, c, h, w))
        labels = labels.reshape((meta_batch_size, support_size))
        
        num_tasks = meta_batch_size 
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
        }
        
        mean_outer_loss = torch.tensor(0., device=self.device)
        self.z_model.eval()
        for task_id in range(num_tasks):
            inputs = images[task_id]
            targets = labels[task_id]
            
#             import ipdb; ipdb.set_trace()
            pred_z_logits = self.z_model(inputs)
            pred_z = np.argmax(pred_z_logits.detach().cpu().numpy(), axis=1)
            z_dist = self.get_z_dist(pred_z)
            z_dist = np.tile(z_dist, (len(self.z_loader), 1, 1))
            self.z_loader.batch_sampler.set_group_sub_dists(z_dist)
            
            recalled_inputs, recalled_targets, recalled_group_ids = next(iter(self.z_loader))
            recalled_inputs = recalled_inputs.to(self.device)
            recalled_targets = recalled_targets.to(self.device)
            params, adaptation_results = self.adapt(recalled_inputs, recalled_targets, 
                                                    num_adaptation_steps=self.num_adaptation_steps, 
                                                    step_size=self.step_size, first_order=self.first_order)
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['accuracies_before'][task_id] = adaptation_results['accuracy_before']
            
            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(inputs, params=params)
                outer_loss = self.loss_function(test_logits, targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss
               
            results['accuracies_after'][task_id] = compute_accuracy(test_logits, targets)
            
        
        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results
    
    def adapt(self, inputs, targets, num_adaptation_steps=1, step_size=0.1, first_order=False):
        params = None
        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if step == 0:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                step_size=step_size, params=params,
                first_order=(not self.model.training) or first_order)

        return params, results
    
    def train(self, dataloader, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(
                        mean_accuracy)
                pbar.set_postfix(**postfix)
                
        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    def train_iter(self, dataloader):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        for (images, labels, group_ids) in dataloader:

            if self.scheduler is not None:
                self.scheduler.step(epoch=num_batches)

            self.optimizer.zero_grad()

            images = images.to(self.device)
            labels = labels.to(self.device)
            outer_loss, results = self.get_outer_loss(images, labels, group_ids)
            yield results

            outer_loss.backward()
            self.optimizer.step()

            num_batches += 1
       
    def evaluate(self, loader, n_samples_per_dist=None, split='val', epoch=None, log_wandb=False, verbose=True, **kwargs):
        accuracies = np.zeros(len(loader.dataset.groups))
        num_examples = []
        groups = []
        
        for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
            dist_id = group
            example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
            example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

            # Create batches
            batches = []
            X, Y, G = [], [], []
            counter = 0
            for i, idx in enumerate(example_ids):
                x, y, g = loader.dataset[idx]
                X.append(x); Y.append(y); G.append(g)
                if (i + 1) % self.support_size == 0:
                    X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
                    batches.append((X, Y, G))
                    X, Y, G = [], [], []

                if i == (n_samples_per_dist - 1):
                    break
            if X:
                X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
                batches.append((X, Y, G))
                
            mean_outer_loss, mean_accuracy, count = 0., 0., 0
            count_examples = 0
            with tqdm(disable=not verbose, **kwargs) as pbar:
                for results, num in self.evaluate_iter(batches):
                    count_examples += num
                    pbar.update(1)
                    count += 1
                    mean_outer_loss += (results['mean_outer_loss']
                        - mean_outer_loss) / count
                    postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                    if 'accuracies_after' in results:
                        mean_accuracy += (np.mean(results['accuracies_after'])
                            - mean_accuracy) / count
                        postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                    pbar.set_postfix(**postfix)
            
            accuracies[dist_id] = mean_accuracy
            groups.append(dist_id)
            num_examples.append(count_examples)

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

        if log_wandb:
            wandb.log(stats)

        return worst_case_acc, stats

    def evaluate_iter(self, batches):
        num_batches = 0
        self.model.eval()
        for (images, labels, group_ids) in batches:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, results = self.get_outer_loss(images, labels, group_ids)
            yield results, labels.shape[0]
            num_batches += 1
    
    
    
    

