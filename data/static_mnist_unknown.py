import torchvision.transforms.functional as TF
import numpy as np
import scipy as sp
import pandas as pd
import colorsys

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
import torch
from tabulate import tabulate
from tensorflow import keras
from pathlib import Path

# Rotation config
CONFIGS = {}
config = {}
config['group_type'] = 'rotation'
n_groups = 14
config['n_groups'] = n_groups
config['group_values'] = np.array(range(n_groups)) * 10
config['group_probs'] = np.zeros(n_groups)
config['group_probs'][:3] = 70/100 # 0 - 20
config['group_probs'][3:6] = 20/100 # 30 - 50
config['group_probs'][6:9] = 6/100 # 60 - 80
config['group_probs'][9:12] = 3/100 # 90 - 110
config['group_probs'][12:] = 1/100

CONFIGS['rotation'] = config.copy()

## Color config
#config = {}
#config['group_type'] = 'color_hue'
#n_groups = 8
#config['n_groups'] = n_groups
#config['group_values'] = np.array(range(n_groups)) * 24
#config['group_probs'] = np.zeros(15)
#config['group_probs'][:3] = 60/100 # 0 - 20
#config['group_probs'][3:6] = 25/100 # 30 - 50
#config['group_probs'][6:9] = 10/100 # 60 - 80
#config['group_probs'][9:12] = 5/100 # 90 - 110
#config['group_probs'][12:] = 0
#
#CONFIGS['color_hue'] = config.copy()
#
### Color config
##config = {}
##config['group_type'] = 'color_hue'
##n_groups = 15
##config['n_groups'] = n_groups
##config['group_values'] = np.array(range(n_groups)) * 24
##config['group_probs'] = np.zeros(15)
##config['group_probs'][:3] = 72/100 # 0 - 20
##config['group_probs'][3:6] = 20/100 # 30 - 50
##config['group_probs'][6:9] = 4/100 # 60 - 80
##config['group_probs'][9:12] = 4/100 # 90 - 110
##config['group_probs'][12:] = 0
##
##CONFIGS['color_hue'] = config.copy()
#
##  rotation config
#config = {}
#config['group_type'] = 'rotation2'
#n_groups = 8
#config['n_groups'] = n_groups
#config['group_values'] = np.array(range(n_groups)) * 20
##config['group_probs'] = np.zeros(n_groups)
#config['group_probs'] = np.ones(n_groups) * 1 / n_groups
##config['group_probs'][:2] = 60/100 # 0 - 20
##config['group_probs'][2:4] = 25/100 # 30 - 50
##config['group_probs'][4:6] = 10/100 # 30 - 50
##config['group_probs'][6:8] = 5/100 # 60 - 80
##config['group_probs'][8:10] = 0/100 # 90 - 110
#CONFIGS['rotation2'] = config.copy()
#
#
## Color 2
#
## Color config
#config = {}
#config['group_type'] = 'color2'
#n_groups = 8
#config['n_groups'] = n_groups
#config['group_values'] = np.array(range(n_groups)) * 45
#config['group_probs'] = np.ones(n_groups) * 1 / n_groups
##config['group_probs'][:2] = 60/100 # 0 - 20
##config['group_probs'][2:4] = 25/100 # 30 - 50
##config['group_probs'][4:6] = 10/100 # 30 - 50
##config['group_probs'][6:8] = 5/100 # 60 - 80
#
#CONFIGS['color2'] = config.copy()
#
#
## color + rotation
#
#config = {}
#config['group_type'] = 'color_rotation'
#config['n_groups'] = CONFIGS['color2']['n_groups'] * CONFIGS['rotation2']['n_groups']
#config['n_groups_color'] = CONFIGS['color2']['n_groups']
#config['n_groups_rotation'] = CONFIGS['rotation2']['n_groups']
#
#config['group_values'] = [CONFIGS['rotation2']['group_values'], CONFIGS['color2']['group_values']]
#config['group_probs'] = [CONFIGS['rotation2']['group_probs'], CONFIGS['color2']['group_probs']]
#CONFIGS['color_rotation'] = config.copy()


TRAIN_SIZE = 60000
IMG_SIZE = 28


def preprocess(X, y):
    return X.reshape([-1, 28, 28, 1]).astype(np.float64), y

def to_rgb(X):

    return np.concatenate([X,X,X], axis=3)

def rescale(X):
    return X.astype(np.float32) / 255.

def rotate(X, rotation, single_image=False):
    if single_image:
        return np.array(sp.ndimage.rotate(X, rotation, reshape=False, order=0))
    else:
        return np.array(
            [sp.ndimage.rotate(X[i], rotation[i], reshape=False, order=0)
             for i in range(X.shape[0])]
        )


def background_color(X, hue):
    """ Expectes hue in range (0, 360) """

    sat = 1
    val = 1
    hue = hue / 360

    rgb = colorsys.hsv_to_rgb(hue, sat, val)

    mask = X[:, :, 0] > 100

    for j in range(3):
        X[:, :, j][mask] = rgb[j] * 255

    return X

def get_data(shuffle=True):
    """Returns train, val and test for mnist"""
    (X_train, y_train), (X_test, y_test) = [preprocess(*data) for data in
                                            keras.datasets.mnist.load_data()]

    #labels = [2,3,4,5,7]

    #labels = [1, 2, 3, 4, 5, 6, 7, 8]

    ##mapping = {2: 0,
    ##           3: 1,
    ##           4: 2,
    ##           5: 3,
    ##           7: 4
    ##           }

    #mapping = {1: 0,
    #           2: 1,
    #           3: 2,
    #           4: 3,
    #           5: 4,
    #           7: 5,
    #           8: 6
    #           }

    #indices_train = np.nonzero(np.isin(y_train, labels))
    #indices_test = np.nonzero(np.isin(y_test, labels))

    #X_train, y_train = X_train[indices_train], y_train[indices_train]
    #X_test, y_test = X_test[indices_test], y_test[indices_test]

    #y_train_new = y_train.copy()
    #y_test_new = y_test.copy()

    #for i in range(len(y_train)):
    #    y_train_new[i] = mapping[y_train[i]]

    #for i in range(len(y_test)):
    #    y_test_new[i] = mapping[y_test[i]]


    #y_train = y_train_new
    #y_test = y_test_new

    if shuffle:
        train_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[train_perm], y_train[train_perm]

        train_frac = 0.90
        n_train = int(len(X_train) * train_frac)

        X_val = X_train[n_train:]
        y_val = y_train[n_train:]

        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

        test_perm = np.random.permutation(X_test.shape[0])
        X_test, y_test = X_test[test_perm], y_test[test_perm]


    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class StaticMNISTUnknown(Dataset):

    def __init__(self, data, split, args,
                 data_folder='datasets'):

        super(StaticMNISTUnknown, self).__init__()

        self.images, self.labels = data
        self.original_size = len(self.images)

        self.num_classes = 10

        self.all_indices = range(self.original_size)

        # Generate the right dataset based on config
        # Create skew
        np.random.seed(1)
        config = CONFIGS['rotation'] # [args.mnist_type]
        if config['group_type'] in ['color_rotation', 'color_hue', 'color2']:
            self.images = to_rgb(self.images)
            self.image_shape = (3, 28, 28)
        else:
            self.image_shape = (1, 28, 28)

        self.config = config


        self.group_type = config['group_type']
        self.group_values = config['group_values']

        if split == 'train':
            self.indices, self.group_ids = self._get_train_skew(config)
        else:
            self.indices, self.group_ids = self._get_test(config)

        # Retrieve set
        self.images, self.labels = self.images[self.indices], self.labels[self.indices]

        self.group_ids_gt = self.group_ids.copy()


        # Map to groups
        # Get group ids
        self.groups = np.unique(self.group_ids)
        self.n_groups = config['n_groups']

        #self.n_train_groups = 12

        if split == 'train' and 'learned_groups' in args and args.learned_groups:

            path = Path('output/clusterings/') / 'mnist_unknown' / args.clustering_filename
            df_clusters = pd.read_csv(path)
            self.group_ids = np.array(df_clusters['cluster_id0'])
            self.n_groups = len(np.unique(self.group_ids))
            self.groups = np.unique(self.group_ids)

        self.group_stats = np.zeros((self.n_groups, 2))

        self.group_counts, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=False)
        self.group_dist, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=True)

        for group_id in range(self.n_groups):
            indices = np.nonzero(np.asarray(self.group_ids == group_id))[0]
            num_in_group = len(indices)
            self.group_stats[group_id, 0] = num_in_group # Num in group
            self.group_stats[group_id, 1] = num_in_group / len(self.labels) # Frac in group
            #self.group_stats[group_id, 1] = self.group_values[group_id]


        print("len dataset: ", len(self.labels))

        self.df_stats = pd.DataFrame(self.group_stats, columns=['n', 'frac'])
        self.df_stats['group_id'] = self.df_stats.index
        self.df_stats['binary'] = self.df_stats['group_id'].apply(lambda x: '{0:b}'.format(x).zfill(int(np.log(self.n_groups))))

        self.binarize = False

        # Print dataset stats
        print("Number of examples", len(self.indices))
        print(tabulate(self.df_stats, headers='keys', tablefmt='psql'))


    def _get_test(self, skew_config):
        """Returns the test set by duplicating the original
            MNIST test set for each rotation angle.

            There is no skew.

            TODO: Clean up this function"""

        group_ids = []
        indices = []

        for group_id in range(skew_config['n_groups']):

            num_examples = int(self.original_size / 3)
            group_ids.extend([group_id] * num_examples)
            indices.extend(self.all_indices)

        group_ids = np.array(group_ids)
        indices = np.array(indices)

        return indices, group_ids

    def _get_train_skew(self, skew_config):
        """Returns a skewed train set"""


        num_examples_total = len(self.labels)

        indices = []
        group_ids = []
        for group_id in range(skew_config['n_groups']):


            if skew_config['group_type'] == 'color_rotation':

                skew_config['n_groups_rotation']
                skew_config['n_groups_color']
                color_id = group_id // skew_config['n_groups_rotation']
                rotation_id = group_id % skew_config['n_groups_rotation']
                group_prob = config['group_probs'][0][rotation_id] * config['group_probs'][1][color_id]

            else:
                group_prob = skew_config['group_probs'][group_id]

            if group_prob == 0:
                continue

            if self.group_type == 'rotation':
                num_examples = int(group_prob * self.original_size / 5)
            elif self.group_type == 'color_rotation':
                num_examples = int(group_prob * self.original_size / 3)
                print("num example: ", num_examples)
                print("self.original size: ", self.original_size)
            elif self.group_type == 'color_hue':
                num_examples = int(group_prob * self.original_size / 5)
            else:
                num_examples = int(group_prob * self.original_size / 10)

            print("group type: ", self.group_type)

            indices_for_group = np.random.choice(self.original_size, size=num_examples)
            group_ids.append(len(indices_for_group) * [group_id])

            print("Group id", group_id)

            indices.append(indices_for_group)

        group_ids = np.concatenate(group_ids)
        indices = np.concatenate(indices)

        return indices, group_ids

    def __len__(self):
        """Returns number of examples in the dataset"""
        return len(self.labels)

    def __getitem__(self, index):

        group_id = self.group_ids_gt[index]

        img = self.images[index]

        self.apply_transform = True
        if self.apply_transform:

            if self.group_type == 'color_rotation':
                color_id = group_id // self.config['n_groups_rotation']
                rotation_id = group_id % self.config['n_groups_rotation']

                rotation_value = self.group_values[0][rotation_id]
                color_value = self.group_values[1][color_id]
            else:
                group_value = self.group_values[group_id]

            if self.group_type in ['rotation', 'rotation2']:
                img = rotate(img, group_value, single_image=True)
            elif self.group_type in ['color_hue', 'color2']:
                img = background_color(img, group_value)
            else:
                img = rotate(img, rotation_value, single_image=True)
                img = background_color(img, color_value)

        img = rescale(img) # =/ 256
        if self.binarize:
            img = np.random.binomial(1, img).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float)

        # Put color channel first
        img = img.permute(-1, 0, 1)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)


        return img, label, group_id


