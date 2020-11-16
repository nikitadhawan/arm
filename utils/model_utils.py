import models

def get_model(args, image_shape):

    num_channels, W, H = image_shape

    print("num channels: ", num_channels)

    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset in ('celeba'):
        num_classes = 4
    elif args.dataset == 'femnist':
        num_classes = 62
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
    elif args.dataset == 'cifar':
        num_classes = 10

    # This is the one used in the paper
    if args.bn:
#         import ipdb; ipdb.set_trace()
        print("using BNConvNet")
        model = models.BNConvNet(num_channels, bn_track_running_stats=False,
             num_classes=num_classes, support_size=args.support_size,
                                     prediction_net=args.prediction_net,
                                     pretrained=args.pretrained, context_net=args.context_net)
    else:
        model = models.ContextualConvNet(num_channels, n_context_channels=args.n_context_channels,
             num_classes=num_classes, support_size=args.support_size, use_context=args.use_context,
                                     prediction_net=args.prediction_net,
                                     pretrained=args.pretrained, context_net=args.context_net)

    return model


"""import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaConvModel(MetaModule):
   """ """4-layer Convolutional Neural Network architecture from [1].
    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.
    out_features : int
        Number of classes (output of the model).
    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.
    feature_size : int (default: 64)
        Number of features returned by the convolutional head.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
   """ """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
"""
