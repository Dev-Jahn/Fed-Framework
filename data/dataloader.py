from math import sqrt

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.models import resnet50

from data.datasets import *

DATASETS = {
    'mnist': MNIST_truncated,
    'femnist': FEMNIST,
    'fmnist': FashionMNIST_truncated,
    'cifar10': CIFAR10_truncated,
    'svhn': SVHN_custom,
    'generated': Generated,
}


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_idx=None, total=0):
        self.std = std
        self.mean = mean
        self.net_idx = net_idx
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_idx is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_idx / size)
            col = self.net_idx % size
            for i in range(size):
                for j in range(size):
                    filt[:, row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataset(dataset_name: str, datadir, datamap, net_idx, train, args):
    if dataset_name not in ('mnist', 'femnist', 'fmnist', 'cifar10', 'svhn', 'generated'):
        raise NotImplementedError(f'Unsupported Dataset "{dataset_name}".')

    # lambda functions for noise making
    if args.noise_type == 'space':
        noise = lambda net_idx: 0 if net_idx == args.n_clients - 1 else args.noise
        noise_args = lambda net_idx: {'net_idx': net_idx, 'total': args.n_clients - 1}
    else:
        noise = lambda net_idx: args.noise / (args.n_clients - 1) * net_idx
        noise_args = lambda net_idx: {}

    # build transforms
    if dataset_name in ('mnist', 'femnist', 'fmnist', 'svhn'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise(net_idx), **noise_args(net_idx))
        ])

    elif dataset_name == 'cifar10':
        if train and args.augment:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise(net_idx), **noise_args(net_idx))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise(net_idx), **noise_args(net_idx))
            ])
    else:
        transform = None

    return DATASETS[dataset_name](
        datadir, dataidxs=datamap[net_idx] if train else None, train=train, transform=transform
    )