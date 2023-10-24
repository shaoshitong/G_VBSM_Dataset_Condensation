from __future__ import print_function

import os
import socket
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8),  # refer to the cifar case
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_tinyimagenet_dataloaders(batch_size=64, num_workers=8, data_folder='./data/tinyimagenet', is_instance=False):
    # train set and loder
    train_set = torchvision.datasets.ImageFolder(root=data_folder + '/train', transform=transform_train)
    train_loader = DataLoaderX(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test set and loder
    test_set = torchvision.datasets.ImageFolder(root=data_folder + '/val', transform=transform_test)
    test_loader = DataLoaderX(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader
