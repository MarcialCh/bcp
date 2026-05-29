# dataloader.py



import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np


import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset


def get_cifar10_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


def load_cifar10(root="./data"):
    transform_train, transform_test = get_cifar10_transforms()

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    return trainset, testset

def get_stl10_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4467, 0.4398, 0.4066),
            (0.2241, 0.2215, 0.2239)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4467, 0.4398, 0.4066),
            (0.2241, 0.2215, 0.2239)
        ),
    ])

    return transform_train, transform_test


def load_stl10(root="./data"):
    transform_train, transform_test = get_stl10_transforms()

    trainset = torchvision.datasets.STL10(
        root=root,
        split='train',
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.STL10(
        root=root,
        split='test',
        download=True,
        transform=transform_test
    )

    return trainset, testset


def get_subset(dataset, size, seed=0):
    """
    randomly sample size samples from the dataset
    """
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=size, replace=False)
    return Subset(dataset, indices)





def get_dataloader(
    trainset,
    testset,
    batch_size=128,
    subset_size=None,
    indices=None,
    seed=0,
    shuffle=True
):

    # --------------------------------------------------------
    # training subset
    # --------------------------------------------------------

    if indices is not None:

        train_subset = Subset(trainset, indices)

    elif subset_size is not None:

        rng = np.random.default_rng(seed)

        sampled_indices = rng.choice(
            len(trainset),
            subset_size,
            replace=False
        )

        train_subset = Subset(
            trainset,
            sampled_indices
        )

    else:

        train_subset = trainset

    # --------------------------------------------------------
    # dataloaders
    # --------------------------------------------------------

    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader