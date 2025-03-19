import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


def cifar_dataloaders_distribute(data_type, batch_size):
    if data_type == '10':
        trainloader, testloader = cifar10_dataloaders_distribute(batch_size=batch_size)
    elif data_type == '100':
        trainloader, testloader = cifar100_dataloaders_distribute(batch_size=batch_size)
    return trainloader, testloader


def cifar10_dataloaders_distribute(batch_size, n_workers=10):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_workers,
                                              pin_memory=True,
                                              sampler=train_sampler)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=n_workers,
                                             pin_memory=True,
                                             sampler=test_sampler)
    return trainloader, testloader


def cifar100_dataloaders_distribute(batch_size, n_workers=10):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_workers,
                                              pin_memory=True,
                                              sampler=train_sampler)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=n_workers,
                                             pin_memory=True,
                                             sampler=test_sampler)
    return trainloader, testloader


def cifar10_dataloaders_parallel(batch_size, n_workers=10):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=n_workers)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=n_workers)
    return trainloader, testloader
