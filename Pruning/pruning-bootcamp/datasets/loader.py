"""
Dataset Loaders - CIFAR-10 and ImageNet-mini
Provides standardized data loading for pruning experiments
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4, data_dir: str = './data') -> Tuple[DataLoader, DataLoader, int]:
    """
    Get CIFAR-10 train and validation data loaders

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to store/load data

    Returns:
        train_loader, val_loader, num_classes
    """
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, 10  # CIFAR-10 has 10 classes


def get_imagenet_mini_loaders(batch_size: int = 128, num_workers: int = 4, data_dir: str = './data/imagenet-mini') -> Tuple[DataLoader, DataLoader, int]:
    """
    Get ImageNet-mini train and validation data loaders
    ImageNet-mini is a subset of ImageNet with fewer images per class

    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory containing ImageNet-mini data

    Returns:
        train_loader, val_loader, num_classes
    """
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Check if data directory exists
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'

    if not train_dir.exists() or not val_dir.exists():
        print(f'Warning: ImageNet-mini not found at {data_dir}')
        print('Please download ImageNet-mini and organize as:')
        print(f'  {data_dir}/train/classXXX/*.jpg')
        print(f'  {data_dir}/val/classXXX/*.jpg')
        print('\nFalling back to CIFAR-10 for demonstration...')
        return get_cifar10_loaders(batch_size, num_workers)

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=str(val_dir),
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)

    return train_loader, val_loader, num_classes


def get_dataset(name: str, batch_size: int = 128, num_workers: int = 4, data_dir: str = './data') -> Tuple[DataLoader, DataLoader, int]:
    """
    Get dataset loaders by name

    Args:
        name: Dataset name ('cifar10' or 'imagenet-mini')
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Data directory

    Returns:
        train_loader, val_loader, num_classes
    """
    name = name.lower()

    if name == 'cifar10' or name == 'cifar-10':
        return get_cifar10_loaders(batch_size, num_workers, data_dir)
    elif name == 'imagenet-mini' or name == 'imagenetmini':
        return get_imagenet_mini_loaders(batch_size, num_workers, data_dir)
    else:
        raise ValueError(f'Unknown dataset: {name}. Supported: cifar10, imagenet-mini')


# For convenience
def get_input_size(dataset_name: str) -> int:
    """Get input image size for dataset"""
    dataset_name = dataset_name.lower()
    if 'cifar' in dataset_name:
        return 32
    elif 'imagenet' in dataset_name:
        return 224
    else:
        return 224


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for dataset"""
    dataset_name = dataset_name.lower()
    if 'cifar10' in dataset_name:
        return 10
    elif 'cifar100' in dataset_name:
        return 100
    elif 'imagenet' in dataset_name:
        return 1000  # Default for ImageNet
    else:
        return 1000
