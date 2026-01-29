from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass
class CIFAR100Config:
    data_dir: str
    img_size: int = 32
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


def _build_transforms(img_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)

    if train:
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        t = []

    if img_size != 32:
        # Support resizing for memory benchmarks (e.g., 64, 224, ...)
        t.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))

    t.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transforms.Compose(t)


def build_cifar100_dataloaders(
    data_dir: str,
    img_size: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 train and test dataloaders.
    """
    cfg = CIFAR100Config(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_transform = _build_transforms(cfg.img_size, train=True)
    test_transform = _build_transforms(cfg.img_size, train=False)

    train_dataset = datasets.CIFAR100(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, test_loader


__all__ = ["build_cifar100_dataloaders", "CIFAR100Config"]

