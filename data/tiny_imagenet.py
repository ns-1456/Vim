from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class TinyImageNetConfig:
    data_dir: str
    img_size: int = 64
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


def _build_transforms(img_size: int, train: bool) -> transforms.Compose:
    t = []
    if train:
        t.extend(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        t.extend(
            [
                transforms.Resize(img_size + 8),
                transforms.CenterCrop(img_size),
            ]
        )
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def build_tiny_imagenet_dataloaders(
    data_dir: str,
    img_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tiny-ImageNet dataloaders.

    Expects the standard Tiny-ImageNet directory layout under data_dir.
    """
    cfg = TinyImageNetConfig(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_dir = Path(cfg.data_dir) / "train"
    val_dir = Path(cfg.data_dir) / "val"

    train_dataset = datasets.ImageFolder(train_dir, transform=_build_transforms(cfg.img_size, train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=_build_transforms(cfg.img_size, train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader


__all__ = ["build_tiny_imagenet_dataloaders", "TinyImageNetConfig"]

