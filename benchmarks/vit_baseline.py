from __future__ import annotations

import torch
from torch import nn

import timm


def deit_tiny_cifar100(pretrained: bool = True) -> nn.Module:
    """
    Create a DeiT-Tiny model adapted for CIFAR-100.
    """
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=100,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    model = deit_tiny_cifar100(pretrained=False)
    params = count_parameters(model)
    print(f"DeiT-Tiny parameters: {params / 1e6:.2f}M")

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()

