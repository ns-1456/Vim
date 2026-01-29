from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from data import build_cifar100_dataloaders
from utils import AverageMeter, accuracy
from vim import vim_tiny_cifar100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vision Mamba (Vim) on CIFAR-100")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CIFAR-100 data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    acc5_meter = AverageMeter("acc5")

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

    print(f"Test Loss: {loss_meter.avg:.4f}")
    print(f"Top-1 Acc: {acc1_meter.avg:.2f}%")
    print(f"Top-5 Acc: {acc5_meter.avg:.2f}%")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Data
    _, test_loader = build_cifar100_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = vim_tiny_cifar100(img_size=args.img_size)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()

