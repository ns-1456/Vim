from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import build_cifar100_dataloaders
from utils import AverageMeter, accuracy
from vim import vim_tiny_cifar100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vision Mamba (Vim) on CIFAR-100")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CIFAR-100 data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default="runs/vim_cifar100")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: AdamW, epochs: int, warmup_epochs: int
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, int]:
    # Simple cosine schedule with warmup handled manually in the loop.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return scheduler, warmup_epochs


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: amp.GradScaler,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None = None,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc1, = accuracy(outputs.detach(), targets, topk=(1,))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))

        pbar.set_postfix(loss=loss_meter.avg, acc1=acc1_meter.avg)

    if writer is not None:
        writer.add_scalar("train/loss", loss_meter.avg, epoch)
        writer.add_scalar("train/acc1", acc1_meter.avg, epoch)

    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg}


def validate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None = None,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    acc5_meter = AverageMeter("acc5")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [val]", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

            pbar.set_postfix(loss=loss_meter.avg, acc1=acc1_meter.avg, acc5=acc5_meter.avg)

    if writer is not None:
        writer.add_scalar("val/loss", loss_meter.avg, epoch)
        writer.add_scalar("val/acc1", acc1_meter.avg, epoch)
        writer.add_scalar("val/acc5", acc5_meter.avg, epoch)

    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Data
    train_loader, val_loader = build_cifar100_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = vim_tiny_cifar100(img_size=args.img_size)
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler, warmup_epochs = create_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    scaler = amp.GradScaler()

    best_acc1 = 0.0
    epochs_no_improve = 0
    patience = 20

    for epoch in range(1, args.epochs + 1):
        # Warmup: linear LR ramp-up for first warmup_epochs.
        if epoch <= warmup_epochs:
            warmup_factor = epoch / float(max(1, warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * warmup_factor
        else:
            scheduler.step()

        train_stats = train_one_epoch(
            model, criterion, optimizer, scaler, train_loader, device, epoch, writer
        )
        val_stats = validate(model, criterion, val_loader, device, epoch, writer)

        # Checkpointing.
        is_best = val_stats["acc1"] > best_acc1
        if is_best:
            best_acc1 = val_stats["acc1"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc1": best_acc1,
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / f"checkpoint_{epoch:03d}.pth")
        if is_best:
            torch.save(ckpt, output_dir / "checkpoint_best.pth")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    writer.close()


if __name__ == "__main__":
    main()

