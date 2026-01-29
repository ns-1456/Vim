from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], save_path: str | None = None) -> None:
    """
    Plot training and validation loss/accuracy curves.

    Expected keys in history:
        - 'train_loss', 'val_loss', 'train_acc1', 'val_acc1'
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if "train_loss" in history:
        plt.plot(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    if "train_acc1" in history:
        plt.plot(epochs, history["train_acc1"], label="Train@1")
    if "val_acc1" in history:
        plt.plot(epochs, history["val_acc1"], label="Val@1")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Top-1 Accuracy")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


__all__ = ["plot_training_curves"]

