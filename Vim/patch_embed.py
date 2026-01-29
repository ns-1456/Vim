from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class PatchEmbedConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 192

    @property
    def grid_size(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        g = self.grid_size
        return g * g


class PatchEmbed(nn.Module):
    """
    Image to patch embeddings.

    Input:  X ∈ R^(B×C×H×W)
    Output: X_seq ∈ R^(B×L×D), where L = (H×W) / P²

    Operations:
      1. Conv2d with kernel_size=P, stride=P (non-overlapping patches)
      2. Flatten spatial dims: (B, D, H/P, W/P) → (B, L, D)
      3. Add learnable positional embeddings
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.config = PatchEmbedConfig(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = self.config.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            seq: (B, L, D)
        """
        b, _, h, w = x.shape
        assert (
            h == self.config.img_size and w == self.config.img_size
        ), f"Expected image size {self.config.img_size}x{self.config.img_size}, got {h}x{w}"

        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)

        # Broadcast positional embeddings
        x = x + self.pos_embed
        return x

    def get_grid_size(self) -> Tuple[int, int]:
        g = self.config.grid_size
        return g, g

