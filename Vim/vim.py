from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from .patch_embed import PatchEmbed
from .vim_block import VimBlock

Tensor = torch.Tensor


@dataclass
class VisionMambaConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 100
    embed_dim: int = 192
    depth: int = 12
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    use_parallel_scan: bool = True


class VisionMamba(nn.Module):
    """
    Vision Mamba for Image Classification.

    Architecture:
      1. PatchEmbed: Image → Sequence of patch embeddings
      2. Positional embeddings added (inside PatchEmbed)
      3. N × VimBlock layers
      4. LayerNorm
      5. Classification head (mean pool → Linear)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        use_parallel_scan: bool = True,
    ) -> None:
        super().__init__()
        self.config = VisionMambaConfig(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_parallel_scan=use_parallel_scan,
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        blocks: List[nn.Module] = []
        for _ in range(depth):
            blocks.append(
                VimBlock(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_parallel_scan=use_parallel_scan,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            feat: (B, D)
        """
        x = self.patch_embed(x)  # (B, L, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=1)  # Global average pooling over tokens
        return feat

    def forward(self, x: Tensor) -> Tensor:
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits


def vim_tiny_cifar100(**kwargs) -> VisionMamba:
    """Convenience constructor for Vim-Tiny on CIFAR-100."""
    cfg = dict(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        use_parallel_scan=True,
    )
    cfg.update(kwargs)
    return VisionMamba(**cfg)


__all__ = ["VisionMamba", "VisionMambaConfig", "vim_tiny_cifar100"]

