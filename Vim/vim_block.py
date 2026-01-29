from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ssm import SelectiveSSM

Tensor = torch.Tensor


@dataclass
class VimBlockConfig:
    d_model: int = 192
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    use_parallel_scan: bool = True
    dropout: float = 0.0


class VimBlock(nn.Module):
    """
    Bidirectional Vision Mamba Block.

    Input:  (B, L, D)
    Output: (B, L, D)

    Steps:
      1. Split into two branches: (xz projection)
      2. Forward scan on x
      3. Backward scan on flip(x), then flip result
      4. Combine: out = (forward + backward) * SiLU(z)
      5. Output projection
    """

    def __init__(
        self,
        d_model: int = 192,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_parallel_scan: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = VimBlockConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_parallel_scan=use_parallel_scan,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(d_model)

        # Projection to x and z branches.
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # Shared SelectiveSSM (applied forward and backward).
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_parallel_scan=use_parallel_scan,
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            out: (B, L, D)
        """
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Forward SSM.
        h_fwd = self.ssm(x_branch)  # (B, L, D)

        # Backward SSM: flip sequence, apply SSM, then flip back.
        x_flip = torch.flip(x_branch, dims=[1])
        h_bwd_flip = self.ssm(x_flip)
        h_bwd = torch.flip(h_bwd_flip, dims=[1])

        # Gating.
        z = F.silu(z_branch)
        y = (h_fwd + h_bwd) * z

        out = self.out_proj(y)
        out = self.dropout(out)
        return residual + out


__all__ = ["VimBlock", "VimBlockConfig"]

