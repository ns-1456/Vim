"""
Selective State Space Model (SSM) block.
Sequence length is normalized after conv so all tensors stay (B, L, *) with L = input length.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch
from torch import nn
from torch.nn import functional as F

from .scan import naive_scan, parallel_scan

Tensor = torch.Tensor


@dataclass
class SelectiveSSMConfig:
    d_model: int = 192
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    use_parallel_scan: bool = True


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with input-dependent parameters.
    1. Depthwise Conv1D (length normalized to input L)
    2. Project to (Δ, B, C) -> reshape B,C to d_state
    3. ZOH discretize -> scan -> output
    """

    def __init__(
        self,
        d_model: int = 192,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_parallel_scan: bool = True,
    ) -> None:
        super().__init__()
        self.config = SelectiveSSMConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_parallel_scan=use_parallel_scan,
        )
        self.d_model = d_model
        self.d_state = d_state
        inner_dim = expand * d_state

        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=d_model,
        )
        self.in_proj = nn.Linear(d_model, d_state + 2 * inner_dim)
        self.out_proj = nn.Linear(d_state, d_model)
        self.A_log = nn.Parameter(torch.zeros(d_state))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _discretize(self, delta: Tensor) -> tuple[Tensor, Tensor]:
        """ZOH: A_bar, B_bar_scale from delta (B, L, d_state)."""
        A = -torch.exp(self.A_log).view(1, 1, -1)
        delta = F.softplus(delta)
        dA = delta * A
        A_bar = torch.exp(dA)
        eps = 1e-4
        A_safe = torch.where(A.abs() < eps, eps * torch.sign(A) + eps, A)
        B_bar_scale = (A_bar - 1.0) / A_safe
        small = A.abs() < eps
        if small.any():
            x = dA[small]
            taylor = 1.0 + x / 2.0 + x * x / 6.0
            B_bar_scale = torch.where(small, taylor, B_bar_scale)
        return A_bar, B_bar_scale

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D) -> y: (B, L, D)
        """
        bsz, L_in, d_model = x.shape
        assert d_model == self.d_model

        # Conv on (B, D, L)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        # Force length L_in (conv can give L_in-1 or L_in+1)
        if x_conv.size(2) != L_in:
            if x_conv.size(2) < L_in:
                x_conv = F.pad(x_conv, (0, L_in - x_conv.size(2)), mode="constant", value=0.0)
            else:
                x_conv = x_conv[:, :, :L_in]
        x_conv = x_conv.transpose(1, 2)
        # x_conv: (B, L_in, D)

        inner_dim = self.config.expand * self.d_state
        proj = self.in_proj(x_conv)
        delta = proj[..., : self.d_state]
        B_in = proj[..., self.d_state : self.d_state + inner_dim]
        C = proj[..., self.d_state + inner_dim :]

        # B_in, C: (B, L_in, inner_dim) -> (B, L_in, d_state) by (L_in, d_state, 2) mean
        n_groups = inner_dim // self.d_state
        B_in = B_in.reshape(bsz, L_in, self.d_state, n_groups).mean(dim=-1)
        C = C.reshape(bsz, L_in, self.d_state, n_groups).mean(dim=-1)

        A_bar, B_bar_scale = self._discretize(delta)
        B_bar = B_bar_scale * B_in

        scan_fn = parallel_scan if self.config.use_parallel_scan else naive_scan
        h = scan_fn(A_bar, B_bar)

        y_state = C * h
        y = self.out_proj(y_state)
        return y


__all__ = ["SelectiveSSM", "SelectiveSSMConfig"]
