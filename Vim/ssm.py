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

    Architecture:
      1. Local depthwise Conv1D over sequence
      2. Project input to (Δ, B, C) via linear layers
      3. ZOH discretization: Ā = exp(Δ·A), B̄ = ...
      4. Run scan over recurrence h_t = Ā_t * h_{t-1} + B̄_t
      5. Compute output y_t = C_t · h_t
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

        # Depthwise convolution over sequence (local mixing).
        # Use padding so output length equals input length (kernel_size=4 -> padding=1).
        self.conv_pad = (d_conv - 1) // 2  # 1 for d_conv=4 -> out_len = in_len
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=self.conv_pad,
            groups=d_model,
        )

        # Project to Δ, B, C.
        inner_dim = expand * d_state
        self.in_proj = nn.Linear(d_model, 2 * inner_dim + d_state)
        self.out_proj = nn.Linear(d_state, d_model)

        # Diagonal SSM parameter A (log-parameterized for stability).
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

    def _discretize(self, delta: Tensor) -> Tensor:
        """
        ZOH discretization for diagonal A.

        Args:
            delta: (B, L, d_state)
        Returns:
            A_bar: (B, L, d_state)
            B_bar_scale: (B, L, d_state) such that B̄ = B_bar_scale * B
        """
        # Ensure A has negative real parts for stability.
        A = -torch.exp(self.A_log)  # (d_state,)
        A = A.view(1, 1, -1)  # (1, 1, d_state)

        # delta >= 0
        delta = F.softplus(delta)
        dA = delta * A  # (B, L, d_state)

        A_bar = torch.exp(dA)

        # (exp(Δ·A) - 1) / A
        # Use a numerically stable formulation around A ≈ 0.
        eps = 1e-4
        A_safe = torch.where(A.abs() < eps, eps * torch.sign(A) + eps, A)
        B_bar_scale = (A_bar - 1.0) / A_safe

        # For very small |A|, use Taylor expansion (exp(x)-1)/x ≈ 1 + x/2 + x^2/6
        small = A.abs() < eps
        if small.any():
            x = dA[small]
            taylor = 1.0 + x / 2.0 + x * x / 6.0
            B_bar_scale = torch.where(small, taylor, B_bar_scale)

        return A_bar, B_bar_scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        bsz, seq_len_in, d_model = x.shape
        assert d_model == self.d_model

        # Local convolution over sequence (B, D, L) -> (B, D, L) -> (B, L, D)
        # padding=(d_conv-1)//2: k=4 gives pad=1 -> out_len = L-1; pad back to L
        x_conv = x.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv(x_conv)  # (B, D, L_out), L_out may be L-1 or L+1
        if x_conv.size(2) != seq_len_in:
            pad_len = seq_len_in - x_conv.size(2)
            if pad_len > 0:
                x_conv = F.pad(x_conv, (0, pad_len), mode="constant", value=0.0)
            else:
                x_conv = x_conv[:, :, :seq_len_in]
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)

        # Project to Δ, B, C parameters.
        proj = self.in_proj(x_conv)  # (B, L, 2*inner_dim + d_state)
        inner_dim = self.config.expand * self.d_state
        delta, B_in, C = torch.split(proj, [self.d_state, inner_dim, inner_dim], dim=-1)

        # Reduce inner_dim to d_state for B and C via reshaping + mean pooling.
        # Use actual sequence length from tensors (conv can change length; 2080 = 1*65*32)
        L = B_in.size(1)
        B_in = B_in.view(bsz, L, self.d_state, -1).mean(dim=-1)  # (B, L, d_state)
        C = C.view(bsz, L, self.d_state, -1).mean(dim=-1)  # (B, L, d_state)

        A_bar, B_bar_scale = self._discretize(delta)  # (B, L, d_state)
        B_bar = B_bar_scale * B_in  # (B, L, d_state)

        # Run scan over state dimension.
        scan_fn = parallel_scan if self.config.use_parallel_scan else naive_scan
        h = scan_fn(A_bar, B_bar)  # (B, L, d_state)

        # Output y_t = C_t · h_t (element-wise); project state to model dim.
        y_state = C * h  # (B, L, d_state)
        y = self.out_proj(y_state)  # (B, L, d_model)
        # Return same length as input (L may differ if conv changed length before pad/slice).
        if y.size(1) != seq_len_in:
            y = y[:, :seq_len_in].contiguous()
        return y


__all__ = ["SelectiveSSM", "SelectiveSSMConfig"]

