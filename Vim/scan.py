from __future__ import annotations

import math
from typing import Tuple

import torch

Tensor = torch.Tensor


def naive_scan(coeffs_a: Tensor, coeffs_b: Tensor) -> Tensor:
    """
    Naive for-loop scan for the recurrence:

        h_t = a_t * h_{t-1} + b_t,  h_{-1} = 0

    Shapes:
        coeffs_a, coeffs_b: (B, L, *state_shape)
    Returns:
        h: (B, L, *state_shape)
    """
    assert coeffs_a.shape == coeffs_b.shape
    bsz, seq_len = coeffs_a.shape[:2]
    state_shape = coeffs_a.shape[2:]

    h = coeffs_a.new_zeros((bsz, seq_len) + state_shape)
    h_prev = coeffs_a.new_zeros((bsz,) + state_shape)

    for t in range(seq_len):
        a_t = coeffs_a[:, t]
        b_t = coeffs_b[:, t]
        h_t = a_t * h_prev + b_t
        h[:, t] = h_t
        h_prev = h_t
    return h


def _associative_combine(
    a1: Tensor, b1: Tensor, a2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)
    """
    a = a2 * a1
    b = a2 * b1 + b2
    return a, b


def parallel_scan(coeffs_a: Tensor, coeffs_b: Tensor) -> Tensor:
    """
    Parallel associative scan (Blelloch-style) for the same recurrence as naive_scan.

    This performs O(L) work with O(log L) sequential depth, leveraging GPU
    parallelism along the sequence dimension.

    Shapes:
        coeffs_a, coeffs_b: (B, L, *state_shape)
    Returns:
        h: (B, L, *state_shape)
    """
    assert coeffs_a.shape == coeffs_b.shape
    bsz, seq_len = coeffs_a.shape[:2]
    state_shape = coeffs_a.shape[2:]

    # We work on copies to keep inputs unchanged.
    a = coeffs_a.clone()
    b = coeffs_b.clone()

    # Up-sweep: build partial products using the associative operator.
    levels = int(math.ceil(math.log2(seq_len))) if seq_len > 1 else 0
    for k in range(levels):
        offset = 1 << k
        # indices i that have a left neighbor at distance offset
        i = torch.arange(seq_len, device=a.device)
        mask = i >= offset
        if not mask.any():
            continue
        idx = i[mask]
        left = idx - offset

        # Broadcast indices for batch and state dims
        a_left = a[:, left]
        b_left = b[:, left]
        a_right = a[:, idx]
        b_right = b[:, idx]

        a_comb, b_comb = _associative_combine(a_left, b_left, a_right, b_right)
        a[:, idx] = a_comb
        b[:, idx] = b_comb

    # After the scan, with initial state h_{-1} = 0, the hidden state h_t
    # equals the accumulated "b" term at each position.
    h = b
    return h


__all__ = ["naive_scan", "parallel_scan"]

