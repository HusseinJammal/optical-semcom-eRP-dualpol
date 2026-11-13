"""Pilot symbols: QPSK on the corners of the square M-QAM grid, one before every ``spacing`` data symbols.

Frame layout per polarization (paper: 256 data symbols, spacing 32 -> 9 pilots, 265 symbols):

    p d d ... d p d d ... d ... p d d ... d p
    ^ 0         ^ 33            ^ 231       ^ 264  (the last pilot closes the frame, so the phase is pinned at both edges)

The pilot sequence is fixed (seed 20261005), known to the receiver, and different on X and Y.
"""
from __future__ import annotations

import math

import torch

__all__ = ["PILOT_SEED", "pilot_amplitude", "pilot_symbols", "pilot_positions", "insert_pilots", "remove_pilots"]

PILOT_SEED = 20261005


def pilot_amplitude(m: int) -> float:
    """Per-axis value of the unit-mean-power square M-QAM corner (1.080 for 64-QAM, |p|^2 = 2.33)."""
    side = math.isqrt(m)
    if side * side != m:
        raise ValueError(f"pilots sit on a square QAM grid's corners; M={m} is not square")
    return math.sqrt(3.0 / (2.0 * (m - 1))) * (side - 1)


def pilot_symbols(n_pol: int, n_pilots: int, m: int, device=None, dtype=torch.complex64) -> torch.Tensor:
    """The known pilot sequence, ``(n_pol, n_pilots)`` QPSK on the grid corners."""
    g = torch.Generator().manual_seed(PILOT_SEED)
    bits = torch.randint(0, 2, (2, n_pol, n_pilots), generator=g).to(torch.float64) * 2 - 1
    a = pilot_amplitude(m)
    return torch.complex(a * bits[0], a * bits[1]).to(device=device, dtype=dtype)


def pilot_positions(k: int, spacing: int) -> torch.Tensor:
    """Frame indices of the pilots for ``k`` data symbols: 0, s+1, 2(s+1), ..., (k/s)(s+1)."""
    return torch.arange(k // spacing + 1) * (spacing + 1)


def insert_pilots(s: torch.Tensor, pilots: torch.Tensor, spacing: int) -> torch.Tensor:
    """``(B, P, k)`` data -> ``(B, P, k + k/spacing + 1)`` frame. Differentiable in ``s``."""
    b, p, k = s.shape
    nb = k // spacing
    blocks = s.reshape(b, p, nb, spacing)
    pil = pilots.to(s.dtype)
    head = pil[:, :nb].reshape(1, p, nb, 1).expand(b, p, nb, 1)
    body = torch.cat([head, blocks], dim=-1).reshape(b, p, nb * (spacing + 1))
    tail = pil[:, nb:nb + 1].reshape(1, p, 1).expand(b, p, 1)
    return torch.cat([body, tail], dim=-1)


def remove_pilots(f: torch.Tensor, spacing: int) -> torch.Tensor:
    """Inverse of :func:`insert_pilots`: ``(B, P, F)`` frame -> ``(B, P, k)`` data."""
    b, p, n = f.shape
    nb = (n - 1) // (spacing + 1)
    return f[..., : nb * (spacing + 1)].reshape(b, p, nb, spacing + 1)[..., 1:].reshape(b, p, nb * spacing)
