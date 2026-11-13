"""Laser phase noise: a Wiener process per SAMPLE with increment variance 2 pi linewidth / fs.

The TX laser rotates the launched field (same laser on both polarizations, plus a uniform initial carrier phase);
the local oscillator rotates the received field BEFORE chromatic-dispersion compensation (same LO on both
polarizations), so equalization-enhanced phase noise (EEPN) appears naturally.
"""
from __future__ import annotations

import math

import torch

from ..utils.random import PerImageRng

__all__ = ["wiener_phase", "rotate"]


def wiener_phase(n_samples: int, linewidth_hz: torch.Tensor, fs_hz: float, rng: PerImageRng, stream: str, dtype,
                 phase0: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, n_samples)`` phase: cumulative sum of N(0, 2 pi linewidth / fs) increments (+ an initial phase)."""
    std = torch.sqrt(2.0 * math.pi * linewidth_hz.to(torch.float64) / fs_hz).to(dtype).to(rng.device)
    inc = rng.randn(stream, (n_samples,), dtype) * std.unsqueeze(-1)
    phi = torch.cumsum(inc, dim=-1)
    if phase0 is not None:
        phi = phi + phase0.to(dtype).to(rng.device).unsqueeze(-1)
    return phi


def rotate(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """``(B, P, N)`` field times exp(j phi), with ``phi`` ``(B, N)`` common to both polarizations."""
    return x * torch.exp(torch.complex(torch.zeros_like(phi), phi)).to(x.dtype).unsqueeze(1)
