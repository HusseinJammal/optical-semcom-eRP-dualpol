"""Residual carrier frequency offset: what a frequency-offset estimator leaves, as a phase ramp on the SAMPLED symbols.

A real receiver estimates the offset after CD compensation (clock recovery absorbs the walk-off a raw offset causes
through the CD equalizer, which this fixed-sampling receiver would otherwise turn into ISI), so the residual is
modelled after sampling.
"""
from __future__ import annotations

import math

import torch

__all__ = ["apply_frequency_offset"]


def apply_frequency_offset(y: torch.Tensor, fo_hz: torch.Tensor, symbol_rate_hz: float) -> torch.Tensor:
    """``(B, P, F)`` symbols times exp(j 2 pi fo t_n), t_n = n / Rs, one offset per image."""
    f_len = y.shape[-1]
    rdtype = y.real.dtype
    t = torch.arange(f_len, dtype=torch.float64, device=y.device) / symbol_rate_hz
    ramp = (2 * math.pi * fo_hz.to(y.device).unsqueeze(-1) * t).to(rdtype)
    return y * torch.exp(torch.complex(torch.zeros_like(ramp), ramp)).to(y.dtype).unsqueeze(1)
