"""Matched filter, guard removal and symbol-rate sampling.

The matched filter is the TX RRC itself (real and symmetric). The TX-RX cascade has zero delay, so symbol k sits at
sample k * sps: fixed timing, no clock recovery.
"""
from __future__ import annotations

import torch

from ..tx_dsp.pulse_shaping import apply_filter

__all__ = ["matched_filter", "strip_guard", "downsample", "extract_frame"]


def matched_filter(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return apply_filter(x, h)


def strip_guard(x: torch.Tensor, guard_samples: int) -> torch.Tensor:
    if guard_samples == 0:
        return x
    return x[..., guard_samples: x.shape[-1] - guard_samples]


def downsample(x: torch.Tensor, sps: int, offset: int = 0) -> torch.Tensor:
    return x[..., offset::sps]


def extract_frame(y: torch.Tensor, pad: int, frame_len: int) -> torch.Tensor:
    """Drop the TX pad (and the tail) around the frame: ``(B, P, n_sym) -> (B, P, frame_len)``."""
    return y[..., pad:pad + frame_len]
