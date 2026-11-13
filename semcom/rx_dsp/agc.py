"""Automatic gain control: each polarization of each image is normalised to unit mean power.

The absolute received scale depends on the launch power and pulse shaping and carries no information, so the decoder
always sees unit-power symbols whatever the link.
"""
from __future__ import annotations

import torch

__all__ = ["agc"]


def agc(s: torch.Tensor) -> torch.Tensor:
    """Normalise ``(..., k)`` complex symbols to unit mean power over the last axis."""
    p = torch.mean(s.real**2 + s.imag**2, dim=-1, keepdim=True).clamp(min=1e-30)
    return s / torch.sqrt(p).to(s.dtype)
