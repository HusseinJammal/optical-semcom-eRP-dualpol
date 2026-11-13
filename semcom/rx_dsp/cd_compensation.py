"""Static chromatic-dispersion compensation: the ideal frequency-domain inverse of the fibre dispersion.

Phase only (the loss is compensated by the inline amplifiers). ``L_km`` must be the distance actually propagated.
"""
from __future__ import annotations

import torch

from ..channel.linear import omega_axis
from ..channel.params import FiberConfig

__all__ = ["cd_compensate"]


def cd_compensate(x: torch.Tensor, fiber: FiberConfig, L_km: float, fs_hz: float) -> torch.Tensor:
    """Multiply the spectrum of ``(B, P, N)`` by exp(-j (beta2/2) w^2 L)."""
    w = omega_axis(x.shape[-1], fs_hz, device=x.device)
    phase = -(fiber.beta2_s2_km / 2.0) * w**2 * L_km
    h = torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(x.dtype)
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * h, dim=-1)
