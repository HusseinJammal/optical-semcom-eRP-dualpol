"""Carrier-phase recovery (CPR).

``pilot_cpr`` is the receiver used everywhere: per pilot, the correlation y p* summed over BOTH polarizations (the
lasers and the Kerr rotation are common to both), averaged over 3 neighbouring pilots, unwrapped, linearly
interpolated across the frame, and removed. The estimate is DETACHED from the autograd graph: it is a fixed DSP block
the encoder cannot learn to steer.

``genie_cpr`` uses the transmitted frame (every symbol known): an evaluation-only upper bound, not a receiver.
"""
from __future__ import annotations

import math

import torch

from ..tx_dsp.pilots import pilot_positions

__all__ = ["pilot_cpr", "genie_cpr"]


def _interp_phase(th: torch.Tensor, n_frame: int, spacing: int) -> torch.Tensor:
    """Linear interpolation of per-pilot phases ``(B, nb+1)`` onto every frame index ``(B, F)``."""
    nb = th.shape[-1] - 1
    n = torch.arange(n_frame, device=th.device)
    k = torch.clamp(n // (spacing + 1), max=nb - 1)
    frac = ((n - k * (spacing + 1)).to(th.dtype)) / (spacing + 1)
    return th[:, k] * (1 - frac) + th[:, k + 1] * frac


def pilot_cpr(y: torch.Tensor, pilots: torch.Tensor, spacing: int, avg: int = 3) -> torch.Tensor:
    """Pilot-aided CPR on a ``(B, P, F)`` frame with known ``pilots`` ``(P, n_pilots)``."""
    pos = pilot_positions((y.shape[-1] - 1) // (spacing + 1) * spacing, spacing).to(y.device)
    yd = y.detach()
    corr = (yd[..., pos] * pilots.to(y.dtype).conj().unsqueeze(0)).sum(dim=1)         # (B, nb+1), joint over pols
    if avg > 1:
        half = avg // 2
        sm = corr.clone()
        for d in range(1, half + 1):
            sm[:, d:] = sm[:, d:] + corr[:, :-d]
            sm[:, :-d] = sm[:, :-d] + corr[:, d:]
        corr = sm
    th = torch.angle(corr).detach()
    d = th[:, 1:] - th[:, :-1]
    d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
    th = torch.cat([th[:, :1], th[:, :1] + torch.cumsum(d, dim=-1)], dim=-1)            # unwrap
    theta = _interp_phase(th, y.shape[-1], spacing)
    rot = torch.exp(torch.complex(torch.zeros_like(theta), -theta)).to(y.dtype)
    return y * rot.unsqueeze(1)


def genie_cpr(y: torch.Tensor, x: torch.Tensor, window: int = 65) -> torch.Tensor:
    """UPPER BOUND: sliding-window data-aided phase from the transmitted frame ``x`` (evaluation only)."""
    corr = (y.detach() * x.detach().conj()).sum(dim=1)                                   # (B, F)
    ker = torch.ones(1, 1, window, dtype=torch.float64, device=y.device)
    re = torch.nn.functional.conv1d(corr.real.to(torch.float64).unsqueeze(1), ker, padding=window // 2).squeeze(1)
    im = torch.nn.functional.conv1d(corr.imag.to(torch.float64).unsqueeze(1), ker, padding=window // 2).squeeze(1)
    theta = torch.atan2(im, re).to(y.real.dtype).detach()
    return y * torch.exp(torch.complex(torch.zeros_like(theta), -theta)).to(y.dtype).unsqueeze(1)
