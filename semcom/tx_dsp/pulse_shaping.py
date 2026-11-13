"""Pulse shaping: zero-insertion upsampling and a root-raised-cosine (RRC) filter.

The RRC has an ODD number of taps on ``t = (arange(n_taps) - n_taps // 2) / sps``: exactly symmetric, a tap at
t = 0, peak on the centre tap, unit energy. With 'same' convolution the TX-RX cascade has zero delay, so symbol k is
sampled at sample k * sps with no timing search (OptiCommPy's ``pulseShape`` grid is stretched by 257/256 and never
samples t = 0, which is why it is not used here).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["upsample", "rrc_filter", "apply_filter", "pad_symbols", "strip_symbols", "pad_guard"]


def upsample(sym: torch.Tensor, sps: int) -> torch.Tensor:
    """``(B, P, n) -> (B, P, n * sps)``: each symbol followed by sps - 1 zeros."""
    b, p, n = sym.shape
    out = torch.zeros(b, p, n * sps, dtype=sym.dtype, device=sym.device)
    out[..., ::sps] = sym
    return out


def rrc_filter(n_taps: int, sps: int, rolloff: float, device=None, dtype=torch.float64) -> torch.Tensor:
    """Root-raised-cosine taps, unit energy, exactly symmetric (``n_taps`` must be odd)."""
    if n_taps % 2 == 0:
        raise ValueError(f"n_taps must be odd for a symmetric grid; got {n_taps}")
    t = (torch.arange(n_taps, dtype=dtype, device=device) - n_taps // 2) / sps
    b = float(rolloff)
    h = torch.empty_like(t)
    at_zero = t == 0.0
    at_sing = (torch.abs(torch.abs(t) - 1.0 / (4 * b)) < 1e-12) if b > 0 else torch.zeros_like(t, dtype=torch.bool)
    regular = ~(at_zero | at_sing)
    tr = t[regular]
    num = torch.sin(math.pi * tr * (1 - b)) + 4 * b * tr * torch.cos(math.pi * tr * (1 + b))
    den = math.pi * tr * (1 - (4 * b * tr) ** 2)
    h[regular] = num / den
    h[at_zero] = 1.0 + b * (4 / math.pi - 1)
    if bool(at_sing.any()):
        h[at_sing] = (b / math.sqrt(2)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * b))
                                           + (1 - 2 / math.pi) * math.cos(math.pi / (4 * b)))
    return h / torch.sqrt(torch.sum(h**2))


def apply_filter(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """'Same'-length linear convolution of ``(B, P, n)`` complex samples with a real symmetric FIR.

    The kernel follows the input's real dtype (complex64 training, complex128 evaluation).
    """
    b, p, n = x.shape
    pad = (h.numel() - 1) // 2
    rdtype = x.real.dtype
    k = h.flip(0).to(rdtype).view(1, 1, -1)

    def _conv(real_part):
        v = real_part.reshape(b * p, 1, n)
        return F.conv1d(F.pad(v, (pad, pad)), k).reshape(b, p, n)

    return torch.complex(_conv(x.real), _conv(x.imag))


def pad_symbols(sym: torch.Tensor, n_pad: int) -> torch.Tensor:
    """Zero-pad ``n_pad`` SYMBOLS on both sides (RRC edge transients; ``n_taps // sps + 8`` = 24)."""
    if n_pad == 0:
        return sym
    z = torch.zeros(*sym.shape[:-1], n_pad, dtype=sym.dtype, device=sym.device)
    return torch.cat([z, sym, z], dim=-1)


def strip_symbols(sym: torch.Tensor, n_pad: int) -> torch.Tensor:
    """Inverse of :func:`pad_symbols`."""
    if n_pad == 0:
        return sym
    return sym[..., n_pad: sym.shape[-1] - n_pad]


def pad_guard(x: torch.Tensor, guard_samples: int) -> torch.Tensor:
    """Zero-pad ``guard_samples`` SAMPLES on both sides (frequency-domain dispersion is periodic in time)."""
    if guard_samples == 0:
        return x
    z = torch.zeros(*x.shape[:-1], guard_samples, dtype=x.dtype, device=x.device)
    return torch.cat([z, x, z], dim=-1)
