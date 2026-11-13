"""Linear fibre propagation: attenuation + chromatic dispersion, D(z) = F^-1 exp(-alpha z/2 + j beta2 w^2 z/2) F.

Only FFTs and complex multiplies, so autograd differentiates it. Shapes are ``(batch, n_pol, n_samples)`` and the
FFT acts on the last axis. Held to OptiCommPy at gamma = 0 by tests/test_channel.py.
"""
from __future__ import annotations

import torch

from .params import FiberConfig

__all__ = ["omega_axis", "dispersion_mask", "D_op", "span_gain", "propagate_link"]


def omega_axis(n: int, fs_hz: float, device=None, dtype=torch.float64) -> torch.Tensor:
    """w = 2 pi fs fftfreq(N) [rad/s], in torch.fft order. Written as OptiCommPy writes it (bit-identical)."""
    return 2.0 * torch.pi * fs_hz * torch.fft.fftfreq(n, device=device, dtype=dtype)


def dispersion_mask(fiber: FiberConfig, z_km: float, n: int, fs_hz: float, device=None) -> torch.Tensor:
    """H(z, w) = exp(-(alpha/2) z + j (beta2/2) w^2 z). Negative ``z_km`` gives the inverse operator."""
    w = omega_axis(n, fs_hz, device=device)
    exponent = torch.complex(
        torch.full_like(w, -(fiber.alpha_1_per_km / 2.0) * z_km),
        (fiber.beta2_s2_km / 2.0) * w**2 * z_km,
    )
    return torch.exp(exponent)


def D_op(x: torch.Tensor, fiber: FiberConfig, z_km: float, fs_hz: float) -> torch.Tensor:
    """Linear propagation over ``z_km``: IFFT(H(z, w) FFT(x)). Periodic in time: pad a guard first."""
    h = dispersion_mask(fiber, z_km, x.shape[-1], fs_hz, device=x.device).to(x.dtype)
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * h, dim=-1)


def span_gain(fiber: FiberConfig, lspan_km: float) -> float:
    """Field gain sqrt(G) = exp(alpha L_span / 2) of a noiseless amplifier that exactly compensates the span loss."""
    return float(torch.exp(torch.tensor(fiber.alpha_1_per_km * lspan_km / 2.0, dtype=torch.float64)))


def propagate_link(x: torch.Tensor, cfg, fs_hz: float | None = None) -> torch.Tensor:
    """Whole link, linear only: one D_op per span and a noiseless gain (``amp`` 'ideal' or None)."""
    fs = cfg.sim.fs_hz if fs_hz is None else fs_hz
    z_span = cfg.link.n_steps_per_span * cfg.link.hz_km
    g = span_gain(cfg.fiber, z_span)
    for _ in range(cfg.link.n_spans):
        x = D_op(x, cfg.fiber, z_span, fs)
        if cfg.link.amp == "ideal":
            x = x * g
        elif cfg.link.amp is None or cfg.link.amp == "none":
            pass
        else:
            raise ValueError(f"amp={cfg.link.amp!r}: the noiseless path takes 'ideal' or None "
                             "(ASE is added span by span by channel.propagate)")
    return x
