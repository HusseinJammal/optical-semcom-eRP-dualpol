r"""First-order regular perturbation (RP) and enhanced RP (eRP) of the Manakov equation: the TRAINING channel.

Expanding the field to first order in the Kerr coefficient, for one span of length L_s,

    E_RP(L_s) = D(L_s) E_in + j gamma_M INT_0^Ls D(L_s - xi) [ ||E0(xi)||^2 E0(xi) ] dxi,   E0(xi) = D(xi) E_in

and evaluating the integral with a midpoint rule over B sections (Delta z = L_s / B, midpoints zeta_b):

    E_out = e^{j phi_NL} [ D(L_s) E_in + j gamma_M Delta z sum_b D(L_s - zeta_b) ( ||E_b||^2 - 2 P_b ) E_b ]

with E_b = D(zeta_b) E_in, P_b its mean power, and phi_NL = 2 gamma_M Delta z sum_b P_b. Removing the mean power
(c = 2) from the perturbation and applying it as an exact rotation is the "enhanced" part: it takes the fastest-
growing piece of the nonlinearity out of the truncated term. All B terms are independent, so they parallelize.

* Kerr sign: +j, as in OptiCommPy (tests/test_channel.py checks it against the split-step).
* Manakov: the kick uses the TOTAL power ||E||^2 = |Ex|^2 + |Ey|^2 and gamma_M = (8/9) gamma.
* P0 is averaged over the payload only (guard excluded): averaging over the zero guard dilutes it and under-rotates
  the frame -- most of what eRP is worth at high power.

"""
from __future__ import annotations

from typing import Callable

import torch

from .linear import D_op, propagate_link, span_gain
from .params import FiberConfig

__all__ = ["kerr_kick", "erp_kick", "rp_propagate_link"]

KickFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _inst_power(ex: torch.Tensor, manakov: bool = False) -> torch.Tensor:
    """|E|^2 per pol (scalar) or the total |Ex|^2 + |Ey|^2 summed over the pol axis (Manakov, keepdim)."""
    p = ex.real**2 + ex.imag**2
    if manakov:
        p = p.sum(dim=1, keepdim=True)
    return p


def kerr_kick(ex: torch.Tensor, p0: torch.Tensor, manakov: bool = False) -> torch.Tensor:
    """Plain RP source term ||E||^2 E (``p0`` unused; every kick shares one signature)."""
    return _inst_power(ex, manakov).to(ex.dtype) * ex


def erp_kick(coeff: float, manakov: bool = False) -> KickFn:
    """eRP source term (||E||^2 - c P0) E; the removed part is applied exactly as a rotation."""

    def _kick(ex: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        p2 = _inst_power(ex, manakov)
        return (p2 - coeff * p0).to(ex.dtype) * ex

    return _kick


def _mean_power(ex: torch.Tensor, guard: int = 0, manakov: bool = False) -> torch.Tensor:
    """Mean power per (batch, pol), or total over pols (Manakov), with ``guard`` samples excluded at both ends."""
    p2 = ex.real**2 + ex.imag**2
    if guard > 0 and 2 * guard < ex.shape[-1]:
        p2 = p2[..., guard: ex.shape[-1] - guard]
    if manakov:
        p2 = p2.sum(dim=1, keepdim=True)
    return torch.mean(p2, dim=-1, keepdim=True)


def rp_propagate_link(x: torch.Tensor, cfg, fs_hz: float | None = None, model: str | None = None,
                      kick: KickFn | None = None, p0_guard_samples: int | None = None) -> torch.Tensor:
    """Propagate ``(batch, n_pol, n_samples)`` over ``cfg.link.n_spans`` spans with a noiseless gain per span.

    ``model``: "linear" | "rp" | "erp" (default ``cfg.link.channel_model``). ``p0_guard_samples``: samples ignored at
    each end when computing P0 (default ``cfg.guard_samples``; 0 for a waveform without a guard).

    Within a span, the linear field at the span input E0 generates a kick at local distance xi from D(xi) E0, carried
    to the span output by D(L_s - xi). At each amplifier the gain multiplies the signal and every kick accumulated so
    far, and the accumulated first-order term is propagated linearly through the following spans (it never feeds
    back into the Kerr source: that is what first order means). With gamma = 0 this is exactly the linear model.
    """
    fiber: FiberConfig = cfg.fiber
    fs = cfg.sim.fs_hz if fs_hz is None else fs_hz
    model = (cfg.link.channel_model if model is None else model).lower()
    if model not in ("linear", "rp", "erp"):
        raise ValueError(f"channel_model must be one of linear|rp|erp; got {model!r}")
    if model == "linear":
        return propagate_link(x, cfg, fs_hz=fs)

    manakov = bool(fiber.manakov)
    gamma = fiber.gamma_eff_per_w_km
    is_erp = model == "erp"
    coeff = float(cfg.link.erp_mean_phase_coeff)
    if kick is None:
        kick = (erp_kick(coeff, manakov=manakov) if is_erp
                else (lambda ex, p0: kerr_kick(ex, p0, manakov=manakov)))

    z_span = cfg.link.L_span_eff_km
    n_br = cfg.link.n_branches_per_span
    dz = z_span / n_br
    g = span_gain(fiber, z_span)
    guard = cfg.guard_samples if p0_guard_samples is None else int(p0_guard_samples)

    e0 = x
    acc = torch.zeros_like(x)
    # one mean phase per (batch, pol): images of one batch may have different launch powers
    phi = torch.zeros(*x.shape[:-1], 1, dtype=x.real.dtype, device=x.device)

    for _ in range(cfg.link.n_spans):
        span_contrib = torch.zeros_like(x)
        for b in range(n_br):
            xi = (b + 0.5) * dz
            ex = D_op(e0, fiber, xi, fs)
            p0 = _mean_power(ex, guard, manakov=manakov)
            span_contrib = span_contrib + D_op(kick(ex, p0), fiber, z_span - xi, fs) * dz
            if is_erp:
                phi = phi + coeff * gamma * p0 * dz
        # carry everything from this span's input to its output, then amplify
        acc = D_op(acc, fiber, z_span, fs) * g
        acc = acc + span_contrib * g * (1j * gamma)
        e0 = D_op(e0, fiber, z_span, fs) * g

    out = e0 + acc
    if is_erp:
        out = out * torch.exp(torch.complex(torch.zeros_like(phi), phi)).to(out.dtype)
    return out
