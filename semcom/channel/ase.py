"""Per-span EDFA amplified spontaneous emission (ASE) and the ASE-limited SNR.

Reproduces the EDFA of OptiCommPy (optic/models/devices.py) with the EXACT spontaneous-emission factor:

    G = alpha[dB/km] * L_span   [dB]          nsp = (G NF - 1) / (2 (G - 1))       (not the NF/2 approximation)
    N_ase = (G - 1) nsp h Fc                    noise variance per sample = N_ase * Fs   (white over the simulation band)

ASE is added at the OUTPUT of each amplifier, independently on each polarization, so noise from span k propagates
through spans k+1..N and accumulates dispersion like the signal. The in-band floor per polarization is
n_spans * N_ase * Rs (a unit-energy RRC matched filter has noise bandwidth Rs); the TOTAL floor over both
polarizations is twice that, and it is the one the dual-pol model-accuracy (eRP validity) checks must use.
"""
from __future__ import annotations

import math

import scipy.constants as const
import torch

__all__ = ["span_gain_linear", "nsp_exact", "ase_variance_per_span", "ase_floor_watt", "ase_snr_db",
           "ase_floor_watt_total", "ase_snr_db_total", "add_ase"]


def span_gain_linear(cfg) -> float:
    """Amplifier POWER gain, linear: 10^(alpha[dB/km] L_span / 10)."""
    return 10.0 ** (cfg.fiber.alpha_db_km * cfg.link.L_span_eff_km / 10.0)


def nsp_exact(g_lin: float, nf_lin: float) -> float:
    if g_lin <= 1.0:
        raise ValueError(f"span gain must exceed 1 for ASE to be defined; got {g_lin}")
    return (g_lin * nf_lin - 1.0) / (2.0 * (g_lin - 1.0))


def ase_variance_per_span(cfg) -> float:
    """Total complex ASE variance per sample added by ONE amplifier, per polarization [W] (full simulation band)."""
    g_lin = span_gain_linear(cfg)
    nf_lin = 10.0 ** (cfg.link.NF_db / 10.0)
    n_ase = (g_lin - 1.0) * nsp_exact(g_lin, nf_lin) * const.h * cfg.fiber.Fc_hz
    return n_ase * cfg.sim.fs_hz


def ase_floor_watt(cfg, bandwidth_hz: float | None = None) -> float:
    """In-band ASE power at the receiver, ONE polarization, summed over all spans [W]."""
    b = cfg.sim.symbol_rate_hz if bandwidth_hz is None else bandwidth_hz
    per_span_in_band = ase_variance_per_span(cfg) * (b / cfg.sim.fs_hz)
    return cfg.link.n_spans * per_span_in_band


def ase_snr_db(cfg, p_tx_watt: float | None = None) -> float:
    """ASE-limited SNR [dB], single-polarization floor."""
    p = cfg.sim.P_tx_watt if p_tx_watt is None else p_tx_watt
    return 10.0 * math.log10(p / ase_floor_watt(cfg))


def ase_floor_watt_total(cfg, bandwidth_hz: float | None = None) -> float:
    """In-band ASE power summed over ALL polarizations [W]."""
    return int(cfg.sim.n_pol) * ase_floor_watt(cfg, bandwidth_hz)


def ase_snr_db_total(cfg, p_tx_watt: float | None = None) -> float:
    """ASE-limited SNR [dB] of the TOTAL launch power against the TOTAL noise (3.01 dB below the per-pol figure)."""
    p = cfg.sim.P_tx_watt if p_tx_watt is None else p_tx_watt
    return 10.0 * math.log10(p / ase_floor_watt_total(cfg))


def add_ase(x: torch.Tensor, cfg, generator: torch.Generator | None = None) -> torch.Tensor:
    """Add one amplifier's ASE (the gain is applied by the propagation code). Independent draw per polarization."""
    sigma2 = ase_variance_per_span(cfg)
    rdtype = x.real.dtype
    scale = math.sqrt(sigma2 / 2.0)

    def _n():
        return torch.randn(x.shape, generator=generator, dtype=rdtype, device=x.device) * scale

    return x + torch.complex(_n(), _n())
