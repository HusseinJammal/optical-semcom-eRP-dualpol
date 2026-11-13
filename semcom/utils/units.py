"""Unit conversions shared by every block.

Time is in SECONDS and distance in KM throughout the propagation code, so

    omega  [rad/s]     beta2  [s^2/km]     z  [km]

and the dispersive phase ``(beta2/2) * omega**2 * z`` is dimensionless. ``beta2`` is only ever *reported* in
ps^2/km (multiply by 1e24). These formulas are written exactly as OptiCommPy writes them (channels.py), so the
differentiable model and the split-step reference agree to the last bit.
"""
from __future__ import annotations

import math

import scipy.constants as const

__all__ = ["C_KM_S", "alpha_db_km_to_1_per_km", "alpha_1_per_km_to_db_km", "beta2_from_D", "D_from_beta2",
           "dbm_to_watt", "watt_to_dbm"]

C_KM_S = const.c / 1e3  # speed of light in vacuum [km/s]


def alpha_db_km_to_1_per_km(alpha_db_km: float) -> float:
    """POWER attenuation coefficient [1/km]: 0.2 dB/km -> 0.04605 1/km.

    The field decays as ``exp(-alpha*z/2)`` and the power as ``exp(-alpha*z)``.
    """
    return alpha_db_km / (10 * math.log10(math.e))


def alpha_1_per_km_to_db_km(alpha_1_per_km: float) -> float:
    """Inverse of :func:`alpha_db_km_to_1_per_km`."""
    return alpha_1_per_km * (10 * math.log10(math.e))


def beta2_from_D(D_ps_nm_km: float, Fc_hz: float) -> float:
    """beta2 = -D * lambda**2 / (2*pi*c) in s^2/km. NEGATIVE for standard SMF (-20.47 ps^2/km at 16 ps/(nm km))."""
    lam_km = C_KM_S / Fc_hz
    return -(D_ps_nm_km * lam_km**2) / (2 * math.pi * C_KM_S)


def D_from_beta2(beta2_s2_km: float, Fc_hz: float) -> float:
    """Inverse of :func:`beta2_from_D`, in ps/(nm km)."""
    lam_km = C_KM_S / Fc_hz
    return -(beta2_s2_km * 2 * math.pi * C_KM_S) / lam_km**2


def dbm_to_watt(p_dbm: float) -> float:
    """0 dBm -> 1 mW."""
    return 10.0 ** (p_dbm / 10.0) / 1000.0


def watt_to_dbm(p_watt: float) -> float:
    return 10.0 * math.log10(p_watt * 1000.0)
