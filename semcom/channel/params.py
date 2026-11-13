"""Physical link parameters: ONE frozen config shared by the differentiable models and the SSFM reference.

``PropagationConfig(fiber, link, sim)`` is what every propagation function takes. It is built from the system
``semcom.config.ChannelConfig`` by :func:`semcom.system.link.build_propagation_config`, once per call, so the training
model (eRP) and the evaluation channel (SSFM) cannot drift apart.

Conventions (see docs/physics_and_conventions.md):

* NLSE / Manakov equation with a POSITIVE Kerr sign, as in OptiCommPy:
      dE/dz = -(alpha/2) E - j (beta2/2) d2E/dt2 + j gamma_M ||E||^2 E
* Manakov factor: gamma_M = (8/9) gamma. It is applied ONCE in the differentiable model
  (``FiberConfig.gamma_eff_per_w_km``); the SSFM reference applies it itself on the bare gamma.
* Launch power and ASE floors are TOTAL over both polarizations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import scipy.constants as const

from ..utils.units import alpha_db_km_to_1_per_km, beta2_from_D, dbm_to_watt

__all__ = ["MANAKOV_POL_FACTOR", "GUARD_SYMBOLS_PER_KM", "GUARD_MARGIN", "GUARD_QUANTUM", "FiberConfig",
           "LinkConfig", "SimConfig", "PropagationConfig"]

MANAKOV_POL_FACTOR = 8.0 / 9.0

# Guard-interval calibration -- MEASURED (broadband probe, opposite-edge leakage < 1e-8):
#     L [km]      80     240     400     800
#     min guard   48     144     224     432      symbols   (0.60 / 0.60 / 0.56 / 0.54 symbols per km)
# The rule uses the largest coefficient (0.60 /km) x 1.15 margin, rounded up to a multiple of 16 symbols. The
# analytic group-delay spread under-predicts this tail criterion and must not be used to size the guard.
GUARD_SYMBOLS_PER_KM = 0.60
GUARD_MARGIN = 1.15
GUARD_QUANTUM = 16


@dataclass(frozen=True)
class FiberConfig:
    """Fibre physics. ``gamma_per_w_km`` is the BARE Kerr coefficient."""

    alpha_db_km: float = 0.2
    D_ps_nm_km: float = 16.0
    gamma_per_w_km: float = 0.0       # 0 = linear fibre; 1.3 /(W km) is the paper's fibre
    Fc_hz: float = 193.1e12
    manakov: bool = False             # True: dual-pol Manakov coupling through ||E||^2 = |Ex|^2 + |Ey|^2

    @property
    def gamma_eff_per_w_km(self) -> float:
        """Kerr coefficient of OUR model: (8/9) gamma in Manakov mode (the only place the 8/9 enters)."""
        if self.manakov:
            return MANAKOV_POL_FACTOR * self.gamma_per_w_km
        return self.gamma_per_w_km

    @property
    def alpha_1_per_km(self) -> float:
        """POWER attenuation coefficient [1/km]."""
        return alpha_db_km_to_1_per_km(self.alpha_db_km)

    @property
    def beta2_s2_km(self) -> float:
        """Group-velocity dispersion [s^2/km], negative for SMF at 1550 nm."""
        return beta2_from_D(self.D_ps_nm_km, self.Fc_hz)

    @property
    def wavelength_m(self) -> float:
        return const.c / self.Fc_hz


@dataclass(frozen=True)
class LinkConfig:
    """Span geometry, amplification and the propagation model."""

    Ltotal_km: float = 400.0
    Lspan_km: float = 80.0
    hz_km: float = 1.0                # split-step size of the SSFM reference
    amp: str = "ideal"                # "ideal" (noiseless gain) | "edfa" (gain + ASE) | None
    NF_db: float = 4.5
    channel_model: str = "linear"     # differentiable model: linear | rp | erp
    # Branches (Kerr kicks) per span of the first-order quadrature. MEASURED: the model error against SSFM saturates
    # at 8 per 80 km span (-28.2 dB for plain RP at 0 dBm), because the residual is first-order truncation in gamma,
    # not quadrature error. The system uses ceil(span / 10 km).
    n_branches_per_span: int = 8
    # eRP mean-phase coefficient c in (||E||^2 - c P0): 2 for a circular-Gaussian field. To first order the removed
    # term and the rotation exp(j c gamma P0 z) cancel for ANY c, so eRP equals RP1 to first order.
    erp_mean_phase_coeff: float = 2.0
    ase_seed: int | None = None       # must stay None for the batched SSFM (seeds are per image)

    @property
    def n_spans(self) -> int:
        """OptiCommPy floors Ltotal / Lspan; the system refuses reaches that are not whole spans."""
        return int(math.floor(self.Ltotal_km / self.Lspan_km))

    @property
    def n_steps_per_span(self) -> int:
        return int(math.floor(self.Lspan_km / self.hz_km))

    @property
    def L_eff_km(self) -> float:
        """Distance actually propagated after both floors (used by CD compensation)."""
        return self.n_spans * self.n_steps_per_span * self.hz_km

    @property
    def L_span_eff_km(self) -> float:
        """Distance actually propagated per span."""
        return self.n_steps_per_span * self.hz_km

    @property
    def n_branches(self) -> int:
        return self.n_branches_per_span * self.n_spans


@dataclass(frozen=True)
class SimConfig:
    """Waveform and simulation grid."""

    n_sym: int = 512                  # symbols per polarization seen by the pulse shaper (frame + pads)
    n_pol: int = 1
    sps: int = 16
    rolloff: float = 0.1
    n_taps: int = 257
    symbol_rate_hz: float = 32e9
    P_tx_dBm: float = 0.0             # TOTAL launch power over all polarizations
    guard_symbols_override: int | None = None   # pin the guard (the system always does: reach-scaled + 5-smooth)

    @property
    def fs_hz(self) -> float:
        return self.symbol_rate_hz * self.sps

    @property
    def n_payload_samples(self) -> int:
        return self.n_sym * self.sps

    @property
    def P_tx_watt(self) -> float:
        return dbm_to_watt(self.P_tx_dBm)


@dataclass(frozen=True)
class PropagationConfig:
    """The single object handed to every propagation function (differentiable and reference)."""

    fiber: FiberConfig = field(default_factory=FiberConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @property
    def guard_symbols(self) -> int:
        """Guard interval [symbols per side]: the override, else the measured reach-scaled rule."""
        if self.sim.guard_symbols_override is not None:
            return int(self.sim.guard_symbols_override)
        raw = GUARD_SYMBOLS_PER_KM * self.link.L_eff_km * GUARD_MARGIN
        return int(math.ceil(raw / GUARD_QUANTUM) * GUARD_QUANTUM)

    @property
    def guard_samples(self) -> int:
        return self.guard_symbols * self.sim.sps

    def cd_spread_symbols(self, L_km: float | None = None, bandwidth_hz: float | None = None) -> float:
        """One-sided CD group-delay spread |beta2| omega_max L in symbol periods (a cross-check, not a guard rule)."""
        L = self.link.L_eff_km if L_km is None else L_km
        b = self.sim.symbol_rate_hz * (1.0 + self.sim.rolloff) if bandwidth_hz is None else bandwidth_hz
        omega_max = 2 * math.pi * (b / 2.0)
        return abs(self.fiber.beta2_s2_km) * omega_max * L * self.sim.symbol_rate_hz
