"""Per-image transceiver parameters: nominal (evaluation) or randomly drawn and hidden from the networks (training).

    snr_db        back-to-back transceiver SNR, SNR_TRx       nominal 23 dB      training U[20, 26] dB
    rx_fraction   kappa_R, share of that noise at the RX       0.5                0.5
    lw_tx_hz      TX laser linewidth                           100 kHz            U[0, 500] kHz
    lw_lo_hz      local-oscillator linewidth                   100 kHz            U[0, 500] kHz
    fo_hz         residual frequency offset                    0                  U[-10, +10] MHz
    random_phase  uniform initial carrier phase per image      on                 on
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["TrxParams", "nominal_trx", "draw_trx"]


@dataclass(frozen=True)
class TrxParams:
    """Transceiver impairments of one batch: ``(B,)`` float64 tensors, or None = that block off."""

    snr_db: torch.Tensor | None = None
    rx_fraction: float = 0.5
    lw_tx_hz: torch.Tensor | None = None
    lw_lo_hz: torch.Tensor | None = None
    fo_hz: torch.Tensor | None = None
    random_phase: bool = False

    def any_active(self) -> bool:
        def on(t):
            return t is not None and bool(torch.any(t != 0)) if torch.is_tensor(t) else False
        snr_on = self.snr_db is not None and bool(torch.any(torch.isfinite(self.snr_db)))
        return snr_on or on(self.lw_tx_hz) or on(self.lw_lo_hz) or on(self.fo_hz) or self.random_phase


def _full(b: int, v: float) -> torch.Tensor:
    return torch.full((b,), float(v), dtype=torch.float64)


def nominal_trx(cfg, batch: int, *, snr_db: float | None = None, rx_fraction: float | None = None,
                lw_tx_hz: float | None = None, lw_lo_hz: float | None = None, fo_hz: float | None = None,
                random_phase: bool | None = None) -> TrxParams | None:
    """The configured (nominal) transceiver for ``batch`` images, each field overridable; None when all is off."""
    c = cfg.channel
    snr = c.trx_snr_db if snr_db is None else snr_db
    p = TrxParams(
        snr_db=None if math.isinf(snr) else _full(batch, snr),
        rx_fraction=c.trx_rx_fraction if rx_fraction is None else float(rx_fraction),
        lw_tx_hz=_full(batch, c.linewidth_tx_hz if lw_tx_hz is None else lw_tx_hz),
        lw_lo_hz=_full(batch, c.linewidth_lo_hz if lw_lo_hz is None else lw_lo_hz),
        fo_hz=_full(batch, c.freq_offset_hz if fo_hz is None else fo_hz),
        random_phase=c.random_carrier_phase if random_phase is None else bool(random_phase),
    )
    return p if p.any_active() else None


def draw_trx(cfg, batch: int, gen: torch.Generator) -> TrxParams:
    """HIDDEN per-image transceiver draw for training (SNR_TRx, both linewidths, frequency offset, random phase)."""
    c = cfg.channel

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(batch, generator=gen, dtype=torch.float64)

    return TrxParams(snr_db=u(c.trx_snr_min_db, c.trx_snr_max_db), rx_fraction=c.trx_rx_fraction,
                     lw_tx_hz=u(0.0, c.linewidth_max_hz), lw_lo_hz=u(0.0, c.linewidth_max_hz),
                     fo_hz=u(-c.freq_offset_max_hz, c.freq_offset_max_hz), random_phase=True)
