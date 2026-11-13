"""Transceiver noise: signal-proportional circular AWGN (a back-to-back SNR ceiling, independent of launch power).

The total transceiver noise 1/SNR_TRx is split between the two ends,

    1/SNR_TRx = (1 - kappa_R)/SNR_TRx  [TX, on the symbols before pulse shaping: in-band, propagates through the fibre]
              +      kappa_R /SNR_TRx  [RX, on the sampled symbols]

so each end adds noise at SNR_TRx - 10 log10(share) (= SNR_TRx + 3.01 dB per end for kappa_R = 0.5).
"""
from __future__ import annotations

import math

import torch

from ..utils.random import PerImageRng

__all__ = ["add_symbol_noise", "tx_share_snr_db", "rx_share_snr_db"]


def tx_share_snr_db(snr_db: torch.Tensor, rx_fraction: float) -> torch.Tensor:
    """SNR of the TX share of the transceiver noise."""
    return snr_db - 10.0 * math.log10(max(1.0 - rx_fraction, 1e-300))


def rx_share_snr_db(snr_db: torch.Tensor, rx_fraction: float) -> torch.Tensor:
    """SNR of the RX share of the transceiver noise."""
    return snr_db - 10.0 * math.log10(rx_fraction)


def add_symbol_noise(s: torch.Tensor, snr_db: torch.Tensor, ref: torch.Tensor, rng: PerImageRng,
                     stream: str) -> torch.Tensor:
    """``s`` + circular AWGN of per-image variance mean|ref|^2 / 10^(snr_db/10)."""
    rdtype = s.real.dtype
    pw = torch.mean(ref.real ** 2 + ref.imag ** 2, dim=tuple(range(1, ref.ndim))).detach()          # (B,)
    sigma2 = pw / torch.pow(torch.tensor(10.0, dtype=torch.float64, device=s.device),
                            snr_db.to(device=s.device, dtype=torch.float64) / 10.0).to(pw.dtype)
    scale = torch.sqrt(sigma2 / 2.0).to(rdtype).reshape(-1, *([1] * (s.ndim - 1)))
    n = rng.randn(stream, (2, *s.shape[1:]), rdtype)
    return s + torch.complex(n[:, 0] * scale, n[:, 1] * scale).to(s.dtype)
