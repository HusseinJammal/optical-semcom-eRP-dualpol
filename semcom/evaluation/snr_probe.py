"""Link SNR: uniform random 64-QAM sent through the same transmitter, link and receiver DSP (incl. pilot CPR).

This is the SNR a conventional system's coded symbols see (their coded bits are ~uniform), so it places both the
semantic transceiver and JPEG + LDPC on one SNR axis. Reported two ways:

    snr_eff_ls_db   least-squares complex gain removed (the conventional effective SNR; used in the paper)
    snr_eff_db      unit-power normalised EVM (biased high at low SNR)

Noise seeds use image indices 10**6 + i, disjoint from the test images'.
"""
from __future__ import annotations

import math

import torch

from ..impairments.params import nominal_trx
from ..mapper.qam import qam_grid
from ..system.link import transmit
from .metrics import evm_energies, ls_snr_energies

__all__ = ["link_snr"]


@torch.no_grad()
def link_snr(cfg, device, *, p_tx_dbm=None, reach_km=None, span_km=None, nf_db=None, trx_overrides: dict | None = None,
             n_img: int = 64, seed: int | None = None, backend: str = "ssfm", amp: str | None = None) -> dict:
    seed = cfg.eval.seed if seed is None else seed
    g = torch.Generator().manual_seed(4242)
    grid = qam_grid(64)
    idx = torch.randint(0, 64, (n_img, 1, cfg.channel.n_pol, cfg.k_per_pol), generator=g)
    x = torch.complex(grid[idx, 0], grid[idx, 1]).to(torch.complex64).to(device).reshape(n_img, cfg.channel.n_pol,
                                                                                         cfg.k_per_pol)
    trx = nominal_trx(cfg, n_img, **(trx_overrides or {}))
    kw = dict(ssfm_seed=seed, ssfm_index0=10**6) if backend == "ssfm" else dict(trx_seed=seed, ssfm_index0=10**6)
    y = transmit(x, cfg, backend=backend, p_tx_dbm=p_tx_dbm, reach_km=reach_km, span_km=span_km, nf_db=nf_db,
                 trx=trx, amp=amp, **kw)
    num, den = evm_energies(x, y)
    sig, err = ls_snr_energies(x, y)
    return {"n": n_img, "snr_eff_db": 10.0 * math.log10(float(den.sum()) / max(float(num.sum()), 1e-300)),
            "snr_eff_ls_db": 10.0 * math.log10(float(sig.sum()) / max(float(err.sum()), 1e-300)),
            "p_tx_dbm": p_tx_dbm, "reach_km": reach_km, "span_km": span_km, "nf_db": nf_db,
            "trx_overrides": trx_overrides or {}}
