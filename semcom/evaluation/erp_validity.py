"""eRP validity on the LEARNED symbols: is the training channel still an accurate model of the true channel?

The noise-free model error

    epsilon = E || E_out^eRP - E_out^SSFM ||^2 / E || E_out^SSFM ||^2

is measured on the encoder's own symbols (with the same neighbour frames on both sides, ASE off on both sides, no
transceiver impairments, raw received frame without CPR) and compared with the TOTAL (both-polarization) ASE-limited
noise floor. The margin is  floor_dB - 10 log10(epsilon)  (positive = model error below the noise).

* Training powers are limited to P_max(L, L_s), the highest power with margin >= 9 dB at the lowest noise figure
  (``semcom.config.PAPER_POWER_LIMIT_TABLE``): the model error is then at most ~1/8 of the ASE noise.
* Verdict used by ``scripts/evaluate.py``: PASS >= +6 dB; STOP_AND_REPORT in [0, +6); FAIL < 0.

ASE must be OFF on both sides: two independent noise draws would make the comparison measure noise (an effective floor
3.01 dB worse) instead of model error.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..system.link import ase_snr_db_total, transmit
from .metrics import papr_db

__all__ = ["PASS_MARGIN_DB", "TRAINING_RULE_MARGIN_DB", "verdict", "erp_validity"]

PASS_MARGIN_DB = 6.0
TRAINING_RULE_MARGIN_DB = 9.0


def verdict(margin_db: float) -> str:
    if margin_db >= PASS_MARGIN_DB:
        return "PASS"
    return "STOP_AND_REPORT" if margin_db >= 0.0 else "FAIL"


@torch.no_grad()
def erp_validity(model, loader, cfg, device, n_batches: int = 8, p_tx_dbm: float | None = None,
                 reach_km: float | None = None, span_km: float | None = None, nf_db: float | None = None) -> dict:
    """Worst-case eRP model error vs the total ASE floor over ``n_batches`` batches of ``loader``.

    ``nf_db`` only sets the floor (ASE is off on both sides): pass the LOWEST noise figure of a randomized range for
    the worst case (less ASE = tighter floor = less margin).
    """
    floor_db = -ase_snr_db_total(cfg, p_tx_dbm, reach_km, span_km, nf_db)
    errs, paprs, offset = [], [], 0
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        tx = model.encoder(x, model.conditioning(p_tx_dbm, reach_km, device=device))
        common = dict(amp="ideal", p_tx_dbm=p_tx_dbm, reach_km=reach_km, span_km=span_km, trx=None, cpr="none",
                      trx_seed=97531, ssfm_index0=offset)
        rx_erp = transmit(tx, cfg, backend="erp", generator=torch.Generator().manual_seed(1234 + i), **common)
        rx_ssfm = transmit(tx, cfg, backend="ssfm", **common)
        offset += x.shape[0]
        num = torch.sum(torch.abs(rx_erp - rx_ssfm) ** 2)
        den = torch.sum(torch.abs(rx_ssfm) ** 2)
        errs.append(10.0 * math.log10(float(num / den)))
        paprs.append(papr_db(tx))
    e = np.array(errs)
    worst = float(e.max())
    return {"nmse_db_mean": float(e.mean()), "nmse_db_std": float(e.std()), "nmse_db_worst": worst,
            "n_batches": len(errs), "ase_floor_db": floor_db, "margin_db_worst": floor_db - worst,
            "papr_db_mean": float(np.mean(paprs)), "verdict": verdict(floor_db - worst),
            "p_tx_dbm": p_tx_dbm, "reach_km": reach_km, "span_km": span_km, "floor_nf_db": nf_db}
