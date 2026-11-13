"""Evaluation on the TRUE channel (Manakov split-step), with per-image seeding.

``evaluate_ssfm`` is the function behind every reported number. It has no backend argument: it can only use the
split-step channel. Image ``i`` of the test set always receives the same amplifier and transceiver noise for a given
seed (default 1234), whatever the batch size, so a test-set block [offset, offset + n) reproduces a full run image by
image and the same images/noise are used for every system.

``evaluate_training_channel`` runs the same chain on the training channel (eRP) or with no channel at all: these are
DIAGNOSTICS (the eRP -> SSFM transfer gap, the fibre-removed control), never results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..impairments.params import nominal_trx
from .metrics import MetricAccumulator

__all__ = ["evaluate_ssfm", "evaluate_training_channel", "per_image_path", "save_per_image"]


@torch.no_grad()
def evaluate_ssfm(model, loader, cfg, device, *, p_tx_dbm: float | None = None, reach_km: float | None = None,
                  span_km: float | None = None, nf_db: float | None = None, trx_overrides: dict | None = None,
                  cpr: str | None = None, told_p_tx_dbm: float | None = None, told_reach_km: float | None = None,
                  limit: int | None = None, seed: int | None = None, index0: int = 0, keep: dict | None = None) -> dict:
    """Metrics of ``model`` over ``loader`` on the split-step channel at one link condition.

    p_tx_dbm, reach_km, span_km, nf_db   the link (None = the nominal value of ``cfg.channel``)
    trx_overrides     changes to the nominal transceiver: keys snr_db, rx_fraction, lw_tx_hz, lw_lo_hz, fo_hz
    told_*            what the networks are told, if different from the true link (wrong-conditioning test)
    limit             number of images (exact); index0 = test-set index of the loader's first image
    keep              dict filled with per-image results (index, label, pred, logits, ssim, psnr, 8-bit reconstruction)
    """
    seed = cfg.eval.seed if seed is None else seed
    model.eval()
    acc, offset = MetricAccumulator(), 0
    for x, y in loader:
        if limit and offset >= limit:
            break
        if limit and offset + x.shape[0] > limit:
            x, y = x[: limit - offset], y[: limit - offset]
        x, y = x.to(device), y.to(device)
        trx = nominal_trx(cfg, x.shape[0], **(trx_overrides or {}))
        x_hat, logits, tx, rx = model.forward_ssfm(x, ssfm_seed=seed, ssfm_index0=index0 + offset, p_tx_dbm=p_tx_dbm,
                                                   return_symbols=True, reach_km=reach_km, span_km=span_km,
                                                   nf_db=nf_db, trx=trx, cpr=cpr, told_p_tx_dbm=told_p_tx_dbm,
                                                   told_reach_km=told_reach_km)
        acc.add(x, x_hat, y, logits, tx_sym=tx, rx_sym=rx)
        if keep is not None:
            keep.setdefault("index", []).append(np.arange(index0 + offset, index0 + offset + x.shape[0]))
            keep.setdefault("logits", []).append(logits.float().cpu().numpy().astype(np.float16))
            keep.setdefault("image_u8", []).append(
                (x_hat.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
        offset += x.shape[0]
    if keep is not None:
        for k in ("index", "logits", "image_u8"):
            keep[k] = np.concatenate(keep[k]) if keep.get(k) else np.zeros(0)
        keep.update(acc.per_image())
    r = acc.result()
    r.update(channel="ssfm", seed=seed, reach_km=reach_km, span_km=span_km, nf_db=nf_db,
             trx_overrides=trx_overrides or {}, cpr=cpr, told_p_tx_dbm=told_p_tx_dbm, told_reach_km=told_reach_km,
             p_tx_dbm=cfg.channel.P_tx_dBm if p_tx_dbm is None else p_tx_dbm)
    return r


@torch.no_grad()
def evaluate_training_channel(model, loader, cfg, backend: str, device, limit=None, seed: int = 4321,
                              reach_km: float | None = None, p_tx_dbm: float | None = None,
                              span_km: float | None = None, nf_db: float | None = None,
                              trx_overrides: dict | None = None) -> dict:
    """DIAGNOSTIC metrics on ``backend`` ("erp", "rp", "linear" or "none"). Never a reported result."""
    if backend == "ssfm":
        raise ValueError("the split-step channel goes through evaluate_ssfm (per-image seeding)")
    model.eval()
    acc = MetricAccumulator()
    gen = torch.Generator().manual_seed(seed)
    for x, y in loader:
        if limit and acc.n >= limit:
            break
        x, y = x.to(device), y.to(device)
        x_hat, logits, tx, rx = model(x, backend=backend, generator=gen, return_symbols=True, reach_km=reach_km,
                                      p_tx_dbm=p_tx_dbm, span_km=span_km, nf_db=nf_db,
                                      trx=nominal_trx(cfg, x.shape[0], **(trx_overrides or {})))
        acc.add(x, x_hat, y, logits, tx_sym=tx, rx_sym=rx)
    r = acc.result()
    r.update(channel=backend, diagnostic=True)
    return r


def per_image_path(dirpath, tag: str, reach_km, p_tx_dbm) -> Path:
    rk = "nominal" if reach_km is None else f"{float(reach_km):g}"
    return Path(dirpath) / f"{tag}__L{rk}_P{float(p_tx_dbm):+g}.npz"


def save_per_image(dirpath, tag: str, reach_km, p_tx_dbm, keep: dict, meta: dict) -> Path:
    """One compressed .npz per (tag, condition): the per-image arrays plus a JSON ``meta`` string."""
    path = per_image_path(dirpath, tag, reach_km, p_tx_dbm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=str)), **keep)
    return path
