"""Evaluate a trained semantic transceiver on the TRUE channel (Manakov split-step), per-image seeded.

    # the three reference links of the paper, all 10,000 test images
    python scripts/evaluate.py --ckpt runs/paper_v3_s0/best.pt --cond 400:0 1440:0 2480:-1 --link-snr-images 64

    # transceiver-SNR stress at 400 km (Fig. 2c): one call per SNR_TRx
    python scripts/evaluate.py --ckpt ... --cond 400:0 --trx-snr-db 11 --link-snr-images 64

    # a test-set block (images [500, 5250)), per-image results saved
    python scripts/evaluate.py --ckpt ... --cond 2480:-1 --offset 500 --limit 4750 --save-per-image out/per_image

For each condition (reach km : launch power dBm): optionally the eRP validity on the learned symbols
(--erp-validity-batches, reported, labels the result), the SSFM metrics, optionally the link SNR of uniform 64-QAM
over the same chain, and optionally training-channel diagnostics. Writes one JSON report (--out/--tag).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.evaluation import (erp_validity, evaluate_ssfm, evaluate_training_channel, link_snr,  # noqa: E402
                               save_per_image)
from semcom.system.checkpoint import load_checkpoint  # noqa: E402
from semcom.training.data import build_datasets  # noqa: E402


def trx_overrides(a) -> dict:
    ov = {}
    if a.trx_snr_db is not None:
        ov["snr_db"] = a.trx_snr_db
    if a.kappa_r is not None:
        ov["rx_fraction"] = a.kappa_r
    if a.linewidth_khz is not None:
        ov["lw_tx_hz"] = ov["lw_lo_hz"] = a.linewidth_khz * 1e3
    if a.fo_mhz is not None:
        ov["fo_hz"] = a.fo_mhz * 1e6
    return ov


def main():
    ap = argparse.ArgumentParser(description="Evaluate on the split-step Fourier channel.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond", nargs="+", required=True, help="reach_km:power_dBm pairs, e.g. 400:0 1440:0 2480:-1")
    ap.add_argument("--limit", type=int, default=None, help="number of test images (default: all 10,000)")
    ap.add_argument("--offset", type=int, default=0, help="first test image (block evaluation)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-seed", type=int, default=None, help="noise seed (default 1234)")
    # link / transceiver overrides (default: the nominal link of the checkpoint)
    ap.add_argument("--span-km", type=float, default=None, help="span length; reaches must be whole spans of it")
    ap.add_argument("--nf-db", type=float, default=None)
    ap.add_argument("--trx-snr-db", type=float, default=None, help="transceiver SNR (inf = none)")
    ap.add_argument("--kappa-r", type=float, default=None, help="receiver share of the transceiver noise")
    ap.add_argument("--linewidth-khz", type=float, default=None, help="linewidth of EACH laser")
    ap.add_argument("--fo-mhz", type=float, default=None, help="residual frequency offset")
    ap.add_argument("--cpr", choices=("pilot", "genie", "none"), default=None)
    ap.add_argument("--told-p-offset-db", type=float, default=None, help="tell the networks P + this (wrong-c test)")
    ap.add_argument("--told-reach-offset-km", type=float, default=None, help="tell the networks L + this")
    # extras
    ap.add_argument("--erp-validity-batches", type=int, default=0,
                    help="measure the eRP model error on the learned symbols with this many batches (0 = skip)")
    ap.add_argument("--link-snr-images", type=int, default=0, help="link SNR from this many uniform 64-QAM images")
    ap.add_argument("--diagnostics", action="store_true", help="also run the eRP and no-channel diagnostics")
    ap.add_argument("--save-per-image", default=None, help="directory for per-image .npz files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    model, cfg, ck = load_checkpoint(a.ckpt, device)
    test = build_datasets(cfg, a.data_root).test
    n = len(test) - a.offset if a.limit is None else a.limit
    loader = DataLoader(Subset(test, range(a.offset, a.offset + n)), batch_size=a.batch_size, shuffle=False)
    ov = trx_overrides(a)
    tag = a.tag or f"eval_{cfg.run_name}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {cfg.describe()}\n[ckpt] {a.ckpt} epoch {ck.get('epoch')} | {n} images from #{a.offset}", flush=True)

    reports = []
    for cond in a.cond:
        L, P = (float(v) for v in cond.split(":"))
        rep = {"reach_km": L, "p_tx_dbm": P, "span_km": a.span_km, "nf_db": a.nf_db, "trx_overrides": ov}
        if a.erp_validity_batches > 0:
            val_loader = DataLoader(test, batch_size=a.batch_size, shuffle=False)
            v = erp_validity(model, val_loader, cfg, device, a.erp_validity_batches, p_tx_dbm=P, reach_km=L,
                             span_km=a.span_km, nf_db=a.nf_db)
            rep["erp_validity"] = v
            print(f"[{L:g} km, {P:+g} dBm] eRP model error {v['nmse_db_worst']:.2f} dB vs floor {v['ase_floor_db']:.2f} "
                  f"dB -> margin {v['margin_db_worst']:+.2f} dB ({v['verdict']})", flush=True)
        keep = {} if a.save_per_image else None
        told_p = None if a.told_p_offset_db is None else P + a.told_p_offset_db
        told_l = None if a.told_reach_offset_km is None else L + a.told_reach_offset_km
        r = evaluate_ssfm(model, loader, cfg, device, p_tx_dbm=P, reach_km=L, span_km=a.span_km, nf_db=a.nf_db,
                          trx_overrides=ov, cpr=a.cpr, told_p_tx_dbm=told_p, told_reach_km=told_l, limit=n,
                          seed=a.eval_seed, index0=a.offset, keep=keep)
        r.pop("confusion", None)
        rep["ssfm"] = r
        print(f"[{L:g} km, {P:+g} dBm] SSFM: accuracy {r['accuracy']:.2f} %  SSIM {r['ssim']:.4f}  "
              f"PSNR {r['psnr_per_image']:.2f} dB  SSIM<0.5 {100 * r['frac_ssim_below_0p5']:.2f} %  "
              f"(n={r['n']})", flush=True)
        if keep is not None:
            meta = {"ckpt": a.ckpt, "tag": tag, "reach_km": L, "p_tx_dbm": P, "offset": a.offset, "n": r["n"],
                    "seed": r["seed"], "span_km": a.span_km, "nf_db": a.nf_db, "trx_overrides": ov}
            rep["per_image_file"] = str(save_per_image(a.save_per_image, tag, L, P, keep, meta))
        if a.link_snr_images:
            s = link_snr(cfg, device, p_tx_dbm=P, reach_km=L, span_km=a.span_km, nf_db=a.nf_db, trx_overrides=ov,
                         n_img=a.link_snr_images, seed=a.eval_seed)
            rep["link_snr"] = s
            print(f"[{L:g} km, {P:+g} dBm] link SNR {s['snr_eff_ls_db']:.2f} dB (LS)", flush=True)
        if a.diagnostics:
            kw = dict(reach_km=L, p_tx_dbm=P, span_km=a.span_km, nf_db=a.nf_db, trx_overrides=ov)
            diag = {b: evaluate_training_channel(model, loader, cfg, b, device, n, **kw) for b in ("erp", "none")}
            for v in diag.values():
                v.pop("confusion", None)
            rep["diagnostics"] = diag
            print(f"[{L:g} km, {P:+g} dBm] DIAGNOSTICS (not results): eRP SSIM {diag['erp']['ssim']:.4f} "
                  f"(gap to SSFM {diag['erp']['ssim'] - r['ssim']:+.5f}); fibre removed SSIM {diag['none']['ssim']:.4f}",
                  flush=True)
        reports.append(rep)
        (out / f"{tag}.json").write_text(json.dumps({"ckpt": a.ckpt, "epoch": ck.get("epoch"), "offset": a.offset,
                                                    "limit": n, "conditions": reports}, indent=1, default=str))
    print(f"wrote {out / f'{tag}.json'}")


if __name__ == "__main__":
    main()
