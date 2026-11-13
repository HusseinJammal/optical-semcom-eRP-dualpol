"""The conventional chain (JPEG + IEEE 802.11n LDPC + Gray 64-QAM) over the SAME link as the semantic transceiver.

    python scripts/run_baseline.py --rates 1/2 2/3 3/4 5/6 --cond 400:0 1440:0 2480:-1 \
        --classifier-dir runs/task_clf --out reports/baseline_reference_links.json

Same test images, frame (512 symbols + 9 pilots per polarization), transceiver, split-step channel and per-image noise
seeds as scripts/evaluate.py. Each rate is classified by the ResNet-18 trained on that rate's error-free JPEG output
(<classifier-dir>/jpeg_r<rate>/best.pt, scripts/train_baseline_classifier.py). The link configuration is the paper's
(or the channel part of --cfg-from <checkpoint>, so both systems see byte-identical channels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.baseline import FrameCodec, load_task_classifier, run_jpeg_ldpc  # noqa: E402
from semcom.config import Config  # noqa: E402
from semcom.evaluation import save_per_image  # noqa: E402
from semcom.training.data import build_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="JPEG + 802.11n LDPC over the split-step channel.")
    ap.add_argument("--rates", nargs="+", default=["1/2", "2/3", "3/4", "5/6"])
    ap.add_argument("--cond", nargs="+", required=True, help="reach_km:power_dBm pairs")
    ap.add_argument("--classifier-dir", required=True, help="directory with jpeg_r12/, jpeg_r23/, ... best.pt")
    ap.add_argument("--cfg-from", default=None, help="semcom checkpoint whose configuration to use")
    ap.add_argument("--images", type=int, default=10000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--span-km", type=float, default=None)
    ap.add_argument("--nf-db", type=float, default=None)
    ap.add_argument("--trx-snr-db", type=float, default=None)
    ap.add_argument("--linewidth-khz", type=float, default=None)
    ap.add_argument("--fo-mhz", type=float, default=None)
    ap.add_argument("--no-ceiling", action="store_true", help="skip the error-free (no channel) ceiling")
    ap.add_argument("--save-per-image", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    cfg = Config.from_dict(torch.load(a.cfg_from, map_location="cpu", weights_only=False)["config"]) \
        if a.cfg_from else Config()
    ov = {}
    if a.trx_snr_db is not None:
        ov["snr_db"] = a.trx_snr_db
    if a.linewidth_khz is not None:
        ov["lw_tx_hz"] = ov["lw_lo_hz"] = a.linewidth_khz * 1e3
    if a.fo_mhz is not None:
        ov["fo_hz"] = a.fo_mhz * 1e6
    test = build_datasets(cfg, a.data_root).test
    idx = range(a.offset, a.offset + a.images)
    x = torch.stack([test[i][0] for i in idx])
    y = torch.tensor([test[i][1] for i in idx])
    out = {"images": a.images, "offset": a.offset, "span_km": a.span_km, "nf_db": a.nf_db, "trx_overrides": ov,
           "rates": {}}
    tag = Path(a.out).stem
    for rate in a.rates:
        codec = FrameCodec(rate)
        clf = load_task_classifier(Path(a.classifier_dir) / f"jpeg_r{rate.replace('/', '')}" / "best.pt", device)
        blk = {"eff_rate": codec.eff_rate, "budget_bytes": codec.budget_bytes, "rows": []}
        if not a.no_ceiling:
            r = run_jpeg_ldpc(x, y, cfg, clf, device, codec, backend="none")
            blk["no_channel"] = r
            print(f"[rate {rate}] error-free: acc {r['accuracy']:.2f} ssim {r['ssim']:.4f} "
                  f"psnr {r['psnr_per_image']:.2f} quality {r['jpeg_quality_mean']:.1f}", flush=True)
        for cnd in a.cond:
            L, P = (float(v) for v in cnd.split(":"))
            t0 = time.time()
            keep = {} if a.save_per_image else None
            r = run_jpeg_ldpc(x, y, cfg, clf, device, codec, backend="ssfm", p_tx_dbm=P, reach_km=L, span_km=a.span_km,
                              nf_db=a.nf_db, index0=a.offset, trx_overrides=ov, keep=keep)
            r["wall_s"] = time.time() - t0
            if keep is not None:
                meta = {"tag": tag, "rate": rate, "reach_km": L, "p_tx_dbm": P, "offset": a.offset, "n": r["n"],
                        "seed": r["seed"], "span_km": a.span_km, "nf_db": a.nf_db, "trx_overrides": ov}
                r["per_image_file"] = str(save_per_image(a.save_per_image, f"{tag}__R{rate.replace('/', '')}", L, P,
                                                         keep, meta))
            blk["rows"].append(r)
            print(f"[rate {rate}] {L:6.0f} km {P:+5.1f} dBm: acc {r['accuracy']:.2f} ssim {r['ssim']:.4f} "
                  f"psnr {r['psnr_per_image']:.2f} SSIM<0.5 {100 * r['frac_ssim_below_0p5']:.2f} % "
                  f"codeword errors {r['codeword_error_rate']:.3f} ({r['wall_s']:.0f} s)", flush=True)
            out["rates"][rate] = blk
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        out["rates"][rate] = blk
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
