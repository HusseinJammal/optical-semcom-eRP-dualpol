"""eRP validity of a TRAINED model over the corners of its training range.

For each target reach and span length (reach moved to whole spans), the noise-free eRP model error on the learned
symbols is compared with the total ASE floor at the LOWEST noise figure of the training range, at the lowest, middle
and highest training power P_max(L, L_s). Rule:

    margin >= +9 dB everywhere    PASS
    +6 <= margin < +9 somewhere   ACCEPT_AND_REPORT
    margin < +6 dB somewhere      FAIL (lower that row of the power-limit table by 0.5 dB and retrain)

    python scripts/erp_validity_gate.py --ckpt runs/paper_v3_s0/best.pt --reach-km 400 960 1440 2000 2480 \
        --out reports/erp_validity_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.evaluation import erp_validity  # noqa: E402
from semcom.system.checkpoint import load_checkpoint  # noqa: E402
from semcom.training.data import build_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="eRP validity over the training range of a trained model.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reach-km", type=float, nargs="+", required=True)
    ap.add_argument("--spans", type=int, nargs="+", default=[60, 80, 100])
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    model, cfg, ck = load_checkpoint(a.ckpt, device)
    loader = DataLoader(build_datasets(cfg, a.data_root).test, batch_size=a.batch_size, shuffle=False)
    ch = cfg.channel
    floor_nf = ch.nf_min_db if ch.randomize_nf else ch.NF_db
    rows = []
    for target in a.reach_km:
        for span in a.spans:
            reach = float(max(1, round(target / span)) * span)
            lo, hi = ch.power_range(reach, span)
            for p in (lo, 0.5 * (lo + hi), hi):
                t0 = time.time()
                r = erp_validity(model, loader, cfg, device, n_batches=a.batches, p_tx_dbm=p, reach_km=reach,
                                 span_km=float(span), nf_db=floor_nf)
                m = r["margin_db_worst"]
                r.update(target_km=target, rule=("PASS" if m >= 9 else "ACCEPT_AND_REPORT" if m >= 6 else "FAIL"),
                         wall_s=time.time() - t0)
                rows.append(r)
                print(f"{reach:6.0f} km / {span:3d} km spans, P {p:+5.2f} dBm (range {lo:+.1f}..{hi:+.1f}): "
                      f"margin {m:+6.2f} dB -> {r['rule']}", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "floor_nf_db": floor_nf, "rows": rows}, indent=1))
    worst = min(r["margin_db_worst"] for r in rows)
    verdict = "PASS" if worst >= 9 else "ACCEPT_AND_REPORT" if worst >= 6 else "FAIL"
    print(f"\nworst margin {worst:+.2f} dB -> {verdict}")
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "floor_nf_db": floor_nf, "rows": rows,
                                       "worst_margin_db": worst, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
