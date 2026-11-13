"""Train the semantic transceiver through the differentiable eRP channel.

    python scripts/train.py --run-name paper_v3_s0 --seed 0                    # the paper model (~16 h on one A40)
    python scripts/train.py --run-name ablation_base --arch base               # 4x4-latent ablation
    python scripts/train.py --run-name ablation_nocond --conditioning constant # no channel-state information
    python scripts/train.py --run-name smoke --device cpu --smoke              # 2 epochs x 3 steps, CPU, minutes

Defaults are the paper configuration (semcom.config / configs/paper.json). Outputs go to <out>/<run-name>/:
config.json, history.json, best.pt (best 3-epoch moving average of monitor accuracy + 100 SSIM), last.pt.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.config import Config  # noqa: E402
from semcom.training import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train the semantic transceiver (eRP channel).")
    ap.add_argument("--config", default=None, help="JSON config (default: the paper configuration)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--arch", choices=("latent8", "base"), default=None)
    ap.add_argument("--conditioning", choices=("power_reach", "constant"), default=None)
    ap.add_argument("--n-points", type=int, default=None, help="QAM order M of the STE mapper (16, 64, 256)")
    ap.add_argument("--backend", choices=("erp", "rp", "linear"), default=None, help="training channel model")
    ap.add_argument("--reach-min", type=float, default=None)
    ap.add_argument("--reach-max", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None, help="CIFAR-10 directory (default $SEMCOM_DATA_ROOT or ./data)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 3 steps, 2 monitor batches (pipeline check)")
    a = ap.parse_args()

    cfg = Config.load_json(a.config) if a.config else Config()
    rp = dataclasses.replace
    m = {k: v for k, v in (("arch", a.arch), ("conditioning", a.conditioning), ("n_points", a.n_points))
         if v is not None}
    c = {k: v for k, v in (("backend", a.backend), ("reach_min_km", a.reach_min), ("reach_max_km", a.reach_max))
         if v is not None}
    t = {k: v for k, v in (("seed", a.seed), ("epochs", a.epochs), ("batch_size", a.batch_size),
                           ("num_workers", a.num_workers)) if v is not None}
    if a.smoke:
        t.update(epochs=t.get("epochs", 2), batch_size=t.get("batch_size", 8), eval_batch_size=8, log_every=1)
    cfg = rp(cfg, model=rp(cfg.model, **m), channel=rp(cfg.channel, **c), train=rp(cfg.train, **t),
             run_name=a.run_name or cfg.run_name)
    train(cfg, device=a.device, out_dir=a.out, data_root=a.data_root, overwrite=a.overwrite,
          max_steps_per_epoch=3 if a.smoke else None, monitor_batches=2 if a.smoke else None)


if __name__ == "__main__":
    main()
