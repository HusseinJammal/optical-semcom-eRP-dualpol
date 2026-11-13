"""End-to-end training through the differentiable eRP channel: one optimizer, one loss, plain backprop.

Each step: draw one link for the batch (reach, span, NF, power) and one transceiver per image (all hidden except P and
L), encode, transmit through TX DSP -> eRP + ASE -> RX DSP, decode, and backpropagate from the decoder through the
receiver DSP, the eRP channel and the straight-through quantizer into the encoder (the carrier-phase estimate is
detached). float32 throughout (no mixed precision near the channel).

After every epoch the validation split is run through the training channel at the monitor conditions (shortest /
middle / longest reach x lowest / highest power). The checkpoint with the best moving average (``smooth_window``
epochs) of accuracy [%] + 100 x SSIM is kept as ``best.pt``. The monitor is a training-channel diagnostic, not a result.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..impairments.params import draw_trx
from ..mapper.qam import usage_metrics
from ..system.checkpoint import save_checkpoint
from ..system.transceiver import SemanticTransceiver
from ..evaluation.metrics import MetricAccumulator
from .data import build_loaders
from .losses import semantic_loss
from .sampling import draw_link, monitor_conditions, validate_ranges

__all__ = ["set_seed", "build_optimizer", "run_monitor", "train"]

N_EFF_STOP = 2.0    # constellation usage collapse: effective points used <= this -> stop the run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, tr) -> torch.optim.Optimizer:
    """AdamW with weight decay on weight matrices / kernels only (ndim > 1); plain Adam when weight_decay = 0."""
    if tr.weight_decay <= 0.0:
        return torch.optim.Adam(model.parameters(), lr=tr.lr)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": tr.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=tr.lr)


def _cond_key(reach, power):
    return (f"{int(reach)}km" if reach is not None else "nominal") + (f"@{power:+.1f}dBm" if power is not None else "")


@torch.no_grad()
def run_monitor(model, loader, cfg, device, gen, limit_batches=None) -> dict:
    """Validation split through the TRAINING channel; batch i uses monitor condition i mod n_conditions."""
    model.eval()
    acc = MetricAccumulator()
    conds = monitor_conditions(cfg)
    per_cond = {c: MetricAccumulator() for c in conds} if conds else {}
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x, y = x.to(device), y.to(device)
        reach, power = conds[i % len(conds)] if conds else (None, None)
        x_hat, logits, tx, rx = model(x, backend=cfg.channel.backend, generator=gen, return_symbols=True,
                                      reach_km=reach, p_tx_dbm=power)
        acc.add(x, x_hat, y, logits, tx_sym=tx, rx_sym=rx)
        if conds:
            per_cond[(reach, power)].add(x, x_hat, y, logits, tx_sym=tx, rx_sym=rx)
    model.train()
    r = acc.result()
    r.pop("confusion", None)
    if conds:
        keep = ("accuracy", "ssim", "psnr_per_image", "evm_pct", "n")
        r["by_condition"] = {_cond_key(*k): {kk: v for kk, v in a.result().items() if kk in keep}
                             for k, a in per_cond.items() if a.n > 0}
    return r


@torch.no_grad()
def _usage_check(model, val_loader, device) -> dict:
    """How many of the M grid points the encoder actually uses (one validation batch); stop on collapse."""
    x, _ = next(iter(val_loader))
    was_training = model.training
    model.eval()
    m = usage_metrics(model.encoder.symbol_indices(x.to(device), model.conditioning(device=device)),
                      model.encoder.mapper.n_points)
    model.train(was_training)
    if m["n_effective"] <= N_EFF_STOP:
        raise RuntimeError(f"constellation usage collapsed: {m['n_effective']:.2f} effective points of "
                           f"{m['n_points']}; aborting")
    return m


def train(cfg: Config, device: str = "cuda", out_dir: str = "runs", data_root: str | None = None,
          overwrite: bool = False, max_steps_per_epoch: int | None = None, monitor_batches: int | None = None):
    """Train ``cfg`` and write ``<out_dir>/<run_name>/{config.json, history.json, best.pt, last.pt}``.

    ``max_steps_per_epoch`` / ``monitor_batches`` shorten an epoch (smoke tests only).
    """
    set_seed(cfg.train.seed)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    out = Path(out_dir) / cfg.run_name
    if (out / "history.json").exists() and not overwrite:
        raise FileExistsError(f"{out} already holds a run; choose another run name or pass overwrite=True")
    out.mkdir(parents=True, exist_ok=True)
    validate_ranges(cfg)
    cfg.save_json(out / "config.json")

    train_loader, val_loader, _ = build_loaders(cfg, data_root)
    model = SemanticTransceiver(cfg).to(device)
    gen = torch.Generator(device="cpu").manual_seed(cfg.train.seed)             # channel noise (ASE, transceiver)
    rgen = torch.Generator(device="cpu").manual_seed(cfg.train.seed + 7919)     # link draws
    tgen = torch.Generator(device="cpu").manual_seed(cfg.train.seed + 104729)   # hidden transceiver draws

    opt = build_optimizer(model, cfg.train)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cfg] {cfg.describe()}\n[cfg] params={n_params / 1e6:.2f}M device={device}", flush=True)

    history, scores = [], []
    best, best_epoch = -math.inf, -1
    for epoch in range(cfg.train.epochs):
        t0, n_seen, run = time.time(), 0, {}
        for it, (x, y) in enumerate(train_loader):
            if max_steps_per_epoch and it >= max_steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            reach, span, nf, power = draw_link(cfg, rgen)
            trx = draw_trx(cfg, x.shape[0], tgen) if cfg.channel.randomize_transceiver else "nominal"
            x_hat, logits = model(x, backend=cfg.channel.backend, generator=gen, reach_km=reach, p_tx_dbm=power,
                                  span_km=span, nf_db=nf, trx=trx)
            loss, parts = semantic_loss(x, x_hat, y, logits, cfg.train)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            n_seen += x.shape[0]
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v * x.shape[0]
            if it % cfg.train.log_every == 0:
                link = "".join([f" L={reach:.0f}km" if reach else "", f" Ls={span:.0f}km" if span else "",
                                f" NF={nf:.2f}dB" if nf else "", f" P={power:+.2f}dBm" if power is not None else ""])
                print(f"  ep{epoch:03d} it{it:04d} loss={parts['total']:.4f} ce={parts['ce']:.4f} "
                      f"mse={parts['mse']:.5f} ssim={parts['ssim']:.4f}{link}", flush=True)
        sched.step()

        tr = {k: v / max(n_seen, 1) for k, v in run.items()}
        va = run_monitor(model, val_loader, cfg, device, gen, limit_batches=monitor_batches)
        usage = _usage_check(model, val_loader, device)
        scores.append(va["accuracy"] + 100.0 * va["ssim"])
        smooth = float(np.mean(scores[-cfg.train.smooth_window:]))
        rec = dict(epoch=epoch, lr=sched.get_last_lr()[0], train=tr, monitor=va, score=scores[-1],
                   score_smooth=smooth, constellation_usage=usage, wall_s=time.time() - t0)
        history.append(rec)
        print(f"[ep{epoch:03d}] train_loss={tr.get('total', float('nan')):.4f} | monitor(eRP) acc={va['accuracy']:.2f}% "
              f"ssim={va['ssim']:.4f} psnr={va['psnr_per_image']:.2f} papr={va.get('papr_db', float('nan')):.2f}dB "
              f"n_eff={usage['n_effective']:.1f}/{usage['n_points']} | {rec['wall_s']:.0f}s", flush=True)
        if "by_condition" in va:
            print("        " + "  ".join(f"{k} acc={v['accuracy']:.2f} ssim={v['ssim']:.4f}"
                                         for k, v in va["by_condition"].items()), flush=True)

        if smooth > best:
            best, best_epoch = smooth, epoch
            save_checkpoint(out / "best.pt", model, epoch=epoch, monitor=va, score_smooth=smooth)
        save_checkpoint(out / "last.pt", model, epoch=epoch, monitor=va)
        (out / "history.json").write_text(json.dumps(history, indent=2, default=str))
        if cfg.train.patience and epoch - best_epoch >= cfg.train.patience:
            print(f"[early stop] best smoothed score at epoch {best_epoch}", flush=True)
            break
    return model, history
