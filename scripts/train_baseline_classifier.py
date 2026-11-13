"""Train the conventional chain's task classifier: a CIFAR ResNet-18 on that chain's own error-free output.

    python scripts/train_baseline_classifier.py --kind jpeg --rate 1/2      # one per code rate (used in the paper)
    python scripts/train_baseline_classifier.py --kind clean                # uncompressed images (reference)

Same 45k / 5k split as the semantic transceiver; random crop (padding 4) + horizontal flip; 40 epochs; SGD Nesterov
momentum 0.9, lr 0.1 (one-cycle cosine), weight decay 5e-4, batch 128, mixed precision on GPU; the epoch with the best
validation accuracy is kept. Output: <out>/<jpeg_r12 | ... | clean>/{best.pt, report.json}; the report gives the
test accuracy on uncompressed images and on the error-free JPEG output of every rate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.baseline import RATES, FrameCodec, build_cifar_resnet18, classify_u8, jpeg_round_trip_u8, to_u8  # noqa
from semcom.config import Config  # noqa: E402
from semcom.training.data import build_datasets  # noqa: E402


def _stack(ds):
    x = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    y = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    return x, y


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop with 4-pixel zero padding + horizontal flip, on the device batch."""
    b = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=x.device)
    j = torch.randint(0, 9, (b,), device=x.device)
    rows = i[:, None] + torch.arange(32, device=x.device)[None, :]
    cols = j[:, None] + torch.arange(32, device=x.device)[None, :]
    xc = xp[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
            rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(b, device=x.device) < 0.5
    xc[flip] = xc[flip].flip(-1)
    return xc


def accuracy(model, x_u8, y, device) -> float:
    return float(100.0 * np.mean(classify_u8(model, x_u8, device).argmax(1) == y))


def main():
    ap = argparse.ArgumentParser(description="Train a task classifier for the conventional chain.")
    ap.add_argument("--kind", choices=("jpeg", "clean"), required=True)
    ap.add_argument("--rate", default=None, help="code rate whose JPEG budget the training images get (--kind jpeg)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: only this many images of each split")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="runs/task_clf")
    a = ap.parse_args()
    if a.kind == "jpeg" and a.rate not in RATES:
        raise SystemExit(f"--kind jpeg needs --rate in {RATES}")
    name = "clean" if a.kind == "clean" else f"jpeg_r{a.rate.replace('/', '')}"
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    t0 = time.time()
    d = build_datasets(Config(), a.data_root)
    xtr, ytr = _stack(d.train_plain)
    xva, yva = _stack(d.val)
    xte, yte = _stack(d.test)
    if a.limit:
        xtr, ytr, xva, yva, xte, yte = (v[: a.limit] for v in (xtr, ytr, xva, yva, xte, yte))
    budgets = {r: FrameCodec(r).budget_bytes for r in RATES}
    if a.kind == "jpeg":
        tr_u8, q_tr, _ = jpeg_round_trip_u8(xtr, budgets[a.rate], a.workers)
        va_u8, _, _ = jpeg_round_trip_u8(xva, budgets[a.rate], a.workers)
    else:
        tr_u8, va_u8, q_tr = to_u8(xtr), to_u8(xva), None
    test_sets = {"uncompressed": to_u8(xte)}
    for r in RATES:
        test_sets[f"jpeg_r{r.replace('/', '')}"], _, _ = jpeg_round_trip_u8(xte, budgets[r], a.workers)
    print(f"[{name}] data ready in {time.time() - t0:.0f} s"
          + (f"; mean JPEG quality {q_tr.mean():.1f} at {budgets[a.rate]} B" if q_tr is not None else ""), flush=True)

    model = build_cifar_resnet18().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    steps = a.epochs * (len(ytr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15,
                                                anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    xtr_t = torch.from_numpy(tr_u8).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    best, best_ep, hist = -1.0, -1, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=device)
        tl, n = 0.0, 0
        for s in range(0, len(ytr) - a.batch + 1, a.batch):
            idx = perm[s:s + a.batch]
            xb = _augment(xtr_t[idx].float().div_(255.0)).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += float(loss) * len(idx)
            n += len(idx)
        model.eval()
        va = accuracy(model, va_u8, yva, device)
        hist.append({"epoch": ep, "train_loss": tl / max(n, 1), "val_acc": va})
        if va > best:
            best, best_ep = va, ep
            torch.save({"model": model.state_dict(), "kind": a.kind, "rate": a.rate, "epoch": ep, "val_acc": va},
                       out / "best.pt")
        print(f"[{name}] epoch {ep:2d} loss {tl / max(n, 1):.4f} val {va:.2f} % (best {best:.2f} @ {best_ep})",
              flush=True)

    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    test = {k: accuracy(model, v, yte, device) for k, v in test_sets.items()}
    rep = {"name": name, "kind": a.kind, "rate": a.rate, "budget_bytes": budgets.get(a.rate), "best_epoch": best_ep,
           "val_acc": best, "test_acc_error_free": test, "history": hist, "wall_s": time.time() - t0}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"[{name}] test accuracy without channel errors: " + ", ".join(f"{k} {v:.2f} %" for k, v in test.items()))


if __name__ == "__main__":
    main()
