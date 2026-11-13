# Training

| File | Content |
|---|---|
| `data.py` | CIFAR-10 loaders in [0, 1]; 45k / 5k train / validation split; crop + flip on the training split only |
| `losses.py` | `semantic_loss`: λ_CE·CE (label smoothing 0.1) + λ_MSE·MSE + λ_SSIM·(1 − SSIM), (1, 100, 2) |
| `sampling.py` | domain randomization: `draw_link` (span, length, NF, power ≤ P_max per batch), monitor conditions, range validation |
| `trainer.py` | `train(cfg)`: AdamW + cosine, eRP forward/backward, per-epoch eRP monitor, smoothed checkpoint selection |

Entry point: `scripts/train.py`. Details: [`docs/training.md`](../../docs/training.md).
