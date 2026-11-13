# Training

```bash
python scripts/train.py --run-name paper_v3_s0 --seed 0          # paper model; seeds 0, 1, 2 were trained
python scripts/train.py --run-name smoke --device cpu --smoke    # pipeline check on CPU (minutes)
```

Code: `semcom/training/` (`trainer.py`, `sampling.py`, `losses.py`, `data.py`).

## Objective

    L = λ_CE · CE(p̂, y) + λ_MSE · ‖x − x̂‖² / (HWC) + λ_SSIM · (1 − SSIM(x, x̂)),     (λ_CE, λ_MSE, λ_SSIM) = (1, 100, 2)

CE uses label smoothing 0.1 (training only); SSIM is `pytorch_msssim.ssim` (Gaussian window 11, σ 1.5, data range 1).

## One training step

1. Draw the link for the batch (`sampling.draw_link`): span length L_s (whole km, U{60..100}), link length
   L ~ U[400, 2480] km rounded to the nearest whole number of spans, noise figure ~ U[4, 6] dB, launch power uniform in dB
   on [−6 dBm, P_max(L, L_s)].
2. Draw the transceiver per image (`semcom.impairments.draw_trx`): SNR_TRx ~ U[20, 26] dB, TX and LO linewidths
   ~ U[0, 500] kHz, frequency offset ~ U[−10, 10] MHz, carrier phase ~ U[0, 2π).
3. Forward: encoder(x; c) → `transmit(..., backend="erp")` → decoder(·; c), with c = [P, L] only.
4. Backward from the decoder through the receiver DSP, the eRP channel and the STE quantizer into the encoder (the CPR
   estimate is detached); gradient clipping at 1.0.

Three generators make a run reproducible from its seed: channel noise (`seed`), link draws (`seed + 7919`),
transceiver draws (`seed + 104729`).

## Recipe

| Item | Value |
|---|---|
| Data | CIFAR-10, 45,000 / 5,000 train / validation split (split seed 0), random crop (pad 4) + horizontal flip |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 on weights only (not biases, BatchNorm or FiLM biases) |
| Schedule | cosine annealing to 0 over 40 epochs, batch 64 (≈ 28,000 steps) |
| Monitor | validation split through eRP at 400 km −6 / +2 dBm, 1440 km −6 / 0 dBm, 2480 km −6 / −1 dBm |
| Checkpoint | best 3-epoch moving average of monitor accuracy [%] + 100 × SSIM (`best.pt`); `last.pt` every epoch |
| Usage guard | the run stops if the encoder collapses onto ≤ 2 effective constellation points |
| Cost | about 16 h for 40 epochs on one NVIDIA A40 (peak memory ≈ 26 GB at 2460 km with 60 km spans) |

The monitor numbers are training-channel diagnostics; reported results come from `scripts/evaluate.py` (SSFM).

## Ablations exposed as options

| Option | Effect |
|---|---|
| `--conditioning constant` | c = [0, 0]: same architecture, no channel-state information |
| `--arch base` | stride-2 bottleneck (64 × 4 × 4 latent) instead of 8 × 8 |
| `--backend rp` / `linear` | train through plain first-order RP / the linear fibre |
| `--n-points 16` / `256` | QAM order of the STE mapper |
| `--reach-min/--reach-max` | training link-length range (single-length models: set `randomize_reach`, `randomize_span` and `randomize_power` to false and `Ltotal_km` / `P_tx_dBm` in a JSON config) |

Any other field can be changed through a JSON config (`--config`, see `configs/paper.json`).
