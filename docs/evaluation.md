# Evaluation and reproduction

Every reported number is measured on the **split-step Fourier channel** (`SemanticTransceiver.forward_ssfm`, which has
no backend argument). eRP numbers are diagnostics only (`--diagnostics`).

```bash
python scripts/evaluate.py --ckpt runs/paper_v3_s0/best.pt --cond 400:0 1440:0 2480:-1 --link-snr-images 64
```

## What is reported (`semcom/evaluation/metrics.py`)

| Metric | Definition |
|---|---|
| accuracy | own classification head, integer count over the images |
| SSIM | scikit-image, data range 1, per image, then averaged (also 5th / 10th / 50th percentiles) |
| PSNR | per image (data range 1), then averaged; also the PSNR of the pooled MSE |
| lost images | fraction with SSIM < 0.5 (`frac_ssim_below_0p5`) |
| link SNR | uniform 64-QAM through the same TX DSP, link and RX DSP (incl. pilot CPR), effective SNR after removing the least-squares complex gain (`snr_eff_ls_db`) |
| effective SNR of the learned symbols | decoder-side (EVM, no gain removed) and LS, from the transmitted vs received symbols |

**Nominal test link**: 80 km spans, NF 4.5 dB, SNR_TRx 23 dB, 100 kHz lasers, no frequency offset, random carrier
phase, pilot CPR. **Seeding**: amplifier and transceiver noise of test image i is derived from (seed 1234, i), so blocks of
the test set (`--offset`, `--limit`) reproduce a full run image by image (up to the SSFM iteration tolerance, about
−180 dB), and the baseline sees the same noise.
`--save-per-image DIR` stores per image: index, label, prediction, logits, SSIM, PSNR and the 8-bit reconstruction.

## Conditions of the paper

Reference links (near-optimum power, measured): **400 km / 0 dBm, 1440 km / 0 dBm, 2480 km / −1 dBm**.

| Figure | Conditions (`--cond L:P`, extra flags) |
|---|---|
| Fig. 2(a) near-optimum power | 400:0 560:0 720:0 880:0 1040:0 1440:0 1680:-0.2 2000:-0.5 2480:-1 2800:-1 3120:-1 |
| Fig. 2(b) −6 dBm | 400:-6 720:-6 1040:-6 1440:-6 2000:-6 2160:-6 2320:-6 2480:-6 2800:-6 3120:-6 |
| Fig. 2(c) transceiver SNR at 400 km | 400:0 with `--trx-snr-db` 20, 17, 14, 11, 8, 5, 2, −1 (and the nominal 23) |
| Table 1 | 400:0 1440:0 2480:-1 (accuracy) + `scripts/benchmark_latency.py` (CPU time) |

Each condition is evaluated for the semantic transceiver (`scripts/evaluate.py`) and the four code rates
(`scripts/run_baseline.py`); `--link-snr-images 64` adds the link SNR used as the x-axis of Fig. 2(c).
[`scripts/reproduce_paper.sh`](../scripts/reproduce_paper.sh) has the full command list. Lengths beyond 2480 km and
powers above P_max are outside the training range (a mismatch test for the model, still measured on the true channel).

**Compute.** The SSFM cost grows with link length: about 2.5e-7 A40-hours per image-km (≈ 6 h for 10,000 images at
2480 km on one A40; A100 ≈ 3× and V100 ≈ 2–3× faster, FP64-bound). Split long conditions into test-set blocks with
`--offset/--limit` and run them in parallel; with per-image seeding the blocks see the same noise as a full run.

## eRP validity of a trained model

```bash
python scripts/erp_validity_gate.py --ckpt runs/paper_v3_s0/best.pt --reach-km 400 960 1440 2000 2480 \
    --out reports/erp_validity_gate.json
```

For 5 lengths × spans 60 / 80 / 100 km × 3 powers (lowest, middle, P_max), the noise-free eRP model error on the model's
own symbols vs the total ASE floor at NF 4 dB: PASS if every margin ≥ +9 dB. `scripts/evaluate.py
--erp-validity-batches N` measures the same quantity at each evaluated condition.

## Robustness and diagnostics flags (`scripts/evaluate.py`)

`--span-km`, `--nf-db`, `--trx-snr-db`, `--kappa-r`, `--linewidth-khz` (each laser), `--fo-mhz` change one hidden
parameter at a time; `--cpr genie` gives the data-aided upper bound; `--told-p-offset-db` / `--told-reach-offset-km` tell
the networks a wrong [P, L]; `--diagnostics` adds the eRP-channel and fibre-removed results.
