# Evaluation

| File | Content |
|---|---|
| `metrics.py` | accuracy, SSIM (scikit-image), PSNR, percentiles, lost-image fraction, EVM / effective SNR (decoder-side and LS), PAPR; `MetricAccumulator` (independent of the batching) |
| `evaluate.py` | `evaluate_ssfm`: metrics on the true channel with per-image seeding, test-set blocks, per-image saving; `evaluate_training_channel`: eRP / no-channel diagnostics |
| `erp_validity.py` | `erp_validity`: noise-free eRP vs SSFM model error on the learned symbols vs the total ASE floor |
| `snr_probe.py` | `link_snr`: uniform 64-QAM through the same link and DSP (the SNR axis shared with the baseline) |

Entry points: `scripts/evaluate.py`, `scripts/erp_validity_gate.py`. Details: [`docs/evaluation.md`](../../docs/evaluation.md).
