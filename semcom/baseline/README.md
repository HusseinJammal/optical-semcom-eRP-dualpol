# Conventional baseline: JPEG + IEEE 802.11n LDPC + Gray 64-QAM

| File | Content |
|---|---|
| `jpeg.py` | JPEG encode / decode (4:4:4), header / payload split, per-image quality search for a byte budget |
| `ldpc.py` | `LDPC80211n` (OptiCommPy codes), `FrameCodec`: payload ↔ two punctured codewords ↔ 512 Gray-64-QAM symbols, exact LLRs, sum-product decoding |
| `chain.py` | `run_jpeg_ldpc`: the chain over `semcom.system.link.transmit` (same frame, pilots, DSP, SSFM and noise seeds as the semantic transceiver) |
| `classifier.py` | CIFAR ResNet-18 task classifier, error-free JPEG round trip (training data for the classifier) |

Budgets: 202 / 270 / 303 / 337 bytes for rates 1/2, 2/3, 3/4, 5/6 (effective rates 0.527 / 0.703 / 0.791 / 0.879).
Entry points: `scripts/run_baseline.py`, `scripts/train_baseline_classifier.py`. Details:
[`docs/baseline.md`](../../docs/baseline.md).
