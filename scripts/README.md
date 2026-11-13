# Scripts

| Script | Purpose |
|---|---|
| `train.py` | train the semantic transceiver through the eRP channel (defaults = paper configuration) |
| `evaluate.py` | evaluate a checkpoint on the split-step channel at `L:P` conditions, with link/transceiver overrides, link SNR, eRP validity, per-image saving |
| `erp_validity_gate.py` | eRP model error on the learned symbols over the corners of the training range |
| `train_baseline_classifier.py` | the conventional chain's ResNet-18 task classifier for one code rate |
| `run_baseline.py` | JPEG + 802.11n LDPC over the same link (all four rates) |
| `benchmark_latency.py` | CPU processing time per image, both systems |
| `reproduce_paper.sh` | every command behind Fig. 2 and Table 1 (`bash scripts/reproduce_paper.sh <step>`) |

All scripts take `--help`. Run them from the repository root (or after `pip install -e .`).
