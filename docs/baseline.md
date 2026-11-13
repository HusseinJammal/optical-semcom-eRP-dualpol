# Conventional baseline: JPEG + IEEE 802.11n LDPC + Gray 64-QAM

Code: `semcom/baseline/`. Both systems send each image as the same 512 data symbols of 64-QAM (256 per polarization) plus
the same 9 pilots per polarization, through the same `semcom.system.link.transmit()` (neighbour frames, transceiver noise,
lasers, SSFM, pilot CPR, AGC) with the same per-image noise seeds.

## Chain

| Step | Detail |
|---|---|
| Source coding | JPEG (PIL / libjpeg, 4:4:4), per image the highest quality factor whose entropy-coded payload fits the byte budget; the header, quality factor and payload length are known to the receiver (not transmitted) |
| Channel coding | two IEEE 802.11n codewords (1944 + 1296 bits), 168 parity bits punctured (101 / 67) → exactly 3072 coded bits |
| Modulation | Gray-mapped 64-QAM, unit mean power → 512 symbols |
| Demapping | exact log-sum-exp LLRs with a per-image decision-directed noise estimate; punctured bits enter with LLR 0 |
| Decoding | sum-product belief propagation, up to 50 iterations (OptiCommPy) |
| Failure | a JPEG stream that does not decode is scored as a black image; codeword errors and decoding failures are reported |

| Mother code rate | 1/2 | 2/3 | 3/4 | 5/6 |
|---|---:|---:|---:|---:|
| Effective rate | 0.527 | 0.703 | 0.791 | 0.879 |
| JPEG payload budget | 202 B | 270 B | 303 B | 337 B |

## Task classifier

For a fair classification comparison each rate gets its own CIFAR ResNet-18 (3×3 stem, no max-pool) trained on the
training images compressed exactly as that rate compresses them, without channel errors
(`scripts/train_baseline_classifier.py --kind jpeg --rate R`): SGD Nesterov 0.9, one-cycle cosine lr 0.1, weight decay
5e-4, 40 epochs, batch 128, crop + flip, best validation epoch kept. This is the conventional chain's best case.

```bash
for r in 1/2 2/3 3/4 5/6; do python scripts/train_baseline_classifier.py --kind jpeg --rate $r; done
python scripts/run_baseline.py --rates 1/2 2/3 3/4 5/6 --cond 400:0 1440:0 2480:-1 --classifier-dir runs/task_clf \
    --out reports/baseline_reference_links.json
```

`run_baseline.py` also reports the error-free (no channel) result of every rate unless `--no-ceiling` is given.
