# Semantic encoder

`encoder.py` — `SemanticEncoder(model_cfg, n_pol=2)`: image `(B, 3, 32, 32)` in [0, 1] → `(B, 2, 256)` complex 64-QAM
symbols, conditioned on `c = [P, L]`.

| Stage | Operation | Output |
|---|---|---|
| `stem` + `film_stem` | 3×3 conv 3→64, BN, ReLU; FiLM(c) | 64 × 32 × 32 |
| `down1` | 3×3 conv stride 2, 64→128, BN, ReLU | 128 × 16 × 16 |
| `down2` | 3×3 conv stride 2, 128→256, BN, ReLU | 256 × 8 × 8 |
| `down3` | 3×3 conv stride 1 (stride 2 for `arch="base"`), 256→256, BN, ReLU | 256 × 8 × 8 |
| `res` + `film_bottleneck` | residual block (256); FiLM(c) | 256 × 8 × 8 |
| `to_symbols` | 1×1 conv 256→16 | 16 × 8 × 8 = 1024 reals |
| `mapper` | `semcom.mapper.QAMMapper`: 512 complex, unit power per image, STE 64-QAM | `(B, 2, 256)` |

2.19 M parameters, 306 MFLOP per image. `features()` returns the 1024 reals before the mapper; `symbol_indices()` the
emitted constellation-point indices.
