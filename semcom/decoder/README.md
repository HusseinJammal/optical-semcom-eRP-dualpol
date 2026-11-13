# Semantic decoder

`decoder.py` — `SemanticDecoder(model_cfg)`: received soft symbols `(B, 2, 256)` (after CPR and AGC) and `c = [P, L]` →
reconstruction `(B, 3, 32, 32)` in [0, 1] and class logits `(B, 10)`. Mirrors the encoder; both heads share the trunk.

| Stage | Operation | Output |
|---|---|---|
| `demapper` | 512 complex → 1024 reals (I, Q interleaved) → reshape | 16 × 8 × 8 |
| `from_symbols` | 1×1 conv 16→256 | 256 × 8 × 8 |
| `film_bottleneck` + `res` | FiLM(c); residual block (256) | 256 × 8 × 8 |
| `cls_head` | global average pool → Linear 256→256, ReLU → Linear 256→10 | 10 logits |
| `up1` + `res1` | 3×3 conv stride 1 (transposed stride 2 for `arch="base"`), 256→256; residual | 256 × 8 × 8 |
| `up2` + `res2` | 3×3 transposed conv stride 2, 256→128; residual | 128 × 16 × 16 |
| `up3` + `film_out` | 3×3 transposed conv stride 2, 128→64; FiLM(c) | 64 × 32 × 32 |
| `recon_head` | 3×3 conv 64→3, sigmoid | 3 × 32 × 32 |

3.73 M parameters, 608 MFLOP per image. The decoder reads the soft symbols directly: no hard decision, no equalizer.
