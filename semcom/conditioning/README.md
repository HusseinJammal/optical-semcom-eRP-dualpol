# Conditioning (channel-state information)

`film.py`

* `link_condition(model_cfg, p_dbm, reach_km)` → `c = [(P + 1.5) / 4.5, (L − 1440) / 1040]` (maps P ∈ [−6, +3] dBm and
  L ∈ [400, 2480] km onto [−1, 1]); `conditioning="constant"` gives `c = [0, 0]` (ablation).
* `FiLMGenerator(channels)`: MLP 2 → 64 (ReLU) → 2·channels giving (Δγ, Δβ); the feature map becomes
  `(1 + Δγ) ⊙ z + Δβ`. The last layer is zero-initialised, so every FiLM layer starts as the identity.

FiLM sits at the encoder stem (64 ch) and bottleneck (256 ch) and at the decoder bottleneck (256 ch) and output stage
(64 ch). Encoder and decoder receive the same `c`. Span length, noise figure, transceiver SNR, linewidths, frequency
offset and carrier phase are never given to the networks.
