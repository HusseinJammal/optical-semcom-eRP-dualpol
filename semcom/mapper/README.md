# Symbol mapping

`qam.py`

* `QAMMapper(n_points=64, n_pol=2)` — TX. `(B, 2k)` reals → k complex values (I, Q interleaved) → unit mean power per
  image → each axis clipped and rounded to the nearest level of the unit-power square M-QAM (64-QAM: ±0.154, ±0.463,
  ±0.772, ±1.080) with the straight-through estimator `s = u + sg[Q(u) − u]` → `(B, n_pol, k / n_pol)`.
  Deterministic (same symbols in training and inference); the grid is fixed and standard (no bit labelling, no FEC).
* `SymbolDemapper` — RX. `(B, n_pol, k / n_pol)` soft symbols → `(B, 2k)` reals in the mapper's layout.
* `qam_grid(M)`, `usage_metrics(index, M)` — the grid and the effective number of points the encoder uses.
