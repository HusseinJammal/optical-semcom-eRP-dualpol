# System: composing the blocks

| File | Content |
|---|---|
| `link.py` | `transmit(symbols, cfg, backend=...)`: TX DSP → TX impairments → fibre (`erp` / `rp` / `linear` / `ssfm`, or `none`) → RX impairments → RX DSP, in physical order; `build_propagation_config`, the reach-scaled 5-smooth guard, `check_reach`, `ase_snr_db_total` |
| `transceiver.py` | `SemanticTransceiver`: encoder → `transmit` → decoder. `forward(x, backend="erp", ...)` for training and diagnostics; `forward_ssfm(x, ssfm_seed, ssfm_index0, ...)` for evaluation (no backend argument) |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint`: state dict + configuration as plain data (`format: semcom/1`) |

Per-call link arguments (`reach_km`, `p_tx_dbm`, `span_km`, `nf_db`, `trx`, `cpr`) default to the nominal values of
`cfg.channel`. `told_reach_km` / `told_p_tx_dbm` change only what the networks are told.
