# Transceiver impairments (both link ends)

| File | What it models | Where it acts |
|---|---|---|
| `params.py` | `TrxParams` (per-image values), `nominal_trx` (evaluation: SNR_TRx 23 dB, 100 kHz lasers, no offset, random phase), `draw_trx` (training: SNR_TRx U[20, 26] dB, linewidths U[0, 500] kHz, offset U[±10 MHz]) | — |
| `noise.py` | signal-proportional transceiver noise, split κ_R = 0.5 between TX and RX (each end at SNR_TRx + 3.01 dB) | TX: symbols before pulse shaping (in-band, propagates); RX: sampled symbols |
| `laser.py` | Wiener phase noise per sample, increment variance 2π·Δν/f_s; initial carrier phase | TX laser: launched field; LO: received field BEFORE CD compensation (→ EEPN) |
| `frequency_offset.py` | residual carrier frequency offset as a phase ramp | sampled symbols (after CD compensation) |

All draws go through `semcom.utils.random.PerImageRng`: one generator in training, per-image seeds in evaluation.
