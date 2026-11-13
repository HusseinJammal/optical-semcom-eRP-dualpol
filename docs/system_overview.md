# System overview

The semantic transceiver solves

    min_{theta, phi}  E_{(x,y), c, xi, n} [ L ],      (x_hat, p_hat) = g_phi( h_{c,xi}( f_theta(x; c); n ); c )

with encoder `f_theta`, decoder `g_phi`, end-to-end channel `h_{c,xi}` (TX DSP + fibre + RX DSP), told link
`c = [P, L]`, hidden link parameters `xi = (L_s, NF, SNR_TRx, linewidth_TX, linewidth_LO, freq. offset, carrier phase)`
and noise realizations `n`. All values below are the paper configuration (`semcom.config.Config()`,
[`configs/paper.json`](../configs/paper.json)).

## Signal chain, block by block

Shapes are per batch of `B` images; complex tensors are `(B, n_pol = 2, ...)`.

| # | Block | Code | Output |
|---|---|---|---|
| 1 | **Semantic encoder**: 3×3 conv 3→64 + FiLM(c); conv s2 64→128; conv s2 128→256; conv s1 256→256; residual block; FiLM(c); 1×1 conv 256→16 | `semcom/encoder/encoder.py` | `(B, 1024)` reals |
| 2 | **Symbol mapper**: pair to 512 complex values (I, Q interleaved), unit mean power per image, STE rounding to square 64-QAM, 2 × 256 layout | `semcom/mapper/qam.py` (`QAMMapper`) | `(B, 2, 256)` complex |
| 3 | **Pilots**: QPSK pilot on the 64-QAM corners (±1.080 per axis) before every 32 data symbols + 1 closing pilot | `semcom/tx_dsp/pilots.py` | `(B, 2, 265)` frame |
| 4 | **Neighbour frames**: TX pad (24 symbols) + guard (reach-scaled) on both sides filled with random 64-QAM + pilots at the frame's power; 1 tail symbol | `semcom/tx_dsp/framing.py` | `(B, 2, 314 + 2G)` |
| 5 | **TX transceiver noise**: circular AWGN at SNR_TRx + 3.01 dB (κ_R = 0.5) | `semcom/impairments/noise.py` | |
| 6 | **Pulse shaping**: upsample ×16, RRC roll-off 0.1, 257 taps | `semcom/tx_dsp/pulse_shaping.py` | `(B, 2, N)`, N = 16 (314 + 2G) |
| 7 | **Launch power**: TOTAL P over both polarizations, measured over the frame | `semcom/tx_dsp/launch_power.py` | |
| 8 | **TX laser**: Wiener phase noise per sample + uniform initial carrier phase (common to both polarizations) | `semcom/impairments/laser.py` | |
| 9 | **Fibre**: N = L / L_s spans; per span Manakov propagation (eRP for training, SSFM for evaluation) + EDFA (gain = span loss, ASE per polarization) | `semcom/channel/` | `(B, 2, N)` |
| 10 | **LO laser**: Wiener phase noise applied BEFORE CD compensation (→ EEPN) | `semcom/impairments/laser.py` | |
| 11 | **CD compensation**: ideal frequency-domain inverse over L | `semcom/rx_dsp/cd_compensation.py` | |
| 12 | **Matched filter + sampling**: same RRC, strip guard, 1 sample/symbol at the ideal instant, keep the frame | `semcom/rx_dsp/sampling.py` | `(B, 2, 265)` |
| 13 | **Residual frequency offset**: phase ramp on the sampled symbols | `semcom/impairments/frequency_offset.py` | |
| 14 | **RX transceiver noise**: circular AWGN at SNR_TRx + 3.01 dB | `semcom/impairments/noise.py` | |
| 15 | **Pilot CPR**: per pilot, correlation summed over both polarizations, 3-pilot average, unwrap, linear interpolation, derotation (detached from the gradient) | `semcom/rx_dsp/cpr.py` | |
| 16 | **Drop pilots + AGC** (unit power per polarization) | `semcom/tx_dsp/pilots.py`, `semcom/rx_dsp/agc.py` | `(B, 2, 256)` |
| 17 | **Semantic decoder**: demapper → 16×8×8 → 1×1 conv 16→256 → FiLM(c) → residual → {GAP + MLP 256→256→10 ; up-convs → 3×32×32 sigmoid} | `semcom/decoder/decoder.py` | image `(B, 3, 32, 32)`, logits `(B, 10)` |

Blocks 3 to 16 are composed, in this order, by `semcom.system.link.transmit()`; blocks 1 to 17 by
`semcom.system.SemanticTransceiver`. Training and evaluation use the same function and differ only in block 9.

## Frame and bandwidth budget

| Item | Value |
|---|---|
| Data symbols | 512 per image = 256 per polarization (bandwidth ratio ρ = 512 / 3072 = 1/6) |
| Pilots | 9 per polarization (3.4 % overhead), paid identically by the baseline |
| Symbol rate / roll-off | 32 GBd / 0.1 |
| Oversampling | 16 samples per symbol (512 GS/s simulation bandwidth) |
| Pad + tail | 24 symbols each side + 1 tail symbol (frame + pads = 314 symbols per polarization) |

The guard is `ceil(0.6 /km × L × 1.15 / 16) × 16` symbols per side (measured CD wraparound rule), increased to the next
value that makes the FFT length 5-smooth:

| Link | Guard (symbols / side) | Samples / polarization | Duration |
|---|---:|---:|---:|
| 400 km (80 km spans) | 293 | 14,400 | 28.1 ns |
| 1440 km | 1,043 | 38,400 | 75.0 ns |
| 2480 km | 1,718 | 60,000 | 117.2 ns |
| 2500 km (100 km spans) | 1,763 | 61,440 | 120.0 ns |

## Told vs hidden

| Parameter | Training (drawn per) | Evaluation (nominal) | Told to the networks |
|---|---|---|---|
| Link length L | U[400, 2480] km, rounded to whole spans (batch) | 400 / 1440 / 2480 km, sweeps | **yes** |
| Launch power P | uniform in dB on [−6 dBm, P_max(L, L_s)] (batch) | 0 / 0 / −1 dBm, sweeps | **yes** |
| Span length L_s | whole km, U{60..100} (batch) | 80 km | no |
| Noise figure | U[4, 6] dB (batch) | 4.5 dB | no |
| Transceiver SNR | U[20, 26] dB (image) | 23 dB | no |
| TX / LO linewidth | U[0, 500] kHz each (image) | 100 kHz each | no |
| Frequency offset | U[−10, 10] MHz (image) | 0 | no |
| Carrier phase | U[0, 2π) (image) | U[0, 2π) (image) | no |

Not modelled: polarization rotation and PMD (ideal polarization demultiplexing; the Manakov model is covariant under a
fixed unitary rotation, so it is undone exactly), PDL, WDM neighbours, DAC/ADC quantization and bandwidth, timing errors.
