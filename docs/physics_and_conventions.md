# Physics, models and conventions

## Fibre and link

| Parameter | Value |
|---|---|
| Attenuation α | 0.2 dB/km (power) |
| Dispersion D | 16 ps/(nm·km) → β2 = −20.47 ps²/km |
| Kerr coefficient γ | 1.3 W⁻¹km⁻¹; Manakov γ_M = (8/9) γ = 1.156 W⁻¹km⁻¹ |
| Carrier | 193.1 THz |
| Amplification | one EDFA per span, gain = span loss (12 / 16 / 20 dB at 60 / 80 / 100 km) |
| ASE | n_sp = (G·NF − 1) / (2 (G − 1)) (exact), PSD (G − 1) n_sp h ν, white over the simulation band, independent per polarization |

## Equation and conventions

    dE/dz = −(α/2) E − j (β2/2) ∂²E/∂t² + j γ_M ‖E‖² E,        E = [Ex, Ey]ᵀ,  ‖E‖² = |Ex|² + |Ey|²

* **Units**: time in seconds, distance in km, so ω [rad/s], β2 [s²/km]; α is the POWER coefficient (field decays as
  exp(−αz/2)). Written exactly as OptiCommPy writes them (`semcom/utils/units.py`).
* **Kerr sign +j**, as in OptiCommPy's `manakovSSF`; with the opposite sign the perturbation model is further from the
  split-step than the linear model (checked in `tests/test_channel.py`).
* **The 8/9 factor is applied once**: in the differentiable model through `FiberConfig.gamma_eff_per_w_km`; the SSFM
  (ours and OptiCommPy's) receives the bare γ and applies 8/9 itself. With Ey = 0 the Manakov model reduces exactly to
  the scalar model with (8/9) γ (tested).
* **Launch power is TOTAL** over both polarizations (P/2 per polarization), measured over the frame (data + pilots).
* **ASE floors are TOTAL**: the in-band noise of both polarizations, 3.01 dB above the per-polarization floor.
  At the nominal link (NF 4.5 dB, 80 km spans) the ASE-limited SNR is 26.4 dB at 400 km / 0 dBm, 20.9 dB at
  1440 km / 0 dBm and 17.5 dB at 2480 km / −1 dBm (`semcom.system.link.ase_snr_db_total`).
* Reaches must be whole spans and spans whole split steps (`check_reach` raises otherwise, instead of silently
  truncating as a floor of L / L_s would).

## Training channel: enhanced regular perturbation (eRP) — `semcom/channel/erp.py`

First order in γ_M, per span of length L_s, with a midpoint rule over B sections (Δz = L_s / B, midpoints ζ_b):

    E_out = e^{j φ_NL} [ D(L_s) E_in + j γ_M Δz Σ_b D(L_s − ζ_b) ( ‖E_b‖² − 2 P_b ) E_b ]

where `D(z) = F⁻¹ exp(−αz/2 + j β2 ω² z / 2) F` (`semcom/channel/linear.py`), `E_b = D(ζ_b) E_in`, `P_b` its mean
total power (guard excluded) and `φ_NL = 2 γ_M Δz Σ_b P_b`. The mean nonlinear phase is removed from the perturbation and
applied as an exact rotation; to first order the two cancel for any coefficient, so eRP equals RP1 to first order while
being much more accurate at high power. B = ⌈L_s / 10 km⌉ (6 / 8 / 10 sections at 60 / 80 / 100 km; the model error
saturates at 8 per 80 km span because the residual is first-order truncation, not quadrature error). Spans are cascaded
with ASE added after each amplifier. All B terms are independent FFT operations, differentiable with autograd; the
model runs in complex64.

### Validity on the LEARNED symbols — `semcom/evaluation/erp_validity.py`

The learned encoder can emit any symbol distribution, so validity is re-measured on its own symbols:
ε = E‖E_out^eRP − E_out^SSFM‖² / E‖E_out^SSFM‖², noise-free on both sides (two independent ASE draws would make the
comparison measure noise), same neighbour frames, raw received frame. Training uses only powers P ≤ P_max(L, L_s), the
highest power at which −10 log10 ε exceeds the ASE-limited SNR at the lowest noise figure (4 dB) by at least 9 dB, i.e. the
model error is at most about 1/8 of the ASE noise (`PAPER_POWER_LIMIT_TABLE` in `semcom/config.py`):

| Span | P_max (dBm) at link length |
|---|---|
| 60 km | 420 km: +0.5 · 960: −0.5 · 1440: −1.5 · 1980: −2 · 2460: −2.5 |
| 80 km | 400: +2 · 960: +1 · 1440: 0 · 2000: −0.5 · 2240: −1 · 2480: −1 |
| 100 km | 400: +3 · 1000: +2 · 1200: +1.5 · 1400: +1.5 · 2000: +0.5 · 2500: +0.5 |

(interpolated linearly in length at each span, then in span; capped at +3 dBm). `scripts/erp_validity_gate.py` checks a
trained model at the corners of this range.

## Evaluation channel: batched Manakov SSFM — `semcom/channel/ssfm.py`

A port of OptiCommPy's `manakovSSF` (fixed step 1 km, symmetric split step, iterated trapezoidal nonlinear step with up
to 10 iterations and tolerance 1e-5, EDFA + ASE per span and polarization) that propagates a whole batch per step, in
complex128, on GPU (CuPy) when available. Parity with `manakovSSF` is below −150 dB in the tests.

**Seeding.** OptiCommPy's noise generator is numba-compiled with its own RNG; only its `seed` argument reaches it. Every
(image, polarization, span) gets its own seed derived from (evaluation seed, test-set index); transceiver draws
(neighbour frames, noise, lasers) use the same scheme with other stream ids (`semcom/utils/random.py`). Image i of the
test set therefore always sees the same noise, on CPU or GPU, in any batch, and the semantic transceiver and the
baseline see the same noise realizations. (With γ ≠ 0 the trapezoid iteration stops when all images of a chunk have
converged, which couples images of a chunk at the level of the iteration tolerance.)

## Guard interval

Frequency-domain dispersion is periodic in time, so the waveform carries a guard on each side. The rule
`ceil(0.6 symbols/km × L × 1.15 / 16) × 16` comes from measured minima (broadband probe, opposite-edge leakage < 1e-8:
48 / 144 / 224 / 432 symbols at 80 / 240 / 400 / 800 km); the analytic group-delay spread must not be used for this
tail criterion. The guard is then raised to the next value giving a 5-smooth FFT length.

## Precision

Training: float32 / complex64 everywhere, no mixed precision near the channel. Evaluation channel: float64 / complex128.
