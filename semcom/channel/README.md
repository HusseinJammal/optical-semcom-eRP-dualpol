# Fibre channel

All functions take a `PropagationConfig` (`params.py`: fibre, link, simulation grid), built from the system config by
`semcom.system.link.build_propagation_config`, and fields of shape `(B, 2, N)`.

| File | Content | Used for |
|---|---|---|
| `params.py` | `FiberConfig` (α, D, bare γ, Manakov switch; `gamma_eff_per_w_km` = (8/9)γ), `LinkConfig` (spans, amplifier, NF, model, branches), `SimConfig`, guard rule constants | everything |
| `linear.py` | `D_op` = F⁻¹ exp(−αz/2 + jβ2ω²z/2) F, noiseless gain, linear link | building block |
| `erp.py` | **first-order RP and enhanced RP of the Manakov equation** (differentiable) | **training** |
| `ase.py` | per-span EDFA ASE (exact n_sp), per- and total-polarization ASE floors / SNRs | training, floors |
| `propagate.py` | `propagate(x, cfg, model=...)`: one dispatch over linear / rp / erp (+ ASE after every span) and ssfm | everything |
| `ssfm.py` | **batched Manakov split-step** (port of OptiCommPy `manakovSSF`, NumPy or CuPy, per-image seeded ASE) | **evaluation** |
| `reference.py` | OptiCommPy `manakovSSF` / `ssfm` wrapper | tests (parity) |

Physics and equations: [`docs/physics_and_conventions.md`](../../docs/physics_and_conventions.md).
