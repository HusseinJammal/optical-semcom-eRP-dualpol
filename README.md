# Optical Semantic Communication over Dual-Polarization Coherent Links via Differentiable Nonlinear Fiber Modeling

Code for the OFC paper *"End-to-End Optical Semantic Communication across Dual-Polarization Coherent Links via
Differentiable Nonlinear Fiber Modeling"* (H. Jammal, Y. Jiang, A. Bianco, P. Poggiolini, C. Rottondi, Politecnico di
Torino).

A semantic image transceiver for 32 GBd dual-polarization coherent links of 400 to 2480 km. It sends standard 64-QAM
symbols with pilots and receives only the **launch power P and link length L** as channel-state information. Every other
link parameter (span length, amplifier noise figure, transceiver SNR, laser linewidths, frequency offset, carrier phase)
is randomized during training and hidden from the networks. It is trained end to end **through a differentiable enhanced
regular-perturbation (eRP) model of the Manakov equation** and evaluated independently with the **split-step Fourier
method (SSFM)**. The baseline is JPEG with IEEE 802.11n LDPC coding over exactly the same symbols, pilots, DSP and link.

```text
 image ─► ENCODER ─► STE 64-QAM ─► TX DSP ─────────────► FIBRE ─────────────► RX DSP ──────────────► DECODER ─► image
 32x32x3  CNN+FiLM   512 symbols   pilots, neighbour      N spans x            LO laser, CD comp.,     CNN+FiLM   + class
             ▲       (2 x 256)     frames, TX noise,      [eRP (training) |    matched filter,           ▲
             │                     RRC, launch power,      SSFM (evaluation)]  sampling, FO, RX noise,   │
             └──────── c = [P, L] ─ TX laser ──────────── + EDFA ASE ───────── pilot CPR, AGC ───────────┘
                         (the only channel-state information given to the networks)
```

## Repository layout: one folder per block

| Block | Folder | What it contains |
|---|---|---|
| Semantic encoder | [`semcom/encoder/`](semcom/encoder) | CNN (3→64→128→256→256, 8×8 latent) + FiLM, projection to 1024 reals |
| Conditioning | [`semcom/conditioning/`](semcom/conditioning) | c = [P, L] and the FiLM layers shared by encoder and decoder |
| Symbol mapping | [`semcom/mapper/`](semcom/mapper) | straight-through (STE) square M-QAM mapper and the soft demapper |
| TX DSP | [`semcom/tx_dsp/`](semcom/tx_dsp) | pilot insertion, neighbour-frame embedding, RRC pulse shaping, launch power |
| Transceiver impairments | [`semcom/impairments/`](semcom/impairments) | transceiver noise (TX/RX shares), laser phase noise, frequency offset |
| Fibre channel | [`semcom/channel/`](semcom/channel) | linear operator, **eRP / RP model** (training), EDFA ASE, **batched Manakov SSFM** (evaluation), OptiCommPy reference |
| RX DSP | [`semcom/rx_dsp/`](semcom/rx_dsp) | CD compensation, matched filter + sampling, pilot-aided carrier-phase recovery, AGC |
| Semantic decoder | [`semcom/decoder/`](semcom/decoder) | CNN + FiLM with reconstruction and classification heads |
| System | [`semcom/system/`](semcom/system) | the link chain `transmit()` (TX DSP → fibre → RX DSP) and the full `SemanticTransceiver` |
| Training | [`semcom/training/`](semcom/training) | CIFAR-10 data, loss, domain randomization of the link, training loop |
| Evaluation | [`semcom/evaluation/`](semcom/evaluation) | metrics, SSFM evaluation (per-image seeding), eRP validity check, link SNR |
| Baseline | [`semcom/baseline/`](semcom/baseline) | JPEG, 802.11n LDPC + Gray 64-QAM frame codec, the chain over the same link, task classifiers |

Command-line entry points are in [`scripts/`](scripts), tests in [`tests/`](tests), the paper configuration in
[`configs/paper.json`](configs/paper.json) and the documentation in [`docs/`](docs):

* [`docs/system_overview.md`](docs/system_overview.md): the signal chain block by block, tensor shapes, frame budget
* [`docs/physics_and_conventions.md`](docs/physics_and_conventions.md): Manakov equation, eRP, ASE, SSFM, units and signs
* [`docs/training.md`](docs/training.md): domain randomization, loss, optimizer, checkpoint selection
* [`docs/evaluation.md`](docs/evaluation.md): metrics, seeding, eRP validity, and how to reproduce every figure
* [`docs/baseline.md`](docs/baseline.md): the JPEG + LDPC chain and its classifiers

## Installation

Python ≥ 3.9 and PyTorch ≥ 2.0. A CUDA GPU is strongly recommended (training ~16 h on one NVIDIA A40; the SSFM runs in
float64, so evaluation is fastest on GPUs with strong FP64 such as A100/V100).

```bash
git clone <this repository> && cd OFC_optical_semcom_differentiable_dualpol_github
pip install -r requirements.txt                 # torch, torchvision, numpy, scipy, scikit-image, ...
# OptiCommPy: SSFM reference, ASE noise generator and the 802.11n LDPC codes. Install from a source checkout so that
# its LDPC parity-check matrices (optic/comm/ALIST) are available; the code was validated at this commit:
git clone https://github.com/edsonportosilva/OptiCommPy.git
git -C OptiCommPy checkout 9f5b7d8ce1ebb9e458438300eaf4778d942d4bbe
pip install -e OptiCommPy
pip install cupy-cuda12x                         # optional: runs the SSFM on the GPU
pip install -e .                                 # optional: install semcom itself
```

CIFAR-10 is downloaded to `./data` on first use (override with `--data-root` or `SEMCOM_DATA_ROOT`).

## Quick start

```bash
# tests (CPU, ~30 s; add --runslow for the split-step checks at the paper link)
pytest -q

# pipeline check on the CPU (2 short epochs, a few minutes)
python scripts/train.py --run-name smoke --device cpu --smoke

# train the paper model (defaults = paper configuration); writes runs/paper_v3_s0/{config.json,history.json,best.pt,last.pt}
python scripts/train.py --run-name paper_v3_s0 --seed 0

# evaluate it on the true channel (SSFM) at the three reference links, all 10,000 test images
python scripts/evaluate.py --ckpt runs/paper_v3_s0/best.pt --cond 400:0 1440:0 2480:-1 --link-snr-images 64
```

```python
import torch
from semcom import load_checkpoint

model, cfg, _ = load_checkpoint("runs/paper_v3_s0/best.pt")
x = torch.rand(4, 3, 32, 32)                                     # images in [0, 1]
with torch.no_grad():
    x_hat, logits = model.forward_ssfm(x, ssfm_seed=1234, reach_km=1440.0, p_tx_dbm=0.0)   # true channel
x_hat, logits = model(x, backend="erp", reach_km=1440.0, p_tx_dbm=0.0)                    # differentiable channel
```

## Reproducing the paper

[`scripts/reproduce_paper.sh`](scripts/reproduce_paper.sh) lists every command: training, the eRP validity check, the four
baseline classifiers, Fig. 2(a) near-optimum power vs link length, Fig. 2(b) −6 dBm, Fig. 2(c) transceiver-SNR sweep at
400 km, and Table 1 (accuracy and CPU processing time). See [`docs/evaluation.md`](docs/evaluation.md) for the conditions
and the compute cost.

## Citation

If you use this code, please cite the accompanying OFC paper (bibliographic details to be added on publication):

> H. Jammal, Y. Jiang, A. Bianco, P. Poggiolini, C. Rottondi, "End-to-End Optical Semantic Communication across
> Dual-Polarization Coherent Links via Differentiable Nonlinear Fiber Modeling," OFC.

## Acknowledgements

The split-step reference, the EDFA noise generator and the IEEE 802.11n LDPC codes come from
[OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) (E. P. da Silva et al.); the batched Manakov SSFM in
`semcom/channel/ssfm.py` is a port of its `manakovSSF`.
