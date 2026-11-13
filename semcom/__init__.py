"""semcom: end-to-end optical semantic communication over dual-polarization coherent links, trained through a
differentiable enhanced regular-perturbation (eRP) Manakov fibre model and evaluated with the split-step Fourier method.

Blocks (one sub-package each):

    encoder/       CNN semantic encoder (+ FiLM)            mapper/        STE M-QAM mapper / soft demapper
    conditioning/  c = [P, L] and FiLM layers               tx_dsp/        pilots, framing, RRC, launch power
    impairments/   transceiver noise, lasers, freq. offset  channel/       linear, eRP, ASE, Manakov SSFM
    rx_dsp/        CDC, matched filter, CPR, AGC            decoder/       CNN semantic decoder (two heads)
    system/        the link chain and the full transceiver  training/      data, loss, domain randomization, loop
    evaluation/    metrics, SSFM evaluation, eRP validity   baseline/      JPEG + IEEE 802.11n LDPC + Gray 64-QAM
"""
from .config import ChannelConfig, Config, EvalConfig, ModelConfig, TrainConfig
from .system import SemanticTransceiver, load_checkpoint, save_checkpoint, transmit

__version__ = "1.0.0"

__all__ = ["ChannelConfig", "Config", "EvalConfig", "ModelConfig", "TrainConfig", "SemanticTransceiver",
           "load_checkpoint", "save_checkpoint", "transmit", "__version__"]
