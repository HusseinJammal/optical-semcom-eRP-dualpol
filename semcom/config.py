"""Configuration of the whole system: one frozen dataclass tree.

The DEFAULTS ARE THE PAPER MODEL. ``Config()`` is exactly the configuration stored in the paper checkpoint
(8x8-latent architecture, STE 64-QAM, FiLM on c = [P, L], pilots + neighbour frames, randomized link and transceiver).
Ablations are one field away (``arch="base"``, ``conditioning="constant"``, ...).

    ModelConfig    encoder / mapper / decoder geometry and the conditioning vector
    ChannelConfig  fibre link, waveform, transceiver (nominal values) and the training randomization ranges
    TrainConfig    loss weights and optimizer recipe
    EvalConfig     the base seed of the per-image evaluation noise

Waveform budget per polarization (paper configuration):

    payload          256 data symbols         k_symbols / n_pol, k = 512 total -> CBR = 512 / 3072 = 1/6
    pilots         +   9 QPSK pilots          one before every 32 data symbols + one closing pilot
    tail           +   1 symbol               keeps the per-pol length even (5-smooth FFT search)
    TX pad         + 2 x 24                   n_taps // sps + 8 (RRC edge transients)
    guard          + 2 x G(L)                 reach-scaled, filled with random neighbour frames
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ModelConfig", "ChannelConfig", "TrainConfig", "EvalConfig", "Config", "PAPER_POWER_LIMIT_TABLE"]


def _interp(x: float, pts) -> float:
    """Piecewise-linear interpolation through sorted (x, y) points, clamped at both ends."""
    pts = list(pts)
    if not pts:
        raise ValueError("empty interpolation table")
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("unreachable")


# MEASURED eRP-validity power limit P_max(span, reach) [dBm]: the highest launch power at which the noise-free eRP
# model error on the learned symbols stays >= 9 dB below the ASE-limited SNR at the lowest noise figure (4 dB),
# rounded down to 0.5 dB and capped at +3 dBm. Interpolated linearly in reach at each span, then in span.
PAPER_POWER_LIMIT_TABLE = (
    (60.0, 420.0, 0.5), (60.0, 960.0, -0.5), (60.0, 1440.0, -1.5), (60.0, 1980.0, -2.0), (60.0, 2460.0, -2.5),
    (80.0, 400.0, 2.0), (80.0, 960.0, 1.0), (80.0, 1440.0, 0.0), (80.0, 2000.0, -0.5), (80.0, 2240.0, -1.0),
    (80.0, 2480.0, -1.0),
    (100.0, 400.0, 3.0), (100.0, 1000.0, 2.0), (100.0, 1200.0, 1.5), (100.0, 1400.0, 1.5), (100.0, 2000.0, 0.5),
    (100.0, 2500.0, 0.5),
)


@dataclass(frozen=True)
class ModelConfig:
    """Encoder / symbol mapper / decoder."""

    image_hw: int = 32
    image_ch: int = 3
    widths: tuple = (64, 128, 256, 256)
    # "latent8" (paper): the last encoder conv and the first decoder conv have stride 1 -> 16 x 8 x 8 latent.
    # "base": stride 2 there -> 64 x 4 x 4 latent (the ablation). Same 512 symbols either way.
    arch: str = "latent8"
    k_symbols: int = 512              # complex channel symbols per image, TOTAL over both polarizations
    n_points: int = 64                # square M-QAM order of the STE mapper (16, 64, 256, ...)
    n_class: int = 10
    film_enabled: bool = True
    film_hidden: int = 64
    # What the FiLM layers are told: "power_reach" c = [(P - p_ref)/p_scale, (L - l_ref)/l_scale] (paper), or
    # "constant" c = [0, 0] (the no-conditioning ablation, same architecture).
    conditioning: str = "power_reach"
    cond_p_ref: float = -1.5          # maps P in [-6, +3] dBm onto [-1, +1]
    cond_p_scale: float = 4.5
    cond_l_ref: float = 1440.0        # maps L in [400, 2480] km onto [-1, +1]
    cond_l_scale: float = 1040.0

    @property
    def feat_hw(self) -> int:
        """Spatial size of the latent grid."""
        if self.arch == "latent8":
            return 8
        if self.arch == "base":
            return 4
        raise ValueError(f"unknown arch {self.arch!r}; choose latent8 or base")

    @property
    def cond_dim(self) -> int:
        if self.conditioning not in ("power_reach", "constant"):
            raise ValueError(f"unknown conditioning {self.conditioning!r}; choose power_reach or constant")
        return 2

    @property
    def symbol_channels(self) -> int:
        """Channels C of the latent grid such that C * feat_hw^2 = 2 * k_symbols real numbers."""
        c, r = divmod(2 * self.k_symbols, self.feat_hw**2)
        if r:
            raise ValueError(f"2*k_symbols={2 * self.k_symbols} is not divisible by feat_hw^2={self.feat_hw ** 2}")
        return c

    @property
    def cbr(self) -> float:
        """Bandwidth ratio: complex channel symbols per source value (1/6 for the paper)."""
        return self.k_symbols / (self.image_ch * self.image_hw**2)


@dataclass(frozen=True)
class ChannelConfig:
    """Fibre link, waveform and transceiver; nominal values (evaluation) and training randomization ranges."""

    # ---- training propagation model (evaluation always uses the split-step Fourier method) -----------------------
    backend: str = "erp"              # erp (paper) | rp | linear

    # ---- fibre ----------------------------------------------------------------------------------------------------
    gamma_per_w_km: float = 1.3       # bare Kerr coefficient; the Manakov model applies (8/9) itself
    alpha_db_km: float = 0.2
    D_ps_nm_km: float = 16.0
    Fc_hz: float = 193.1e12
    n_pol: int = 2
    manakov: bool = True

    # ---- nominal link (evaluation) ---------------------------------------------------------------------------------
    Ltotal_km: float = 400.0          # reach used when a call does not give one
    Lspan_km: float = 80.0
    hz_km: float = 1.0                # split-step size of the SSFM reference
    NF_db: float = 4.5
    amp: str = "edfa"                 # per-span EDFA with ASE
    P_tx_dBm: float = 0.0             # TOTAL launch power over both polarizations, when a call does not give one

    # ---- eRP model ------------------------------------------------------------------------------------------------
    n_branches_per_span: int = 8      # used at the nominal span; other spans use ceil(span / branch_spacing_km)
    branch_spacing_km: float = 10.0
    erp_mean_phase_coeff: float = 2.0

    # ---- waveform -------------------------------------------------------------------------------------------------
    sps: int = 16
    rolloff: float = 0.1
    n_taps: int = 257                 # odd: exactly symmetric RRC with a tap at t = 0
    symbol_rate_hz: float = 32e9

    # ---- framing --------------------------------------------------------------------------------------------------
    pilots: bool = True               # QPSK pilots on the M-QAM corners + pilot-aided carrier phase recovery
    pilot_spacing: int = 32
    neighbour_frames: bool = True     # guards carry random neighbour frames (continuous stream) instead of zeros

    # ---- nominal transceiver (evaluation) ---------------------------------------------------------------------------
    trx_snr_db: float = 23.0          # back-to-back transceiver SNR (inf = none)
    trx_rx_fraction: float = 0.5      # kappa_R: share of the transceiver noise power at the receiver
    linewidth_tx_hz: float = 100e3
    linewidth_lo_hz: float = 100e3
    freq_offset_hz: float = 0.0
    random_carrier_phase: bool = True

    # ---- training randomization (domain randomization) -------------------------------------------------------------
    # One (reach, span, NF, power) per batch; the transceiver per image. Only P and L are told to the networks.
    randomize_reach: bool = True
    reach_min_km: float = 400.0
    reach_max_km: float = 2480.0
    randomize_span: bool = True       # whole-km span length, all spans of a link equal; reach -> whole spans
    span_min_km: int = 60
    span_max_km: int = 100
    randomize_nf: bool = True
    nf_min_db: float = 4.0
    nf_max_db: float = 6.0
    randomize_power: bool = True      # uniform in dB on [power_min_dbm, P_max(reach, span)]
    power_min_dbm: float = -6.0
    power_max_dbm: float = 3.0
    power_limit_table: tuple = PAPER_POWER_LIMIT_TABLE
    randomize_transceiver: bool = True
    trx_snr_min_db: float = 20.0
    trx_snr_max_db: float = 26.0
    linewidth_max_hz: float = 500e3   # each laser ~ U[0, linewidth_max_hz]
    freq_offset_max_hz: float = 10e6  # residual frequency offset ~ U[-max, +max]

    def power_range(self, reach_km: float | None = None, span_km: float | None = None) -> tuple:
        """(lo, hi) launch power [dBm] that training may draw at this reach and span length."""
        hi = self.power_max_dbm
        r = self.Ltotal_km if reach_km is None else float(reach_km)
        if self.power_limit_table:
            s = self.Lspan_km if span_km is None else float(span_km)
            spans = sorted({float(t[0]) for t in self.power_limit_table})
            per_span = [(sp, _interp(r, sorted((float(t[1]), float(t[2])) for t in self.power_limit_table
                                               if float(t[0]) == sp))) for sp in spans]
            hi = min(hi, _interp(s, per_span))
        if hi < self.power_min_dbm:
            raise ValueError(f"no valid launch power at {reach_km} km / span {span_km} km: limit {hi:+.2f} dBm is "
                             f"below power_min_dbm {self.power_min_dbm:+.2f} dBm")
        return (self.power_min_dbm, hi)

    @property
    def reach_grid(self) -> tuple:
        """Legal reaches [km] at the nominal span: whole spans from reach_min to reach_max."""
        lo = int(round(self.reach_min_km / self.Lspan_km))
        hi = int(round(self.reach_max_km / self.Lspan_km))
        if abs(lo * self.Lspan_km - self.reach_min_km) > 1e-6 or abs(hi * self.Lspan_km - self.reach_max_km) > 1e-6:
            raise ValueError(f"reach_min/max ({self.reach_min_km}, {self.reach_max_km}) are not whole multiples "
                             f"of the {self.Lspan_km:g} km span")
        return tuple(float(n * self.Lspan_km) for n in range(lo, hi + 1))

    @property
    def tx_pad_symbols(self) -> int:
        """Symbol pad on each side of the frame: n_taps // sps + 8."""
        return self.n_taps // self.sps + 8

    @property
    def fs_hz(self) -> float:
        """Simulation sampling rate [Sa/s]."""
        return self.symbol_rate_hz * self.sps


@dataclass(frozen=True)
class TrainConfig:
    """Loss and optimizer. Defaults = the paper recipe."""

    lambda_ce: float = 1.0
    lambda_mse: float = 100.0
    lambda_ssim: float = 2.0
    label_smoothing: float = 0.1      # training loss only; every reported CE is the plain one
    lr: float = 1e-3
    weight_decay: float = 0.05        # AdamW, on weights only (not biases / BatchNorm / FiLM biases)
    epochs: int = 40                  # cosine annealing to 0 over the run
    batch_size: int = 64
    eval_batch_size: int = 128
    grad_clip: float = 1.0
    augment: bool = True              # random crop (pad 4) + horizontal flip, training split only
    val_fraction: float = 0.1         # 45k / 5k split of the CIFAR-10 training set
    smooth_window: int = 3            # checkpoint selection: moving average of (accuracy + 100 SSIM) on the monitor
    patience: int = 0                 # early stop on the smoothed score (0 = never)
    seed: int = 0                     # initialisation, shuffling, augmentation, training noise
    split_seed: int = 0               # the train/val split only
    num_workers: int = 2
    log_every: int = 50


@dataclass(frozen=True)
class EvalConfig:
    seed: int = 1234                  # base seed of the per-image evaluation noise (amplifier + transceiver)


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "paper_v3"

    # ---- frame geometry (per polarization) ---------------------------------------------------------------------- #
    @property
    def k_per_pol(self) -> int:
        k, n_pol = self.model.k_symbols, self.channel.n_pol
        if k % n_pol:
            raise ValueError(f"k_symbols={k} not divisible by n_pol={n_pol}")
        return k // n_pol

    @property
    def n_pilots(self) -> int:
        """Pilots per polarization: one before each pilot_spacing data block plus a closing one (0 if off)."""
        c = self.channel
        if not c.pilots:
            return 0
        if self.k_per_pol % c.pilot_spacing:
            raise ValueError(f"k_per_pol={self.k_per_pol} is not a whole number of {c.pilot_spacing}-symbol blocks")
        return self.k_per_pol // c.pilot_spacing + 1

    @property
    def frame_symbols(self) -> int:
        """Data + pilot symbols per polarization (265 for the paper)."""
        return self.k_per_pol + self.n_pilots

    @property
    def frame_tail(self) -> int:
        """1 extra symbol when frame + 2 * pad is odd, so the per-pol length stays even."""
        return (self.frame_symbols + 2 * self.channel.tx_pad_symbols) % 2

    @property
    def n_symbols_total(self) -> int:
        """Frame + TX pad + tail per polarization: what the pulse shaper sees (314 for the paper)."""
        return self.frame_symbols + 2 * self.channel.tx_pad_symbols + self.frame_tail

    def describe(self) -> str:
        m, c = self.model, self.channel
        return (f"arch={m.arch} k={m.k_symbols} (CBR 1/{1 / m.cbr:.0f}, {self.k_per_pol}/pol) {m.n_points}-QAM STE, "
                f"conditioning={m.conditioning}; frame {self.frame_symbols} sym/pol ({self.n_pilots} pilots), "
                f"train backend={c.backend}, reach {c.reach_min_km:g}-{c.reach_max_km:g} km, "
                f"nominal {c.Lspan_km:g} km spans NF {c.NF_db:g} dB SNR_TRx {c.trx_snr_db:g} dB")

    # ---- serialization ------------------------------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Inverse of :meth:`to_dict` (lists become tuples again; unknown keys raise)."""
        def build(sub_cls, sub: dict):
            names = {f.name: f for f in dataclasses.fields(sub_cls)}
            unknown = set(sub) - set(names)
            if unknown:
                raise ValueError(f"unknown {sub_cls.__name__} fields: {sorted(unknown)}")
            kw = {}
            for k, v in sub.items():
                default = names[k].default
                kw[k] = _to_tuple(v) if isinstance(default, tuple) else v
            return sub_cls(**kw)

        return cls(model=build(ModelConfig, d.get("model", {})), channel=build(ChannelConfig, d.get("channel", {})),
                   train=build(TrainConfig, d.get("train", {})), eval=build(EvalConfig, d.get("eval", {})),
                   run_name=d.get("run_name", "paper_v3"))

    def save_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _to_tuple(v):
    if isinstance(v, (list, tuple)):
        return tuple(_to_tuple(x) for x in v)
    return v
