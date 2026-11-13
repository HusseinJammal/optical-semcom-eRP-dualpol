"""The end-to-end link h_{c,xi}: transmitter DSP -> fibre -> receiver DSP, in physical order. ONE function, two channels.

    transmit(symbols, cfg, backend="erp" | "ssfm" | "none", ...) -> received soft symbols (B, n_pol, k / n_pol)

Training (eRP) and evaluation (SSFM) share every block except the propagation model, so a train/test difference can
only come from the channel model. The chain (each block lives in its own package):

  TRANSMITTER  (semcom.tx_dsp, semcom.impairments)
    data symbols -> insert pilots -> embed in random neighbour frames (guard) -> + TX share of transceiver noise
    -> upsample + RRC -> launch power (total, on the frame) -> x exp(j phi_TX) (TX laser + initial carrier phase)
  FIBRE        (semcom.channel)
    N spans x [ eRP (training) | Manakov SSFM (evaluation) ] + EDFA ASE per span and polarization
  RECEIVER     (semcom.impairments, semcom.rx_dsp)
    x exp(-j phi_LO) (LO laser, before CDC -> EEPN) -> CD compensation -> matched filter -> sampling
    -> residual frequency offset -> + RX share of transceiver noise -> pilot CPR -> drop pilots -> AGC

Randomness: training passes ``generator`` (all draws come from it, in a fixed order); evaluation passes integer seeds
and the images' test-set index (``ssfm_seed``, ``trx_seed``, ``ssfm_index0``) so every image gets the same noise
whatever the batch size.
"""
from __future__ import annotations

import math

import torch

from ..channel import ase as ase_mod
from ..channel.params import (GUARD_MARGIN, GUARD_QUANTUM, GUARD_SYMBOLS_PER_KM, FiberConfig, LinkConfig,
                              PropagationConfig, SimConfig)
from ..channel.propagate import propagate
from ..config import Config
from ..impairments.frequency_offset import apply_frequency_offset
from ..impairments.laser import rotate, wiener_phase
from ..impairments.noise import add_symbol_noise, rx_share_snr_db, tx_share_snr_db
from ..rx_dsp.agc import agc
from ..rx_dsp.cd_compensation import cd_compensate
from ..rx_dsp.cpr import genie_cpr, pilot_cpr
from ..rx_dsp.sampling import downsample, extract_frame, matched_filter, strip_guard
from ..tx_dsp.framing import neighbour_stream, zero_padded_stream
from ..tx_dsp.launch_power import set_frame_launch_power
from ..tx_dsp.pilots import insert_pilots, pilot_symbols, remove_pilots
from ..tx_dsp.pulse_shaping import apply_filter, pad_guard, rrc_filter, upsample
from ..utils.random import PerImageRng
from ..utils.units import dbm_to_watt

__all__ = ["BACKENDS", "check_reach", "n_branches_for_span", "guard_symbols_for_reach", "build_propagation_config",
           "ase_snr_db_total", "transmit"]

BACKENDS = ("none", "linear", "rp", "erp", "ssfm")


# --------------------------------------------------------------------------------------------------------------- #
# Link geometry
# --------------------------------------------------------------------------------------------------------------- #
def check_reach(cfg: Config, reach_km: float, span_km: float | None = None) -> int:
    """Number of spans; raises unless ``reach_km`` is a whole number of spans and the span a whole number of steps.

    (OptiCommPy floors Ltotal / Lspan silently: 1400 km with 80 km spans would propagate 1360 km.)
    """
    c = cfg.channel
    ls = c.Lspan_km if span_km is None else float(span_km)
    steps = ls / c.hz_km
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"span {ls} km is not a whole number of {c.hz_km:g} km split steps")
    n = reach_km / ls
    if abs(n - round(n)) > 1e-6 or round(n) < 1:
        raise ValueError(f"reach {reach_km} km is not a whole number of {ls:g} km spans")
    return int(round(n))


def n_branches_for_span(cfg: Config, span_km: float | None) -> int:
    """eRP branches per span: the configured count at the nominal span, else ceil(span / 10 km) (6 / 8 / 10 at
    60 / 80 / 100 km), which keeps the validated ~10 km quadrature step."""
    c = cfg.channel
    if span_km is None:
        return c.n_branches_per_span
    return int(math.ceil(float(span_km) / c.branch_spacing_km - 1e-9))


def _is_5_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def guard_symbols_for_reach(cfg: Config, reach_km: float, span_km: float | None = None) -> int:
    """Guard [symbols per side]: the measured reach-scaled rule ceil(0.6 /km * L * 1.15 / 16) * 16, then the smallest
    larger value that makes the per-polarization FFT length 5-smooth (fast radix in every FFT)."""
    c = cfg.channel
    ls = c.Lspan_km if span_km is None else float(span_km)
    n_spans = check_reach(cfg, reach_km, span_km)
    l_eff = n_spans * math.floor(ls / c.hz_km + 1e-9) * c.hz_km
    need = int(math.ceil(GUARD_SYMBOLS_PER_KM * l_eff * GUARD_MARGIN / GUARD_QUANTUM) * GUARD_QUANTUM)
    n_payload = cfg.n_symbols_total
    for g in range(need, need + 256):
        if _is_5_smooth((n_payload + 2 * g) * c.sps):
            return g
    raise RuntimeError(f"no 5-smooth waveform length within 256 symbols of guard {need}")


def build_propagation_config(cfg: Config, backend: str | None = None, p_tx_dbm: float | None = None,
                             amp: str | None = None, reach_km: float | None = None, span_km: float | None = None,
                             nf_db: float | None = None) -> PropagationConfig:
    """Translate the system config (+ this call's link) into the physical ``PropagationConfig``.

    ``amp="ideal"`` turns the ASE off for MODEL-ACCURACY comparisons (two independent noise draws would otherwise make
    the comparison measure noise rather than model error).
    """
    c = cfg.channel
    backend = (backend or c.backend).lower()
    model = "linear" if backend in ("linear", "ssfm", "none") else backend
    ltotal = c.Ltotal_km if reach_km is None else float(reach_km)
    guard = guard_symbols_for_reach(cfg, ltotal, span_km)
    lspan = c.Lspan_km if span_km is None else float(span_km)
    return PropagationConfig(
        fiber=FiberConfig(alpha_db_km=c.alpha_db_km, D_ps_nm_km=c.D_ps_nm_km,
                          gamma_per_w_km=0.0 if backend == "linear" else c.gamma_per_w_km,
                          Fc_hz=c.Fc_hz, manakov=c.manakov),
        link=LinkConfig(Ltotal_km=ltotal, Lspan_km=lspan, hz_km=c.hz_km, amp=(c.amp if amp is None else amp),
                        NF_db=(c.NF_db if nf_db is None else float(nf_db)), channel_model=model,
                        n_branches_per_span=n_branches_for_span(cfg, span_km),
                        erp_mean_phase_coeff=c.erp_mean_phase_coeff),
        sim=SimConfig(n_sym=cfg.n_symbols_total, n_pol=c.n_pol, sps=c.sps, rolloff=c.rolloff, n_taps=c.n_taps,
                      symbol_rate_hz=c.symbol_rate_hz, P_tx_dBm=c.P_tx_dBm if p_tx_dbm is None else p_tx_dbm,
                      guard_symbols_override=guard),
    )


def ase_snr_db_total(cfg: Config, p_tx_dbm: float | None = None, reach_km: float | None = None,
                     span_km: float | None = None, nf_db: float | None = None) -> float:
    """ASE-limited SNR of the link [dB], total launch power over total (both-polarization) noise."""
    return ase_mod.ase_snr_db_total(build_propagation_config(cfg, "linear", p_tx_dbm, reach_km=reach_km,
                                                             span_km=span_km, nf_db=nf_db))


# --------------------------------------------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------------------------------------------- #
def transmit(symbols: torch.Tensor, cfg: Config, backend: str | None = None, generator: torch.Generator | None = None,
             p_tx_dbm: float | None = None, amp: str | None = None, ssfm_seed: int | None = None,
             ssfm_index0: int = 0, reach_km: float | None = None, span_km: float | None = None,
             nf_db: float | None = None, trx=None, trx_seed: int | None = None, cpr: str | None = None,
             return_frames: bool = False):
    """``(B, n_pol, k / n_pol)`` complex symbols -> received soft symbols of the same shape (after CPR and AGC).

    backend      "erp" | "rp" | "linear" (differentiable, ASE from ``generator``), "ssfm" (ground truth, per-image
                 seeds ``ssfm_seed``), or "none" (identity channel + AGC). Default ``cfg.channel.backend``.
    p_tx_dbm, reach_km, span_km, nf_db
                 this call's link; None = the nominal value of ``cfg.channel``.
    trx          a ``semcom.impairments.TrxParams`` (per-image transceiver) or None (ideal transceiver).
    trx_seed     per-image seeding of every transceiver draw (neighbours, noise, lasers); default ``ssfm_seed``.
    cpr          "pilot" (default with pilots), "genie" (upper bound) or "none" (raw frame, for model comparisons).
    amp          "ideal" switches the ASE off (model-accuracy comparisons).
    return_frames  also return (transmitted frame, received frame before CPR).
    """
    backend = (backend or cfg.channel.backend).lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}")
    if backend == "none":
        return agc(symbols)
    out = _chain(symbols, cfg, backend, generator, p_tx_dbm, amp, ssfm_seed, ssfm_index0, reach_km, span_km, nf_db,
                 trx, ssfm_seed if trx_seed is None else trx_seed, cpr, return_frames)
    if return_frames:
        y, frames = out
        return agc(y), frames
    return agc(out)


def _chain(symbols, cfg, backend, generator, p_tx_dbm, amp, ssfm_seed, ssfm_index0, reach_km, span_km, nf_db, trx,
           trx_seed, cpr, return_frames):
    c = cfg.channel
    prop = build_propagation_config(cfg, backend, p_tx_dbm, amp=amp, reach_km=reach_km, span_km=span_km, nf_db=nf_db)
    pad = c.tx_pad_symbols
    b, n_pol, k = symbols.shape
    cdtype = torch.complex128 if backend == "ssfm" else torch.complex64
    rdtype = torch.float64 if backend == "ssfm" else torch.float32
    dev = symbols.device
    rng = PerImageRng(b, dev, generator=generator, seed=trx_seed, index0=ssfm_index0)
    m = cfg.model.n_points

    # ---- transmitter DSP ----------------------------------------------------------------------------------------- #
    s = symbols.to(cdtype)
    pilots = pilot_symbols(n_pol, cfg.n_pilots, m, device=dev, dtype=cdtype) if c.pilots else None
    spacing = c.pilot_spacing if c.pilots else None
    frame = insert_pilots(s, pilots, c.pilot_spacing) if c.pilots else s
    f_len = frame.shape[-1]
    tail = cfg.frame_tail
    n_side = pad + prop.guard_symbols
    if c.neighbour_frames:
        stream = neighbour_stream(frame, n_side, n_side + tail, m, spacing, rng)
    else:
        stream = zero_padded_stream(frame, pad, pad + tail)              # the guard is padded in samples below
    snr = None if trx is None else trx.snr_db
    if snr is not None and trx.rx_fraction < 1.0:                         # TX share of the transceiver noise
        tx_snr = tx_share_snr_db(snr, trx.rx_fraction)
        if c.neighbour_frames:
            stream = add_symbol_noise(stream, tx_snr, frame, rng, "tx_noise")
        else:
            noisy = add_symbol_noise(frame, tx_snr, frame, rng, "tx_noise")
            stream = torch.cat([stream[..., :pad], noisy, stream[..., pad + f_len:]], dim=-1)

    h = rrc_filter(c.n_taps, c.sps, c.rolloff, device=dev, dtype=rdtype)
    x = apply_filter(upsample(stream, c.sps), h)
    lead = (n_side if c.neighbour_frames else pad) * c.sps
    x = set_frame_launch_power(x, dbm_to_watt(c.P_tx_dBm if p_tx_dbm is None else p_tx_dbm), lead, f_len * c.sps)
    if not c.neighbour_frames:
        x = pad_guard(x, prop.guard_samples)
    n_samp = x.shape[-1]

    if trx is not None and (trx.random_phase or (trx.lw_tx_hz is not None and bool(torch.any(trx.lw_tx_hz > 0)))):
        ph0 = (2 * math.pi * rng.rand("phase0", (), torch.float64)) if trx.random_phase else None
        lw = trx.lw_tx_hz if trx.lw_tx_hz is not None else torch.zeros(b, dtype=torch.float64)
        x = rotate(x, wiener_phase(n_samp, lw, c.fs_hz, rng, "tx_laser", rdtype, phase0=ph0))

    # ---- fibre --------------------------------------------------------------------------------------------------- #
    if backend == "ssfm":
        y = propagate(x, prop, model="ssfm", seed=ssfm_seed, index0=ssfm_index0).to(x.dtype)
    else:
        from ..utils.random import generator_for
        y = propagate(x, prop, model=prop.link.channel_model, generator=generator_for(generator, x.device))

    # ---- receiver DSP -------------------------------------------------------------------------------------------- #
    if trx is not None and trx.lw_lo_hz is not None and bool(torch.any(trx.lw_lo_hz > 0)):
        y = rotate(y, -wiener_phase(n_samp, trx.lw_lo_hz, c.fs_hz, rng, "lo_laser", rdtype))
    y = cd_compensate(y, prop.fiber, prop.link.L_eff_km, c.fs_hz)
    y = matched_filter(y, h)
    y = strip_guard(y, prop.guard_samples)
    y = downsample(y, c.sps)
    y = extract_frame(y, pad, f_len)
    if trx is not None and trx.fo_hz is not None and bool(torch.any(trx.fo_hz != 0)):
        y = apply_frequency_offset(y, trx.fo_hz, c.symbol_rate_hz)
    if snr is not None and trx.rx_fraction > 0.0:                         # RX share of the transceiver noise
        y = add_symbol_noise(y, rx_share_snr_db(snr, trx.rx_fraction), y, rng, "rx_noise")

    mode = cpr if cpr is not None else ("pilot" if c.pilots else "none")
    if mode == "pilot":
        if not c.pilots:
            raise ValueError("pilot CPR needs pilots (ChannelConfig.pilots=True)")
        y_c = pilot_cpr(y, pilots, c.pilot_spacing)
    elif mode == "genie":
        y_c = genie_cpr(y, frame)
    elif mode == "none":
        y_c = y
    else:
        raise ValueError(f"cpr must be pilot, genie or none; got {mode!r}")
    out = remove_pilots(y_c, c.pilot_spacing) if c.pilots else y_c
    out = out.to(torch.complex64)
    if return_frames:
        return out, (frame, y)
    return out
