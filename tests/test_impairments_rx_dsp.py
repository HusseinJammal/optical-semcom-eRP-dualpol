"""Transceiver impairments and receiver DSP: noise, lasers, frequency offset, CD compensation, CPR, AGC."""
import math

import torch

from semcom.config import ChannelConfig, Config
from semcom.channel.linear import D_op
from semcom.channel.params import FiberConfig
from semcom.impairments import (TrxParams, add_symbol_noise, apply_frequency_offset, draw_trx, nominal_trx,
                                wiener_phase)
from semcom.rx_dsp import agc, cd_compensate, pilot_cpr
from semcom.tx_dsp import insert_pilots, pilot_symbols, remove_pilots
from semcom.utils.random import PerImageRng


def test_transceiver_noise_hits_the_requested_snr():
    s = torch.randn(4, 2, 50000, dtype=torch.complex128)
    rng = PerImageRng(4, "cpu", generator=torch.Generator().manual_seed(0))
    snr = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float64)
    y = add_symbol_noise(s, snr, s, rng, "rx_noise")
    meas = 10 * torch.log10((s.abs() ** 2).mean(dim=(1, 2)) / ((y - s).abs() ** 2).mean(dim=(1, 2)))
    assert float((meas - snr).abs().max()) < 0.05


def test_wiener_increment_variance():
    lw = torch.tensor([100e3, 500e3], dtype=torch.float64)
    rng = PerImageRng(2, "cpu", generator=torch.Generator().manual_seed(0))
    phi = wiener_phase(400000, lw, 512e9, rng, "tx_laser", torch.float64)
    var = torch.diff(phi, dim=-1).var(dim=-1)
    assert torch.allclose(var, 2 * math.pi * lw / 512e9, rtol=0.01)


def test_nominal_and_drawn_transceiver(cfg):
    p = nominal_trx(cfg, 3)
    assert float(p.snr_db[0]) == 23.0 and float(p.lw_tx_hz[0]) == 100e3 and p.random_phase
    off = ChannelConfig(trx_snr_db=math.inf, linewidth_tx_hz=0.0, linewidth_lo_hz=0.0, random_carrier_phase=False)
    assert nominal_trx(Config(channel=off), 3) is None
    d = draw_trx(cfg, 1000, torch.Generator().manual_seed(0))
    assert 20 <= float(d.snr_db.min()) and float(d.snr_db.max()) <= 26
    assert float(d.lw_lo_hz.max()) <= 500e3 and float(d.fo_hz.abs().max()) <= 10e6


def test_per_image_draws_do_not_depend_on_the_batch():
    a = PerImageRng(3, "cpu", seed=5, index0=10).randn("tx_noise", (2, 7), torch.float64)
    b = PerImageRng(1, "cpu", seed=5, index0=12).randn("tx_noise", (2, 7), torch.float64)
    assert torch.equal(a[2:], b)


def test_cd_compensation_inverts_lossless_dispersion():
    f = FiberConfig(alpha_db_km=0.0)
    x = torch.randn(1, 2, 8192, dtype=torch.complex128)
    y = cd_compensate(D_op(x, f, 1200.0, 512e9), f, 1200.0, 512e9)
    assert float((y - x).abs().max()) < 1e-10


def test_pilot_cpr_removes_a_common_phase_and_a_frequency_offset():
    g = torch.Generator().manual_seed(0)
    lv = torch.linspace(-7, 7, 8) / 42 ** 0.5
    idx = torch.randint(0, 8, (2, 3, 2, 256), generator=g)
    s = torch.complex(lv[idx[0]], lv[idx[1]]).to(torch.complex128)
    pil = pilot_symbols(2, 9, 64, dtype=torch.complex128)
    f = insert_pilots(s, pil, 32)
    rot = torch.exp(1j * torch.tensor([0.3, 2.0, -2.9], dtype=torch.float64)).view(3, 1, 1)
    fo = torch.tensor([0.0, 5e6, -10e6], dtype=torch.float64)
    out = pilot_cpr(apply_frequency_offset(f * rot, fo, 32e9), pil, 32)
    # a constant phase is removed exactly everywhere
    assert float((remove_pilots(out, 32)[0] - s[0]).abs().max()) < 1e-9
    # a phase ramp is removed exactly between pilots 1 and 7 (both estimates are centred 3-pilot averages) ...
    assert float((out[1:, :, 33:232] - f[1:, :, 33:232]).abs().max()) < 1e-9
    # ... and up to the one-sided average's bias near the frame edges (~half a pilot spacing of the ramp)
    ramp_per_pilot = 2 * math.pi * 10e6 * 33 / 32e9
    resid = torch.angle(out[2] * f[2].conj()).abs().max()
    assert ramp_per_pilot / 4 < float(resid) < ramp_per_pilot


def test_agc_normalises_each_polarization():
    s = torch.randn(2, 2, 256, dtype=torch.complex64) * torch.tensor([[1.0, 5.0], [0.1, 2.0]]).view(2, 2, 1)
    p = (agc(s).abs() ** 2).mean(dim=-1)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-5)


def test_trx_params_activity():
    assert not TrxParams().any_active()
    assert TrxParams(random_phase=True).any_active()
