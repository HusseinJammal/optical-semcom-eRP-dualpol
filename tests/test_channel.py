"""Fibre channel: linear operator, eRP (Manakov), ASE, batched split-step, and parity with OptiCommPy."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from conftest import needs_opticommpy, nmse_db, random_waveform, short_link
from semcom.channel import add_ase, ase_snr_db_total, propagate
from semcom.channel.ase import ase_floor_watt, ase_variance_per_span
from semcom.channel.erp import rp_propagate_link
from semcom.channel.linear import D_op
from semcom.channel.params import FiberConfig
from semcom.channel.ssfm import manakov_ssfm_batched, propagate_ssfm


def test_dispersion_operator_composes_and_inverts():
    f = FiberConfig()
    x = torch.randn(1, 2, 4096, dtype=torch.complex128)
    fs = 512e9
    assert nmse_db(D_op(D_op(x, f, 30.0, fs), f, 50.0, fs), D_op(x, f, 80.0, fs)) < -250
    assert nmse_db(D_op(D_op(x, f, 80.0, fs), f, -80.0, fs), x) < -250


def test_erp_with_zero_gamma_is_the_linear_model():
    cfg = short_link(gamma=0.0, n_spans=2)
    x = random_waveform(cfg)
    assert torch.equal(rp_propagate_link(x, cfg, model="erp"), rp_propagate_link(x, cfg, model="linear"))


def test_manakov_with_one_polarization_is_the_scalar_model_with_8_9_gamma():
    """Ey = 0: the Manakov model reduces to the scalar NLSE with gamma_M = (8/9) gamma."""
    man = short_link(gamma=1.3, manakov=True, p_dbm=6.0)
    sca = short_link(gamma=1.3 * 8 / 9, manakov=False, p_dbm=6.0)
    x = random_waveform(man, batch=1)
    x[:, 1] = 0
    y_man = rp_propagate_link(x, man, model="erp")
    y_sca = rp_propagate_link(x, sca, model="erp")
    assert nmse_db(y_man[:, :1], y_sca[:, :1]) < -250
    assert float(y_man[:, 1].abs().max()) == 0.0


def test_erp_matches_the_split_step_and_the_kerr_sign_matters():
    """eRP vs the batched Manakov split-step (noise-free, 2 x 80 km, +3 dBm total); flipping the Kerr sign is far worse."""
    cfg = short_link(gamma=1.3, n_spans=2, p_dbm=3.0)
    x = random_waveform(cfg)
    ref = torch.from_numpy(manakov_ssfm_batched(x.numpy(), cfg))
    e_erp = nmse_db(rp_propagate_link(x, cfg, model="erp"), ref)
    e_rp = nmse_db(rp_propagate_link(x, cfg, model="rp"), ref)
    e_lin = nmse_db(rp_propagate_link(x, cfg, model="linear"), ref)
    flipped = replace(cfg, fiber=replace(cfg.fiber, gamma_per_w_km=-1.3))
    e_flip = nmse_db(rp_propagate_link(x, flipped, model="erp"), ref)
    assert e_erp < e_rp < e_lin < e_flip
    assert e_erp < -35.0


def test_ase_variance_and_floor():
    cfg = short_link(amp="edfa", gamma=0.0)
    x = torch.zeros(4, 2, 2**18, dtype=torch.complex128)
    n = add_ase(x, cfg, generator=torch.Generator().manual_seed(0))
    measured = float(torch.mean(torch.abs(n) ** 2))
    assert abs(measured / ase_variance_per_span(cfg) - 1) < 0.005
    # total (both-polarization) floor is 3.01 dB tighter than the single-polarization one
    p = cfg.sim.P_tx_watt
    assert abs((10 * np.log10(p / ase_floor_watt(cfg)) - ase_snr_db_total(cfg)) - 10 * np.log10(2)) < 1e-9


def test_noisy_propagation_is_reproducible_from_the_generator():
    cfg = short_link(amp="edfa", n_spans=2)
    x = random_waveform(cfg).to(torch.complex64)
    a = propagate(x, cfg, generator=torch.Generator().manual_seed(3))
    b = propagate(x, cfg, generator=torch.Generator().manual_seed(3))
    c = propagate(x, cfg, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not torch.equal(a, c)


@needs_opticommpy
def test_split_step_noise_is_per_image_and_per_polarization():
    # linear fibre: the per-image ASE realization is bitwise independent of the batch the image is in
    cfg = short_link(amp="edfa", gamma=0.0, n_sym=32, guard=16)
    x = random_waveform(cfg, batch=2)
    assert torch.equal(propagate_ssfm(x, cfg, seed=11, index0=0)[1:], propagate_ssfm(x[1:], cfg, seed=11, index0=1))
    # nonlinear fibre: the trapezoidal iteration stops when EVERY image of the chunk has converged, so a neighbour can
    # add an iteration; the effect stays at the level of the iteration tolerance
    cfg = short_link(amp="edfa", gamma=1.3, n_sym=32, guard=16)
    both = propagate_ssfm(x, cfg, seed=11, index0=0)
    assert nmse_db(both[1:], propagate_ssfm(x[1:], cfg, seed=11, index0=1)) < -150
    noise_only = propagate_ssfm(torch.zeros_like(x[:1]), cfg, seed=11)
    assert not torch.allclose(noise_only[:, 0], noise_only[:, 1])   # X and Y draw different ASE


@needs_opticommpy
def test_batched_split_step_matches_opticommpy_manakov():
    from semcom.channel.reference import propagate_reference
    cfg = short_link(gamma=1.3, n_spans=2, p_dbm=3.0)
    x = random_waveform(cfg, batch=1)
    ref = propagate_reference(x.numpy(), cfg)
    ours = manakov_ssfm_batched(x.numpy(), cfg)
    assert nmse_db(torch.from_numpy(ours), torch.from_numpy(ref)) < -150


@needs_opticommpy
def test_linear_model_matches_opticommpy_at_zero_gamma():
    from semcom.channel.reference import propagate_reference
    cfg = short_link(gamma=0.0, n_spans=2, model="linear")
    x = random_waveform(cfg, batch=1)
    ref = torch.from_numpy(propagate_reference(x.numpy(), cfg))
    assert nmse_db(propagate(x, cfg), ref) < -200


@pytest.mark.slow
@needs_opticommpy
def test_erp_valid_at_the_paper_link_on_random_64qam(cfg):
    """400 km, 0 dBm total, 64-QAM: noise-free eRP error far below the total ASE floor (the validity criterion)."""
    from semcom.system.link import transmit
    g = torch.Generator().manual_seed(0)
    lv = torch.linspace(-7, 7, 8) / (42 ** 0.5)
    idx = torch.randint(0, 8, (2, 2, 2, 256), generator=g)
    s = torch.complex(lv[idx[0]], lv[idx[1]]).to(torch.complex64)
    kw = dict(amp="ideal", p_tx_dbm=0.0, reach_km=400.0, trx=None, cpr="none", trx_seed=1)
    e = nmse_db(transmit(s, cfg, backend="erp", **kw), transmit(s, cfg, backend="ssfm", **kw))
    from semcom.system.link import ase_snr_db_total as snr_total
    assert -snr_total(cfg, 0.0, 400.0, None, 4.0) - e > 9.0
