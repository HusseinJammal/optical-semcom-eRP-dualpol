"""Transmitter DSP: RRC, pilots, neighbour frames, launch power."""
import math

import torch

from semcom.tx_dsp import (apply_filter, insert_pilots, neighbour_stream, pilot_amplitude, pilot_positions,
                           pilot_symbols, remove_pilots, rrc_filter, set_frame_launch_power, upsample)
from semcom.utils.random import PerImageRng


def test_rrc_is_symmetric_unit_energy_and_nyquist():
    h = rrc_filter(257, 16, 0.1)
    assert torch.equal(h, h.flip(0)) and abs(float(torch.sum(h ** 2)) - 1) < 1e-12
    assert int(torch.argmax(h)) == 128
    # TX-RX cascade sampled every sps: an (almost) ideal Nyquist pulse
    g = torch.nn.functional.conv1d(h.view(1, 1, -1), h.flip(0).view(1, 1, -1), padding=256).view(-1)
    taps = g[256::16]
    assert abs(float(taps[0]) - 1) < 2e-3 and float(taps[1:16].abs().max()) < 2e-2


def test_pilot_layout_and_inverse():
    assert pilot_positions(256, 32).tolist() == [0, 33, 66, 99, 132, 165, 198, 231, 264]
    assert abs(pilot_amplitude(64) - math.sqrt(3 / 126) * 7) < 1e-15
    s = torch.randn(3, 2, 256, dtype=torch.complex64)
    p = pilot_symbols(2, 9, 64)
    f = insert_pilots(s, p, 32)
    assert f.shape == (3, 2, 265)
    assert torch.equal(f[..., pilot_positions(256, 32)], p.expand(3, 2, 9))
    assert torch.equal(remove_pilots(f, 32), s)
    assert not torch.equal(p[0], p[1])                        # different sequences on X and Y


def test_neighbour_stream_keeps_the_frame_and_the_pilot_cadence():
    s = torch.randn(2, 2, 256, dtype=torch.complex64)
    f = insert_pilots(s, pilot_symbols(2, 9, 64), 32)
    rng = PerImageRng(2, "cpu", generator=torch.Generator().manual_seed(0))
    st = neighbour_stream(f, 66, 67, 64, 32, rng)
    assert torch.equal(st[..., 66:66 + 265], f)
    a = pilot_amplitude(64)
    left_pilots = st[..., [0, 33]]                            # frame-relative -66 and -33
    assert torch.allclose(left_pilots.real.abs(), torch.full_like(left_pilots.real, a))


def test_launch_power_is_total_over_both_polarizations():
    x = torch.randn(3, 2, 1000, dtype=torch.complex128) * torch.tensor([1.0, 3.0, 0.5]).view(3, 1, 1)
    p_w = 10 ** (-1.0 / 10) / 1000
    y = set_frame_launch_power(x, p_w, 100, 800)
    seg = y[..., 100:900]
    total = (seg.real ** 2 + seg.imag ** 2).mean(dim=-1).sum(dim=1)        # per image, summed over the pols
    assert torch.allclose(total, torch.full((3,), p_w, dtype=torch.float64), rtol=1e-12)


def test_upsample_then_filter_shapes():
    s = torch.randn(2, 2, 10, dtype=torch.complex64)
    x = apply_filter(upsample(s, 16), rrc_filter(257, 16, 0.1, dtype=torch.float32))
    assert x.shape == (2, 2, 160) and x.dtype == torch.complex64
