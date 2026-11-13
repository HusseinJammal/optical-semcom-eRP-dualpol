"""Shared fixtures. Run with ``pytest -q``; ``--runslow`` adds the slower split-step checks."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.channel.params import FiberConfig, LinkConfig, PropagationConfig, SimConfig  # noqa: E402
from semcom.config import Config  # noqa: E402
from semcom.tx_dsp.pulse_shaping import apply_filter, pad_guard, rrc_filter, upsample  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test (needs --runslow)")
    config.addinivalue_line("markers", "opticommpy: needs the OptiCommPy reference")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def has_opticommpy() -> bool:
    try:
        import optic.models.channels  # noqa: F401
        return True
    except Exception:
        return False


needs_opticommpy = pytest.mark.skipif(not has_opticommpy(), reason="OptiCommPy not installed")


@pytest.fixture(scope="session")
def cfg():
    return Config()


def nmse_db(a: torch.Tensor, ref: torch.Tensor) -> float:
    a, ref = a.to(torch.complex128), ref.to(torch.complex128)
    return 10 * torch.log10(torch.sum(torch.abs(a - ref) ** 2) / torch.sum(torch.abs(ref) ** 2)).item()


def short_link(gamma=1.3, manakov=True, n_spans=1, amp="ideal", model="erp", n_sym=128, guard=96, p_dbm=0.0,
               n_pol=2, n_branches=8):
    """A small PropagationConfig (80 km spans) for fast channel tests."""
    return PropagationConfig(
        fiber=FiberConfig(gamma_per_w_km=gamma, manakov=manakov),
        link=LinkConfig(Ltotal_km=80.0 * n_spans, Lspan_km=80.0, amp=amp, channel_model=model,
                        n_branches_per_span=n_branches),
        sim=SimConfig(n_sym=n_sym, n_pol=n_pol, P_tx_dBm=p_dbm, guard_symbols_override=guard))


def random_waveform(cfg: PropagationConfig, batch=2, seed=0, order=16, dtype=torch.complex128):
    """Random square-QAM, RRC-shaped, launched at cfg.sim.P_tx_dBm (total), guard-padded: (B, n_pol, N)."""
    g = torch.Generator().manual_seed(seed)
    side = int(order ** 0.5)
    lv = torch.arange(-(side - 1), side, 2, dtype=torch.float64)
    idx = torch.randint(0, side, (2, batch, cfg.sim.n_pol, cfg.sim.n_sym), generator=g)
    s = torch.complex(lv[idx[0]], lv[idx[1]]).to(dtype)
    h = rrc_filter(cfg.sim.n_taps, cfg.sim.sps, cfg.sim.rolloff, dtype=s.real.dtype)
    x = apply_filter(upsample(s, cfg.sim.sps), h)
    p = torch.mean(x.real ** 2 + x.imag ** 2, dim=(1, 2), keepdim=True)
    x = x * torch.sqrt((cfg.sim.P_tx_watt / cfg.sim.n_pol) / p).to(x.dtype)
    return pad_guard(x, cfg.guard_samples)
