"""Configuration: the paper defaults, frame geometry, power-limit table, guard rule, serialization."""
import pytest

from semcom.config import Config
from semcom.system.link import check_reach, guard_symbols_for_reach, n_branches_for_span


def test_defaults_are_the_paper_frame(cfg):
    assert cfg.model.k_symbols == 512 and cfg.k_per_pol == 256
    assert abs(cfg.model.cbr - 1 / 6) < 1e-12
    assert cfg.n_pilots == 9 and cfg.frame_symbols == 265
    assert cfg.channel.tx_pad_symbols == 24 and cfg.frame_tail == 1 and cfg.n_symbols_total == 314
    assert cfg.model.feat_hw == 8 and cfg.model.symbol_channels == 16


def test_json_round_trip(tmp_path, cfg):
    p = tmp_path / "c.json"
    cfg.save_json(p)
    assert Config.load_json(p) == cfg
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"not_a_field": 1}})


@pytest.mark.parametrize("reach,span,expected", [(400, None, 2.0), (1440, None, 0.0), (2480, None, -1.0),
                                                 (960, 60.0, -0.5), (2000, 90.0, 0.0), (100, None, 2.0),
                                                 (3120, None, -1.0)])
def test_power_limit_table_interpolates_and_clamps(cfg, reach, span, expected):
    lo, hi = cfg.channel.power_range(reach, span)
    assert lo == -6.0 and abs(hi - expected) < 1e-12


@pytest.mark.parametrize("reach,span,guard,n_samples", [(400, None, 293, 14400), (1440, None, 1043, 38400),
                                                        (2480, None, 1718, 60000), (2500, 100.0, 1763, 61440)])
def test_guard_is_reach_scaled_and_5_smooth(cfg, reach, span, guard, n_samples):
    g = guard_symbols_for_reach(cfg, reach, span)
    assert g == guard
    n = (cfg.n_symbols_total + 2 * g) * cfg.channel.sps
    assert n == n_samples
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    assert n == 1


def test_off_grid_reach_is_refused(cfg):
    assert check_reach(cfg, 2480.0) == 31
    with pytest.raises(ValueError):
        check_reach(cfg, 1400.0)                  # 17.5 spans of 80 km


def test_branches_keep_10_km_spacing(cfg):
    assert [n_branches_for_span(cfg, s) for s in (None, 60.0, 80.0, 100.0)] == [8, 6, 8, 10]
