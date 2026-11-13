"""Domain randomization: the link drawn for each training batch, and the conditions the monitor watches.

Per BATCH (the waveform length depends on the link, so one link per batch):
    span length   whole km, uniform on [60, 100]                              hidden
    reach         uniform on [400, 2480] km, rounded to whole spans            told (L)
    noise figure  uniform on [4, 6] dB, same in every amplifier                hidden
    launch power  uniform in dB on [-6 dBm, P_max(reach, span)]                told (P)
Per IMAGE (``semcom.impairments.draw_trx``): SNR_TRx, both linewidths, frequency offset, carrier phase (hidden).
"""
from __future__ import annotations

import math

import torch

__all__ = ["draw_link", "monitor_conditions", "validate_ranges"]


def draw_link(cfg, rgen: torch.Generator):
    """(reach_km, span_km, nf_db, p_tx_dbm) for one batch; None where the config does not randomize."""
    ch = cfg.channel
    reach = span = nf = power = None
    if ch.randomize_span:
        if not ch.randomize_reach:
            raise ValueError("randomize_span needs randomize_reach (the reach is drawn per span length)")
        ls = int(torch.randint(int(ch.span_min_km), int(ch.span_max_km) + 1, (1,), generator=rgen))
        target = ch.reach_min_km + (ch.reach_max_km - ch.reach_min_km) * float(torch.rand(1, generator=rgen))
        n_lo, n_hi = math.ceil(ch.reach_min_km / ls - 1e-9), math.floor(ch.reach_max_km / ls + 1e-9)
        n = min(max(int(round(target / ls)), n_lo), n_hi)
        reach, span = float(n * ls), float(ls)
    elif ch.randomize_reach:
        grid = ch.reach_grid
        reach = grid[int(torch.randint(len(grid), (1,), generator=rgen))]
    if ch.randomize_power:
        lo, hi = ch.power_range(reach, span)
        power = lo + (hi - lo) * float(torch.rand(1, generator=rgen))
    if ch.randomize_nf:
        nf = ch.nf_min_db + (ch.nf_max_db - ch.nf_min_db) * float(torch.rand(1, generator=rgen))
    return reach, span, nf, power


def monitor_conditions(cfg):
    """(reach, power) pairs the validation monitor cycles through: shortest / middle / longest reach of the nominal-span
    grid, each at the lowest and highest training power (paper: 400 km -6/+2, 1440 km -6/0, 2480 km -6/-1 dBm)."""
    ch = cfg.channel
    if not (ch.randomize_reach or ch.randomize_power):
        return None
    if ch.randomize_reach:
        g = ch.reach_grid
        reaches = (g[0], g[len(g) // 2], g[-1])
    else:
        reaches = (None,)
    if not ch.randomize_power:
        return [(r, None) for r in reaches]
    return [(r, p) for r in reaches for p in ch.power_range(r)]


def validate_ranges(cfg) -> None:
    """Fail fast (before hours of training) if some drawable (reach, span) has an empty power range."""
    ch = cfg.channel
    if not ch.randomize_span:
        ch.reach_grid
    if not ch.randomize_power:
        return
    spans = range(int(ch.span_min_km), int(ch.span_max_km) + 1) if ch.randomize_span else (None,)
    for sp in spans:
        if sp is None:
            reaches = ch.reach_grid if ch.randomize_reach else (None,)
        else:
            reaches = [n * sp for n in range(math.ceil(ch.reach_min_km / sp - 1e-9),
                                             math.floor(ch.reach_max_km / sp + 1e-9) + 1)]
        for r in reaches:
            ch.power_range(r, sp)
