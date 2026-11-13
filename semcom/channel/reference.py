"""OptiCommPy reference (per image, NumPy in / NumPy out). Used by the tests to hold every model to the reference.

Not used for training or evaluation: the batched split-step (``semcom.channel.ssfm``) is the evaluation channel,
and it is gated against this wrapper. OptiCommPy receives the BARE gamma (``manakovSSF`` applies the 8/9 itself).
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_params", "propagate_reference"]


def build_params(cfg):
    """OptiCommPy ``parameters()`` built from a ``PropagationConfig`` (single source of truth)."""
    from optic.utils import parameters

    p = parameters()
    p.Ltotal = cfg.link.Ltotal_km
    p.Lspan = cfg.link.Lspan_km
    p.hz = cfg.link.hz_km
    p.alpha = cfg.fiber.alpha_db_km
    p.D = cfg.fiber.D_ps_nm_km
    p.gamma = cfg.fiber.gamma_per_w_km          # BARE gamma, always
    p.Fc = cfg.fiber.Fc_hz
    p.Fs = cfg.sim.fs_hz
    p.amp = cfg.link.amp
    p.NF = cfg.link.NF_db
    p.seed = cfg.link.ase_seed
    p.prec = np.complex128
    p.prgsBar = False
    p.returnParameters = False
    if cfg.fiber.manakov:
        p.nlprMethod = False                     # fixed step hz, as the batched port
    return p


def propagate_reference(x_np: np.ndarray, cfg) -> np.ndarray:
    """``(batch, n_pol, n_samples)`` through OptiCommPy: ``manakovSSF`` (Manakov) or ``ssfm`` per pol (scalar)."""
    from optic.models import channels

    if x_np.ndim != 3:
        raise ValueError(f"expected (batch, n_pol, n_samples); got {x_np.shape}")
    p = build_params(cfg)
    out = np.empty_like(x_np, dtype=np.complex128)
    if cfg.fiber.manakov:
        if x_np.shape[1] != 2:
            raise ValueError(f"Manakov reference requires n_pol=2; got {x_np.shape}")
        for b in range(x_np.shape[0]):
            y = channels.manakovSSF(np.ascontiguousarray(x_np[b].T, dtype=np.complex128), p)   # (n_samples, 2)
            y = y[0] if isinstance(y, tuple) else y
            out[b] = np.asarray(y).T
        return out
    for b in range(x_np.shape[0]):
        for m in range(x_np.shape[1]):
            y = channels.ssfm(np.ascontiguousarray(x_np[b, m], dtype=np.complex128), p)
            y = y[0] if isinstance(y, tuple) else y
            out[b, m] = np.asarray(y).reshape(-1)
    return out
