"""ONE entry point for fibre propagation.

    propagate(x, cfg, model="erp", generator=...)      differentiable: linear | rp | erp, ASE after every span
    propagate(x, cfg, model="ssfm", seed=..., index0=...)   ground truth: batched Manakov split-step (no gradient)

Both paths take the same ``PropagationConfig`` and the same waveform, so "train on eRP, evaluate on SSFM" differs in
the propagation model and nothing else.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .ase import add_ase
from .erp import rp_propagate_link

__all__ = ["DIFFERENTIABLE_MODELS", "MODELS", "propagate"]

DIFFERENTIABLE_MODELS = ("linear", "rp", "erp")
MODELS = DIFFERENTIABLE_MODELS + ("ssfm",)


def propagate(x: torch.Tensor, cfg, model: str | None = None, generator: torch.Generator | None = None,
              seed: int | None = None, index0: int = 0) -> torch.Tensor:
    """Propagate ``(batch, n_pol, n_samples)`` through the link of ``cfg``.

    ``model`` defaults to ``cfg.link.channel_model``. ``generator`` seeds the ASE of the differentiable models;
    ``seed``/``index0`` seed the per-image ASE of the split-step.
    """
    model = (cfg.link.channel_model if model is None else model).lower()
    if model == "ssfm":
        from .ssfm import propagate_ssfm
        with torch.no_grad():
            return propagate_ssfm(x, cfg, seed=seed, index0=index0)
    if model not in DIFFERENTIABLE_MODELS:
        raise ValueError(f"model must be one of {MODELS}; got {model!r}")
    amp = cfg.link.amp
    if amp not in ("ideal", "edfa", None, "none"):
        raise ValueError(f"amp must be 'ideal', 'edfa' or None; got {amp!r}")
    fs = cfg.sim.fs_hz
    if amp != "edfa":
        # noiseless: the whole link in one call
        return rp_propagate_link(x, cfg, fs_hz=fs, model=model)
    # noisy: span by span, ASE at each amplifier output
    one = _single_span_cfg(cfg)
    y = x
    for _ in range(cfg.link.n_spans):
        y = rp_propagate_link(y, one, fs_hz=fs, model=model, p0_guard_samples=cfg.guard_samples)
        y = add_ase(y, cfg, generator=generator)
    return y


def _single_span_cfg(cfg):
    """A one-span, noiseless-gain view of ``cfg`` (the gain is applied by the model, the noise by add_ase)."""
    span = cfg.link.L_span_eff_km
    return replace(cfg, link=replace(cfg.link, Ltotal_km=span, Lspan_km=span, amp="ideal"))
