"""Batched dual-polarization (Manakov) split-step Fourier method: the EVALUATION channel (ground truth).

A faithful port of OptiCommPy's ``optic.models.channels.manakovSSF`` (fixed step, ``nlprMethod=False``) that
propagates a whole batch per step:

* symmetric split step with the linear half-step exp((-alpha/2 + j beta2 w^2 / 2) hz / 2);
* nonlinear rotation exp(j (8/9) gamma (P(z) + P(z + hz)) / 2 hz) per sample, with the same iterated trapezoidal
  rule (up to ``maxIter`` = 10 iterations, tolerance 1e-5, warm-started per step, reset per span);
* per-span EDFA: field gain sqrt(G) plus INDEPENDENT ASE on each polarization.

Held to ``manakovSSF`` by tests/test_channel.py.

Noise seeding: OptiCommPy's ``gaussianComplexNoise`` is numba-compiled and keeps its own RNG, so ``np.random.seed``
does not reach it; its ``seed`` argument does. Every (image, polarization, span) therefore gets its own seed
(``semcom.utils.random.image_seed`` / ``span_seed``), and the noise is generated on the host with OptiCommPy's own
generator. A result is then identical on CPU and GPU and does not depend on the batch size.

Evaluation only: NumPy/CuPy in and out, no gradients.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.constants as const
import torch

from ..utils.random import STREAM, image_seed, span_seed

__all__ = ["ssfm_available", "manakov_ssfm_batched", "propagate_ssfm"]


def _backend():
    """(array module, is_gpu): CuPy when a CUDA device is usable, NumPy otherwise."""
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, True
    except Exception:
        pass
    return np, False


def ssfm_available() -> bool:
    """True when the split-step will run on a GPU."""
    return _backend()[1]


def _gaussian_complex_noise(n: int, sigma2: float, seed: int | None) -> np.ndarray:
    """OptiCommPy's own complex Gaussian noise generator (numba RNG, seeded inside the compiled call)."""
    from optic.dsp.core import gaussianComplexNoise
    return gaussianComplexNoise((n,), sigma2, seed)


def manakov_ssfm_batched(fields: np.ndarray, cfg, seeds=None, chunk: int = 64, max_iter: int = 10,
                         tol: float = 1e-5) -> np.ndarray:
    """Propagate ``(B, 2, N)`` dual-pol fields over the link of ``cfg`` (a ``PropagationConfig``).

    ``seeds``: length-B sequence of ``(seed_x, seed_y)`` per-image base seeds, or None (unseeded). Returns
    ``(B, 2, N)`` complex128 on the host. ``chunk`` bounds device memory.
    """
    fields = np.ascontiguousarray(fields, dtype=np.complex128)
    if fields.ndim != 3 or fields.shape[1] != 2:
        raise ValueError(f"expected (B, 2, N); got {fields.shape}")
    if seeds is not None and len(seeds) != fields.shape[0]:
        raise ValueError(f"need one (seed_x, seed_y) per image: {len(seeds)} for {fields.shape[0]} images")
    out = np.empty_like(fields)
    for lo in range(0, fields.shape[0], chunk):
        hi = min(lo + chunk, fields.shape[0])
        out[lo:hi] = _manakov_chunk(fields[lo:hi], cfg, None if seeds is None else seeds[lo:hi], max_iter, tol)
    return out


def _manakov_chunk(fields: np.ndarray, cfg, seeds, max_iter: int, tol: float) -> np.ndarray:
    xp, is_gpu = _backend()
    B, _, N = fields.shape

    # link parameters, derived exactly as OptiCommPy derives them (BARE gamma; the 8/9 is applied below)
    Ltotal, Lspan, hz = cfg.link.Ltotal_km, cfg.link.Lspan_km, cfg.link.hz_km
    alpha_db, D, gamma = cfg.fiber.alpha_db_km, cfg.fiber.D_ps_nm_km, cfg.fiber.gamma_per_w_km
    Fc, Fs = cfg.fiber.Fc_hz, cfg.sim.fs_hz
    amp, NF = cfg.link.amp, cfg.link.NF_db
    Nspans = int(math.floor(Ltotal / Lspan))
    Nsteps = int(math.floor(Lspan / hz))

    c_kms = const.c / 1e3
    lam = c_kms / Fc
    alpha = alpha_db / (10 * np.log10(np.exp(1)))
    beta2 = -(D * lam**2) / (2 * np.pi * c_kms)

    w = 2 * np.pi * Fs * np.fft.fftfreq(N)
    argLimOp = -(alpha / 2) + 1j * (beta2 / 2) * (w**2)
    linHalf = xp.asarray(np.exp(argLimOp * (hz / 2)))
    fac = (8.0 / 9.0) * gamma

    G_db = alpha_db * Lspan
    G_lin = 10 ** (G_db / 10)
    if amp == "edfa":
        if not G_db > 0:
            raise ValueError("EDFA gain should be a positive scalar")
        if not NF >= 3:
            raise ValueError("the minimal EDFA noise figure is 3 dB")
        nsp = (G_lin * 10 ** (NF / 10) - 1) / (2 * (G_lin - 1))
        p_noise = (G_lin - 1) * nsp * const.h * Fc * Fs
    ideal_gain = math.exp(alpha / 2 * Lspan)

    Ex = xp.asarray(fields[:, 0, :])
    Ey = xp.asarray(fields[:, 1, :])

    def _fft(a):
        return xp.fft.fft(a, axis=-1)

    def _ifft(a):
        return xp.fft.ifft(a, axis=-1)

    def _totpow(a, b):
        return a.real**2 + a.imag**2 + b.real**2 + b.imag**2

    for span in range(Nspans):
        Ex_conv = Ex.copy()
        Ey_conv = Ey.copy()
        for _ in range(Nsteps):
            Pch = _totpow(Ex, Ey)
            phiRot = fac * (Pch + _totpow(Ex_conv, Ey_conv)) / 2.0
            Ex_hd = _ifft(_fft(Ex) * linHalf)
            Ey_hd = _ifft(_fft(Ey) * linHalf)
            for _it in range(max_iter):
                rot = xp.exp(1j * phiRot * hz)
                Ex_fd = _ifft(_fft(Ex_hd * rot) * linHalf)
                Ey_fd = _ifft(_fft(Ey_hd * rot) * linHalf)
                num = xp.sqrt(xp.sum(xp.abs(Ex_fd - Ex_conv) ** 2 + xp.abs(Ey_fd - Ey_conv) ** 2, axis=-1))
                den = xp.sqrt(xp.sum(xp.abs(Ex_conv) ** 2 + xp.abs(Ey_conv) ** 2, axis=-1))
                lim = num / xp.maximum(den, 1e-30)
                Ex_conv, Ey_conv = Ex_fd, Ey_fd
                if float(xp.max(lim)) < tol:          # every image of the chunk has converged
                    break
                phiRot = fac * (Pch + _totpow(Ex_conv, Ey_conv)) / 2.0
            Ex, Ey = Ex_fd, Ey_fd

        if amp == "edfa":
            noise_x = np.empty((B, N), dtype=np.complex128)
            noise_y = np.empty((B, N), dtype=np.complex128)
            for i in range(B):
                sx = None if seeds is None else span_seed(seeds[i][0], span)
                sy = None if seeds is None else span_seed(seeds[i][1], span)
                noise_x[i] = _gaussian_complex_noise(N, p_noise, sx)
                noise_y[i] = _gaussian_complex_noise(N, p_noise, sy)
            Ex = Ex * math.sqrt(G_lin) + xp.asarray(noise_x)
            Ey = Ey * math.sqrt(G_lin) + xp.asarray(noise_y)
        elif amp == "ideal":
            Ex = Ex * ideal_gain
            Ey = Ey * ideal_gain

    out = xp.stack([Ex, Ey], axis=1)
    return xp.asnumpy(out) if is_gpu else np.asarray(out)


def propagate_ssfm(x: torch.Tensor, cfg, seed: int | None = None, index0: int = 0, chunk: int = 64) -> torch.Tensor:
    """``(B, 2, N)`` torch field -> same shape, propagated by the Manakov SSFM (no gradient).

    Image ``b`` of the call is image ``index0 + b`` of the evaluation set: its ASE seeds are
    ``image_seed(seed, index0 + b, pol)``, so the noise does not depend on the batch size. ``seed=None`` leaves the
    noise unseeded (interactive use only).
    """
    if cfg.link.ase_seed is not None:
        raise ValueError("link.ase_seed must be None: seeds are per image (pass seed=...)")
    if not cfg.fiber.manakov:
        raise ValueError("the batched split-step is the dual-polarization Manakov model (fiber.manakov=True)")
    host = x.detach().cpu().numpy().astype(np.complex128)
    B, P, _ = host.shape
    if P != 2:
        raise ValueError(f"Manakov SSFM needs n_pol=2; got shape {host.shape}")
    seeds = None if seed is None else [
        (image_seed(seed, index0 + b, STREAM["ase_x"]), image_seed(seed, index0 + b, STREAM["ase_y"]))
        for b in range(B)]
    out = manakov_ssfm_batched(host, cfg, seeds=seeds, chunk=chunk)
    return torch.from_numpy(out).to(x.device)
