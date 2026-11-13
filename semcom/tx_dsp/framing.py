"""Frame embedding: the image's frame travels inside a continuous stream of random neighbour frames.

Instead of an isolated burst padded with zeros, the TX pad and the guard on both sides carry random M-QAM data at the
frame's own mean payload power, with pilots continuing the frame's pilot cadence. Every image therefore sees a
stationary, continuous stream (as on a real coherent link): the same nonlinear interference from its neighbours and
the same dispersion-induced inter-frame leakage. The receiver discards everything outside the frame.
"""
from __future__ import annotations

import math

import torch

from ..utils.random import PerImageRng
from .pilots import pilot_amplitude, remove_pilots

__all__ = ["neighbour_stream", "zero_padded_stream"]


def neighbour_stream(frame: torch.Tensor, n_left: int, n_right: int, m: int, spacing: int | None,
                     rng: PerImageRng) -> torch.Tensor:
    """``(B, P, F)`` frame -> ``(B, P, n_left + F + n_right)`` stream with random M-QAM neighbours on both sides.

    Neighbour data symbols are uniform over the grid, scaled per image to the frame's mean payload power (pilots
    excluded). With pilots (``spacing`` not None) the neighbours carry random QPSK corner pilots every spacing + 1
    symbols, aligned with the frame's own pilot grid.
    """
    b, p, f = frame.shape
    side = math.isqrt(m)
    a = math.sqrt(3.0 / (2.0 * (m - 1)))
    rdtype = frame.real.dtype
    n_tot = n_left + n_right
    idx = rng.randint("nb_data", side, (2, p, n_tot))                                    # (B, 2, P, n_tot)
    lv = a * (2.0 * idx.to(rdtype) - (side - 1))
    nb = torch.complex(lv[:, 0], lv[:, 1]).to(frame.dtype)                                # unit mean power
    payload = remove_pilots(frame, spacing) if spacing is not None else frame
    pw = torch.mean(payload.real ** 2 + payload.imag ** 2, dim=(1, 2), keepdim=True).detach()
    nb = nb * torch.sqrt(pw).to(nb.dtype)
    left, right = nb[..., :n_left], nb[..., n_left:]
    if spacing is not None:
        pa = pilot_amplitude(m)
        bits = rng.randint("nb_pilot", 2, (2, p, n_tot)).to(rdtype) * 2 - 1
        pil = torch.complex(pa * bits[:, 0], pa * bits[:, 1]).to(frame.dtype)
        rel_left = torch.arange(-n_left, 0, device=frame.device)                           # frame-relative index
        rel_right = torch.arange(f, f + n_right, device=frame.device)
        ml = torch.remainder(rel_left, spacing + 1) == 0
        mr = torch.remainder(rel_right, spacing + 1) == 0
        left = torch.where(ml, pil[..., :n_left], left)
        right = torch.where(mr, pil[..., n_left:], right)
    return torch.cat([left, frame, right], dim=-1)


def zero_padded_stream(frame: torch.Tensor, n_left: int, n_right: int) -> torch.Tensor:
    """``(B, P, F)`` frame with ``n_left`` / ``n_right`` zero symbols (an isolated burst; neighbour frames off)."""
    b, p, _ = frame.shape
    z = torch.zeros(b, p, n_left, dtype=frame.dtype, device=frame.device)
    zt = torch.zeros(b, p, n_right, dtype=frame.dtype, device=frame.device)
    return torch.cat([z, frame, zt], dim=-1)
