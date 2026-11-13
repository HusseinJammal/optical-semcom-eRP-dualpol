"""Symbol mapping: encoder reals -> square M-QAM symbols (straight-through estimator), and the receiver's inverse.

TRANSMITTER (``QAMMapper``). The encoder's 2k reals are paired into k complex values u (I, Q interleaved), normalised
to unit mean power PER IMAGE, and each axis is clipped to the grid and rounded to the nearest of the sqrt(M) levels
of the unit-power square M-QAM (64-QAM: +-0.154, +-0.463, +-0.772, +-1.080). The rounding uses the straight-through
estimator  s = u + sg[Q(u) - u]: the forward pass emits grid points, the backward pass treats rounding as identity.
It is deterministic (training and inference emit the same symbols) and ordered (neighbouring encoder values land on
neighbouring levels). The grid is fixed and standard; no bit labelling or FEC is involved.

Layout: k symbols -> (n_pol, k / n_pol), polarization p carrying symbols [p k/n_pol, (p+1) k/n_pol).

RECEIVER (``SymbolDemapper``). The soft received symbols (after CPR and AGC) are flattened back in the same order and
split into interleaved I, Q reals; the decoder reads them directly (no hard decision).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["qam_grid", "QAMMapper", "SymbolDemapper", "usage_metrics"]


def qam_grid(n_points: int) -> torch.Tensor:
    """Square M-QAM as an ``(M, 2)`` real tensor with unit mean power, I-major point order."""
    side = math.isqrt(n_points)
    if side * side != n_points:
        raise ValueError(f"square QAM needs a square order (16, 64, 256, ...); got M={n_points}")
    lv = torch.arange(-(side - 1), side, 2, dtype=torch.float32)
    g = torch.stack([lv.repeat_interleave(side), lv.repeat(side)], dim=1)
    p = (g[:, 0] ** 2 + g[:, 1] ** 2).mean().clamp(min=1e-12)
    return g / torch.sqrt(p)


class QAMMapper(nn.Module):
    """``(B, 2k)`` reals -> ``(B, n_pol, k / n_pol)`` complex symbols on the square M-QAM grid (STE)."""

    def __init__(self, n_points: int = 64, n_pol: int = 2):
        super().__init__()
        side = math.isqrt(n_points)
        if side * side != n_points:
            raise ValueError(f"the QAM mapper needs a square order (16, 64, 256, ...); got M={n_points}")
        self.n_points, self.side, self.n_pol = int(n_points), side, int(n_pol)
        a = math.sqrt(3.0 / (2.0 * (n_points - 1)))                  # unit mean power: E[x^2] = 1/2 per axis
        self.register_buffer("levels", a * (2.0 * torch.arange(side, dtype=torch.float32) - (side - 1)))
        self.a = a

    def constellation(self) -> torch.Tensor:
        """The M grid points, ``(M,)`` complex, I-major (same order as :func:`qam_grid`)."""
        lv = self.levels
        return torch.complex(lv.repeat_interleave(self.side), lv.repeat(self.side))

    def _quantize(self, v: torch.Tensor):
        """Nearest level per element with a straight-through gradient: (quantised, integer level index)."""
        s, a = self.side, self.a
        vmax = a * (s - 1)
        vc = v.clamp(-vmax, vmax)                                    # outside the grid: clipped, gradient 0
        idx = torch.round((vc / a + (s - 1)) / 2.0).clamp(0, s - 1)
        q = a * (2.0 * idx - (s - 1))
        return vc + (q - vc).detach(), idx.long()

    def forward(self, z: torch.Tensor, return_index: bool = False):
        if z.shape[-1] % 2 != 0:
            raise ValueError(f"need an even feature count; got {z.shape[-1]}")
        i, q = z[..., 0::2], z[..., 1::2]
        rms = torch.sqrt(torch.mean(i ** 2 + q ** 2, dim=-1, keepdim=True)).clamp(min=1e-12)
        qi, ii = self._quantize(i / rms)
        qq, iq = self._quantize(q / rms)
        s = torch.complex(qi, qq)
        B, k = s.shape
        if k % self.n_pol != 0:
            raise ValueError(f"k={k} not divisible by n_pol={self.n_pol}")
        out = s.view(B, self.n_pol, k // self.n_pol)
        return (out, ii * self.side + iq) if return_index else out


class SymbolDemapper(nn.Module):
    """``(B, n_pol, k / n_pol)`` complex -> ``(B, 2k)`` reals, I and Q interleaved (inverse layout of QAMMapper)."""

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.reshape(s.shape[0], -1)
        z = torch.stack([s.real, s.imag], dim=-1)
        return z.reshape(z.shape[0], -1)


@torch.no_grad()
def usage_metrics(index: torch.Tensor, n_points: int) -> dict:
    """How the encoder uses the grid: effective number of points (inverse Simpson) and normalised usage entropy."""
    usage = torch.bincount(index.reshape(-1), minlength=n_points).float()
    usage = usage / usage.sum().clamp(min=1.0)
    n_eff = 1.0 / torch.sum(usage ** 2).clamp(min=1e-12)
    ent = -torch.sum(usage * torch.log(usage.clamp(min=1e-12))) / math.log(n_points)
    return {"n_effective": float(n_eff), "usage_entropy": float(ent), "n_points": int(n_points)}
