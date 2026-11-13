"""Channel-state conditioning: the link vector c = [P, L] and feature-wise linear modulation (FiLM).

The ONLY channel-state information the networks receive is the total launch power P [dBm] and the link length L [km]:

    c = [ (P - p_ref) / p_scale , (L - l_ref) / l_scale ]   with defaults mapping P in [-6, +3] dBm and
                                                              L in [400, 2480] km onto [-1, +1]

Span length, noise figure, transceiver SNR, linewidths, frequency offset and carrier phase stay hidden. The same c is
given to the encoder and the decoder. Each FiLM layer turns c into per-channel (dgamma, dbeta) through a small MLP and
modulates a feature map as  z <- (1 + dgamma) * z + dbeta.  The last MLP layer is zero-initialised, so every FiLM layer
starts as the exact identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["link_condition", "FiLM", "FiLMGenerator"]


def link_condition(model_cfg, p_dbm: float, reach_km: float, device=None) -> torch.Tensor:
    """The ``(2,)`` conditioning vector for ``model_cfg.conditioning`` ("power_reach" or the "constant" ablation)."""
    if model_cfg.conditioning == "power_reach":
        return torch.tensor([(float(p_dbm) - model_cfg.cond_p_ref) / model_cfg.cond_p_scale,
                             (float(reach_km) - model_cfg.cond_l_ref) / model_cfg.cond_l_scale],
                            device=device, dtype=torch.float32)
    if model_cfg.conditioning == "constant":
        return torch.zeros(2, device=device, dtype=torch.float32)
    raise ValueError(f"unknown conditioning {model_cfg.conditioning!r}; choose power_reach or constant")


class FiLM(nn.Module):
    """y = (1 + dgamma) * x + dbeta, broadcast over the spatial axes."""

    def forward(self, x: torch.Tensor, dgamma: torch.Tensor, dbeta: torch.Tensor) -> torch.Tensor:
        while dgamma.dim() < x.dim():
            dgamma = dgamma.unsqueeze(-1)
            dbeta = dbeta.unsqueeze(-1)
        return (1.0 + dgamma) * x + dbeta


class FiLMGenerator(nn.Module):
    """c -> (dgamma, dbeta) for a feature map of ``channels``: MLP in_dim -> hidden (ReLU) -> 2 * channels."""

    def __init__(self, channels: int, hidden: int = 64, enabled: bool = True, in_dim: int = 2):
        super().__init__()
        self.enabled = enabled
        self.channels = channels
        if not enabled:
            return
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.film = FiLM()

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if not self.enabled or cond is None:
            return x
        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(x.shape[0], -1)
        dgamma, dbeta = self.net(cond).chunk(2, dim=-1)
        return self.film(x, dgamma, dbeta)
