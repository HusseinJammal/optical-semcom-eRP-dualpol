"""Launch power: TOTAL over both polarizations, set per image, measured over the FRAME (data + pilots).

Each polarization is scaled to P / n_pol, so the summed optical power is P. The mean is taken over the frame's
samples only (not the TX pad or the guard): in a continuous stream that is the power the fibre carries. Setting the
power per image (rather than with one mean over the batch) keeps every image at exactly P whatever else is in its batch.
"""
from __future__ import annotations

import torch

__all__ = ["set_frame_launch_power"]


def set_frame_launch_power(x: torch.Tensor, p_tx_watt: float, lead_samples: int, frame_samples: int) -> torch.Tensor:
    """Scale ``(B, P, N)`` so the frame segment ``[lead, lead + frame)`` has total mean power ``p_tx_watt``."""
    n_pol = x.shape[1]
    seg = x[..., lead_samples:lead_samples + frame_samples]
    mean_p = torch.mean(seg.real ** 2 + seg.imag ** 2, dim=(1, 2), keepdim=True)
    return x * torch.sqrt((p_tx_watt / n_pol) / mean_p).to(x.dtype)
