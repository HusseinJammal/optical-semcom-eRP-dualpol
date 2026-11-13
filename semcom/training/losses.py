"""Training loss  L = lambda_CE CE(logits, y) + lambda_MSE MSE(x_hat, x) + lambda_SSIM (1 - SSIM(x_hat, x)).

Paper weights (1, 100, 2), label smoothing 0.1 on the CE term. SSIM is ``pytorch_msssim.ssim`` (differentiable,
Gaussian window 11, sigma 1.5, data range 1); evaluation uses scikit-image's SSIM, a different implementation.
"""
from __future__ import annotations

import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_metric

from ..config import TrainConfig

__all__ = ["semantic_loss"]


def semantic_loss(x, x_hat, labels, logits, cfg: TrainConfig | None = None):
    """-> (total loss, dict of detached parts). The logged CE is the plain (unsmoothed) one."""
    cfg = cfg or TrainConfig()
    mse = F.mse_loss(x_hat, x)
    ssim_val = ssim_metric(x_hat, x, data_range=1.0)
    ce_train = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)
    ce = ce_train if cfg.label_smoothing == 0.0 else F.cross_entropy(logits.detach(), labels)
    total = cfg.lambda_ce * ce_train + cfg.lambda_mse * mse + cfg.lambda_ssim * (1.0 - ssim_val)
    return total, {"ce": float(ce.detach()), "mse": float(mse.detach()), "ssim": float(ssim_val.detach()),
                   "total": float(total.detach())}
