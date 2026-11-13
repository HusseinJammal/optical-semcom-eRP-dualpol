"""Semantic decoder: received soft symbols -> (reconstructed image, class logits). Mirrors the encoder.

    stage           operation (paper, arch "latent8")                             output
    demapper        512 complex -> 1024 reals (I, Q interleaved) -> reshape        16 x 8 x 8
    from_symbols    1x1 conv 16->256                                               256 x 8 x 8
    bottleneck      FiLM(c); residual block (256)                                  256 x 8 x 8
    cls_head        global average pool -> Linear 256->256, ReLU -> Linear 256->10  10 logits
    up1             3x3 conv stride 1 256->256, BN, ReLU; residual block           256 x 8 x 8  (transposed stride 2, "base")
    up2             3x3 transposed conv stride 2 256->128, BN, ReLU; residual      128 x 16 x 16
    up3             3x3 transposed conv stride 2 128->64, BN, ReLU; FiLM(c)        64 x 32 x 32
    recon_head      3x3 conv 64->3, sigmoid                                        3 x 32 x 32 in [0, 1]

Both heads share the trunk. 3.73 M parameters, 608 MFLOP per image.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..conditioning.film import FiLMGenerator
from ..config import ModelConfig
from ..layers import ResBlock
from ..mapper.qam import SymbolDemapper

__all__ = ["SemanticDecoder"]


def _up(c_in: int, c_out: int) -> nn.Sequential:
    """3x3 transposed conv stride 2 (x2 upsampling) + BN + ReLU."""
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class SemanticDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        w = cfg.widths
        self.demapper = SymbolDemapper()
        self.from_symbols = nn.Conv2d(cfg.symbol_channels, w[3], 1)
        self.film_bottleneck = FiLMGenerator(w[3], cfg.film_hidden, cfg.film_enabled, in_dim=cfg.cond_dim)
        self.res = ResBlock(w[3])
        if cfg.arch == "latent8":
            self.up1 = nn.Sequential(nn.Conv2d(w[3], w[2], 3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(w[2]), nn.ReLU(inplace=True))
        else:
            self.up1 = _up(w[3], w[2])
        self.res1 = ResBlock(w[2])
        self.up2 = _up(w[2], w[1])
        self.res2 = ResBlock(w[1])
        self.up3 = _up(w[1], w[0])
        self.film_out = FiLMGenerator(w[0], cfg.film_hidden, cfg.film_enabled, in_dim=cfg.cond_dim)
        self.recon_head = nn.Conv2d(w[0], cfg.image_ch, 3, padding=1)
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(w[3], 256), nn.ReLU(inplace=True), nn.Linear(256, cfg.n_class))

    def forward(self, s_hat: torch.Tensor, cond: torch.Tensor | None = None):
        """``(B, n_pol, k / n_pol)`` received symbols -> (``(B, 3, 32, 32)`` image, ``(B, n_class)`` logits)."""
        z = self.demapper(s_hat)
        hw = self.cfg.feat_hw
        h = z.view(z.shape[0], self.cfg.symbol_channels, hw, hw)
        h = self.from_symbols(h)
        h = self.film_bottleneck(h, cond)
        h = self.res(h)
        logits = self.cls_head(h)
        r = self.res2(self.up2(self.res1(self.up1(h))))
        r = self.film_out(self.up3(r), cond)
        x_hat = torch.sigmoid(self.recon_head(r))
        return x_hat, logits
