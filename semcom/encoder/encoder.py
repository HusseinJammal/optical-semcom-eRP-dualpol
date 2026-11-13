"""Semantic encoder: CIFAR-10 image -> k = 512 complex 64-QAM symbols over two polarizations.

    stage       operation (paper, arch "latent8")                     output
    stem        3x3 conv 3->64, BN, ReLU; FiLM(c)                      64 x 32 x 32
    down1       3x3 conv stride 2, 64->128, BN, ReLU                   128 x 16 x 16
    down2       3x3 conv stride 2, 128->256, BN, ReLU                  256 x 8 x 8
    down3       3x3 conv stride 1, 256->256, BN, ReLU                  256 x 8 x 8   (stride 2 -> 4 x 4 for "base")
    res         residual block (256)                                   256 x 8 x 8
    bottleneck  FiLM(c)                                                256 x 8 x 8
    projection  1x1 conv 256->16                                       16 x 8 x 8 = 1024 reals
    mapper      pair to 512 complex, unit power per image, STE 64-QAM  (2, 256) complex

2.19 M parameters, 306 MFLOP per image.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..conditioning.film import FiLMGenerator
from ..config import ModelConfig
from ..layers import ResBlock, conv_bn_relu
from ..mapper.qam import QAMMapper

__all__ = ["SemanticEncoder"]


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, n_pol: int = 2):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        w = cfg.widths
        self.stem = conv_bn_relu(cfg.image_ch, w[0], 1)
        self.down1 = conv_bn_relu(w[0], w[1], 2)
        self.down2 = conv_bn_relu(w[1], w[2], 2)
        self.down3 = conv_bn_relu(w[2], w[3], 1 if cfg.arch == "latent8" else 2)
        self.res = ResBlock(w[3])
        self.film_stem = FiLMGenerator(w[0], cfg.film_hidden, cfg.film_enabled, in_dim=cfg.cond_dim)
        self.film_bottleneck = FiLMGenerator(w[3], cfg.film_hidden, cfg.film_enabled, in_dim=cfg.cond_dim)
        self.to_symbols = nn.Conv2d(w[3], cfg.symbol_channels, 1)
        self.mapper = QAMMapper(cfg.n_points, n_pol=n_pol)

    def features(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """``(B, 3, 32, 32)`` -> ``(B, 2k)`` reals before the mapper."""
        h = self.film_stem(self.stem(x), cond)
        h = self.res(self.down3(self.down2(self.down1(h))))
        h = self.film_bottleneck(h, cond)
        return self.to_symbols(h).flatten(1)

    def symbol_indices(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, k)`` constellation-point index of every emitted symbol (for usage diagnostics)."""
        return self.mapper(self.features(x, cond), return_index=True)[1]

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 3, 32, 32)`` images in [0, 1] -> ``(B, n_pol, k / n_pol)`` complex symbols."""
        return self.mapper(self.features(x, cond))
