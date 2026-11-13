"""Neural-network building blocks shared by the encoder and the decoder."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ResBlock", "conv_bn_relu"]


class ResBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm, identity shortcut, ReLU after the sum."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return torch.relu(self.block(x) + x)


def conv_bn_relu(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    """3x3 convolution (no bias) + BatchNorm + ReLU."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
