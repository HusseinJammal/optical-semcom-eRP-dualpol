"""Task classifiers for the conventional chain: a CIFAR ResNet-18 trained on that chain's own output.

The semantic transceiver classifies with its jointly trained head. For a fair task comparison the conventional chain
gets, per code rate, a CIFAR ResNet-18 (3x3 stride-1 stem, no max-pool) trained from scratch on the training images
compressed EXACTLY as the chain compresses them at that rate (same byte budget, same per-image quality search, no
channel errors): its best case. ``scripts/train_baseline_classifier.py`` trains them.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .jpeg import choose_jpeg_quality, jpeg_decode_rgb

__all__ = ["build_cifar_resnet18", "to_u8", "jpeg_round_trip_u8", "load_task_classifier", "classify_u8"]


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    from torchvision.models import resnet18
    m = resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def to_u8(x: np.ndarray) -> np.ndarray:
    """float images in [0, 1] -> uint8 (round half to even)."""
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def _one(args):
    img, budget = args
    q, header, payload = choose_jpeg_quality(img, budget)
    payload = payload[:budget]
    return to_u8(jpeg_decode_rgb(header + payload)), q, len(payload)


def jpeg_round_trip_u8(images: np.ndarray, budget_bytes: int, workers: int = 4) -> tuple:
    """(N, 3, 32, 32) float -> the chain's error-free received images (uint8), qualities, payload bytes."""
    jobs = [(images[i], budget_bytes) for i in range(images.shape[0])]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_one, jobs, chunksize=256))
    else:
        out = [_one(j) for j in jobs]
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])


def load_task_classifier(path, device) -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_cifar_resnet18().to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def classify_u8(model: nn.Module, images_u8: np.ndarray, device, batch: int = 1000) -> np.ndarray:
    """uint8 (N, 3, 32, 32) -> float32 logits (N, 10)."""
    out = []
    for s in range(0, images_u8.shape[0], batch):
        x = torch.from_numpy(images_u8[s:s + batch]).to(device).float().div_(255.0)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 10), dtype=np.float32)

