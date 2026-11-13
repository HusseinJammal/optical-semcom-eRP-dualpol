"""Checkpoints: the state dict plus the configuration as plain data (no pickled classes).

    {"format": "semcom/1", "config": Config.to_dict(), "model": state_dict, "epoch": int, ...}
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from .transceiver import SemanticTransceiver

__all__ = ["FORMAT", "save_checkpoint", "load_checkpoint"]

FORMAT = "semcom/1"


def save_checkpoint(path, model: SemanticTransceiver, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "config": model.cfg.to_dict(), "model": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    """-> (model in eval mode, config, the raw checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (format={ck.get('format')!r})")
    cfg = Config.from_dict(ck["config"])
    model = SemanticTransceiver(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck
