"""CIFAR-10 loaders: images kept in [0, 1] (ToTensor only; the decoder ends in a sigmoid and SSIM uses data range 1).

Split: the 50,000 training images -> 45,000 train / 5,000 validation (fixed ``split_seed``); the 10,000 test images
are used only for evaluation. Augmentation (random crop with 4-pixel padding + horizontal flip) is applied to the
training split only, through a second dataset object over the same indices, so validation is never augmented.

Data root: ``--data-root``, else ``$SEMCOM_DATA_ROOT``, else ``./data``. Downloaded on first use when ``download``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

__all__ = ["default_data_root", "Splits", "build_datasets", "build_loaders"]


def default_data_root() -> str:
    return os.environ.get("SEMCOM_DATA_ROOT") or "data"


@dataclass(frozen=True)
class Splits:
    train: Dataset            # augmented when cfg.train.augment
    train_plain: Dataset      # the same images, not augmented
    val: Dataset
    test: Dataset


def build_datasets(cfg, root: str | None = None, download: bool = True) -> Splits:
    root = root or default_data_root()
    plain = transforms.ToTensor()
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()])
    full_plain = CIFAR10(root, train=True, transform=plain, download=download)
    full_train = CIFAR10(root, train=True, transform=aug if cfg.train.augment else plain, download=False)
    test = CIFAR10(root, train=False, transform=plain, download=download)
    n_val = int(len(full_plain) * cfg.train.val_fraction)
    g = torch.Generator().manual_seed(cfg.train.split_seed)
    tr_idx, va_idx = (sub.indices for sub in random_split(range(len(full_plain)), [len(full_plain) - n_val, n_val],
                                                           generator=g))
    return Splits(train=Subset(full_train, tr_idx), train_plain=Subset(full_plain, tr_idx),
                  val=Subset(full_plain, va_idx), test=test)


def build_loaders(cfg, root: str | None = None, download: bool = True):
    """(train, val, test) loaders."""
    d = build_datasets(cfg, root, download)
    kw = dict(num_workers=cfg.train.num_workers, pin_memory=True, persistent_workers=cfg.train.num_workers > 0)
    return (DataLoader(d.train, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(d.val, batch_size=cfg.train.eval_batch_size, shuffle=False, **kw),
            DataLoader(d.test, batch_size=cfg.train.eval_batch_size, shuffle=False, **kw))
