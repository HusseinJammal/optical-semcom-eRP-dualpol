"""Random-number discipline.

Two modes are used everywhere:

* TRAINING draws from one ``torch.Generator`` per stream (channel noise, link draws, transceiver draws), so a run
  is reproducible from its seed.
* EVALUATION derives an independent seed for every (image, stream) from one integer seed and the image's index in
  the test set (:func:`image_seed`). A result therefore never depends on the batch size, and evaluating the test set
  in blocks reproduces a full run image by image.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["image_seed", "span_seed", "generator_for", "PerImageRng", "STREAM"]

# Per-image random streams (the third component of image_seed). 0 and 1 are the SSFM amplifier noise of the X and Y
# polarization; the transceiver streams use 101..107.
STREAM = {"ase_x": 0, "ase_y": 1, "nb_data": 101, "nb_pilot": 102, "tx_noise": 103, "tx_laser": 104, "phase0": 105,
          "lo_laser": 106, "rx_noise": 107}


def image_seed(seed: int, index: int, stream: int = 0) -> int:
    """A well-mixed 32-bit seed for (evaluation seed, image index, stream).

    ``SeedSequence`` rather than ``seed + index``: neighbouring integer seeds give the Mersenne Twister nearly
    identical states, and correlated noise between adjacent images would shrink the effective sample size.
    """
    return int(np.random.SeedSequence([int(seed), int(index), int(stream)]).generate_state(1)[0])


def span_seed(base: int, span: int) -> int:
    """Seed for amplifier ``span`` of an image whose base seed is ``base`` (span 0 uses ``base`` itself)."""
    if span == 0:
        return int(base)
    return int(np.random.SeedSequence([int(base), int(span)]).generate_state(1)[0])


def generator_for(generator: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    """A generator that lives on ``device`` (PyTorch refuses to draw on CUDA with a CPU generator).

    Same device: the SAME object is returned. Different device: a device-local generator seeded by ONE draw from
    the caller's generator, so the run stays reproducible and successive calls get distinct noise.
    """
    if generator is None or generator.device.type == device.type:
        return generator
    seed = int(torch.randint(0, 2**62, (1,), generator=generator))
    return torch.Generator(device=device).manual_seed(seed)


class PerImageRng:
    """Draws ``(B, ...)`` tensors from one generator (training) or from per-image seeds (evaluation).

    ``seed`` given: image ``b`` of the batch is image ``index0 + b`` of the evaluation set and every named stream
    (see :data:`STREAM`) gets its own seed, so draws are independent of batch size and of each other.
    ``seed`` None: everything is drawn from ``generator`` in call order.
    """

    def __init__(self, batch: int, device, generator: torch.Generator | None = None, seed: int | None = None,
                 index0: int = 0):
        self.b, self.device, self.gen, self.seed, self.index0 = batch, torch.device(device), generator, seed, index0

    def _per_image(self, stream: str, fn):
        outs = []
        for i in range(self.b):
            g = torch.Generator().manual_seed(image_seed(self.seed, self.index0 + i, STREAM[stream]))
            outs.append(fn(g))
        return torch.stack(outs).to(self.device)

    def randn(self, stream: str, shape: tuple, dtype) -> torch.Tensor:
        if self.seed is not None:
            return self._per_image(stream, lambda g: torch.randn(shape, generator=g, dtype=dtype))
        g = generator_for(self.gen, self.device)
        return torch.randn((self.b, *shape), generator=g, dtype=dtype, device=self.device)

    def rand(self, stream: str, shape: tuple, dtype) -> torch.Tensor:
        if self.seed is not None:
            return self._per_image(stream, lambda g: torch.rand(shape, generator=g, dtype=dtype))
        g = generator_for(self.gen, self.device)
        return torch.rand((self.b, *shape), generator=g, dtype=dtype, device=self.device)

    def randint(self, stream: str, high: int, shape: tuple) -> torch.Tensor:
        if self.seed is not None:
            return self._per_image(stream, lambda g: torch.randint(high, shape, generator=g))
        g = generator_for(self.gen, self.device)
        return torch.randint(high, (self.b, *shape), generator=g, device=self.device)
