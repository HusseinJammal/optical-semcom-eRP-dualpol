"""The conventional chain over the SAME link as the semantic transceiver.

    image -> JPEG (highest quality fitting the rate's budget) -> 802.11n LDPC -> Gray 64-QAM (512 symbols)
          -> semcom.system.link.transmit (same pilots, neighbour frames, transceiver, SSFM, CPR, per-image seeds)
          -> LLRs -> LDPC decoding -> JPEG decoding (failure = black image) -> task classifier

Both systems therefore see byte-identical channels: same symbol and pilot budget, same noise realizations per image.
"""
from __future__ import annotations

import numpy as np
import torch

from ..evaluation.metrics import MetricAccumulator
from ..impairments.params import nominal_trx
from ..system.link import transmit
from .jpeg import choose_jpeg_quality, jpeg_decode_rgb
from .ldpc import FrameCodec

__all__ = ["run_jpeg_ldpc"]


@torch.no_grad()
def run_jpeg_ldpc(images: torch.Tensor, labels: torch.Tensor, cfg, classifier, device, codec: FrameCodec, *,
                  backend: str = "ssfm", p_tx_dbm: float | None = None, reach_km: float | None = None,
                  span_km: float | None = None, nf_db: float | None = None, trx="nominal", seed: int | None = None,
                  index0: int = 0, chunk: int = 64, trx_overrides: dict | None = None, keep: dict | None = None) -> dict:
    """Metrics of JPEG + LDPC on ``images`` (test-set indices ``index0 ...``) at one link condition.

    ``backend="none"`` gives the error-free (no-channel) ceiling of the rate. ``keep`` receives per-image results
    (index, label, pred, logits, ssim, psnr, received 8-bit image, decode_failed, codeword_errors, jpeg_quality,
    payload_bytes).
    """
    seed = cfg.eval.seed if seed is None else seed
    imgs = images.cpu().numpy()
    B = images.shape[0]
    enc = [choose_jpeg_quality(imgs[i], codec.budget_bytes) for i in range(B)]
    for i, (q, h, p) in enumerate(enc):
        if len(p) > codec.budget_bytes:                             # pathological image even at q = 1: truncate
            enc[i] = (q, h, p[: codec.budget_bytes])
    tx = np.stack([codec.encode(p) for _, _, p in enc])            # (B, 512)
    n_pol = cfg.channel.n_pol
    kp = codec.n_symbols // n_pol

    acc = MetricAccumulator()
    cw_err, img_fail = 0, 0
    if keep is not None:
        keep.update(index=np.arange(index0, index0 + B), image_u8=np.zeros((B, 3, 32, 32), dtype=np.uint8),
                    logits=np.zeros((B, 10), dtype=np.float16), decode_failed=np.zeros(B, dtype=bool),
                    codeword_errors=np.zeros(B, dtype=np.int8), jpeg_quality=np.array([q for q, _, _ in enc]),
                    payload_bytes=np.array([len(p) for _, _, p in enc]))
    for s0 in range(0, B, chunk):
        s1 = min(B, s0 + chunk)
        t = torch.from_numpy(tx[s0:s1]).to(torch.complex64).reshape(s1 - s0, n_pol, kp).to(device)
        kw = dict(ssfm_seed=seed, ssfm_index0=index0 + s0) if backend == "ssfm" else {}
        tr = nominal_trx(cfg, s1 - s0, **(trx_overrides or {})) if trx == "nominal" else trx
        rx = transmit(t, cfg, backend=backend, p_tx_dbm=p_tx_dbm, reach_km=reach_km, span_km=span_km,
                      nf_db=nf_db, trx=tr, trx_seed=seed, **kw).reshape(s1 - s0, -1).cpu().numpy()
        for j in range(s1 - s0):
            i = s0 + j
            q, header, payload = enc[i]
            data, e = codec.decode(rx[j], len(payload))
            cw_err += e
            img = jpeg_decode_rgb(header + data)
            failed = img is None
            if failed:
                img_fail += 1
                img = np.zeros((3, 32, 32), dtype=np.float32)
            xr = torch.from_numpy(img).unsqueeze(0).to(device)
            logits = classifier(xr)
            acc.add(images[i:i + 1].to(device), xr, labels[i:i + 1].to(device), logits)
            if keep is not None:
                keep["image_u8"][i] = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
                keep["logits"][i] = logits[0].float().cpu().numpy().astype(np.float16)
                keep["decode_failed"][i] = failed
                keep["codeword_errors"][i] = e
    if keep is not None:
        keep.update(acc.per_image())
    r = acc.result()
    r.pop("confusion", None)
    r.update(rate=codec.rate, eff_rate=codec.eff_rate, budget_bytes=codec.budget_bytes, backend=backend,
             p_tx_dbm=p_tx_dbm, reach_km=reach_km, span_km=span_km, nf_db=nf_db, n=B, seed=seed,
             trx_overrides=trx_overrides or {}, codeword_error_rate=cw_err / (2 * B), image_failure_rate=img_fail / B,
             jpeg_quality_mean=float(np.mean([q for q, _, _ in enc])),
             payload_bytes_mean=float(np.mean([len(p) for _, _, p in enc])))
    return r
