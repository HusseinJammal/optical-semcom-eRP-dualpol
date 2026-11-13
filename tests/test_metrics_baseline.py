"""Metrics (batching invariance) and the JPEG + 802.11n LDPC baseline."""
import numpy as np
import pytest
import torch

from conftest import needs_opticommpy
from semcom.evaluation.metrics import MetricAccumulator, psnr_from_mse


def test_metrics_do_not_depend_on_the_batching():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(10, 3, 32, 32, generator=g)
    xh = (x + 0.05 * torch.randn(10, 3, 32, 32, generator=g)).clamp(0, 1)
    y = torch.randint(0, 10, (10,), generator=g)
    lg = torch.randn(10, 10, generator=g)
    tx = torch.randn(10, 2, 256, dtype=torch.complex64, generator=g)
    rx = tx + 0.1 * torch.randn(10, 2, 256, dtype=torch.complex64, generator=g)
    outs = []
    for bs in (1, 3, 10):
        a = MetricAccumulator()
        for i in range(0, 10, bs):
            a.add(x[i:i + bs], xh[i:i + bs], y[i:i + bs], lg[i:i + bs], tx[i:i + bs], rx[i:i + bs])
        outs.append(a.result())
    for k in ("accuracy", "ssim", "psnr_per_image", "psnr_pooled", "evm_pct", "snr_eff_ls_db", "ce"):
        assert abs(outs[0][k] - outs[1][k]) < 1e-9 and abs(outs[0][k] - outs[2][k]) < 1e-9
    assert psnr_from_mse(0.0) == 120.0


@needs_opticommpy
@pytest.mark.parametrize("rate,budget,eff", [("1/2", 202, 0.527), ("2/3", 270, 0.703), ("3/4", 303, 0.791),
                                             ("5/6", 337, 0.879)])
def test_codec_fills_exactly_512_symbols_and_round_trips(rate, budget, eff):
    from semcom.baseline import FrameCodec
    c = FrameCodec(rate)
    assert c.budget_bytes == budget and abs(c.eff_rate - eff) < 5e-4 and sum(c.punct) == 168
    payload = np.random.default_rng(0).integers(0, 256, budget, dtype=np.uint8).tobytes()
    sym = c.encode(payload)
    assert sym.shape == (512,) and abs(np.mean(np.abs(sym) ** 2) - 1) < 0.1
    data, errs = c.decode(sym, budget)
    assert data == payload and errs == 0


@needs_opticommpy
def test_jpeg_fits_the_budget_and_the_error_free_chain_decodes(cfg):
    from semcom.baseline import FrameCodec, build_cifar_resnet18, choose_jpeg_quality, run_jpeg_ldpc
    g = torch.Generator().manual_seed(0)
    imgs = torch.rand(2, 3, 32, 32, generator=g)
    codec = FrameCodec("1/2")
    q, h, p = choose_jpeg_quality(imgs[0].numpy(), codec.budget_bytes)
    assert len(p) <= codec.budget_bytes and 1 <= q <= 95
    r = run_jpeg_ldpc(imgs, torch.tensor([1, 2]), cfg, build_cifar_resnet18().eval(), "cpu", codec, backend="none")
    assert r["image_failure_rate"] == 0.0 and r["codeword_error_rate"] == 0.0
