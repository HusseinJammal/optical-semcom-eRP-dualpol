"""Mapper, encoder/decoder, the full transceiver, gradients, checkpoints, and a training smoke run."""
import math
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import needs_opticommpy
from semcom.config import Config
from semcom.mapper import QAMMapper, SymbolDemapper, qam_grid, usage_metrics
from semcom.system import SemanticTransceiver, load_checkpoint, save_checkpoint, transmit


def test_mapper_emits_grid_points_and_passes_gradient():
    m = QAMMapper(64, n_pol=2)
    z = torch.randn(4, 1024, requires_grad=True)
    s = m(z)
    assert s.shape == (4, 2, 256)
    lv = m.levels
    for part in (s.real, s.imag):
        assert torch.all(torch.min((part.unsqueeze(-1) - lv).abs(), dim=-1).values < 1e-6)
    assert abs(float((qam_grid(64) ** 2).sum(1).mean()) - 1) < 1e-6
    s.real.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def test_demapper_inverts_the_layout():
    m, d = QAMMapper(64, n_pol=2), SymbolDemapper()
    z = torch.randn(3, 1024)
    s = m(z)
    r = d(s)
    assert torch.equal(r, torch.stack([s.reshape(3, -1).real, s.reshape(3, -1).imag], -1).reshape(3, -1))
    # inside the grid range, each real lands within half a level step (= a) of the normalised encoder output
    rms = torch.sqrt((z[:, 0::2] ** 2 + z[:, 1::2] ** 2).mean(dim=-1, keepdim=True))
    u = (z / rms.repeat_interleave(1, dim=-1))
    inside = u.abs() <= m.a * (m.side - 1)
    assert float((r - u)[inside].abs().max()) <= m.a + 1e-6
    idx = m(z, return_index=True)[1]
    assert usage_metrics(idx, 64)["n_effective"] > 10


def test_model_shapes_parameters_and_film_identity(cfg):
    torch.manual_seed(0)
    model = SemanticTransceiver(cfg).eval()
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    assert abs(n_enc / 1e6 - 2.19) < 0.01 and abs(n_dec / 1e6 - 3.73) < 0.01
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        s1 = model.encoder(x, model.conditioning(0.0, 400.0))
        s2 = model.encoder(x, model.conditioning(-6.0, 2480.0))
    assert torch.equal(s1, s2)                                # zero-initialised FiLM: identity at init
    with torch.no_grad():
        x_hat, logits = model.decoder(transmit(s1, cfg, backend="none"), model.conditioning())
    assert x_hat.shape == (2, 3, 32, 32) and logits.shape == (2, 10)


def test_base_architecture_ablation():
    c = Config()
    c = replace(c, model=replace(c.model, arch="base"))
    m = SemanticTransceiver(c).eval()
    assert c.model.feat_hw == 4 and c.model.symbol_channels == 64
    with torch.no_grad():
        assert m.encoder(torch.rand(1, 3, 32, 32)).shape == (1, 2, 256)


def test_gradient_reaches_the_encoder_through_the_erp_link(cfg):
    torch.manual_seed(0)
    model = SemanticTransceiver(cfg).train()
    x = torch.rand(2, 3, 32, 32)
    from semcom.impairments import draw_trx
    x_hat, logits = model(x, backend="erp", generator=torch.Generator().manual_seed(0), reach_km=400.0,
                          p_tx_dbm=0.0, trx=draw_trx(cfg, 2, torch.Generator().manual_seed(1)))
    (x_hat.mean() + logits.mean()).backward()
    g = model.encoder.to_symbols.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    assert model.encoder.stem[0].weight.grad.abs().sum() > 0


def test_linear_noiseless_link_returns_the_symbols(cfg):
    """TX DSP -> linear fibre (ideal amplifiers) -> RX DSP is transparent: CDC, matched filter, timing, pilots align."""
    s = QAMMapper(64, 2)(torch.randn(2, 1024))
    r = transmit(s, cfg, backend="linear", amp="ideal", trx=None, reach_km=400.0, p_tx_dbm=0.0,
                 generator=torch.Generator().manual_seed(0))
    s_n = s / torch.sqrt((s.abs() ** 2).mean(dim=-1, keepdim=True))
    err = 10 * math.log10(float(((r - s_n).abs() ** 2).mean()))
    # the floor is the 257-tap RRC truncation ISI (6.3e-4 = -32 dB back to back, measured); a misaligned CDC, matched
    # filter, sampling instant or pilot layout would land far above it
    assert err < -30


def test_checkpoint_round_trip(tmp_path, cfg):
    torch.manual_seed(0)
    m = SemanticTransceiver(cfg).eval()
    save_checkpoint(tmp_path / "m.pt", m, epoch=3)
    m2, cfg2, ck = load_checkpoint(tmp_path / "m.pt")
    assert cfg2 == cfg and ck["epoch"] == 3
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m.encoder(x, m.conditioning()), m2.encoder(x, m2.conditioning()))


@pytest.mark.slow
@needs_opticommpy
def test_ssfm_forward_carries_no_gradient(cfg):
    m = SemanticTransceiver(cfg)
    x = torch.rand(1, 3, 32, 32)
    x_hat, _ = m.forward_ssfm(x, ssfm_seed=1234, reach_km=400.0, p_tx_dbm=0.0)
    x_hat.mean().backward()
    assert m.encoder.to_symbols.weight.grad is None


def test_training_smoke(tmp_path, monkeypatch):
    """Two short epochs on synthetic data over a short randomized link: the loop runs and writes checkpoints."""
    import semcom.training.trainer as trainer
    c = Config()
    c = replace(c, channel=replace(c.channel, reach_min_km=400.0, reach_max_km=480.0, span_min_km=80, span_max_km=80),
                train=replace(c.train, epochs=2, batch_size=4, eval_batch_size=4, num_workers=0, log_every=1),
                run_name="smoke")
    g = torch.Generator().manual_seed(0)
    ds = TensorDataset(torch.rand(24, 3, 32, 32, generator=g), torch.randint(0, 10, (24,), generator=g))

    def fake_loaders(cfg, root=None, download=True):
        return (DataLoader(ds, batch_size=4, shuffle=True, drop_last=True), DataLoader(ds, batch_size=4),
                DataLoader(ds, batch_size=4))

    monkeypatch.setattr(trainer, "build_loaders", fake_loaders)
    _, hist = trainer.train(c, device="cpu", out_dir=str(tmp_path), max_steps_per_epoch=2, monitor_batches=2)
    assert len(hist) == 2 and math.isfinite(hist[-1]["train"]["total"])
    assert (tmp_path / "smoke" / "best.pt").exists() and (tmp_path / "smoke" / "config.json").exists()
    m, cfg2, _ = load_checkpoint(tmp_path / "smoke" / "best.pt")
    assert cfg2 == c
