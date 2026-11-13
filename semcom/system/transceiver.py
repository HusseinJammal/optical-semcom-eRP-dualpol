"""The semantic transceiver: encoder f_theta -> link h_{c,xi} -> decoder g_phi, as one module.

    (x_hat, logits) = g_phi( h_{c,xi}( f_theta(x; c); n ); c ),      c = [P, L]

``forward`` takes the training channel as an argument (eRP during training and for diagnostics). ``forward_ssfm`` has
NO backend argument: it is the path every reported number goes through, and it can only use the split-step channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..conditioning.film import link_condition
from ..config import Config
from ..decoder.decoder import SemanticDecoder
from ..encoder.encoder import SemanticEncoder
from ..impairments.params import nominal_trx
from .link import transmit

__all__ = ["SemanticTransceiver"]


class SemanticTransceiver(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.encoder = SemanticEncoder(self.cfg.model, n_pol=self.cfg.channel.n_pol)
        self.decoder = SemanticDecoder(self.cfg.model)

    def conditioning(self, p_tx_dbm: float | None = None, reach_km: float | None = None, device=None) -> torch.Tensor:
        """c = [P, L] for this call (None = the nominal launch power / reach of the config)."""
        c = self.cfg.channel
        p = c.P_tx_dBm if p_tx_dbm is None else p_tx_dbm
        L = c.Ltotal_km if reach_km is None else reach_km
        return link_condition(self.cfg.model, p, L, device=device)

    def _trx(self, trx, batch: int):
        """'nominal' -> the configured transceiver; a TrxParams or None is used as given."""
        if isinstance(trx, str):
            if trx != "nominal":
                raise ValueError(f"trx must be 'nominal', None or a TrxParams; got {trx!r}")
            return nominal_trx(self.cfg, batch)
        return trx

    def forward(self, x: torch.Tensor, backend: str | None = None, generator: torch.Generator | None = None,
                p_tx_dbm: float | None = None, return_symbols: bool = False, reach_km: float | None = None,
                span_km: float | None = None, nf_db: float | None = None, trx="nominal",
                trx_seed: int | None = None, cpr: str | None = None, told_p_tx_dbm: float | None = None,
                told_reach_km: float | None = None):
        """Images ``(B, 3, 32, 32)`` -> (reconstruction, logits) [, tx symbols, rx symbols].

        ``told_*`` change only what the networks are TOLD (wrong-conditioning tests); the channel uses the true values.
        """
        cond = self.conditioning(p_tx_dbm if told_p_tx_dbm is None else told_p_tx_dbm,
                                 reach_km if told_reach_km is None else told_reach_km, device=x.device)
        tx = self.encoder(x, cond)
        rx = transmit(tx, self.cfg, backend=backend, generator=generator, p_tx_dbm=p_tx_dbm, reach_km=reach_km,
                      span_km=span_km, nf_db=nf_db, trx=self._trx(trx, x.shape[0]), trx_seed=trx_seed, cpr=cpr)
        x_hat, logits = self.decoder(rx, cond)
        if return_symbols:
            return x_hat, logits, tx, rx
        return x_hat, logits

    def forward_ssfm(self, x: torch.Tensor, ssfm_seed: int | None = None, ssfm_index0: int = 0,
                     p_tx_dbm: float | None = None, return_symbols: bool = False, reach_km: float | None = None,
                     span_km: float | None = None, nf_db: float | None = None, trx="nominal",
                     cpr: str | None = None, told_p_tx_dbm: float | None = None, told_reach_km: float | None = None):
        """The evaluation forward pass: the channel is the Manakov split-step, seeded per image.

        Image ``b`` of the batch is image ``ssfm_index0 + b`` of the evaluation set; its amplifier and transceiver noise
        are derived from (``ssfm_seed``, that index), so results do not depend on the batch size.
        """
        cond = self.conditioning(p_tx_dbm if told_p_tx_dbm is None else told_p_tx_dbm,
                                 reach_km if told_reach_km is None else told_reach_km, device=x.device)
        tx = self.encoder(x, cond)
        rx = transmit(tx, self.cfg, backend="ssfm", p_tx_dbm=p_tx_dbm, ssfm_seed=ssfm_seed, ssfm_index0=ssfm_index0,
                      reach_km=reach_km, span_km=span_km, nf_db=nf_db, trx=self._trx(trx, x.shape[0]), cpr=cpr)
        x_hat, logits = self.decoder(rx, cond)
        if return_symbols:
            return x_hat, logits, tx, rx
        return x_hat, logits
