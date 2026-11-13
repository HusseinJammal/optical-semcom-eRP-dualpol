"""IEEE 802.11n LDPC channel coding + Gray 64-QAM, sized to EXACTLY the semantic transceiver's 512 symbols.

512 Gray-64-QAM symbols carry 3072 coded bits. Two 802.11n codewords (1944 + 1296 = 3240 bits) are the closest fit,
so 168 parity bits are punctured (101 / 67, proportional to the codeword lengths, 802.11n's own rate matching):

    mother rate        1/2     2/3     3/4     5/6
    effective rate     0.527   0.703   0.791   0.879
    JPEG budget [B]    202     270     303     337

Receiver: exact log-sum-exp LLRs for Gray 64-QAM with a decision-directed noise estimate per image (punctured bits
enter with LLR 0), then sum-product belief propagation (up to 50 iterations) with OptiCommPy's 802.11n codes.

OptiCommPy's ``encodeLDPC`` permutes H on its first call; the codes are therefore warmed up once, the generator (or
the triangular parity parts) cached, and the encoder primitives called directly afterwards with the same permuted H
used by the decoder.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
from scipy.special import logsumexp

__all__ = ["CODE_LENGTHS", "RATES", "alist_dir", "LDPC80211n", "FrameCodec"]

CODE_LENGTHS = (1944, 1296)
RATES = ("1/2", "2/3", "3/4", "5/6")
BPS = 6                                           # Gray 64-QAM


def alist_dir() -> str:
    """Directory of OptiCommPy's ALIST parity-check matrices ($SEMCOM_ALIST_DIR overrides)."""
    env = os.environ.get("SEMCOM_ALIST_DIR")
    if env:
        return env
    import optic
    d = os.path.join(os.path.dirname(optic.__file__), "comm", "ALIST")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"OptiCommPy ALIST files not found at {d}: install OptiCommPy from a source checkout "
                                "(pip install -e) or set SEMCOM_ALIST_DIR")
    return d


class LDPC80211n:
    """One systematic IEEE 802.11n LDPC code (n in {648, 1296, 1944}, rate '1/2' | '2/3' | '3/4' | '5/6')."""

    def __init__(self, n: int, rate: str, max_iter: int = 50, alg: str = "SPA"):
        from optic.comm.fec import encodeLDPC, encoder, encodeTriang, readAlist
        from optic.utils import parameters
        self._encoder, self._encodeTriang = encoder, encodeTriang
        tag = rate.replace("/", "")
        d = alist_dir()
        H = readAlist(os.path.join(d, f"LDPC_IEEE_802.11nD2_{n}b_R{tag}.txt"))
        m, nn = H.shape
        if nn != n:
            raise ValueError(f"alist codeword length {nn} != {n}")
        self.n, self.k, self.rate = n, n - m, rate
        enc = parameters()
        enc.mode, enc.n, enc.R, enc.H, enc.path = "IEEE_802.11nD2", n, rate, H, d + os.sep
        logging.getLogger("root").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encodeLDPC(np.zeros((self.k, 1), dtype=np.uint8), enc)          # warm-up: caches G or P1/P2, permutes H
        self._G, self._P1, self._P2 = getattr(enc, "G", None), getattr(enc, "P1", None), getattr(enc, "P2", None)
        dec = parameters()
        dec.H, dec.maxIter, dec.alg = enc.H, max_iter, alg
        self._dec = dec

    def encode(self, info: np.ndarray) -> np.ndarray:
        """(k,) bits -> (n,) codeword (systematic: information first, parity last)."""
        b = info.reshape(self.k, 1).astype(np.uint8)
        c = self._encoder(self._G, b, systematic=True) if self._G is not None else self._encodeTriang(b, self._P1,
                                                                                                      self._P2)
        return np.asarray(c).reshape(self.n).astype(np.uint8)

    def decode(self, llr: np.ndarray) -> tuple:
        """(n,) LLRs (positive = bit 0) -> ((k,) information bits, frame error flag)."""
        from optic.comm.fec import decodeLDPC
        out = decodeLDPC(np.asarray(llr, dtype=np.float32).reshape(self.n, 1), self._dec)
        bits, fe = out[0], out[-1]
        return np.asarray(bits)[: self.k, 0].astype(np.uint8), bool(np.asarray(fe).sum() > 0)


class FrameCodec:
    """JPEG payload bytes <-> 512 Gray-64-QAM symbols (two punctured 802.11n codewords)."""

    def __init__(self, rate: str, n_symbols: int = 512, max_iter: int = 50):
        from optic.comm.modulation import demodulateGray, grayMapping
        self.rate, self.n_symbols = rate, n_symbols
        self.codes = [LDPC80211n(n, rate, max_iter) for n in CODE_LENGTHS]
        coded = sum(c.n for c in self.codes)
        budget = n_symbols * BPS
        excess = coded - budget
        if excess < 0:
            raise ValueError(f"{coded} coded bits do not fill {budget}")
        p0 = int(round(excess * self.codes[0].n / coded))
        self.punct = (p0, excess - p0)
        self.k_info = sum(c.k for c in self.codes)
        self.budget_bytes = self.k_info // 8
        self.eff_rate = self.k_info / budget
        const = grayMapping(64, "qam").reshape(-1)
        self.const = (const / np.sqrt(np.mean(np.abs(const) ** 2))).astype(np.complex128)
        self.bitmap = np.asarray(demodulateGray(grayMapping(64, "qam").reshape(-1), 64, "qam")).reshape(64, BPS)
        self._w = 1 << np.arange(BPS - 1, -1, -1)
        label = self.bitmap.dot(self._w)
        self._label_to_idx = np.empty(64, dtype=np.int64)
        self._label_to_idx[label] = np.arange(64)

    def encode(self, payload: bytes) -> np.ndarray:
        """bytes (<= budget_bytes) -> (n_symbols,) unit-mean-power complex symbols."""
        if len(payload) > self.budget_bytes:
            raise ValueError(f"payload {len(payload)} B exceeds the {self.budget_bytes} B budget")
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        info = np.zeros(self.k_info, dtype=np.uint8)
        info[: len(bits)] = bits
        coded, o = [], 0
        for c, p in zip(self.codes, self.punct):
            cw = c.encode(info[o:o + c.k])
            o += c.k
            coded.append(cw[: c.n - p])                           # puncture the last p parity bits
        flat = np.concatenate(coded)
        assert flat.size == self.n_symbols * BPS
        labels = flat.reshape(-1, BPS).dot(self._w)
        return self.const[self._label_to_idx[labels]]

    def llr(self, sym: np.ndarray) -> np.ndarray:
        """Exact Gray-64-QAM LLRs (positive = bit 0) with a per-image decision-directed noise estimate."""
        y = sym.astype(np.complex128)
        y = y / np.sqrt(max(np.mean(np.abs(y) ** 2), 1e-30))
        d2 = np.abs(y[:, None] - self.const[None, :]) ** 2
        sigma2 = max(float(np.mean(d2.min(axis=1))), 1e-6)
        m = -d2 / sigma2
        out = np.empty((y.size, BPS))
        for b in range(BPS):
            zero = self.bitmap[:, b] == 0
            out[:, b] = logsumexp(m[:, zero], axis=1) - logsumexp(m[:, ~zero], axis=1)
        return out.reshape(-1)

    def decode(self, sym: np.ndarray, n_bytes: int) -> tuple:
        """(n_symbols,) received symbols -> (payload bytes, number of codewords in error)."""
        llr = self.llr(sym)
        info, o, errs = [], 0, 0
        for c, p in zip(self.codes, self.punct):
            seg = llr[o:o + c.n - p]
            o += c.n - p
            bits, fe = c.decode(np.concatenate([seg, np.zeros(p)]))
            info.append(bits)
            errs += int(fe)
        bits = np.concatenate(info)[: n_bytes * 8]
        return np.packbits(bits).tobytes(), errs
