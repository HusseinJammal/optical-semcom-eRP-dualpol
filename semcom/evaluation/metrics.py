"""Evaluation metrics. Every metric is a function of the images only, never of how they were batched.

    task             accuracy (integer count), macro-F1, plain cross-entropy
    reconstruction   SSIM (scikit-image, data range 1, per image then averaged), PSNR per image (data range 1) and
                     PSNR of the pooled MSE, 5th/10th/50th percentiles, fraction of images with SSIM < 0.5 / 0.8
    symbols          EVM and decoder-side effective SNR (both sides normalised to unit power, rotation NOT removed),
                     LS effective SNR (best complex gain removed: the conventional coherent "effective SNR"), PAPR
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.metrics import confusion_matrix, f1_score

__all__ = ["accuracy", "macro_f1", "mse_per_image", "psnr_from_mse", "psnr_per_image", "ssim_per_image",
           "ssim_skimage", "evm_percent", "evm_energies", "ls_snr_energies", "papr_db", "papr_db_per_image",
           "MetricAccumulator", "PSNR_MSE_FLOOR"]

PSNR_MSE_FLOOR = 1e-12       # caps PSNR at 120 dB so one bit-exact image cannot make a mean infinite


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Percent correct from an integer count."""
    return 100.0 * int((logits.argmax(1) == labels).sum()) / labels.numel()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mse_per_image(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    """MSE of each image, float64 ``(B,)`` (``x_hat`` not clipped)."""
    d = (x_hat.detach().double() - x.detach().double()) ** 2
    return d.reshape(d.shape[0], -1).mean(dim=1).cpu().numpy()


def psnr_from_mse(mse):
    """10 log10(1 / MSE) for data range 1, capped at 120 dB."""
    return 10.0 * np.log10(1.0 / np.maximum(mse, PSNR_MSE_FLOOR))


def psnr_per_image(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    return psnr_from_mse(mse_per_image(x, x_hat))


def ssim_per_image(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    """SSIM of each CHW image (scikit-image, float64, reconstruction clipped to [0, 1])."""
    a = x.detach().double().cpu().numpy()
    b = np.clip(x_hat.detach().double().cpu().numpy(), 0.0, 1.0)
    return np.array([ssim_fn(a[i], b[i], data_range=1.0, channel_axis=0) for i in range(a.shape[0])], dtype=np.float64)


def ssim_skimage(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return float(ssim_per_image(x, x_hat).mean())


def evm_energies(tx: torch.Tensor, rx: torch.Tensor):
    """Per-image error and reference energy after unit-power normalisation of both sides, float64 ``(B,)`` each.

    The complex gain is NOT removed: a residual rotation is a distortion the decoder has to handle.
    """
    def _n(s):
        s = s.reshape(s.shape[0], -1).to(torch.complex128)
        p = torch.mean(s.real**2 + s.imag**2, dim=-1, keepdim=True).clamp(min=1e-30)
        return s / torch.sqrt(p)

    with torch.no_grad():
        t, r = _n(tx), _n(rx)
        return torch.mean(torch.abs(r - t) ** 2, dim=-1), torch.mean(torch.abs(t) ** 2, dim=-1)


def ls_snr_energies(tx: torch.Tensor, rx: torch.Tensor):
    """Per-image signal and error energy after removing the least-squares complex gain g = <tx, rx> / <tx, tx>."""
    with torch.no_grad():
        t = tx.reshape(tx.shape[0], -1).to(torch.complex128)
        r = rx.reshape(rx.shape[0], -1).to(torch.complex128)
        tt = torch.sum(torch.abs(t) ** 2, dim=-1).clamp(min=1e-30)
        g = torch.sum(t.conj() * r, dim=-1) / tt
        err = torch.sum(torch.abs(r - g.unsqueeze(-1) * t) ** 2, dim=-1)
        return torch.abs(g) ** 2 * tt, err


def evm_percent(tx: torch.Tensor, rx: torch.Tensor) -> float:
    """sqrt(sum |rx - tx|^2 / sum |tx|^2) * 100, pooled over the images given (each normalised to unit power)."""
    num, den = evm_energies(tx, rx)
    return 100.0 * math.sqrt(float(num.sum() / den.sum()))


def papr_db_per_image(x: torch.Tensor) -> torch.Tensor:
    """Peak-to-average power ratio of each image's symbols [dB]."""
    with torch.no_grad():
        x = x.to(torch.complex128)
        p = (x.real**2 + x.imag**2).reshape(x.shape[0], -1)
        ratio = p.max(dim=-1).values / p.mean(dim=-1).clamp(min=1e-30)
        return 10.0 * torch.log10(ratio)


def papr_db(x: torch.Tensor) -> float:
    return float(papr_db_per_image(x).mean())


class MetricAccumulator:
    """Streams batches and returns one dict; exact per-image sums, so the result does not depend on the batching."""

    def __init__(self):
        self.n = 0
        self.correct = 0
        self._sums = {"psnr_img": 0.0, "mse": 0.0, "ssim": 0.0, "ce": 0.0}
        self._sym = None
        self._ls = None
        self._y_true, self._y_pred = [], []
        self._ssim_img, self._psnr_img = [], []

    def add(self, x, x_hat, labels, logits, tx_sym=None, rx_sym=None):
        pred = logits.argmax(1)
        self.n += x.shape[0]
        self.correct += int((pred == labels).sum())
        self._sums["ce"] += float(F.cross_entropy(logits.detach().double(), labels, reduction="sum"))
        mse = mse_per_image(x, x_hat)
        self._sums["mse"] += float(mse.sum())
        psnr_i = psnr_from_mse(mse)
        ssim_i = ssim_per_image(x, x_hat)
        self._sums["psnr_img"] += float(psnr_i.sum())
        self._sums["ssim"] += float(ssim_i.sum())
        self._psnr_img.append(psnr_i)
        self._ssim_img.append(ssim_i)
        if tx_sym is not None and rx_sym is not None:
            num, den = evm_energies(tx_sym, rx_sym)
            sym = self._sym or [0.0, 0.0, 0.0]
            sym[0] += float(num.sum())
            sym[1] += float(den.sum())
            sym[2] += float(papr_db_per_image(tx_sym).sum())
            self._sym = sym
            sig, err = ls_snr_energies(tx_sym, rx_sym)
            ls = self._ls or [0.0, 0.0]
            ls[0] += float(sig.sum())
            ls[1] += float(err.sum())
            self._ls = ls
        self._y_true.append(labels.detach().cpu().numpy())
        self._y_pred.append(pred.detach().cpu().numpy())

    def per_image(self) -> dict:
        """Per-image arrays in insertion order: ssim, psnr, label, pred."""
        def cat(v, dt):
            return np.concatenate(v) if v else np.zeros(0, dtype=dt)
        return {"ssim": cat(self._ssim_img, np.float64), "psnr": cat(self._psnr_img, np.float64),
                "label": cat(self._y_true, np.int64), "pred": cat(self._y_pred, np.int64)}

    def result(self) -> dict:
        n = max(self.n, 1)
        out = {
            "accuracy": 100.0 * self.correct / n,
            "ce": self._sums["ce"] / n,
            "ssim": self._sums["ssim"] / n,
            "psnr_per_image": self._sums["psnr_img"] / n,
            "psnr_pooled": float(psnr_from_mse(self._sums["mse"] / n)),
        }
        if self._sym is not None:
            out["evm_pct"] = 100.0 * math.sqrt(self._sym[0] / self._sym[1])
            out["papr_db"] = self._sym[2] / n
            out["snr_eff_db"] = 10.0 * math.log10(self._sym[1] / max(self._sym[0], 1e-300))
        if self._ls is not None:
            out["snr_eff_ls_db"] = 10.0 * math.log10(self._ls[0] / max(self._ls[1], 1e-300))
        if self._ssim_img:
            si = np.concatenate(self._ssim_img)
            pi = np.concatenate(self._psnr_img)
            for q in (5, 10, 50):
                out[f"ssim_p{q:02d}"] = float(np.percentile(si, q))
                out[f"psnr_p{q:02d}"] = float(np.percentile(pi, q))
            out["frac_ssim_below_0p5"] = float(np.mean(si < 0.5))
            out["frac_ssim_below_0p8"] = float(np.mean(si < 0.8))
        yt = np.concatenate(self._y_true)
        yp = np.concatenate(self._y_pred)
        out["macro_f1"] = macro_f1(yt, yp)
        out["n"] = self.n
        out["confusion"] = confusion_matrix(yt, yp, labels=list(range(10))).tolist()
        return out
