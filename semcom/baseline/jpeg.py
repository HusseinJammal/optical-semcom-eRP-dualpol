"""JPEG source coding for 32 x 32 RGB images (PIL / libjpeg, 4:4:4 chroma).

Only the entropy-coded PAYLOAD is transmitted. The header (quantization and Huffman tables, ~620 bytes, identical for
every image at a given quality) is assumed known to the receiver together with the quality factor and payload
length -- the most favourable accounting for the baseline. Per image, the highest quality whose payload fits the
byte budget of the code rate is used.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

__all__ = ["jpeg_encode_rgb", "jpeg_decode_rgb", "split_jpeg", "choose_jpeg_quality"]


def jpeg_encode_rgb(img: np.ndarray, quality: int) -> bytes:
    """(3, 32, 32) float in [0, 1] (or HWC uint8) -> JPEG bytes, no chroma subsampling."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def split_jpeg(stream: bytes) -> tuple:
    """(header, entropy-coded payload), split after the SOS segment."""
    i = stream.find(b"\xff\xda")
    if i < 0:
        return b"", stream
    ln = int.from_bytes(stream[i + 2:i + 4], "big")
    cut = i + 2 + ln
    return stream[:cut], stream[cut:]


def jpeg_decode_rgb(data: bytes):
    """JPEG bytes -> (3, 32, 32) float in [0, 1], or None when the stream does not decode (scored as a black image)."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        a = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        return np.transpose(a, (2, 0, 1))
    except Exception:
        return None


def choose_jpeg_quality(img: np.ndarray, budget_bytes: int, q_hi: int = 95) -> tuple:
    """Highest quality in [1, q_hi] whose payload fits ``budget_bytes``: (quality, header, payload)."""
    best = None
    for q in range(q_hi, 0, -1):
        header, payload = split_jpeg(jpeg_encode_rgb(img, q))
        if len(payload) <= budget_bytes:
            return q, header, payload
        best = (q, header, payload)
    return best
