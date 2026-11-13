"""Per-image processing time on the CPU (batch 1, fp32): semantic transceiver vs JPEG + LDPC.

    python scripts/benchmark_latency.py --ckpt runs/paper_v3_s0/best.pt --classifier-dir runs/task_clf --threads 1

Semantic transceiver: encoder (TX) + decoder with its classification head (RX); independent of the link.

Conventional chain, per image:
    TX  JPEG encoding at the image's quality + LDPC encoding of both codewords + puncturing + Gray 64-QAM mapping
    RX  exact LLRs + LDPC decoding (sum-product, up to 50 iterations, early stop) + JPEG decoding + the rate's ResNet-18
The JPEG quality search (rate matching) is reported separately. LDPC decoding time depends on how many iterations are
needed, hence on the SNR: the received symbols are AWGN at the link SNRs of the reference links (default 19.9 / 16.8 /
14.8 dB at 400 / 1440 / 2480 km; measure them with scripts/evaluate.py --link-snr-images).

OptiCommPy's ``decodeLDPC`` wrapper rebuilds the Tanner graph on every call; a real decoder builds it once. The graph is
therefore built once here and only the compiled decoding core is timed (the decoded bits are checked against the
library path). This is a compiled CPU software reference, not an optimized decoder; say so wherever it is quoted.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semcom.baseline import (FrameCodec, choose_jpeg_quality, jpeg_decode_rgb, jpeg_encode_rgb,  # noqa: E402
                             load_task_classifier, split_jpeg)
from semcom.system.checkpoint import load_checkpoint  # noqa: E402
from semcom.training.data import build_datasets  # noqa: E402


class CoreDecoder:
    """OptiCommPy's compiled sum-product core with the Tanner graph built once (what decodeLDPC rebuilds per call)."""

    def __init__(self, code, max_iter: int = 50):
        from numba.typed import List
        from optic.comm.fec import sumProductAlgorithm
        H = code._dec.H
        Hcsc = H.tocsc()
        self.cn = List([H[i].indices.astype(np.uint32) for i in range(H.shape[0])])
        self.vn = List([Hcsc[:, j].indices.astype(np.uint32) for j in range(code.n)])
        self.fn, self.max_iter, self.n, self.k = sumProductAlgorithm, max_iter, code.n, code.k

    def __call__(self, llr):
        llrs = np.clip(np.asarray(llr, dtype=np.float32).reshape(self.n, 1), -200, 200)
        out, last_iter, fe = self.fn(llrs, self.cn, self.vn, self.max_iter, np.float32)
        return (out[: self.k, 0] < 0).astype(np.uint8), int(np.asarray(last_iter).ravel()[0])


def timed_us(fn, *args):
    t0 = time.perf_counter_ns()
    out = fn(*args)
    return out, (time.perf_counter_ns() - t0) / 1e3


def med_ms(us) -> float:
    return float(np.median(us)) / 1e3


def main():
    ap = argparse.ArgumentParser(description="CPU processing time per image.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classifier-dir", required=True)
    ap.add_argument("--rates", nargs="+", default=["1/2", "2/3", "3/4", "5/6"])
    ap.add_argument("--snr-db", nargs="+", default=["400:19.9", "1440:16.8", "2480:14.8"], help="reach_km:link_SNR_dB")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import numba
    torch.set_num_threads(a.threads)
    numba.set_num_threads(min(a.threads, numba.config.NUMBA_NUM_THREADS))
    model, cfg, _ = load_checkpoint(a.ckpt, "cpu")
    test = build_datasets(cfg, a.data_root).test
    x = torch.stack([test[i][0] for i in range(a.images)])
    imgs = [x[i].numpy() for i in range(a.images)]
    rng = np.random.default_rng(0)
    report = {"host": platform.node(), "processor": platform.processor(), "threads": a.threads, "images": a.images,
              "semantic": {}, "jpeg_ldpc": {}}

    # ---- semantic transceiver ---------------------------------------------------------------------------------- #
    cond = model.conditioning(0.0, 1440.0)
    with torch.inference_mode():
        for i in range(5):
            model.decoder(model.encoder(x[i:i + 1], cond), cond)
        t_enc, t_dec = [], []
        for i in range(a.images):
            s, t = timed_us(model.encoder, x[i:i + 1], cond)
            t_enc.append(t)
            _, t = timed_us(model.decoder, s, cond)
            t_dec.append(t)
    report["semantic"] = {"tx_ms": med_ms(t_enc), "rx_ms": med_ms(t_dec), "total_ms": med_ms(t_enc) + med_ms(t_dec)}
    print(f"semantic transceiver: TX {med_ms(t_enc):.2f} + RX {med_ms(t_dec):.2f} = "
          f"{med_ms(t_enc) + med_ms(t_dec):.2f} ms per image (independent of the link)", flush=True)

    # ---- conventional chain ------------------------------------------------------------------------------------- #
    for rate in a.rates:
        codec = FrameCodec(rate)
        cores = [CoreDecoder(c) for c in codec.codes]
        clf = load_task_classifier(Path(a.classifier_dir) / f"jpeg_r{rate.replace('/', '')}" / "best.pt", "cpu")
        frames, t_search, t_jenc, t_fenc = [], [], [], []
        for im in imgs:
            (q, _, _), t = timed_us(choose_jpeg_quality, im, codec.budget_bytes)
            t_search.append(t)
            stream, t = timed_us(jpeg_encode_rgb, im, q)
            t_jenc.append(t)
            header, payload = split_jpeg(stream)
            payload = payload[: codec.budget_bytes]
            sym, t = timed_us(codec.encode, payload)
            t_fenc.append(t)
            frames.append((header, payload, sym))
        tx_ms = med_ms(t_jenc) + med_ms(t_fenc)
        res = {"tx_ms": tx_ms, "rate_matching_search_ms": med_ms(t_search), "by_link": {}}
        for item in a.snr_db:
            reach, snr = (float(v) for v in item.split(":"))
            t_llr, t_dec, t_jdec, t_clf, iters = [], [], [], [], []
            for n_img, (header, payload, sym) in enumerate(frames):
                sd = np.sqrt(np.mean(np.abs(sym) ** 2) / 10 ** (snr / 10) / 2)
                y = sym + sd * (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
                llr, t = timed_us(codec.llr, y)
                t_llr.append(t)
                t0, o, bits = time.perf_counter_ns(), 0, []
                for d, p in zip(cores, codec.punct):
                    b, it = d(np.concatenate([llr[o:o + d.n - p], np.zeros(p)]))
                    o += d.n - p
                    bits.append(b)
                    iters.append(it)
                t_dec.append((time.perf_counter_ns() - t0) / 1e3)
                data = np.packbits(np.concatenate(bits)[: len(payload) * 8]).tobytes()
                if n_img < 3:                                              # same bits as the library decoding path
                    assert data == codec.decode(y, len(payload))[0], "core decoder disagrees with decodeLDPC"
                img, t = timed_us(jpeg_decode_rgb, header + data)
                t_jdec.append(t)
                img = np.zeros((3, 32, 32), dtype=np.float32) if img is None else img
                with torch.inference_mode():
                    _, t = timed_us(clf, torch.from_numpy(img).unsqueeze(0))
                t_clf.append(t)
            rx_ms = med_ms(t_llr) + med_ms(t_dec) + med_ms(t_jdec) + med_ms(t_clf)
            res["by_link"][f"{reach:g}"] = {"snr_db": snr, "llr_ms": med_ms(t_llr), "ldpc_decode_ms": med_ms(t_dec),
                                            "jpeg_decode_ms": med_ms(t_jdec), "classifier_ms": med_ms(t_clf),
                                            "rx_ms": rx_ms, "total_ms": tx_ms + rx_ms,
                                            "ldpc_iterations_median": float(np.median(iters))}
            print(f"JPEG+LDPC {rate} at {reach:g} km ({snr:g} dB): TX {tx_ms:.2f} + RX {rx_ms:.2f} "
                  f"(LDPC {med_ms(t_dec):.2f}, {np.median(iters):.0f} it; classifier {med_ms(t_clf):.2f}) = "
                  f"{tx_ms + rx_ms:.2f} ms per image", flush=True)
        report["jpeg_ldpc"][rate] = res
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
