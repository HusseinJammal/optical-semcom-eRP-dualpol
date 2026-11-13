#!/usr/bin/env bash
# Commands behind every result of the paper. Each step is independent once its inputs exist; on a cluster, submit the
# evaluation lines as separate jobs (and split long links into test-set blocks with --offset/--limit).
#
#   CKPT     trained semantic transceiver (step 1)                    CLF   baseline classifiers (step 3)
#   OUT      where the JSON reports go                               DEV   cuda | cpu
#
# Cost (one NVIDIA A40): training ~16 h; SSFM evaluation ~2.5e-7 h per image-km (10,000 images at 2480 km ~6 h).
set -euo pipefail

CKPT=${CKPT:-runs/paper_v3_s0/best.pt}
CLF=${CLF:-runs/task_clf}
OUT=${OUT:-reports}
DEV=${DEV:-cuda}
STEP=${1:-all}

run() { echo "+ $*"; "$@"; }

# 1) Train the semantic transceiver through the eRP channel (paper: seeds 0, 1, 2; figures use seed 0).
if [[ $STEP == all || $STEP == train ]]; then
  run python scripts/train.py --run-name paper_v3_s0 --seed 0 --device "$DEV"
fi

# 2) eRP validity of the trained model at the corners of its training range (every margin should be >= +9 dB).
if [[ $STEP == all || $STEP == validity ]]; then
  run python scripts/erp_validity_gate.py --ckpt "$CKPT" --reach-km 400 960 1440 2000 2480 --device "$DEV" \
      --out "$OUT/erp_validity_gate.json"
fi

# 3) Baseline classifiers: one CIFAR ResNet-18 per code rate, trained on that rate's error-free JPEG output.
if [[ $STEP == all || $STEP == classifiers ]]; then
  for r in 1/2 2/3 3/4 5/6; do
    run python scripts/train_baseline_classifier.py --kind jpeg --rate "$r" --device "$DEV" --out "$CLF"
  done
fi

# 4) Fig. 2(a): near-optimum launch power vs link length (the three reference links are points of this sweep).
FIG_A="400:0 560:0 720:0 880:0 1040:0 1440:0 1680:-0.2 2000:-0.5 2480:-1 2800:-1 3120:-1"
if [[ $STEP == all || $STEP == fig2a ]]; then
  for c in $FIG_A; do
    run python scripts/evaluate.py --ckpt "$CKPT" --cond "$c" --link-snr-images 64 --device "$DEV" \
        --out "$OUT/fig2a" --tag "semantic_L${c%%:*}"
    run python scripts/run_baseline.py --rates 1/2 2/3 3/4 5/6 --cond "$c" --classifier-dir "$CLF" --no-ceiling \
        --device "$DEV" --out "$OUT/fig2a/jpeg_ldpc_L${c%%:*}.json"
  done
fi

# 5) Fig. 2(b): -6 dBm launch power vs link length.
FIG_B="400:-6 720:-6 1040:-6 1440:-6 2000:-6 2160:-6 2320:-6 2480:-6 2800:-6 3120:-6"
if [[ $STEP == all || $STEP == fig2b ]]; then
  for c in $FIG_B; do
    run python scripts/evaluate.py --ckpt "$CKPT" --cond "$c" --link-snr-images 64 --device "$DEV" \
        --out "$OUT/fig2b" --tag "semantic_L${c%%:*}"
    run python scripts/run_baseline.py --rates 1/2 2/3 3/4 5/6 --cond "$c" --classifier-dir "$CLF" --no-ceiling \
        --device "$DEV" --out "$OUT/fig2b/jpeg_ldpc_L${c%%:*}.json"
  done
fi

# 6) Fig. 2(c): transceiver SNR lowered at 400 km / 0 dBm (x-axis: the measured link SNR, LS estimate).
if [[ $STEP == all || $STEP == fig2c ]]; then
  for t in 20 17 14 11 8 5 2 -1; do
    run python scripts/evaluate.py --ckpt "$CKPT" --cond 400:0 --trx-snr-db "$t" --link-snr-images 64 \
        --device "$DEV" --out "$OUT/fig2c" --tag "semantic_trx${t}"
    run python scripts/run_baseline.py --rates 1/2 2/3 3/4 5/6 --cond 400:0 --trx-snr-db "$t" --classifier-dir "$CLF" \
        --no-ceiling --device "$DEV" --out "$OUT/fig2c/jpeg_ldpc_trx${t}.json"
  done
fi

# 7) Table 1: accuracy at the reference links (from step 4) and CPU processing time per image.
if [[ $STEP == all || $STEP == table1 ]]; then
  run python scripts/benchmark_latency.py --ckpt "$CKPT" --classifier-dir "$CLF" --threads 1 \
      --out "$OUT/table1_latency_cpu.json"
fi
