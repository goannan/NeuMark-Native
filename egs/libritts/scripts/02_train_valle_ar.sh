#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

exp_dir="${1:-exp/valle_ar}"
manifest_dir="${2:-data/tokenized}"
sampling_rate="${3:-16000}"

mkdir -p "${exp_dir}"

echo "=========================================================="
echo " VALL-E Stage 1: Auto-Regressive (AR) Model Training"
echo " Time:        $(date)"
echo " Exp Dir:     ${exp_dir}"
echo " Manifests:   ${manifest_dir}"
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
else
    NUM_GPUS=1
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --nproc_per_node="${NUM_GPUS}" bin/trainer.py \
      --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --manifest-dir "${manifest_dir}" --sampling-rate "${sampling_rate}" \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir "${exp_dir}"
else
  python3 bin/trainer.py \
      --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --manifest-dir "${manifest_dir}" --sampling-rate "${sampling_rate}" \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir "${exp_dir}"
fi
