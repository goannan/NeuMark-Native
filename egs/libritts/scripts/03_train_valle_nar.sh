#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

exp_dir="${1:-exp/valle_nar}"
ar_ckpt="${2:-exp/valle_ar/best-valid-loss.pt}"
manifest_dir="${3:-data/tokenized}"
sampling_rate="${4:-16000}"

mkdir -p "${exp_dir}"

echo "=========================================================="
echo " VALL-E Stage 2: Non-Auto-Regressive (NAR) Model Training"
echo " Time:        $(date)"
echo " Exp Dir:     ${exp_dir}"
echo " AR Checkpoint: ${ar_ckpt}"
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

if [ -f "${ar_ckpt}" ]; then
  cp "${ar_ckpt}" "${exp_dir}/epoch-2.pt"
fi

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
else
    NUM_GPUS=1
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --nproc_per_node="${NUM_GPUS}" bin/trainer.py \
      --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --manifest-dir "${manifest_dir}" --sampling-rate "${sampling_rate}" \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir "${exp_dir}"
else
  python3 bin/trainer.py \
      --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --manifest-dir "${manifest_dir}" --sampling-rate "${sampling_rate}" \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir "${exp_dir}"
fi
