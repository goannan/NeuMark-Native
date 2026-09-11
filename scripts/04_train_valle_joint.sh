#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

exp_dir="${1:-exp/valle_joint}"
manifest_dir="${2:-data/tokenized}"

mkdir -p "${exp_dir}/log"

echo "=========================================================="
echo " VALL-E Joint AR + NAR Training Pipeline"
echo " Time:        $(date)"
echo " Exp Dir:     ${exp_dir}"
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
else
    NUM_GPUS=1
fi

echo "[1/2] Training Stage 1: AR Model..."
python3 bin/joint_trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --world-size "${NUM_GPUS}" \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir "${exp_dir}"

echo "[2/2] Training Stage 2: NAR Model..."
if [ -f "${exp_dir}/best-valid-loss.pt" ]; then
    cp "${exp_dir}/best-valid-loss.pt" "${exp_dir}/epoch-2.pt"
    python3 bin/joint_trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
          --world-size "${NUM_GPUS}" \
          --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
          --model-name valle --share-embedding true --norm-first true --add-prenet false \
          --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
          --base-lr 0.05 --warmup-steps 200 --average-period 0 \
          --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
          --exp-dir "${exp_dir}"
else
    echo "Error: Stage 1 training failed to produce best-valid-loss.pt in ${exp_dir}"
    exit 1
fi
