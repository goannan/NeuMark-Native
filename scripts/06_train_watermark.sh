#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="${1:-configs/config_tts_native.json}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "=========================================================="
echo " NeuMark-Native: TTS-Native Watermark Training"
echo " Time:        $(date)"
echo " Config:      ${CONFIG_PATH}"
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
else
    NUM_GPUS=1
fi

if [ -n "$CUDA_DEVICES" ]; then
    IFS=',' read -ra ADDR <<< "$CUDA_DEVICES"
    NUM_GPUS=${#ADDR[@]}
fi

echo " Using ${NUM_GPUS} GPU(s)..."

if [ "${NUM_GPUS}" -gt 1 ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision bf16 \
        --dynamo_backend no \
        tts_native_train.py --config "${CONFIG_PATH}"
else
    accelerate launch \
        --num_processes 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        tts_native_train.py --config "${CONFIG_PATH}"
fi
