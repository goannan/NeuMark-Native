#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${1:-data/cuts_test_valle_native.jsonl.gz}"
CKPT_PATH="${2:-checkpoints/NeuMark_native_latest.pt}"
OUTPUT_DIR="${3:-exp/eval_results}"
DEVICE="${4:-cuda:0}"

echo "=========================================================="
echo " NeuMark-Native: Watermark Benchmark Evaluation"
echo " Time:        $(date)"
echo " Manifest:    ${MANIFEST_PATH}"
echo " Checkpoint:  ${CKPT_PATH}"
echo " Output Dir:  ${OUTPUT_DIR}"
echo " Device:      ${DEVICE}"
echo "=========================================================="

mkdir -p "${OUTPUT_DIR}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 test_valle_native_watermark.py \
    --manifest "${MANIFEST_PATH}" \
    --watermark-checkpoint "${CKPT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples -1 \
    --save-audio-samples 10 \
    --device "${DEVICE}"

echo "=========================================================="
echo " Evaluation Finished. Results saved to ${OUTPUT_DIR}"
echo "=========================================================="
