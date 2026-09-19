#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_PATH="${1:-data/tokenized_valle_native/cuts_test_valle_native.jsonl.gz}"
CKPT_PATH="${2:-checkpoints/NeuMark-Native.pt}"
OUTPUT_DIR="${3:-exp/eval_results}"
DEVICE="${4:-cuda:0}"

echo "=========================================================="
echo " NeuMark-Native: Benchmark Robustness & Audio Evaluation"
echo " Manifest:    ${MANIFEST_PATH}"
echo " Checkpoint:  ${CKPT_PATH}"
echo " Output Dir:  ${OUTPUT_DIR}"
echo " Device:      ${DEVICE}"
echo "=========================================================="

if [ ! -e "${MANIFEST_PATH}" ]; then
    echo " [Error] Test manifest not found at: ${MANIFEST_PATH}"
    echo " Token manifests are generated after VALL-E training."
    echo " Please complete Stages 1-3 in README.md (run scripts/05_prepare_native_tokens.sh)."
    exit 1
fi

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

echo " Benchmark evaluation completed! Results in ${OUTPUT_DIR}"
