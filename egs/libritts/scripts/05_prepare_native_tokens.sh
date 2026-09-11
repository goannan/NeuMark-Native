#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MANIFEST_IN="${1:-data/tokenized/cuts_train.jsonl.gz}"
OUTPUT_DIR="${2:-data/tokenized_valle_native}"
NUM_SAMPLES="${3:--1}"

echo "=========================================================="
echo " NeuMark-Native: VALL-E Native Token Extraction"
echo " Input Manifest:  ${MANIFEST_IN}"
echo " Output Dir:      ${OUTPUT_DIR}"
echo " Num Samples:     ${NUM_SAMPLES}"
echo "=========================================================="

mkdir -p "${OUTPUT_DIR}"

python3 generate_valle_native_dataset.py \
    --input-manifest "${MANIFEST_IN}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples "${NUM_SAMPLES}"

echo " Native token extraction finished! Saved to ${OUTPUT_DIR}"
