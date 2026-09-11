#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${1:-exp/demo_samples}"
TEXT="${2:-To be or not to be, that is the question.}"
PROMPT_AUDIO="${3:-../../docs/audio/libritts_sample_1/00_prompt.wav}"
WATERMARK_BITS="${4:-1011001110001101}"

echo "=========================================================="
echo " NeuMark-Native: Zero-Shot TTS with Watermark Embedding"
echo " Text:           ${TEXT}"
echo " Prompt Audio:   ${PROMPT_AUDIO}"
echo " Watermark Bits: ${WATERMARK_BITS}"
echo " Output Dir:     ${OUTPUT_DIR}"
echo "=========================================================="

mkdir -p "${OUTPUT_DIR}"

python3 generate_demo_audios.py

echo " Synthesis complete! Check ${OUTPUT_DIR}"
