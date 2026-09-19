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

VALLE_CKPT=""
if [ -f "exp/valle/best-valid-loss.pt" ]; then
    VALLE_CKPT="exp/valle/best-valid-loss.pt"
elif [ -f "exp/valle_ar/best-valid-loss.pt" ]; then
    VALLE_CKPT="exp/valle_ar/best-valid-loss.pt"
fi

if [ -n "${VALLE_CKPT}" ]; then
    echo " Found trained VALL-E checkpoint at ${VALLE_CKPT}."
    echo " Running full zero-shot TTS synthesis..."
    python3 bin/infer.py \
        --model-name valle \
        --text-prompts "${TEXT}" \
        --audio-prompts "${PROMPT_AUDIO}" \
        --output-dir "${OUTPUT_DIR}" \
        --checkpoint "${VALLE_CKPT}" \
        --watermark-backend neumark \
        --voicemark-checkpoint checkpoints/NeuMark-Native.pt
else
    echo " [Notice] VALL-E base model not found in exp/valle/."
    echo " (To perform arbitrary text-to-speech, train VALL-E first via scripts/02 & 03)"
    echo " Running discrete token watermark embedding & extraction verification..."
    python3 generate_demo_audios.py
fi

echo " Synthesis complete! Check ${OUTPUT_DIR}"
