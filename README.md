<div align="center">

# NeuMark-Native: In-Model Watermarking for Discrete Latent Acoustic Tokens in Zero-Shot Speech Synthesis

<p align="center">
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat&logo=pytorch"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat"></a>
  <a href="https://huggingface.co/docs/accelerate/index"><img alt="Accelerate" src="https://img.shields.io/badge/HuggingFace-Accelerate-yellow?style=flat"></a>
  <a href="https://goannan.github.io/NeuMark-Native/"><img alt="Demo Page" src="https://img.shields.io/badge/Audio%20Demo-GitHub%20Pages-blue?style=flat&logo=github"></a>
</p>

</div>

**NeuMark-Native** is an end-to-end framework integrating zero-shot neural speech synthesis (**VALL-E**) with in-model **native latent audio watermarking**. 

Unlike conventional post-hoc audio watermarking that superimposes perturbations onto synthesized waveforms, NeuMark-Native embeds covert cryptographic watermark signals directly into discrete latent acoustic tokens during generative synthesis, ensuring inherent provenance and traceability.

- **Interactive Audio Demos**: [https://goannan.github.io/NeuMark-Native/](https://goannan.github.io/NeuMark-Native/)
- **Pretrained Watermark Checkpoint**: The trained native watermark model (`checkpoints/NeuMark-Native.pt`, 80.5 MB) is provided directly in this repository.
- **VALL-E Base Model & Training Pipeline**: Readers can train the base VALL-E model from scratch and generate token manifests using the complete recipe in `egs/libritts/`.

---

## Repository Structure

```text
NeuMark-Native/
├── setup.py                           # Python package setup for valle
├── requirements.txt                   # Dependencies
├── README.md                          # Full guide & documentation
├── .gitignore
├── checkpoints -> egs/libritts/checkpoints # Symlink to model checkpoints
├── valle/                             # Core VALL-E neural engine
│   ├── models/                        # AR & NAR Transformers (valle.py, transformer.py)
│   ├── modules/                       # Custom layer modules & attention
│   ├── data/                          # Dataset collators, samplers, datamodule
│   ├── utils/                         # Checkpointing & tensor utilities
│   └── bin/                           # trainer.py, joint_trainer.py, tokenizer.py, infer.py
├── docs/                              # Demo showcase page (GitHub Pages)
│   ├── index.html
│   └── audio/                         # Demo audio samples (.wav)
└── egs/
    └── libritts/                      # Recipes, training scripts, configs & watermark models
        ├── bin -> ../../valle/bin     # Symlink to valle/bin
        ├── checkpoints/               # Checkpoint directory
        │   └── NeuMark-Native.pt      # Pretrained watermark embedder & detector (80.5 MB)
        ├── shared/                    # parse_options.sh
        ├── STmodels/                  # SpeechTokenizer architecture & discriminators
        ├── models.py                  # Watermark embedder & detector (WMEmbedder, WMDetector)
        ├── configs/                   # Training & ablation configs
        │   ├── config_tts_native.json
        │   ├── config_ablation_real_tokens.json
        │   └── config_ablation_valle_neumark_loss.json
        ├── tts_native_train.py        # Native watermark training (Accelerate DDP)
        ├── tts_native_loss.py         # Multi-scale Mel, VAD margin, adversarial losses
        ├── tts_native_dataset.py      # PyTorch Dataset for native watermark training
        ├── tts_native_attacks.py      # Differentiable distortion attack channels
        ├── test_valle_native_watermark.py # Extraction & robustness evaluation
        ├── generate_valle_native_dataset.py # Tokenized speech pairs generator
        ├── generate_demo_audios.py    # Quick audio synthesis & verification script
        └── scripts/                   # Numbered bash scripts (01 ~ 08)
            ├── 01_prepare_libritts.sh
            ├── 02_train_valle_ar.sh
            ├── 03_train_valle_nar.sh
            ├── 04_train_valle_joint.sh
            ├── 05_prepare_native_tokens.sh
            ├── 06_train_watermark.sh
            ├── 07_infer_zero_shot.sh
            └── 08_evaluate_watermark.sh
```

---

## Installation & Environment Setup

```bash
# 1. System packages & Python environment
sudo apt-get update && sudo apt-get install -y espeak-ng sox libsox-fmt-all git-lfs
conda create -n neumark-native python=3.10 -y
conda activate neumark-native

# 2. Clone repo & install base dependencies
git clone https://github.com/goannan/NeuMark-Native.git
cd NeuMark-Native
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install k2, icefall & NeuMark-Native (valle)
pip install k2 -f https://k2-fsa.github.io/k2/cuda.html || true
git clone https://github.com/k2-fsa/icefall.git ../icefall
export PYTHONPATH=$PWD/../icefall:$PYTHONPATH
pip install -e .

# 4. Download pretrained SpeechTokenizer weights
cd egs/libritts
mkdir -p STmodels/pretrained_model
curl -L -o STmodels/pretrained_model/SpeechTokenizer.pt \
    https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt

# (Optional) Download WavLM for speaker similarity evaluation
mkdir -p models
curl -L -o models/wavlm_large_finetune.pth \
    https://github.com/goannan/NeuMark/releases/download/v1.0/wavlm_large_finetune.pth
```

---

## Quick Start: Watermark Verification (Pretrained Checkpoint)

Full zero-shot speech synthesis from text requires training the VALL-E base model (~1.4 GB, described in the Full Pipeline below). 

To immediately verify NeuMark-Native's core capability without any prior training, the official watermark checkpoint **`checkpoints/NeuMark-Native.pt`** (80.5 MB) is provided directly in the repository. You can verify discrete token watermark embedding, audio synthesis, and bit extraction on a sample audio (or any custom `.wav`) in seconds:

```bash
cd egs/libritts

# Verify discrete acoustic token watermark embedding & bit extraction
python3 generate_demo_audios.py \
    --audio-path ../../docs/audio/libritts_sample_1/01_clean_tts.wav \
    --message 1011001110001101 \
    --output-dir exp/demo_samples

# Or run the zero-shot wrapper script directly
bash scripts/07_infer_zero_shot.sh
```

**Expected verification output:**
```text
======================================================================
 VERIFICATION RESULTS:
   Embedded Watermark: 1011001110001101
   Extracted Bits:     1011001110001101
   Bit Accuracy:       100.00% (16/16 bits match)
   Detection Score:    0.999991
----------------------------------------------------------------------
 Saved Audio Files:
   1. Clean Reconstruction: exp/demo_samples/clean_reconstructed.wav
   2. Watermarked Audio:    exp/demo_samples/watermarked_native.wav
   3. Residual (Diff x 10): exp/demo_samples/watermark_diff_x10.wav
======================================================================
```

---

## Full Pipeline: Train VALL-E Base Model & End-to-End Inference

To synthesize voice-cloned speech from arbitrary text prompts, generate token manifests, and evaluate benchmark robustness, execute the recipe stages in `egs/libritts`:

```bash
cd egs/libritts

# Stage 1: LibriTTS data download, Lhotse manifests, and SpeechTokenizer tokenization
bash scripts/01_prepare_libritts.sh \
    --stage 0 \
    --stop-stage 3 \
    --dataset-parts "--dataset-parts all" \
    --audio-extractor "SpeechTokenizer" \
    --audio-feats-dir "data/tokenized"

# Stage 2: Train base VALL-E models (Stage 1 AR & Stage 2 NAR)
bash scripts/02_train_valle_ar.sh exp/valle_ar data/tokenized 16000
bash scripts/03_train_valle_nar.sh exp/valle_nar exp/valle_ar/best-valid-loss.pt data/tokenized 16000
# (Alternative joint training: bash scripts/04_train_valle_joint.sh exp/valle_joint data/tokenized)

# Stage 3: Extract native tokens from trained VALL-E for watermark training & evaluation
bash scripts/05_prepare_native_tokens.sh \
    data/tokenized/cuts_train.jsonl.gz \
    data/tokenized_valle_native \
    -1

# Stage 4: (Optional) Retrain native watermark model from scratch
# (You can also skip this stage and directly use the provided checkpoints/NeuMark-Native.pt)
bash scripts/06_train_watermark.sh configs/config_tts_native.json

# Stage 5: End-to-end zero-shot TTS synthesis with native watermark embedding
bash scripts/07_infer_zero_shot.sh \
    exp/demo_samples \
    "To be or not to be, that is the question." \
    ../../docs/audio/libritts_sample_1/00_prompt.wav \
    "1011001110001101"

# Stage 6: Benchmark robustness evaluation across distortion channels on native test tokens
bash scripts/08_evaluate_watermark.sh \
    data/tokenized_valle_native/cuts_test_valle_native.jsonl.gz \
    checkpoints/NeuMark-Native.pt \
    exp/eval_results \
    cuda:0
```

---

## Local Demo Showcase Preview

Preview the interactive audio comparison demo locally:
```bash
cd docs
python3 -m http.server 8080
```
Open `http://localhost:8080` in your browser.

---

## Citation & Acknowledgments

This project builds upon:
- [VALL-E](https://github.com/lifeiteng/valle)
- [SpeechTokenizer](https://github.com/ZhangXingjian/SpeechTokenizer)
- [NeuMark](https://github.com/goannan/NeuMark)
- [Lhotse](https://github.com/lhotse-speech/lhotse)
- [icefall & k2](https://github.com/k2-fsa/icefall)
