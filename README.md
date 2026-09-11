# NeuMark-Native: End-to-End VALL-E Generative Speech Synthesis with Native Latent Audio Watermarking

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/docs/accelerate/index"><img alt="Accelerate" src="https://img.shields.io/badge/HuggingFace-Accelerate-yellow?style=flat"></a>
  <a href="https://goannan.github.io/NeuMark-Native/"><img alt="Demo Page" src="https://img.shields.io/badge/Audio%20Demo-GitHub%20Pages-blue?style=flat&logo=github"></a>
</p>

**NeuMark-Native** is an end-to-end framework integrating zero-shot neural speech synthesis (**VALL-E**) with in-model **native latent audio watermarking**. 

Unlike conventional post-hoc audio watermarking algorithms (e.g., AudioSeal, WavMark) that operate on synthesized waveforms and suffer heavy degradation under neural vocoders and codec compression, **NeuMark-Native embeds covert, multi-bit cryptographic watermark signals directly into the discrete latent acoustic tokens and residual quantization streams during generative synthesis**.

This repository provides the complete, self-contained pipeline from data downloading and base VALL-E model training (Auto-Regressive + Non-Auto-Regressive) to native watermark training, zero-shot inference, and robustness benchmarking.

---

## 🎧 Interactive Audio Demos

Listen to side-by-side comparative audio demonstrations across **LibriTTS** and **SeedTTS** benchmarks on our interactive GitHub Pages demo:
👉 **[Online Demo Page (https://goannan.github.io/NeuMark-Native/)](https://goannan.github.io/NeuMark-Native/)**

---

## 📁 Repository Structure

```text
NeuMark-Native/
├── setup.py                           # Python package setup for valle
├── requirements.txt                   # Complete dependencies
├── README.md                          # Full-pipeline tutorial
├── .gitignore
├── shared/                            # CLI option parsing utilities
│   └── parse_options.sh
├── valle/                             # Core VALL-E neural engine
│   ├── models/                        # AR & NAR Transformers (valle.py, transformer.py)
│   ├── modules/                       # Custom layer modules & attention
│   ├── data/                          # Dataset collators, samplers, datamodule
│   ├── utils/                         # Checkpointing & tensor utilities
│   └── bin/                           # trainer.py, joint_trainer.py, tokenizer.py, infer.py
├── bin -> valle/bin                   # Top-level symlink for easy CLI access
├── models.py                          # Watermark embedder & detector (WMEmbedder, WMDetector)
├── STmodels/                          # SpeechTokenizer & GAN discriminators
├── tts_native_train.py                # Standard native watermark training (Accelerate DDP)
├── tts_native_energy_gated_train.py   # Energy-gated watermark training pipeline
├── tts_native_loss.py                 # Multi-scale Mel, VAD margin, cosine, adversarial losses
├── tts_native_dataset.py              # PyTorch Dataset for native watermark training
├── tts_native_attacks.py              # Differentiable acoustic & distortion attack channels
├── test_valle_native_watermark.py     # Watermark extraction & robustness benchmark
├── generate_valle_native_dataset.py   # Tokenized speech pairs generator
├── generate_demo_audios.py            # Quick audio synthesis
├── configs/                           # Training & ablation configurations
│   ├── config_tts_native.json
│   ├── config_tts_native_energy_gated.json
│   ├── config_ablation_real_tokens.json
│   └── config_ablation_valle_neumark_loss.json
├── scripts/                           # Numbered step-by-step bash scripts
│   ├── 01_prepare_libritts.sh         # Download & tokenize LibriTTS
│   ├── 02_train_valle_ar.sh           # Train VALL-E Stage 1 (AR model)
│   ├── 03_train_valle_nar.sh          # Train VALL-E Stage 2 (NAR model)
│   ├── 04_train_valle_joint.sh        # Joint VALL-E AR + NAR training
│   ├── 05_prepare_native_tokens.sh    # Generate paired tokens for watermark training
│   ├── 06_train_watermark.sh          # Train native watermark model
│   ├── 07_train_watermark_energy.sh   # Train energy-gated watermark model
│   ├── 08_infer_zero_shot.sh          # Zero-shot inference with watermarking
│   └── 09_evaluate_watermark.sh       # Benchmark extraction & audio quality
└── docs/                              # Academic demo page for GitHub Pages
    ├── index.html
    └── audio/
```

---

## 🛠️ Step 1: Environment & Installation

### 1.1 System Packages (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y espeak-ng sox libsox-fmt-all git-lfs
```

### 1.2 Python Environment
Python 3.10+ is recommended:
```bash
conda create -n neumark-native python=3.10 -y
conda activate neumark-native
```

### 1.3 Install PyTorch & Dependencies
```bash
# Clone the repository
git clone https://github.com/goannan/NeuMark-Native.git
cd NeuMark-Native

# Install base dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install k2 (match with your CUDA version, e.g. CUDA 11.8/12.1)
# See https://k2-fsa.github.io/k2/installation/from_wheels.html
pip install k2 -f https://k2-fsa.github.io/k2/cuda.html || true

# Install icefall
git clone https://github.com/k2-fsa/icefall.git ../icefall
export PYTHONPATH=$PWD/../icefall:$PYTHONPATH

# Install NeuMark-Native (valle) in editable mode
pip install -e .
```

---

## 📦 Step 2: Pretrained Models & Assets

NeuMark-Native uses **SpeechTokenizer** for discrete acoustic token representations:

```bash
mkdir -p STmodels/pretrained_model

# Download SpeechTokenizer model weights (460MB)
curl -L -o STmodels/pretrained_model/SpeechTokenizer.pt \
    https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt
```

*(Optional)* Download WavLM for speaker similarity loss calculation:
```bash
mkdir -p models
curl -L -o models/wavlm_large_finetune.pth \
    https://github.com/goannan/NeuMark/releases/download/v1.0/wavlm_large_finetune.pth # or official WavLM URL
```

---

## 📊 Step 3: LibriTTS Data Preparation

Prepare the LibriTTS dataset (downloads audio, builds Lhotse manifests, extracts SpeechTokenizer acoustic tokens, and phonemizes transcripts):

```bash
# Run complete data preparation (Stage 0 to Stage 3)
bash scripts/01_prepare_libritts.sh
```

**Custom subsets or directories:**
```bash
bash scripts/01_prepare_libritts.sh \
    --stage 0 \
    --stop-stage 3 \
    --dataset-parts "--dataset-parts all" \
    --audio-extractor "SpeechTokenizer" \
    --audio-feats-dir "data/tokenized"
```
This generates:
- `data/tokenized/cuts_train.jsonl.gz`
- `data/tokenized/cuts_dev.jsonl.gz`
- `data/tokenized/cuts_test.jsonl.gz`

---

## 🎙️ Step 4: Base VALL-E Model Training

VALL-E decomposes zero-shot text-to-speech into two stages:
1. **Auto-Regressive (AR) Model**: Predicts the first-layer acoustic code from text phonemes and acoustic prompt.
2. **Non-Auto-Regressive (NAR) Model**: Iteratively refines the remaining codebook layers (stages 2 to 8).

### 4.1 Train AR Model (Stage 1)
```bash
# Automatically detects all available GPUs:
bash scripts/02_train_valle_ar.sh exp/valle_ar data/tokenized 16000

# Or run with specific GPUs:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/02_train_valle_ar.sh exp/valle_ar data/tokenized 16000
```

### 4.2 Train NAR Model (Stage 2)
```bash
bash scripts/03_train_valle_nar.sh exp/valle_nar exp/valle_ar/best-valid-loss.pt data/tokenized 16000
```

### 4.3 (Alternative) Joint AR + NAR Training
```bash
bash scripts/04_train_valle_joint.sh exp/valle_joint data/tokenized
```

---

## 🧩 Step 5: Preparing Native Token Pairs for Watermark Training

Generate paired token representations produced by the VALL-E model for training the watermark embedder and detector:

```bash
bash scripts/05_prepare_native_tokens.sh \
    data/tokenized/cuts_train.jsonl.gz \
    data/tokenized_valle_native \
    -1
```

---

## 🔐 Step 6: Native Watermark Model Training

Train the **WMEmbedder** and **WMDetector** models to embed multi-bit watermark payloads into the discrete acoustic representations.

### Option A: Standard Native Watermark Training
```bash
bash scripts/06_train_watermark.sh configs/config_tts_native.json
```

### Option B: Energy-Gated Native Watermark Training
Dynamically adjusts watermark injection strength based on speech energy, avoiding audible distortion during silence/unvoiced segments:
```bash
bash scripts/07_train_watermark_energy.sh configs/config_tts_native_energy_gated.json
```

---

## 🔊 Step 7: Zero-Shot Speech Synthesis with Watermarking

Synthesize zero-shot voice-cloned speech from a target text and 3-second prompt audio with an embedded 16-bit cryptographic watermark:

```bash
bash scripts/08_infer_zero_shot.sh \
    exp/demo_samples \
    "To be or not to be, that is the question." \
    docs/audio/libritts_sample_1/00_prompt.wav \
    "1011001110001101"
```

---

## 📈 Step 8: Extraction & Robustness Benchmark Evaluation

Evaluate watermark extraction bit accuracy, ROC-AUC, detection latency, and audio quality (PESQ, STOI, SNR, UTMOS) across diverse acoustic distortion attacks:

```bash
bash scripts/09_evaluate_watermark.sh \
    data/tokenized_valle_native/cuts_test_valle_native.jsonl.gz \
    exp/tts_native_neumark/NeuMark_epoch_010.pt \
    exp/eval_results \
    cuda:0
```

---

## 🌐 Step 9: Running the Demo Page Locally

Preview the academic demo showcase page locally:

```bash
cd docs
python3 -m http.server 8080
```
Open `http://localhost:8080` in your web browser.

---

## 📄 Citation & Acknowledgments

This project integrates and builds upon work from:
- [VALL-E](https://github.com/lifeiteng/valle)
- [SpeechTokenizer](https://github.com/ZhangXingjian/SpeechTokenizer)
- [NeuMark](https://github.com/goannan/NeuMark)
- [Lhotse](https://github.com/lhotse-speech/lhotse)
- [icefall & k2](https://github.com/k2-fsa/icefall)

For research questions or collaborations, please open an issue in this repository.
