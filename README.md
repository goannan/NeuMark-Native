# NeuMark-Native: End-to-End VALL-E Generative Speech Synthesis with Native Latent Audio Watermarking

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/docs/accelerate/index"><img alt="Accelerate" src="https://img.shields.io/badge/HuggingFace-Accelerate-yellow?style=flat"></a>
  <a href="https://goannan.github.io/NeuMark-Native/"><img alt="Demo Page" src="https://img.shields.io/badge/Audio%20Demo-GitHub%20Pages-blue?style=flat&logo=github"></a>
</p>

**NeuMark-Native** is an end-to-end framework integrating zero-shot neural speech synthesis (**VALL-E**) with in-model **native latent audio watermarking**. 

Unlike conventional post-hoc audio watermarking algorithms (e.g., AudioSeal, WavMark) that operate on synthesized waveforms and suffer heavy degradation under neural vocoders and codec compression, **NeuMark-Native embeds covert, multi-bit cryptographic watermark signals directly into the discrete latent acoustic tokens and residual quantization streams during generative synthesis**.

This repository preserves the standard **VALL-E / Kaldi recipe architecture** (`valle/` package at root, all experiment recipes, training code, and execution scripts inside `egs/libritts/`) and provides the complete, self-contained pipeline from base VALL-E training (Auto-Regressive + Non-Auto-Regressive) to native watermark training, zero-shot inference, and robustness benchmarking.

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
├── checkpoints -> egs/libritts/checkpoints # Symlink to model checkpoints
├── valle/                             # Core VALL-E neural engine
│   ├── models/                        # AR & NAR Transformers (valle.py, transformer.py)
│   ├── modules/                       # Custom layer modules & attention
│   ├── data/                          # Dataset collators, samplers, datamodule
│   ├── utils/                         # Checkpointing & tensor utilities
│   └── bin/                           # trainer.py, joint_trainer.py, tokenizer.py, infer.py
├── docs/                              # Academic demo page for GitHub Pages
│   ├── index.html                     # Interactive audio comparison web page
│   └── audio/                         # Demo audio samples (.wav)
└── egs/
    └── libritts/                      # Recipes, training scripts, configs & watermark models
        ├── bin -> ../../valle/bin     # Symlink to valle/bin
        ├── checkpoints/               # Pretrained & trained model checkpoints
        │   └── NeuMark-Native.pt      # Official pretrained watermark embedder & detector
        ├── shared/                    # parse_options.sh
        ├── STmodels/                  # SpeechTokenizer & GAN discriminators
        ├── models.py                  # Watermark embedder & detector (WMEmbedder, WMDetector)
        ├── configs/                   # Training & ablation configurations
        │   ├── config_tts_native.json
        │   ├── config_ablation_real_tokens.json
        │   └── config_ablation_valle_neumark_loss.json
        ├── tts_native_train.py        # Standard native watermark training (Accelerate DDP)
        ├── tts_native_loss.py         # Multi-scale Mel, VAD margin, adversarial losses
        ├── tts_native_dataset.py      # PyTorch Dataset for native watermark training
        ├── tts_native_attacks.py      # Differentiable distortion attack channels
        ├── test_valle_native_watermark.py # Extraction & robustness evaluation
        ├── generate_valle_native_dataset.py # Tokenized speech pairs generator
        ├── generate_demo_audios.py    # Quick audio synthesis script
        └── scripts/                   # Step-by-step portable bash scripts
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

## 🚀 Step 2: Navigate to Recipe Directory & Pretrained Weights

Following standard VALL-E / Kaldi convention, all training, data preparation, and evaluation commands are executed inside `egs/libritts`:

```bash
cd egs/libritts
```

### 2.1 Official Pretrained NeuMark-Native Model Checkpoint
The official trained checkpoint is included directly in this repository:
- **Location**: `egs/libritts/checkpoints/NeuMark-Native.pt` (or `checkpoints/NeuMark-Native.pt`)
- **Contents**: Trained `WMEmbedder` (`msg_processor`) and `WMDetector` (`detector`) trained on LibriTTS for 130,950 steps (10 epochs).
- **Usage**: You can **directly skip training** and verify watermark embedding and extraction using this checkpoint!

### 2.2 Pretrained SpeechTokenizer Model Weights
NeuMark-Native uses **SpeechTokenizer** for discrete acoustic token representations:

```bash
mkdir -p STmodels/pretrained_model

# Download SpeechTokenizer model weights (460MB)
curl -L -o STmodels/pretrained_model/SpeechTokenizer.pt     https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt
```

*(Optional)* Download WavLM for speaker similarity loss calculation:
```bash
mkdir -p models
curl -L -o models/wavlm_large_finetune.pth     https://github.com/goannan/NeuMark/releases/download/v1.0/wavlm_large_finetune.pth # or official WavLM URL
```

---

## ⚡ Direct Verification with Pretrained Checkpoint

If you want to immediately test and verify the watermark embedder and detector without retraining from scratch, run either of the following commands:

### Option A: Zero-Shot Speech Synthesis with Watermark Embedding
Synthesize voice-cloned speech with a 16-bit watermark payload and immediately extract the watermark to verify bit accuracy:
```bash
# In egs/libritts:
bash scripts/07_infer_zero_shot.sh     exp/demo_samples     "To be or not to be, that is the question."     ../../docs/audio/libritts_sample_1/00_prompt.wav     "1011001110001101"
```

### Option B: Benchmark Robustness & Audio Quality Evaluation
Evaluate watermark extraction accuracy, ROC-AUC, SNR, UTMOS, and PESQ across multiple acoustic distortion channels:
```bash
# In egs/libritts:
bash scripts/08_evaluate_watermark.sh     data/tokenized_valle_native/cuts_test_valle_native.jsonl.gz     checkpoints/NeuMark-Native.pt     exp/eval_results     cuda:0
```

---

## 📊 Step 3: LibriTTS Data Preparation (For Training from Scratch)

Prepare the LibriTTS dataset (downloads audio, builds Lhotse manifests, extracts SpeechTokenizer acoustic tokens, and phonemizes transcripts):

```bash
# In egs/libritts:
bash scripts/01_prepare_libritts.sh
```

**Custom subsets or directories:**
```bash
bash scripts/01_prepare_libritts.sh     --stage 0     --stop-stage 3     --dataset-parts "--dataset-parts all"     --audio-extractor "SpeechTokenizer"     --audio-feats-dir "data/tokenized"
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
bash scripts/05_prepare_native_tokens.sh     data/tokenized/cuts_train.jsonl.gz     data/tokenized_valle_native     -1
```

---

## 🔐 Step 6: Native Watermark Model Training

Train the **WMEmbedder** and **WMDetector** models to embed multi-bit watermark payloads into the discrete acoustic representations:

```bash
bash scripts/06_train_watermark.sh configs/config_tts_native.json
```

---

## 🔊 Step 7: Zero-Shot Speech Synthesis with Watermarking

Synthesize zero-shot voice-cloned speech from a target text and 3-second prompt audio with an embedded 16-bit cryptographic watermark:

```bash
bash scripts/07_infer_zero_shot.sh     exp/demo_samples     "To be or not to be, that is the question."     ../../docs/audio/libritts_sample_1/00_prompt.wav     "1011001110001101"
```

---

## 📈 Step 8: Extraction & Robustness Benchmark Evaluation

Evaluate watermark extraction bit accuracy, ROC-AUC, detection latency, and audio quality (PESQ, STOI, SNR, UTMOS) across diverse acoustic distortion attacks:

```bash
bash scripts/08_evaluate_watermark.sh     data/tokenized_valle_native/cuts_test_valle_native.jsonl.gz     checkpoints/NeuMark-Native.pt     exp/eval_results     cuda:0
```

---

## 🌐 Step 9: Running the Demo Page Locally

Preview the academic demo showcase page locally:

```bash
# Return to root directory
cd ../..

# Start HTTP server for docs/
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
