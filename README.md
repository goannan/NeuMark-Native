# NeuMark-Native: Robust In-Model Latent Audio Watermarking for Generative Speech Synthesis

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/docs/accelerate/index"><img alt="Accelerate" src="https://img.shields.io/badge/HuggingFace-Accelerate-yellow?style=flat"></a>
  <a href="https://goannan.github.io/NeuMark-Native/"><img alt="Demo Page" src="https://img.shields.io/badge/Audio%20Demo-GitHub%20Pages-blue?style=flat&logo=github"></a>
</p>

**NeuMark-Native** is an open-source framework for embedding robust, imperceptible, multi-bit cryptographic watermarks directly into the discrete latent acoustic space and residual quantization streams of generative zero-shot text-to-speech (TTS) models (e.g., VALL-E).

Unlike conventional post-hoc audio watermarking methods (e.g., AudioSeal, WavMark) that process rendered waveform audio and suffer severe degradation under neural codec compression and generative synthesis, NeuMark-Native operates directly within the neural codec's latent representations, ensuring high acoustic naturalness, energy-gated transparency, and superior resilience against neural resynthesis attacks.

---

## 🎧 Interactive Audio Demos

Listen to audio comparisons across **LibriTTS** and **SeedTTS** benchmarks against state-of-the-art baselines on our interactive demo page:
👉 **[Online Demo Page](https://goannan.github.io/NeuMark-Native/)**

---

## 📁 Repository Structure

```text
NeuMark-Native/
├── configs/                            # Training & ablation configurations
│   ├── config_tts_native.json          # Standard TTS-native training config
│   ├── config_tts_native_energy_gated.json # Energy-gated training config
│   ├── config_ablation_real_tokens.json
│   └── config_ablation_valle_neumark_loss.json
├── scripts/                            # User-friendly Bash execution scripts
│   ├── train.sh                        # Multi-GPU / Single-GPU training launcher
│   ├── train_energy_gated.sh           # Energy-gated training launcher
│   ├── evaluate.sh                     # Full test evaluation benchmark
│   └── prepare_dataset.sh              # Manifest and token extraction
├── models.py                           # Watermark Embedder & Detector architectures
├── STmodels/                           # SpeechTokenizer architecture & discriminators
├── tts_native_train.py                 # Core training script with Accelerate DDP
├── tts_native_energy_gated_train.py    # Energy-gated watermark training pipeline
├── tts_native_loss.py                  # Losses: Multi-scale Mel, VAD, Cosine, Adv, Sim, UTMOS
├── tts_native_dataset.py               # Lhotse-based DataLoader for tokenized speech
├── tts_native_attacks.py               # Differentiable acoustic & distortion channels
├── test_valle_native_watermark.py      # Authoritative benchmark evaluation script
├── generate_valle_native_dataset.py    # Offline tokenization & manifest extraction
├── generate_demo_audios.py             # Demo audio synthesis script
├── docs/                               # Demo page assets (HTML & WAV samples)
├── requirements.txt                    # Python package dependencies
└── README.md
```

---

## 🚀 1. Installation & Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/goannan/NeuMark-Native.git
cd NeuMark-Native
```

### Step 2: Create Environment
Python 3.10+ is required. You can use `conda` or `pyenv`:

```bash
# Using Conda:
conda create -n neumark-native python=3.10 -y
conda activate neumark-native

# Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 2. Pretrained Models & Assets

NeuMark-Native uses **SpeechTokenizer** as the underlying discrete acoustic representation:

1. Create model directory:
   ```bash
   mkdir -p STmodels/pretrained_model
   ```
2. Download `SpeechTokenizer.pt` into `STmodels/pretrained_model/`:
   ```bash
   # Download from HuggingFace
   curl -L -o STmodels/pretrained_model/SpeechTokenizer.pt \
       https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt
   ```
3. (Optional for SIM loss) Download WavLM checkpoint if training with speaker similarity regularization:
   ```bash
   mkdir -p models
   curl -L -o models/wavlm_large_finetune.pth \
       https://github.com/goannan/NeuMark/releases/download/v1.0/wavlm_large_finetune.pth # or standard WavLM URL
   ```

---

## 📊 3. Data Preparation

Training requires paired speech manifests containing VALL-E discrete acoustic tokens and corresponding audio:

```bash
bash scripts/prepare_dataset.sh \
    data/libritts_cuts.jsonl.gz \
    data/tokenized_native \
    -1
```

This processes the dataset and produces:
- `cuts_train_valle_native.jsonl.gz`
- `cuts_dev_valle_native.jsonl.gz`
- `cuts_test_valle_native.jsonl.gz`

Make sure the manifest paths in `configs/config_tts_native.json` point to your prepared manifest locations.

---

## 🏋️ 4. Training

We provide standalone Bash scripts that automatically detect available GPUs and launch single-GPU or multi-GPU Distributed Data Parallel (DDP) training via HuggingFace `accelerate`.

### Option A: Standard Native Watermark Training
```bash
# Uses all available GPUs by default:
bash scripts/train.sh configs/config_tts_native.json

# Or specify GPUs explicitly:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/config_tts_native.json
```

### Option B: Energy-Gated Native Watermark Training
Energy-gating dynamically regularizes watermark injection amplitude according to local acoustic energy levels, preserving silence and whisper frames without audible artifacts:
```bash
bash scripts/train_energy_gated.sh configs/config_tts_native_energy_gated.json
```

### Key Hyperparameters (`configs/*.json`):
- `cos_loss_lambda`: Cosine embedding alignment loss weight.
- `adv_loss_lambda`: Multi-period and multi-scale STFT GAN discriminator loss weight.
- `dec_loss_lambda`: Watermark detection bit-recovery loss weight.
- `vad_loss_lambda`: Energy-gated Voice Activity Detection margin loss.
- `mel_loss_lambda`: Multi-scale Mel-spectrogram reconstruction loss weight.
- `utmos_loss_lambda` / `sim_loss_lambda`: Perceptual speech quality and speaker similarity weights.

Checkpoints and TensorBoard event logs are automatically saved under `exp/tts_native_neumark/`.

---

## 📈 5. Testing & Evaluation

To evaluate watermark extraction accuracy, bit error rate, and speech fidelity against acoustic attacks:

```bash
bash scripts/evaluate.sh \
    data/cuts_test_valle_native.jsonl.gz \
    exp/tts_native_neumark/NeuMark_epoch_010.pt \
    exp/eval_results \
    cuda:0
```

The benchmark computes:
- **Bit Accuracy (%)**: Exact watermark extraction accuracy over 16-bit payloads.
- **ROC-AUC & TPR@0.1%FPR**: Watermark detection probability under positive/negative speech distributions.
- **Objective Quality**: PESQ (Wideband), STOI, and SNR (dB).
- **Latency**: Embedding latency and extraction throughput (ms/sec).

---

## 🌐 6. Running the Demo Page Locally

You can preview the academic showcase page locally with any static HTTP server:

```bash
cd docs
python3 -m http.server 8080
```
Open `http://localhost:8080` in your web browser to play and compare samples.

---

## 📄 License & Acknowledgments

This project builds upon insights and architectures from:
- [SpeechTokenizer](https://github.com/ZhangXingjian/SpeechTokenizer)
- [VALL-E](https://github.com/lifeiteng/vall-e)
- [NeuMark](https://github.com/goannan/NeuMark)

For research queries and questions, please open an issue in this repository.
