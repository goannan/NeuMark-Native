import sys
import os
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = PROJECT_DIR.parent / "NeuMark"

# Insert NeuMark root at index 0 to ensure priority
for p in [str(NEUMARK_ROOT / "train"), str(NEUMARK_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
    if os.path.exists(p):
        sys.path.insert(0, p)

try:
    from losses import (
        adversarial_loss_d,
        adversarial_loss_g,
        bits_to_chunks,
        cos_loss,
        decoding_loss,
        feature_loss,
        vad_based_loss,
    )
except ImportError:
    from loss import (
    adversarial_loss_d,
    adversarial_loss_g,
    bits_to_chunks,
    cos_loss,
    decoding_loss,
    feature_loss,
    vad_based_loss,
)


# ==========================================
# 1. UTMOS Loss (Subjective Naturalness)
# ==========================================
class UTMOSLoss(nn.Module):
    """Predicts MOS naturalness score using SpeechMOS UTMOS22 model and maximizes it."""

    def __init__(self, repository: str = "tarepan/SpeechMOS:v1.2.0", model_name: str = "utmos22_strong", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        try:
            self.model = torch.hub.load(repository, model_name, trust_repo=True, verbose=False)
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.to(self.device)
            self.available = True
        except Exception as e:
            print(f"[Warning] Failed to load UTMOS model: {e}. UTMOS loss will be disabled.")
            self.available = False

    def forward(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Input: audio waveform [B, 1, T] or [B, T]
        Output: scalar loss (- mean MOS score)
        """
        if not self.available:
            return torch.zeros((), device=audio.device, requires_grad=True)

        if audio.ndim == 3:
            audio = audio.squeeze(1)  # [B, T]

        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        # Minimum receptive field check (400 samples)
        if audio.shape[-1] < 400:
            return torch.zeros((), device=audio.device, requires_grad=True)

        scores = self.model(audio, 16000)  # [B, 1] or [B]
        loss = -scores.mean()
        return loss


# ==========================================
# 2. Speaker Similarity Loss (WavLM-Large SV)
# ==========================================
class SpeakerSimLoss(nn.Module):
    def get_similarity(self, wm_audio: torch.Tensor, prompt_audio: torch.Tensor, sample_rate: int = 16000) -> float:
        if not self.available or self.model is None:
            return 0.0
        if wm_audio.ndim == 3:
            wm_audio = wm_audio.squeeze(1)
        if prompt_audio.ndim == 3:
            prompt_audio = prompt_audio.squeeze(1)

        # Safety: if prompt_audio is empty or pure silence, return 1.0 (no degradation reference)
        if prompt_audio.numel() == 0 or prompt_audio.abs().max() < 1e-4:
            return 1.0

        if sample_rate != 16000:
            wm_audio = torchaudio.functional.resample(wm_audio, sample_rate, 16000)
            prompt_audio = torchaudio.functional.resample(prompt_audio, sample_rate, 16000)

        wm_audio = wm_audio.to(self.device).float()
        prompt_audio = prompt_audio.to(self.device).float()

        with torch.no_grad():
            emb_wm = self.model(wm_audio)
            emb_prompt = self.model(prompt_audio)
            sim = F.cosine_similarity(emb_wm, emb_prompt, dim=-1).mean().item()
        return sim

    """Calculates cosine similarity loss against target speaker prompt embedding."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        if checkpoint_path is None or not os.path.isabs(checkpoint_path):
            checkpoint_path = str(SCRIPT_DIR / (checkpoint_path or "models/wavlm_large_finetune.pth"))

        self.available = False
        if os.path.exists(checkpoint_path):
            try:
                import importlib.util
                ecapa_file = SCRIPT_DIR / "tools/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py"
                spec = importlib.util.spec_from_file_location("ecapa_tdnn_standalone", str(ecapa_file))
                ecapa_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ecapa_mod)
                ECAPA_TDNN_SMALL = ecapa_mod.ECAPA_TDNN_SMALL

                self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")
                state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state_dict["model"], strict=False)
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.to(self.device)
                self.available = True
            except Exception as e:
                print(f"[Warning] Failed to initialize WavLM SV model: {e}. SpeakerSimLoss will fall back.")
                self.available = False
        else:
            print(f"[Warning] WavLM checkpoint not found at {checkpoint_path}. SpeakerSimLoss will fall back.")

    def forward(self, wm_audio: torch.Tensor, prompt_audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Inputs: wm_audio [B, 1, T], prompt_audio [B, 1, T_p]
        Output: scalar loss (1.0 - cosine_sim)
        """
        # Safety: if prompt_audio is empty or pure silence, no valid speaker target
        if prompt_audio.numel() == 0 or prompt_audio.abs().max() < 1e-4:
            return torch.zeros((), device=self.device)

        if wm_audio.ndim == 3:
            wm_audio = wm_audio.squeeze(1)
        if prompt_audio.ndim == 3:
            prompt_audio = prompt_audio.squeeze(1)

        if sample_rate != 16000:
            wm_audio = torchaudio.functional.resample(wm_audio, sample_rate, 16000)
            prompt_audio = torchaudio.functional.resample(prompt_audio, sample_rate, 16000)

        wm_audio = wm_audio.to(self.device).float()
        prompt_audio = prompt_audio.to(self.device).float()

        if not self.available:
            return cos_loss(wm_audio, prompt_audio[..., :wm_audio.shape[-1]])

        emb_wm = self.model(wm_audio)  # [B, D]
        with torch.no_grad():
            emb_prompt = self.model(prompt_audio)  # [B, D]

        cos_sim = F.cosine_similarity(emb_wm, emb_prompt, dim=-1)
        loss = (1.0 - cos_sim).mean()
        return loss


# ==========================================
# 3. ASR CTC Loss (Wav2Vec2 Pronunciation)
# ==========================================
class ASRLoss(nn.Module):
    def decode_greedy(self, audio: torch.Tensor, sample_rate: int = 16000) -> List[str]:
        if audio.ndim == 3:
            audio = audio.squeeze(1)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        with torch.no_grad():
            emissions, _ = self.model(audio)
            indices = torch.argmax(emissions, dim=-1)
        
        transcripts = []
        for seq in indices:
            collapsed = []
            prev = None
            for idx in seq.tolist():
                if idx != prev:
                    if idx != self.blank_id:
                        collapsed.append(self.labels[idx])
                prev = idx
            text = "".join(collapsed).replace("|", " ").strip()
            transcripts.append(text)
        return transcripts

    """Calculates CTC Loss against ground truth text tokens using Wav2Vec2 ASR."""

    def __init__(self, bundle_name: str = "WAV2VEC2_ASR_BASE_960H", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.model = self.bundle.get_model().to(self.device).eval()
        self.model.requires_grad_(False)
        self.labels = self.bundle.get_labels()
        self.char2id = {c: i for i, c in enumerate(self.labels)}
        self.blank_id = 0
        self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)

    def text_to_ids(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = []
        lengths = []
        for t in texts:
            clean_t = t.upper().replace("\n", " ").strip()
            ids = [self.char2id[c] for c in clean_t if c in self.char2id]
            if not ids:
                ids = [self.char2id.get("|", 4)]
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long, device=self.device), torch.tensor(lengths, dtype=torch.long, device=self.device)

    def forward(self, audio: torch.Tensor, texts: List[str], sample_rate: int = 16000) -> torch.Tensor:
        if audio.ndim == 3:
            audio = audio.squeeze(1)

        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)

        emissions, _ = self.model(audio)  # [B, frames, num_classes]
        log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)  # [frames, B, classes]

        targets, target_lengths = self.text_to_ids(texts)
        input_lengths = torch.full((audio.size(0),), log_probs.size(0), dtype=torch.long, device=self.device)

        loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        return loss


# ==========================================
# 4. Latent Space Cosine Loss (Anchor)
# ==========================================
def latent_cosine_loss(z_wm: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss on continuous latent representation to anchor perturbations."""
    min_len = min(z_wm.shape[-1], z_q.shape[-1])
    z_wm = z_wm[..., :min_len]
    z_q = z_q[..., :min_len]
    return cos_loss(z_wm, z_q)

# -----------------------------------------------------------------------------
# NeuMark Original Multi-Scale Mel Loss Implementation
# -----------------------------------------------------------------------------
_mel_basis_cache = {}
_hann_window_cache = {}

def mel_spectrogram_torch(y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    n_fft = int(n_fft)
    num_mels = int(num_mels)
    sample_rate = int(sample_rate)
    hop_size = int(hop_size)
    win_size = int(win_size)
    fmin = float(fmin)
    fmax = float(fmax)

    key = f"{fmax}_{win_size}_{y.device}"
    global _mel_basis_cache, _hann_window_cache
    if key not in _mel_basis_cache:
        mel_transform = torchaudio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="htk"
        )
        _mel_basis_cache[key] = mel_transform.fb.float().T.to(y.device)
    
    if key not in _hann_window_cache:
        _hann_window_cache[key] = torch.hann_window(win_size).to(y.device)

    # Pad waveform
    y_in = y.squeeze(1) if y.dim() == 3 else y
    y_pad = torch.nn.functional.pad(
        y_in.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect"
    ).squeeze(1)

    device_type = y.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        y_pad = y_pad.float().contiguous()
        if y_pad.shape[-1] < win_size:
            pad_amount = win_size - y_pad.shape[-1]
            y_pad = torch.nn.functional.pad(y_pad, (0, pad_amount), mode="constant", value=0.0)

        window = _hann_window_cache[key].float()
        basis = _mel_basis_cache[key].float()

        spec_complex = torch.stft(
            y_pad, n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True  
        )

        spec = torch.view_as_real(spec_complex)
        spec = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-9)
        spec = torch.matmul(basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec

def mel_loss(x, x_hat, **kwargs):
    x_mel = mel_spectrogram_torch(x, **kwargs)
    x_hat_mel = mel_spectrogram_torch(x_hat, **kwargs)
    length = min(x_mel.size(-1), x_hat_mel.size(-1))
    return torch.nn.functional.l1_loss(x_mel[..., :length], x_hat_mel[..., :length])


# ==========================================
# 5. Large-Margin VAD Detection Loss
# ==========================================
def margin_vad_loss(
    vad_pred: torch.Tensor,
    vad_label: torch.Tensor,
    margin: float = 2.0,
    from_logits: bool = True
) -> torch.Tensor:
    """
    Large-Margin BCE Loss for VAD Watermark Presence Detection.
    Enforces logits >= margin for positive frames (1), and logits <= -margin for negative frames (0).
    Creates a wide classification buffer zone to suppress false alarms under cross-domain evaluation.

    Parameters
    ----------
    vad_pred : torch.Tensor
        Model predicted logits or probabilities [B, T] or [B, 1, T].
    vad_label : torch.Tensor
        Binary label tensor [B, T] or [B, 1, T].
    margin : float
        Logit decision margin (default 2.0).
    from_logits : bool
        Whether vad_pred contains unnormalized raw logits (default True).
    """
    if from_logits:
        if margin > 0:
            # Positive frames (1): require logits >= margin -> shift by -margin
            # Negative frames (0): require logits <= -margin -> shift by +margin
            adjusted_logits = torch.where(vad_label > 0.5, vad_pred - margin, vad_pred + margin)
            return F.binary_cross_entropy_with_logits(adjusted_logits, vad_label.float())
        else:
            return F.binary_cross_entropy_with_logits(vad_pred, vad_label.float())
    else:
        vad_pred = torch.clamp(vad_pred, 1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(vad_pred, vad_label.float())
