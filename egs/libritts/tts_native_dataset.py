import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lhotse import CutSet, load_manifest_lazy
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent


def find_audio_path(candidate_path: str, manifest_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolves an audio path adaptively across different clusters (local, Genkai, etc.)."""
    p = Path(candidate_path)
    if p.exists():
        return p

    str_p = str(candidate_path)
    if "LibriTTS" in str_p:
        rel_sub = str_p[str_p.index("LibriTTS") :]
    elif "download" in str_p:
        rel_sub = str_p[str_p.index("download") :]
    else:
        rel_sub = p.name

    search_roots = [
        manifest_dir.parents[1] if manifest_dir else None,
        SCRIPT_DIR,
        PROJECT_DIR / "egs/libritts",
        Path(os.environ.get("LIBRITTS_ROOT", "")),
        Path.cwd(),
    ]

    for root in search_roots:
        if root and root.exists():
            for target in [root / rel_sub, root / "download" / rel_sub, root / "download/LibriTTS" / rel_sub]:
                if target.exists():
                    return target
    return None


class TTSNativeDataset(Dataset):
    """
    Dataset that loads full-length 8-layer SpeechTokenizer tokens,
    full text transcripts, and full real audio recordings without truncation.
    """

    def __init__(
        self,
        manifest_path: str,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        sample_rate: int = 16000,
        downsample_rate: int = 320,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.downsample_rate = downsample_rate

        print(f"[Dataset] Loading manifest: {self.manifest_path} (full audio mode)...")
        self.cuts = load_manifest_lazy(self.manifest_path)
        self.cut_list = [c for c in self.cuts if min_duration <= c.duration <= max_duration]
        print(f"[Dataset] Loaded {len(self.cut_list)} valid full-length cuts.")

    def __len__(self) -> int:
        return len(self.cut_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cut = self.cut_list[idx]

        # 1. Load full 8-layer SpeechTokenizer codes: [T, 8] -> [8, T]
        try:
            codes_np = cut.load_features()  # [T, 8]
            codes = torch.from_numpy(codes_np).long().transpose(0, 1)  # [8, T]
        except Exception:
            codes = torch.zeros((8, int(self.sample_rate / self.downsample_rate * 2)), dtype=torch.long)

        # 2. Full text transcript
        text = cut.supervisions[0].text if cut.supervisions else ""

        # 3. Audio recording: resolve real audio if available
        audio = None
        if cut.has_recording and len(cut.recording.sources) > 0:
            rec_source = cut.recording.sources[0].source
            resolved_p = find_audio_path(rec_source, self.manifest_path.parent)
            if resolved_p is not None:
                try:
                    audio_t, sr = torchaudio.load(str(resolved_p))
                    if sr != self.sample_rate:
                        audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
                    audio = audio_t
                except Exception:
                    audio = None

        if audio is None or audio.numel() == 0:
            # Reconstruct clean placeholder audio if raw LibriTTS directory is not downloaded on remote node
            audio = torch.zeros((1, codes.shape[-1] * self.downsample_rate), dtype=torch.float32)

        # 4. Prompt audio: resolve speaker reference recording (prompt wav or target speaker audio)
        prompt_audio = None
        if cut.custom and "prompt_wav" in cut.custom and cut.custom["prompt_wav"]:
            cand = Path(cut.custom["prompt_wav"])
            if cand.exists():
                try:
                    p_wav, p_sr = torchaudio.load(str(cand))
                    if p_sr != self.sample_rate:
                        p_wav = torchaudio.functional.resample(p_wav, p_sr, self.sample_rate)
                    if p_wav.shape[0] > 1:
                        p_wav = p_wav.mean(dim=0, keepdim=True)
                    prompt_audio = p_wav
                except Exception:
                    prompt_audio = None

        if cut.custom and "prompt_wav_rel" in cut.custom and prompt_audio is None:
            p_rel = cut.custom["prompt_wav_rel"]
            for p_root in [
                SCRIPT_DIR / "data/seed_tts_eval/en",
                SCRIPT_DIR / "data/seed_tts_eval",
                SCRIPT_DIR / "synthesized_data/seedTTS/prompt",
                SCRIPT_DIR / "prompts",
            ]:
                cand = p_root / p_rel
                if not cand.exists():
                    cand = p_root / Path(p_rel).name
                if cand.exists():
                    try:
                        p_wav, p_sr = torchaudio.load(str(cand))
                        if p_sr != self.sample_rate:
                            p_wav = torchaudio.functional.resample(p_wav, p_sr, self.sample_rate)
                        if p_wav.shape[0] > 1:
                            p_wav = p_wav.mean(dim=0, keepdim=True)
                        prompt_audio = p_wav
                        break
                    except Exception:
                        pass

        p_id = cut.custom.get("prompt_cut_id", "") if (cut.custom and prompt_audio is None) else ""
        if p_id:
            p_rec_id = p_id.rsplit("-", 1)[0] if "-" in p_id else p_id
            parts = p_rec_id.split("_")
            if len(parts) >= 4:
                spk, chap = parts[0], parts[1]
                for root_candidate in [
                    self.manifest_path.parents[1] / "download/LibriTTS",
                    SCRIPT_DIR / "download/LibriTTS",
                    Path(os.environ.get("LIBRITTS_ROOT", "")),
                ]:
                    if root_candidate.exists():
                        for subset in ["dev-clean", "train-clean-100", "train-clean-360", "test-clean", "test-other", "dev-other", "train-other-500"]:
                            cand = root_candidate / subset / spk / chap / f"{p_rec_id}.wav"
                            if cand.exists():
                                try:
                                    p_wav, p_sr = torchaudio.load(str(cand))
                                    if p_sr != self.sample_rate:
                                        p_wav = torchaudio.functional.resample(p_wav, p_sr, self.sample_rate)
                                    prompt_audio = p_wav
                                    break
                                except Exception:
                                    pass
                        if prompt_audio is not None:
                            break

        if prompt_audio is None and audio.abs().max() > 1e-4:
            prompt_audio = audio.clone()

        if prompt_audio is None:
            prompt_audio = torch.empty((1, 0), dtype=torch.float32)

        return {
            "id": cut.id,
            "codes": codes,  # [8, T]
            "text": text,
            "audio": audio,  # [1, T_samples]
            "prompt_audio": prompt_audio,  # [1, T_samples] (or empty if missing)
            "frames": codes.shape[-1],
        }


def collate_tts_native(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for full-length audio and codes using batch dynamic padding."""
    batch_size = len(batch)
    max_frames = max(item["codes"].shape[-1] for item in batch)
    max_audio_len = max_frames * 320
    max_prompt_len = max((item["prompt_audio"].shape[-1] if item["prompt_audio"].numel() > 0 else 0) for item in batch)

    padded_codes = torch.zeros((batch_size, 8, max_frames), dtype=torch.long)
    padded_audio = torch.zeros((batch_size, 1, max_audio_len), dtype=torch.float32)
    padded_prompt = torch.zeros((batch_size, 1, max_prompt_len), dtype=torch.float32) if max_prompt_len > 0 else torch.empty((batch_size, 1, 0), dtype=torch.float32)
    frame_lengths = []
    texts = []
    ids = []

    for i, item in enumerate(batch):
        f_len = item["codes"].shape[-1]
        a_len = min(item["audio"].shape[-1], f_len * 320)
        p_len = item["prompt_audio"].shape[-1] if item["prompt_audio"].numel() > 0 else 0

        padded_codes[i, :, :f_len] = item["codes"]
        padded_audio[i, :, :a_len] = item["audio"][:, :a_len]
        if p_len > 0 and padded_prompt.shape[-1] >= p_len:
            padded_prompt[i, :, :p_len] = item["prompt_audio"]

        frame_lengths.append(f_len)
        texts.append(item["text"])
        ids.append(item["id"])

    return {
        "ids": ids,
        "codes": padded_codes,  # [B, 8, T_max]
        "audio": padded_audio,  # [B, 1, T_audio_max]
        "prompt_audio": padded_prompt,  # [B, 1, T_prompt_max]
        "frame_lengths": torch.tensor(frame_lengths, dtype=torch.long),
        "texts": texts,
    }


def get_tts_native_dataloader(
    manifest_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_duration: float = 30.0,
    min_duration: float = 0.5,
) -> DataLoader:
    dataset = TTSNativeDataset(
        manifest_path=manifest_path,
        max_duration=max_duration,
        min_duration=min_duration,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tts_native,
        pin_memory=(num_workers == 0 or torch.cuda.is_available()),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
