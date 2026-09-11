#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from email import parser
from pyexpat.errors import codes
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union


import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import os
import argparse
import json
from pathlib import Path
# TraceableSpeech project lives at vall-e/traceableSpeech; imports are lowercase on disk.
from traceableSpeech.env import AttrDict
from traceableSpeech.meldataset import mel_spectrogram
from traceableSpeech.models import Generator, Encoder, Quantizer
from traceableSpeech.watermark import Watermark_Encoder, Watermark_Decoder, Random_watermark


try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = (
                                    get_finals(py[0][:-1], strict=False)
                                    + py[0][-1]
                                )
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEUMARK_ROOT = PROJECT_ROOT.parent / "NeuMark"
DEFAULT_VOICEMARK_ROOT = PROJECT_ROOT.parent / "NeuMark"


def str_to_bool(value: Union[str, bool, None], default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def resolve_path(path: Optional[str], default: Union[str, Path], root: Union[str, Path] = PROJECT_ROOT) -> str:
    if not path:
        return str(default)
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    if path_obj.exists():
        return str(path_obj.resolve())
    root_path = Path(root) / path_obj
    if root_path.exists():
        return str(root_path.resolve())
    return str(path_obj)


def _import_voicemark(voicemark_root: str):
    import torch.nn.utils.parametrizations as parametrizations
    from torch.nn.utils import spectral_norm as legacy_spectral_norm
    from torch.nn.utils import weight_norm as legacy_weight_norm

    if not hasattr(parametrizations, "weight_norm"):
        parametrizations.weight_norm = legacy_weight_norm
    if not hasattr(parametrizations, "spectral_norm"):
        parametrizations.spectral_norm = legacy_spectral_norm

    root = str(Path(voicemark_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from STmodels.model import SpeechTokenizer as VoiceMarkSpeechTokenizer
    from models import WMDetector, WMEmbedder

    return VoiceMarkSpeechTokenizer, WMEmbedder, WMDetector


def _load_voicemark_watermark_state(
    checkpoint_path: str,
    msg_processor: torch.nn.Module,
    detector: torch.nn.Module,
) -> None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"VoiceMark watermark checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "msg_processor" in checkpoint and "detector" in checkpoint:
        msg_processor.load_state_dict(checkpoint["msg_processor"], strict=True)
        detector.load_state_dict(checkpoint["detector"], strict=True)
    elif "embedder" in checkpoint and "detector" in checkpoint:
        msg_processor.load_state_dict(checkpoint["embedder"], strict=True)
        detector.load_state_dict(checkpoint["detector"], strict=True)
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        msg_state = {
            k.removeprefix("msg_processor."): v
            for k, v in state.items()
            if k.startswith("msg_processor.")
        }
        det_state = {
            k.removeprefix("detector."): v
            for k, v in state.items()
            if k.startswith("detector.")
        }
        if msg_state:
            msg_processor.load_state_dict(msg_state, strict=True)
        if det_state:
            detector.load_state_dict(det_state, strict=True)
    else:
        msg_state = {
            k.removeprefix("msg_processor."): v
            for k, v in checkpoint.items()
            if k.startswith("msg_processor.")
        }
        det_state = {
            k.removeprefix("detector."): v
            for k, v in checkpoint.items()
            if k.startswith("detector.")
        }
        if not msg_state and not det_state:
            raise ValueError(f"Unsupported VoiceMark checkpoint format: {checkpoint_path}")
        if msg_state:
            msg_processor.load_state_dict(msg_state, strict=True)
        if det_state:
            detector.load_state_dict(det_state, strict=True)


class AudioTokenizer:
    """Audio codec wrapper for EnCodec, TraceableSpeech, and VoiceMark."""

    def __init__(
        self,
        device: Any = None,
        enable_ts: bool = False,
        ts_checkpoint: Optional[str] = None,
        ts_config: Optional[str] = None,
        watermark_backend: str = "encodec",
        voicemark_root: Optional[str] = None,
        voicemark_config: Optional[str] = None,
        voicemark_st_checkpoint: Optional[str] = None,
        voicemark_checkpoint: Optional[str] = None,
        voicemark_embed_vq1: Union[str, bool, None] = None,
        neumark_root: Optional[str] = None,
        neumark_config: Optional[str] = None,
        neumark_st_checkpoint: Optional[str] = None,
        neumark_checkpoint: Optional[str] = None,
        neumark_embed_vq1: Union[str, bool, None] = None,
    ) -> None:
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        backend = (watermark_backend or "encodec").lower()
        if enable_ts and backend == "encodec":
            backend = "traceablespeech"
        if backend in {"ts", "traceable_speech"}:
            backend = "traceablespeech"
        if backend in {"nm", "neu_mark", "neumark", "vm", "voice_mark", "voicemark"}:
            backend = "neumark"
        if backend not in {"encodec", "traceablespeech", "voicemark", "neumark"}:
            raise ValueError(f"Unsupported watermark backend: {watermark_backend}")
        self.watermark_backend = backend

        self.codec = None
        self.sample_rate = 24000
        self.channels = 1
        self.downsample_rate = 320
        self.num_quantizers = 8

        if self.watermark_backend not in {"voicemark", "neumark"}:
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)
            remove_encodec_weight_norm(model)
            self.codec = model.to(device)
            self.sample_rate = model.sample_rate
            self.channels = model.channels

        self._config_path = resolve_path(
            ts_config,
            PROJECT_ROOT / "traceableSpeech" / "config.json",
        )
        self._ts_checkpoint = resolve_path(
            ts_checkpoint,
            PROJECT_ROOT / "traceableSpeech" / "g_00150000",
        )
        self.enable_ts = self.watermark_backend == "traceablespeech"
        self._ts_loaded = False
        self._ts_available = False
        self._h = None
        self._generator = None
        self._encoder = None
        self._quantizer_audio = None
        self._watermark_encoder = None
        self._watermark_decoder = None

        self._vm_root = resolve_path(voicemark_root, DEFAULT_VOICEMARK_ROOT)
        self._vm_config_path = resolve_path(
            voicemark_config,
            Path(self._vm_root) / "STmodels" / "pretrained_model" / "speechtokenizer_hubert_avg_config.json",
            root=self._vm_root,
        )
        self._vm_st_checkpoint = resolve_path(
            voicemark_st_checkpoint,
            Path(self._vm_root) / "STmodels" / "pretrained_model" / "SpeechTokenizer.pt",
            root=self._vm_root,
        )
        self._vm_checkpoint = resolve_path(
            voicemark_checkpoint,
            Path(self._vm_root) / "train" / "Log" / "spt_base" / "20260601-123358" / "WatermarkTrainer_final_00150000.pt",
            root=self._vm_root,
        )
        self._vm_embed_vq1 = str_to_bool(neumark_embed_vq1 if neumark_embed_vq1 is not None else voicemark_embed_vq1, default=True)
        self._vm_loaded = False
        self._vm_available = False
        self._vm_st_model = None
        self._vm_msg_processor = None
        self._vm_detector = None
        self.nbits = 16
        self.nchunk_size = 4

        if self.watermark_backend in {"voicemark", "neumark"} and os.path.isfile(self._vm_config_path):
            with open(self._vm_config_path) as f:
                vm_cfg = json.load(f)
            self.sample_rate = int(vm_cfg.get("sample_rate", 16000))
            self.channels = 1
            self.downsample_rate = int(np.prod(vm_cfg.get("strides", [8, 5, 4, 2])))
            self.num_quantizers = int(vm_cfg.get("n_q", 8))

    @property
    def device(self):
        return self._device

    @property
    def frame_shift(self) -> Seconds:
        return self.downsample_rate / self.sample_rate

    @property
    def has_watermark_decoder(self) -> bool:
        if self.watermark_backend in {"voicemark", "neumark"}:
            return self._load_voicemark()
        if self.watermark_backend == "traceablespeech":
            return self._load_traceable_speech()
        return False

    def _load_traceable_speech(self) -> bool:
        """Lazily load TraceableSpeech models if configuration exists."""
        if self._ts_loaded:
            return self._ts_available

        self._ts_loaded = True
        if not self.enable_ts or not os.path.isfile(self._config_path):
            return False

        print(f"Loading TraceableSpeech config from {self._config_path}")
        with open(self._config_path) as f:
            json_config = json.load(f)

        self._h = AttrDict(json_config)
        torch.manual_seed(self._h.seed)

        device = self.device
        print(f"Loading TraceableSpeech checkpoint from {self._ts_checkpoint} on {device}")

        generator = Generator(self._h).to(device)
        encoder = Encoder(self._h).to(device)
        quantizer_audio = Quantizer(self._h, "Audio").to(device)
        watermark_encoder = Watermark_Encoder(self._h).to(device)
        watermark_decoder = Watermark_Decoder(self._h).to(device)

        state_dict_g = load_checkpoint(self._ts_checkpoint, device)
        generator.load_state_dict(state_dict_g["generator"])
        encoder.load_state_dict(state_dict_g["encoder"])
        quantizer_audio.load_state_dict(state_dict_g["quantizer_Audio"])
        watermark_encoder.load_state_dict(state_dict_g["watermark_encoder"])
        watermark_decoder.load_state_dict(state_dict_g["watermark_decoder"])

        generator.eval()
        generator.remove_weight_norm()
        encoder.eval()
        encoder.remove_weight_norm()
        watermark_encoder.eval()
        watermark_decoder.eval()

        self._ts_available = True
        self._generator = generator
        self._encoder = encoder
        self._quantizer_audio = quantizer_audio
        self._watermark_encoder = watermark_encoder
        self._watermark_decoder = watermark_decoder
        return True

    def _load_voicemark(self) -> bool:
        if self._vm_loaded:
            return self._vm_available

        self._vm_loaded = True
        if self.watermark_backend not in {"voicemark", "neumark"}:
            return False
        if not os.path.isfile(self._vm_config_path):
            raise FileNotFoundError(f"VoiceMark config not found: {self._vm_config_path}")
        if not os.path.isfile(self._vm_st_checkpoint):
            raise FileNotFoundError(f"VoiceMark SpeechTokenizer checkpoint not found: {self._vm_st_checkpoint}")

        VoiceMarkSpeechTokenizer, WMEmbedder, WMDetector = _import_voicemark(self._vm_root)
        print(f"Loading VoiceMark SpeechTokenizer from {self._vm_st_checkpoint}")
        st_model = VoiceMarkSpeechTokenizer.load_from_checkpoint(
            self._vm_config_path,
            self._vm_st_checkpoint,
        ).to(self.device)
        msg_processor = WMEmbedder(nbits=self.nbits, input_dim=1024, nchunk_size=self.nchunk_size).to(self.device)
        detector = WMDetector(1024, self.nbits, nchunk_size=self.nchunk_size).to(self.device)
        _load_voicemark_watermark_state(self._vm_checkpoint, msg_processor, detector)

        st_model.eval()
        msg_processor.eval()
        detector.eval()
        for module in (st_model, msg_processor, detector):
            for param in module.parameters():
                param.requires_grad = False

        self._vm_st_model = st_model
        self._vm_msg_processor = msg_processor
        self._vm_detector = detector
        self._vm_available = True
        return True

    def _extract_codes(self, frames: torch.Tensor) -> torch.Tensor:
        if isinstance(frames, (list, tuple)):
            codes = frames[0][0] if isinstance(frames[0], (list, tuple)) else frames[0]
        else:
            codes = frames
        return codes.to(self.device)

    def _voicemark_codes_to_qbt(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long()
        if codes.dim() != 3:
            raise ValueError(f"Expected VoiceMark codes with 3 dims, got {tuple(codes.shape)}")
        if codes.shape[1] == self.num_quantizers:
            return codes.permute(1, 0, 2).contiguous()
        if codes.shape[0] == self.num_quantizers:
            return codes.contiguous()
        raise ValueError(f"Cannot infer VoiceMark code layout from shape {tuple(codes.shape)}")

    def _decode_voicemark(self, codes: torch.Tensor, message: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._load_voicemark():
            raise RuntimeError("VoiceMark backend is not available.")

        codes_qbt = self._voicemark_codes_to_qbt(codes)
        if message is None:
            return self._vm_st_model.decode(codes_qbt)

        message = message.to(self.device).long()
        quantized_layers = [
            self._vm_st_model.quantizer.decode(codes_qbt[i : i + 1], st=i)
            for i in range(codes_qbt.shape[0])
        ]
        if self._vm_embed_vq1:
            watermarked_latent = sum(
                self._vm_msg_processor(layer, message) for layer in quantized_layers
            )
        else:
            watermarked_latent = quantized_layers[0] + sum(
                self._vm_msg_processor(layer, message) for layer in quantized_layers[1:]
            )
        return self._vm_st_model.decoder(watermarked_latent)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # input: wav: [B, C = 1, T]
        # output: List[(codes, scale)] with codes [B, n_q = 8, T/downsample_rate]
        if self.watermark_backend in {"voicemark", "neumark"}:
            if not self._load_voicemark():
                raise RuntimeError("VoiceMark backend is not available.")
            with torch.no_grad():
                codes = self._vm_st_model.encode(wav.to(self.device))
                return [(codes.permute(1, 0, 2).contiguous(), None)]

        if not self._load_traceable_speech():
            return self.codec.encode(wav.to(self.device))

        with torch.no_grad():
            wav = wav.to(self.device)
            en_y = self._encoder(wav)  # [B, 1024, T/320]
            q, _, c = self._quantizer_audio(en_y)
            q = torch.stack([code.reshape(q.size(0), -1) for code in c], -1)
            q = q.transpose(1, 2)  # [B, n_q = 8, T/320]
            encoded_frames = [(q, None)]
        return encoded_frames

    def decode(
        self, frames: torch.Tensor, watermark_sign: torch.Tensor = None
    ) -> torch.Tensor:
        # frames: List[(codes, scale)] with codes [B, n_q, T]
        if self.watermark_backend in {"voicemark", "neumark"}:
            return self._decode_voicemark(self._extract_codes(frames), watermark_sign)

        if not self._load_traceable_speech():
            return self.codec.decode(frames)

        codes = self._extract_codes(frames)
        quantized = self._quantizer_audio.embed(
            codes.transpose(1, 2),
            self._h.Audio.get("infer_need_layer", self._h.Audio["residul_layer"]),
        )
        sign_trait = torch.zeros((codes.size(0), 256), device=self.device)
        if watermark_sign is not None:
            sign_trait = self._watermark_encoder(watermark_sign.to(self.device).long())
        decoded_wav = self._generator(quantized, sign_trait)
        return decoded_wav

    def decode_pair(
        self, frames: torch.Tensor, watermark_sign: torch.Tensor = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode both reference and optional watermarked audio."""
        if self.watermark_backend in {"voicemark", "neumark"}:
            codes = self._extract_codes(frames)
            decoded_ref = self._decode_voicemark(codes, None)
            decoded_wm = None
            if watermark_sign is not None:
                decoded_wm = self._decode_voicemark(codes, watermark_sign)
            return decoded_ref, decoded_wm

        if not self._load_traceable_speech():
            ref = self.codec.decode(frames)
            wm = self.codec.decode(frames) if watermark_sign is not None else None
            return ref, wm

        codes = self._extract_codes(frames)
        quantized = self._quantizer_audio.embed(
            codes.transpose(1, 2),
            self._h.Audio.get("infer_need_layer", self._h.Audio["residul_layer"]),
        )

        ref_trait = torch.zeros((codes.size(0), 256), device=self.device)
        decoded_ref = self._generator(quantized, ref_trait)

        decoded_wm = None
        if watermark_sign is not None:
            wm_trait = self._watermark_encoder(watermark_sign.to(self.device).long())
            decoded_wm = self._generator(quantized, wm_trait)

        return decoded_ref, decoded_wm

    def random_watermark(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.watermark_backend in {"voicemark", "neumark"} and self._load_voicemark():
            return torch.randint(0, 2, (batch_size, self.nbits), device=self.device)
        if self.watermark_backend == "traceablespeech" and self._load_traceable_speech():
            return Random_watermark(batch_size).to(self.device)
        return None

    def detect_watermark(self, wav: torch.Tensor):
        if self.watermark_backend in {"voicemark", "neumark"} and self._load_voicemark():
            wav = wav.to(self.device)
            try:
                features = self._vm_st_model.forward_feature(wav, embed_vq1=self._vm_embed_vq1)
            except TypeError:
                features = self._vm_st_model.forward_feature(wav)
            return self._vm_detector.detect_watermark(features)

        if self.watermark_backend == "traceablespeech" and self._load_traceable_speech():
            wav = wav.to(self.device)
            if wav.dim() == 3:
                wav_for_mel = wav.squeeze(1)
            else:
                wav_for_mel = wav
            pad_need = int((self._h.n_fft - self._h.hop_size) / 2)
            if wav_for_mel.shape[-1] <= pad_need * 2:
                wav_for_mel = torch.nn.functional.pad(wav_for_mel, (0, max(1600, pad_need * 2 - wav_for_mel.shape[-1] + 1600)))
            mel = mel_spectrogram(
                wav_for_mel,
                self._h.n_fft,
                self._h.num_mels,
                self._h.sampling_rate,
                self._h.hop_size,
                self._h.win_size,
                self._h.fmin,
                self._h.fmax_for_loss,
            )
            sign_score, sign_pred = self._watermark_decoder(mel)
            detect_prob = torch.stack(
                [score.softmax(dim=1).max(dim=1).values for score in sign_score],
                dim=1,
            ).mean(dim=1)
            return detect_prob, sign_pred, sign_score

        return None


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


@dataclass
class AudioTokenConfig:
    frame_shift: Seconds = 320.0 / 24000
    num_quantizers: int = 8
    backend: str = "encodec"
    voicemark_root: Optional[str] = None
    voicemark_config: Optional[str] = None
    voicemark_st_checkpoint: Optional[str] = None
    voicemark_checkpoint: Optional[str] = None
    voicemark_embed_vq1: Union[str, bool, None] = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AudioTokenConfig":
        return AudioTokenConfig(**data)


class AudioTokenExtractor(FeatureExtractor):
    name = "encodec"
    config_type = AudioTokenConfig

    def __init__(self, config: Optional[Any] = None):
        super(AudioTokenExtractor, self).__init__(config)
        self.tokenizer = AudioTokenizer(
            watermark_backend=self.config.backend,
            voicemark_root=self.config.voicemark_root,
            voicemark_config=self.config.voicemark_config,
            voicemark_st_checkpoint=self.config.voicemark_st_checkpoint,
            voicemark_checkpoint=self.config.voicemark_checkpoint,
            voicemark_embed_vq1=self.config.voicemark_embed_vq1,
        )
        self.config.frame_shift = self.tokenizer.frame_shift
        self.config.num_quantizers = self.tokenizer.num_quantizers
        self.name = self.config.backend

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / self.tokenizer.sample_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.tokenizer.sample_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def _prepare_batch_audio(
        self,
        samples: List[Union[np.ndarray, torch.Tensor]],
        sampling_rate: int,
    ) -> List[torch.Tensor]:
        prepared = []
        for wav in samples:
            if not isinstance(wav, torch.Tensor):
                wav = torch.from_numpy(wav)
            wav = wav.detach().cpu().float()

            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2:
                wav = wav.squeeze()
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)
                elif wav.shape[0] not in (1, 2) and wav.shape[-1] in (1, 2):
                    wav = wav.transpose(0, 1)
            else:
                raise ValueError()

            if wav.shape[0] not in (1, 2):
                raise ValueError()

            if (
                sampling_rate != self.tokenizer.sample_rate
                or wav.shape[0] != self.tokenizer.channels
            ):
                wav = convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )

            prepared.append(wav.squeeze(0))
        return prepared

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:
        device = self.tokenizer.device
        samples = self._prepare_batch_audio(samples, sampling_rate)
        samples, lengths = self.pad_tensor_list(samples, "cpu")
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames[0][0]  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / self.tokenizer.sample_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.tokenizer.sample_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]


if __name__ == "__main__":
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    samples = torch.from_numpy(np.random.random([4, 1, 1600])).type(
        torch.float32
    )
    codes_raw = model.encode(samples)

    remove_encodec_weight_norm(model)
    codes_norm = model.encode(samples)

    assert torch.allclose(codes_raw[0][0], codes_norm[0][0])
