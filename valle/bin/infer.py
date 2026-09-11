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
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
import torchaudio.functional as AF
import numpy as np
from icefall.utils import AttributeDict, str2bool

try:
    from torchmetrics.functional.audio import pesq as tm_pesq
    from torchmetrics.functional.audio import stoi as tm_stoi
except Exception:
    tm_pesq = None
    tm_stoi = None

# torchmetrics versions differ; resolve callable fallbacks if imports yielded modules
if tm_pesq is not None and not callable(tm_pesq):
    try:
        from torchmetrics.functional.audio.pesq import pesq as tm_pesq_fn  # type: ignore

        tm_pesq = tm_pesq_fn
    except Exception:
        pass
if tm_stoi is not None and not callable(tm_stoi):
    try:
        from torchmetrics.functional.audio.stoi import stoi as tm_stoi_fn  # type: ignore

        tm_stoi = tm_stoi_fn
    except Exception:
        pass

try:
    from pesq import pesq as pesq_pkg
except Exception:
    pesq_pkg = None

try:
    from pystoi import stoi as stoi_pkg
except Exception:
    stoi_pkg = None

try:
    from .attacks import (
        CODEC_KEYWORDS,
        CodecAttackError,
        build_voicemark_valid_attacks,
        release_codec_models,
    )
except ImportError:
    # Fallback when running as a standalone script (no package context)
    from attacks import (
        CODEC_KEYWORDS,
        CodecAttackError,
        build_voicemark_valid_attacks,
        release_codec_models,
    )

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )

    parser.add_argument(
        "--text-file",
        type=str,
        default="",
        help="Path to a file with one text per line (overrides --text).",
    )

    parser.add_argument(
        "--seed-tts-manifest",
        type=Path,
        default=None,
        help=(
            "Seed-TTS-Eval meta.lst. Each row must be "
            "utterance_id|prompt_text|prompt_wav|target_text. This mode "
            "overrides --text, --text-file, --text-prompts and --audio-prompts."
        ),
    )

    parser.add_argument(
        "--seed-tts-num-samples",
        type=int,
        default=None,
        help="Use the first N Seed-TTS-Eval rows; default uses the full manifest.",
    )

    parser.add_argument(
        "--seed-tts-primary-output",
        choices=["watermarked", "clean"],
        default="watermarked",
        help=(
            "Audio linked as <utterance_id>.wav for the official Seed-TTS "
            "WER/SIM scripts. Watermarked is the system-under-test default."
        ),
    )

    parser.add_argument(
        "--seed-tts-fixed-prompt-audio",
        type=Path,
        default=None,
        help=(
            "Optional fixed prompt WAV used for every Seed-TTS manifest row. "
            "The manifest still supplies utterance IDs and target texts."
        ),
    )

    parser.add_argument(
        "--seed-tts-fixed-prompt-text",
        type=str,
        default=None,
        help="Transcript of --seed-tts-fixed-prompt-audio.",
    )

    # model
    # add_model_arguments(parser)
    # parser.add_argument(
    #     "--text-tokens",
    #     type=str,
    #     default="data/tokenized/unique_text_tokens.k2symbols",
    #     help="Path to the unique text tokens file.",
    # )

    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )

    parser.add_argument(
        "--ts-enable",
        type=str2bool,
        default=False,
        help="Legacy switch for --watermark-backend traceablespeech.",
    )

    parser.add_argument(
        "--ts-checkpoint-file",
        type=str,
        default="traceableSpeech/g_00150000",
        help="The checkpoint file of TraceableSpeech model.",
    )

    parser.add_argument(
        "--watermark-backend",
        type=str,
        default="encodec",
        choices=["encodec", "traceablespeech", "voicemark", "neumark"],
        help="Codec/watermark backend used by AudioTokenizer.",
    )

    parser.add_argument(
        "--voicemark-root",
        type=str,
        default="/home/wu25/mrnas04home/projects/NeuMark",
        help="Path to the VoiceMark project root.",
    )

    parser.add_argument(
        "--voicemark-config",
        type=str,
        default="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json",
        help="VoiceMark SpeechTokenizer config, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-st-checkpoint",
        type=str,
        default="STmodels/pretrained_model/SpeechTokenizer.pt",
        help="VoiceMark SpeechTokenizer checkpoint, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-checkpoint",
        type=str,
        default="train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt",
        help="VoiceMark watermark checkpoint, absolute or relative to --voicemark-root.",
    )

    parser.add_argument(
        "--voicemark-embed-vq1",
        type=str2bool,
        default=True,
        help="Whether VoiceMark embeds/detects watermark in VQ1-8 instead of VQ2-8.",
    )

    parser.add_argument(
        "--runtime-timing-json",
        type=Path,
        default=None,
        help=(
            "Optional output for synchronized per-utterance synthesis, watermark "
            "embedding-path, and extraction timings."
        ),
    )
    parser.add_argument(
        "--skip-post-generation-eval",
        action="store_true",
        help=(
            "Skip PESQ/STOI/SI-SNR and the watermark attack suite after generation. "
            "Intended for runtime benchmarking."
        ),
    )

    return parser.parse_args()


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device, function):
    synchronize_device(device)
    start = time.perf_counter()
    result = function()
    synchronize_device(device)
    return result, time.perf_counter() - start


def summarize_runtime_timing(
    args,
    watermark_backend,
    audio_sample_rate,
    rows,
    skipped,
    setup,
    loop_wall_seconds,
    device,
):
    def total(key):
        return float(sum(row[key] for row in rows))

    audio_seconds = total("audio_duration_seconds")
    prompt_per_row = total("prompt_audio_encode_seconds")
    token_generation = total("valle_token_generation_seconds")
    clean_decode = total("clean_decode_seconds")
    message_generation = total("watermark_message_seconds")
    watermarked_decode = total("watermarked_decode_seconds")
    extraction = total("watermark_extract_seconds")
    prompt_once = float(setup.get("fixed_prompt_encode_seconds", 0.0))
    extraction_count = sum(
        1 for row in rows if row.get("watermark_extract_available", True)
    )
    extraction_skipped_count = len(rows) - extraction_count
    setup_excluded = {
        key: value
        for key, value in setup.items()
        if key != "fixed_prompt_encode_seconds"
    }
    clean_synthesis = prompt_once + prompt_per_row + token_generation + clean_decode
    watermarked_synthesis = (
        prompt_once
        + prompt_per_row
        + token_generation
        + message_generation
        + watermarked_decode
    )
    watermark_path = message_generation + watermarked_decode
    watermark_incremental = watermark_path - clean_decode

    return {
        "schema_version": 2,
        "benchmark_kind": "integrated_valle_watermark",
        "timing_scope": (
            "CUDA-synchronized synthesis operations; model loading, generated "
            "waveform writes, text tokenization, quality metrics, and robustness "
            "attacks are excluded. Fixed-prompt loading and audio encoding are "
            "included once in both synthesis totals."
        ),
        "watermark_backend": watermark_backend,
        "checkpoint": args.checkpoint,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "hardware": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "torch_version": str(torch.__version__),
        },
        "configuration": {
            "seed_tts_manifest": (
                str(args.seed_tts_manifest.expanduser().resolve())
                if args.seed_tts_manifest is not None
                else None
            ),
            "fixed_prompt_audio": (
                str(args.seed_tts_fixed_prompt_audio.expanduser().resolve())
                if args.seed_tts_fixed_prompt_audio is not None
                else None
            ),
            "fixed_prompt_text": args.seed_tts_fixed_prompt_text,
            "top_k": args.top_k,
            "temperature": args.temperature,
        },
        "sample_rate": audio_sample_rate,
        "requested_samples": (
            args.seed_tts_num_samples
            if args.seed_tts_num_samples is not None
            else len(rows) + len(skipped)
        ),
        "count": len(rows),
        "skipped_count": len(skipped),
        "setup_seconds_excluded_from_totals": setup_excluded,
        "prompt_preparation_seconds_included_in_totals": prompt_once,
        "loop_wall_seconds_including_io_and_cpu_overhead": loop_wall_seconds,
        "totals": {
            "audio_duration_seconds": audio_seconds,
            "fixed_prompt_encode_seconds": prompt_once,
            "per_row_prompt_audio_encode_seconds": prompt_per_row,
            "valle_token_generation_seconds": token_generation,
            "clean_decode_seconds": clean_decode,
            "clean_synthesis_seconds": clean_synthesis,
            "watermark_message_seconds": message_generation,
            "watermarked_decode_seconds": watermarked_decode,
            "watermark_embedding_path_seconds": watermark_path,
            "watermark_incremental_over_clean_seconds": watermark_incremental,
            "watermarked_synthesis_seconds": watermarked_synthesis,
            "watermark_extract_seconds": extraction,
            "watermark_extract_count": extraction_count,
            "watermark_extract_skipped_count": extraction_skipped_count,
            "clean_synthesis_rtf": (
                clean_synthesis / audio_seconds if audio_seconds > 0 else None
            ),
            "watermarked_synthesis_rtf": (
                watermarked_synthesis / audio_seconds if audio_seconds > 0 else None
            ),
            "watermark_embedding_path_rtf": (
                watermark_path / audio_seconds if audio_seconds > 0 else None
            ),
            "watermark_incremental_over_clean_rtf": (
                watermark_incremental / audio_seconds
                if audio_seconds > 0
                else None
            ),
            "watermark_extract_rtf": (
                extraction / audio_seconds if audio_seconds > 0 else None
            ),
        },
        "skipped": skipped,
        "details": rows,
    }


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    text_tokens = args.text_tokens

    return model, text_tokens


def load_seed_tts_manifest(manifest_path: Path, num_samples=None):
    """Load the standard Seed-TTS-Eval zero-shot TTS manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Seed-TTS-Eval manifest not found: {manifest_path}")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("--seed-tts-num-samples must be greater than zero.")

    items = []
    seen_ids = set()
    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) != 4:
                raise ValueError(
                    f"{manifest_path}:{line_number}: expected four pipe-separated "
                    f"fields, got {len(fields)}"
                )
            raw_utterance_id, prompt_text, prompt_wav, target_text = fields
            utterance_id = Path(raw_utterance_id).stem
            if not all((utterance_id, prompt_text, prompt_wav, target_text)):
                raise ValueError(
                    f"{manifest_path}:{line_number}: Seed-TTS fields cannot be empty"
                )
            if utterance_id in seen_ids:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate utterance id "
                    f"{utterance_id!r}"
                )

            prompt_path = Path(prompt_wav).expanduser()
            if not prompt_path.is_absolute():
                prompt_path = manifest_path.parent / prompt_path
            prompt_path = prompt_path.resolve()
            if not prompt_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}:{line_number}: prompt audio not found: "
                    f"{prompt_path}"
                )

            seen_ids.add(utterance_id)
            items.append(
                {
                    "utterance_id": utterance_id,
                    "prompt_text": prompt_text,
                    "prompt_wav": str(prompt_path),
                    "target_text": target_text,
                }
            )
            if num_samples is not None and len(items) >= num_samples:
                break

    if not items:
        raise ValueError(f"Seed-TTS-Eval manifest is empty: {manifest_path}")
    return items


def _link_seed_tts_primary_output(source_path: Path, target_path: Path):
    """Create the <utt>.wav expected by the official Seed-TTS metric scripts."""

    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copyfile(source_path, target_path)


def _ensure_mono_and_resample(waveform: torch.Tensor, orig_sr: int, target_sr: int):
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if orig_sr != target_sr:
        waveform = AF.resample(waveform, orig_sr, target_sr)
    return waveform


def compute_si_snr(ref_waveform: torch.Tensor, test_waveform: torch.Tensor) -> float:
    """Use the same scale-invariant projection as VoiceMark valid.py."""

    ref = ref_waveform.reshape(-1)
    test = test_waveform.reshape(-1)
    length = min(ref.numel(), test.numel())
    ref = ref[:length]
    test = test[:length]
    target = (torch.sum(ref * test) / (torch.sum(ref**2) + 1e-8)) * ref
    noise = test - target
    return float(
        10
        * torch.log10(
            (torch.sum(target**2) + 1e-8) / (torch.sum(noise**2) + 1e-8)
        )
    )


def compute_quality(
    ref_waveform: torch.Tensor,
    test_waveform: torch.Tensor,
    orig_sr: int,
):
    """Compute PESQ-WB, STOI and SI-SNR with VoiceMark's alignment policy."""

    target_sr = 16000
    ref_rs = _ensure_mono_and_resample(ref_waveform.cpu(), orig_sr, target_sr)
    test_rs = _ensure_mono_and_resample(test_waveform.cpu(), orig_sr, target_sr)
    length = min(ref_rs.shape[-1], test_rs.shape[-1])
    ref_rs = ref_rs[..., :length].float()
    test_rs = test_rs[..., :length].float()
    ref_np = ref_rs.squeeze().numpy().astype("float32")
    test_np = test_rs.squeeze().numpy().astype("float32")

    pesq_score = 0.0
    stoi_score = 0.0
    try:
        if pesq_pkg is not None:
            pesq_score = float(pesq_pkg(target_sr, ref_np, test_np, "wb"))
        elif callable(tm_pesq):
            pesq_score = float(tm_pesq(ref_rs, test_rs, target_sr, "wb"))
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"PESQ failed; recording 0.0 as VoiceMark valid.py does: {exc}")
    try:
        if stoi_pkg is not None:
            stoi_score = float(
                stoi_pkg(ref_np, test_np, target_sr, extended=False)
            )
        elif callable(tm_stoi):
            stoi_score = float(
                tm_stoi(ref_rs, test_rs, target_sr, extended=False)
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"STOI failed; recording 0.0 as VoiceMark valid.py does: {exc}")

    return {
        "pesq_wb": pesq_score,
        "stoi": stoi_score,
        "si_snr": compute_si_snr(ref_rs, test_rs),
    }


def _unwrap_attacked_audio(attacked):
    """AudioEffects optionally returns ``(audio, mask)``; evaluation needs audio."""

    if isinstance(attacked, torch.Tensor):
        return attacked
    return attacked[0]


def traceable_symbols_to_bits(symbols: torch.Tensor) -> torch.Tensor:
    """Expand TraceableSpeech hexadecimal symbols (0..15) to binary bits."""

    symbols = torch.as_tensor(symbols).long()
    if symbols.numel() == 0 or torch.any(symbols < 0) or torch.any(symbols > 15):
        raise ValueError("TraceableSpeech symbols must be integers in [0, 15].")
    shifts = torch.tensor([3, 2, 1, 0], device=symbols.device, dtype=torch.long)
    return ((symbols.unsqueeze(-1) >> shifts) & 1).reshape(
        *symbols.shape[:-1], symbols.shape[-1] * 4
    )


def traceable_bits_to_symbols(bits: torch.Tensor) -> torch.Tensor:
    """Pack binary bits into the detector's native hexadecimal symbols."""

    bits = torch.as_tensor(bits).long()
    if bits.numel() == 0 or bits.shape[-1] % 4 != 0:
        raise ValueError(
            "TraceableSpeech binary messages must contain a multiple of four bits."
        )
    if torch.any((bits != 0) & (bits != 1)):
        raise ValueError("TraceableSpeech binary messages may contain only 0/1.")
    weights = torch.tensor([8, 4, 2, 1], device=bits.device, dtype=torch.long)
    grouped = bits.reshape(*bits.shape[:-1], bits.shape[-1] // 4, 4)
    return (grouped * weights).sum(dim=-1)


def watermark_sign_from_record(audio_tokenizer, record, device):
    """Return the native detector target while accepting old and new JSON."""

    backend = getattr(audio_tokenizer, "watermark_backend", None)
    if backend != "traceablespeech":
        return torch.tensor(record["watermark_bits"], dtype=torch.long, device=device)

    if record.get("watermark_symbols") is not None:
        return torch.tensor(
            record["watermark_symbols"], dtype=torch.long, device=device
        )

    stored = torch.tensor(record["watermark_bits"], dtype=torch.long, device=device)
    # Legacy TraceableSpeech metadata mislabeled four native hexadecimal
    # symbols as ``watermark_bits``.  New metadata stores 16 real bits.
    if stored.shape[-1] == 4:
        return stored
    return traceable_bits_to_symbols(stored)


def _get_detection_stats(audio_tokenizer, waveform, watermark_sign):
    """Return probability and true binary bit accuracy for both backends."""

    detection = audio_tokenizer.detect_watermark(waveform)
    if detection is None:
        raise RuntimeError("Watermark detector returned no result.")

    detect_prob, sign_pred, _ = detection
    predicted_native = sign_pred.detach().long()
    target_native = watermark_sign.to(predicted_native.device).long()
    is_traceable = (
        getattr(audio_tokenizer, "watermark_backend", None) == "traceablespeech"
    )
    if is_traceable:
        predicted_message = traceable_symbols_to_bits(predicted_native)
        target_message = traceable_symbols_to_bits(target_native)
        predicted_symbols = predicted_native.detach().cpu().int().tolist()
    else:
        predicted_message = predicted_native
        target_message = target_native
        predicted_symbols = None

    predicted = predicted_message.reshape(-1)
    target = target_message.reshape(-1)
    if predicted.numel() != target.numel():
        raise ValueError(
            "Detector/message size mismatch: "
            f"predicted={predicted.numel()}, target={target.numel()}"
        )

    bits_total = target.numel()
    bits_correct = int((predicted == target).sum().item())
    return {
        "prob": float(torch.as_tensor(detect_prob).float().mean().item()),
        "bit_acc": bits_correct / bits_total,
        "bits_correct": bits_correct,
        "bits_total": bits_total,
        "bit_accuracy_unit": "binary_bit",
        "predicted_bits": predicted_message.detach().cpu().int().tolist(),
        "predicted_symbols": predicted_symbols,
    }


def _new_attack_stats(is_codec):
    return {
        "is_codec": is_codec,
        "count": 0,
        "wm_bit_acc_sum": 0.0,
        "wm_prob_sum": 0.0,
        "ori_bit_acc_sum": 0.0,
        "ori_prob_sum": 0.0,
        "recon_bit_acc_sum": 0.0,
        "recon_prob_sum": 0.0,
        "wm_bits_correct": 0,
        "wm_bits_total": 0,
        "error_count": 0,
        "errors": [],
        "disabled_error": None,
    }


def _record_attack_error(stats, message):
    stats["error_count"] += 1
    if len(stats["errors"]) < 5:
        stats["errors"].append(message)


def _load_generated_audio(path, target_sr, device):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = AF.resample(waveform, sample_rate, target_sr)
    return waveform.unsqueeze(0).to(device)


def evaluate_voicemark_attacks(
    audio_tokenizer,
    output_dir,
    wm_records,
    attack_specs,
    sample_rate,
    device,
):
    """Evaluate all generated pairs using VoiceMark valid.py's statistics."""

    attack_stats = {
        name: _new_attack_stats(is_codec)
        for name, _, is_codec in attack_specs
    }
    active_codec_family = None

    try:
        for attack_name, attack_fn, is_codec in attack_specs:
            stats = attack_stats[attack_name]
            codec_family = next(
                (key for key in CODEC_KEYWORDS if attack_name.startswith(key)),
                None,
            )

            # EnCodec reuses one model for its three bandwidths. DAC and SNAC
            # use a different checkpoint per row, so release before each row.
            if is_codec and (
                codec_family != active_codec_family or codec_family != "Encodec"
            ):
                release_codec_models()
            active_codec_family = codec_family if is_codec else None

            logging.info(f"Evaluate attack: {attack_name}")
            for record in wm_records:
                record.setdefault("attacks", {})
                watermark_sign = watermark_sign_from_record(
                    audio_tokenizer, record, device
                )
                clean_path = Path(output_dir) / record["clean"]
                wm_path = Path(output_dir) / record["watermarked"]

                try:
                    decoded_wm = _load_generated_audio(
                        wm_path, sample_rate, device
                    )
                    decoded_clean = _load_generated_audio(
                        clean_path, sample_rate, device
                    )
                    attacked_wm = _unwrap_attacked_audio(
                        attack_fn(decoded_wm.clone())
                    )
                    attacked_clean = _unwrap_attacked_audio(
                        attack_fn(decoded_clean.clone())
                    )

                    wm_result = _get_detection_stats(
                        audio_tokenizer, attacked_wm, watermark_sign
                    )
                    clean_result = _get_detection_stats(
                        audio_tokenizer, attacked_clean, watermark_sign
                    )
                except CodecAttackError as exc:
                    message = str(exc)
                    stats["disabled_error"] = message
                    _record_attack_error(stats, message)
                    logging.error(f"Skip {attack_name}: {message}")
                    for pending_record in wm_records:
                        pending_record.setdefault("attacks", {}).setdefault(
                            attack_name, {"error": message}
                        )
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    message = f"CUDA out of memory: {exc}"
                    stats["disabled_error"] = message
                    _record_attack_error(stats, message)
                    logging.error(f"Skip remaining {attack_name}: {message}")
                    for pending_record in wm_records:
                        pending_record.setdefault("attacks", {}).setdefault(
                            attack_name, {"error": message}
                        )
                    release_codec_models()
                    break
                except Exception as exc:  # noqa: BLE001
                    message = f"sample {record['index']}: {exc}"
                    _record_attack_error(stats, message)
                    record["attacks"][attack_name] = {"error": str(exc)}
                    logging.exception(f"Attack {attack_name} failed on {message}")
                    continue

                # infer.py has no pre-codec source waveform. Its clean decoder
                # output is therefore the explicitly documented Ori/Rec alias.
                ori_result = clean_result
                recon_result = clean_result
                detect_acc = (
                    wm_result["prob"] + 1.0 - ori_result["prob"]
                ) / 2.0

                stats["count"] += 1
                stats["wm_bit_acc_sum"] += wm_result["bit_acc"]
                stats["wm_prob_sum"] += wm_result["prob"]
                stats["ori_bit_acc_sum"] += ori_result["bit_acc"]
                stats["ori_prob_sum"] += ori_result["prob"]
                stats["recon_bit_acc_sum"] += recon_result["bit_acc"]
                stats["recon_prob_sum"] += recon_result["prob"]
                stats["wm_bits_correct"] += wm_result["bits_correct"]
                stats["wm_bits_total"] += wm_result["bits_total"]

                record["attacks"][attack_name] = {
                    # Backward-compatible fields from the earlier evaluator.
                    "accuracy": wm_result["bit_acc"],
                    "bits_correct": wm_result["bits_correct"],
                    "bits_total": wm_result["bits_total"],
                    "probability": wm_result["prob"],
                    # Full VoiceMark valid.py row for this sample.
                    "detect_acc": detect_acc,
                    "wm_bit_acc": wm_result["bit_acc"],
                    "wm_prob": wm_result["prob"],
                    "ori_bit_acc": ori_result["bit_acc"],
                    "ori_prob": ori_result["prob"],
                    "recon_bit_acc": recon_result["bit_acc"],
                    "recon_prob": recon_result["prob"],
                    "predicted_bits": wm_result["predicted_bits"],
                    "predicted_symbols": wm_result["predicted_symbols"],
                    "bit_accuracy_unit": "binary_bit",
                }

                if attack_name == "Clean (Identity)":
                    record.update(
                        {
                            "detect_prob": wm_result["prob"],
                            "accuracy": wm_result["bit_acc"],
                            "bits_correct": wm_result["bits_correct"],
                            "bits_total": wm_result["bits_total"],
                            "predicted_bits": wm_result["predicted_bits"],
                            "predicted_symbols": wm_result["predicted_symbols"],
                            "bit_accuracy_unit": "binary_bit",
                        }
                    )
    finally:
        release_codec_models()

    return attack_stats


def summarize_attacks(attack_stats):
    """Convert running sums into the same row fields as VoiceMark valid.py."""

    summary_report = {}
    for attack_name, stats in attack_stats.items():
        count = stats["count"]
        if count:
            row = {
                "wm_bit_acc": stats["wm_bit_acc_sum"] / count,
                "wm_prob": stats["wm_prob_sum"] / count,
                "ori_bit_acc": stats["ori_bit_acc_sum"] / count,
                "ori_prob": stats["ori_prob_sum"] / count,
                "recon_bit_acc": stats["recon_bit_acc_sum"] / count,
                "recon_prob": stats["recon_prob_sum"] / count,
                "detect_acc": (
                    stats["wm_prob_sum"] + count - stats["ori_prob_sum"]
                )
                / (2 * count),
            }
        else:
            row = {
                "wm_bit_acc": None,
                "wm_prob": None,
                "ori_bit_acc": None,
                "ori_prob": None,
                "recon_bit_acc": None,
                "recon_prob": None,
                "detect_acc": None,
            }
        row.update(
            {
                "accuracy": row["wm_bit_acc"],
                "bits_correct": int(stats["wm_bits_correct"]),
                "bits_total": int(stats["wm_bits_total"]),
                "bit_accuracy_unit": "binary_bit",
                "bits_per_message": 16,
                "count": int(count),
                "is_codec": bool(stats["is_codec"]),
                "error_count": int(stats["error_count"]),
                "errors": list(stats["errors"]),
                "disabled_error": stats["disabled_error"],
            }
        )
        summary_report[attack_name] = row
    return summary_report


def _format_metric(value, width):
    if value is None:
        return f"{'N/A':<{width}}"
    return f"{value:<{width}.4f}"


def make_voicemark_table(checkpoint, summary_report, quality_metrics):
    """Render the valid.py table, widening Attack Type for codec names."""

    clean_name = "Clean (Identity)"
    codec_names = []
    other_names = []
    for name, stats in summary_report.items():
        if name == clean_name:
            continue
        if stats["is_codec"] or any(key in name for key in CODEC_KEYWORDS):
            codec_names.append(name)
        else:
            other_names.append(name)

    ordered_names = []
    if clean_name in summary_report:
        ordered_names.append(clean_name)
    ordered_names.extend(sorted(other_names))
    attack_width = max(
        [22, len("Attack Type")] + [len(name) for name in summary_report]
    )
    widths = [attack_width, 12, 12, 10, 10, 10, 10, 10]
    headers = [
        "Attack Type",
        "Detect ACC",
        "WM Bit Acc",
        "WM Prob",
        "Ori B.Acc",
        "Ori Prob",
        "Rec B.Acc",
        "Rec Prob",
    ]
    header = " | ".join(
        f"{label:<{width}}" for label, width in zip(headers, widths)
    )
    line_width = len(header)
    lines = [
        "=" * line_width,
        f"测试模型: {checkpoint}",
        "=" * line_width,
        header,
        "-" * line_width,
    ]

    def append_row(name):
        stats = summary_report[name]
        values = [
            stats["detect_acc"],
            stats["wm_bit_acc"],
            stats["wm_prob"],
            stats["ori_bit_acc"],
            stats["ori_prob"],
            stats["recon_bit_acc"],
            stats["recon_prob"],
        ]
        cells = [f"{name:<{attack_width}}"]
        cells.extend(
            _format_metric(value, width)
            for value, width in zip(values, widths[1:])
        )
        lines.append(" | ".join(cells))

    for name in ordered_names:
        append_row(name)
    lines.append("-" * line_width)
    for name in sorted(codec_names):
        append_row(name)
    lines.append("-" * line_width)
    lines.append(
        "Bit Acc 定义: 统一按 16 个二进制 bit 逐位匹配；"
        "TraceableSpeech 的 4 个 0-15 符号按 MSB-first 展开。"
    )
    lines.append(
        "注: 本脚本没有独立的 pre-codec 原始波形；"
        "Ori 与 Rec 两组列均使用 *_clean.wav（clean reconstruction）。"
    )

    if quality_metrics:
        avg_pesq = float(np.mean([item["pesq_wb"] for item in quality_metrics]))
        avg_stoi = float(np.mean([item["stoi"] for item in quality_metrics]))
        avg_si_snr = float(np.mean([item["si_snr"] for item in quality_metrics]))
        lines.extend(
            [
                "Transparency (vs Reconstructed - Ref Recon):",
                f"  - Avg PESQ (WB): {avg_pesq:.4f}",
                f"  - Avg STOI:      {avg_stoi:.4f}",
                f"  - Avg SI-SNR:    {avg_si_snr:.4f} dB",
            ]
        )
    lines.append("=" * line_width)
    return "\n".join(lines)


@torch.no_grad()
def main():
    args = get_args()
    timing_enabled = args.runtime_timing_json is not None
    runtime_rows = []
    runtime_skipped = []
    runtime_setup = {}
    text_tokenizer = TextTokenizer(backend=args.text_extractor)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    if timing_enabled:
        (model, text_tokens), model_load_seconds = timed_call(
            device, lambda: load_model(args.checkpoint, device)
        )
        runtime_setup["valle_model_load_seconds"] = model_load_seconds
    else:
        model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)

    watermark_backend = args.watermark_backend
    if args.ts_enable and watermark_backend == "encodec":
        watermark_backend = "traceablespeech"

    ts_config = Path(args.ts_checkpoint_file).parent / "config.json"
    def make_audio_tokenizer():
        return AudioTokenizer(
            enable_ts=args.ts_enable,
            ts_checkpoint=args.ts_checkpoint_file,
            ts_config=str(ts_config),
            watermark_backend=watermark_backend,
            voicemark_root=args.voicemark_root,
            voicemark_config=args.voicemark_config,
            voicemark_st_checkpoint=args.voicemark_st_checkpoint,
            voicemark_checkpoint=args.voicemark_checkpoint,
            voicemark_embed_vq1=args.voicemark_embed_vq1,
        )

    if timing_enabled:
        audio_tokenizer, tokenizer_init_seconds = timed_call(
            device, make_audio_tokenizer
        )
        runtime_setup["audio_tokenizer_init_seconds"] = tokenizer_init_seconds
    else:
        audio_tokenizer = make_audio_tokenizer()
    audio_sample_rate = audio_tokenizer.sample_rate
    if timing_enabled:
        watermark_available, watermark_load_seconds = timed_call(
            device, lambda: audio_tokenizer.has_watermark_decoder
        )
        runtime_setup["watermark_backend_load_seconds"] = watermark_load_seconds
        if not watermark_available:
            raise RuntimeError(
                f"Runtime benchmark requires an available watermark backend: "
                f"{watermark_backend}"
            )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    seed_tts_items = None
    fixed_seed_audio_prompts = None
    if args.seed_tts_manifest is not None:
        if args.continual:
            raise ValueError(
                "Seed-TTS-Eval zero-shot TTS is incompatible with --continual."
            )
        seed_tts_items = load_seed_tts_manifest(
            args.seed_tts_manifest,
            num_samples=args.seed_tts_num_samples,
        )
        logging.info(
            f"Loaded {len(seed_tts_items)} Seed-TTS-Eval rows from "
            f"{args.seed_tts_manifest}"
        )

        fixed_audio = args.seed_tts_fixed_prompt_audio
        fixed_text = args.seed_tts_fixed_prompt_text
        if (fixed_audio is None) != (fixed_text is None):
            raise ValueError(
                "--seed-tts-fixed-prompt-audio and "
                "--seed-tts-fixed-prompt-text must be specified together."
            )
        if fixed_audio is not None:
            fixed_audio = fixed_audio.expanduser().resolve()
            if not fixed_audio.is_file():
                raise FileNotFoundError(
                    f"Fixed Seed-TTS prompt audio not found: {fixed_audio}"
                )
            if not fixed_text.strip():
                raise ValueError("--seed-tts-fixed-prompt-text cannot be empty.")
            for item in seed_tts_items:
                item["manifest_prompt_text"] = item["prompt_text"]
                item["manifest_prompt_wav"] = item["prompt_wav"]
                item["prompt_text"] = fixed_text.strip()
                item["prompt_wav"] = str(fixed_audio)
            logging.info(
                "Use one fixed prompt for all Seed-TTS targets: "
                f"audio={fixed_audio} | text={fixed_text.strip()}"
            )
            if timing_enabled:
                fixed_frames, fixed_prompt_seconds = timed_call(
                    device,
                    lambda: tokenize_audio(audio_tokenizer, str(fixed_audio)),
                )
                runtime_setup["fixed_prompt_encode_seconds"] = fixed_prompt_seconds
            else:
                fixed_frames = tokenize_audio(audio_tokenizer, str(fixed_audio))
            fixed_seed_audio_prompts = (
                fixed_frames[0][0].transpose(2, 1).to(device)
            )
            logging.info(
                "Fixed prompt tokenized once and cached: "
                f"shape={tuple(fixed_seed_audio_prompts.shape)}"
            )

    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts and seed_tts_items is None:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], audio_sample_rate
                )

            audio_prompts.append(encoded_frames[0][0])

        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        # import pdb; pdb.set_trace()
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    texts = None
    # accumulators for batch metrics
    quality_metrics = []
    wm_records = []
    attack_specs = build_voicemark_valid_attacks(audio_sample_rate)

    if seed_tts_items is not None:
        texts = seed_tts_items
    elif args.text_file:
        with open(args.text_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif os.path.isfile(args.text):  # legacy demo file with tab-separated fields
        # https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/prepare.py
        with open(args.text) as f:
            for line in f:
                fields = line.strip().split("\t")
                assert len(fields) == 4
                prompt_text, prompt_audio, text, audio_path = fields
                logging.info(f"synthesize text: {text}")
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text} {text}".strip()
                        )
                    ]
                )
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text}".strip()
                        )
                    ]
                )

                audio_prompts = tokenize_audio(audio_tokenizer, prompt_audio)
                audio_prompts = audio_prompts[0][0].transpose(2, 1).to(device)

                # synthesis
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

                samples = audio_tokenizer.decode(
                    [(encoded_frames.transpose(2, 1), None)]
                )
                # store
                torchaudio.save(audio_path, samples[0].cpu(), audio_sample_rate)
        return
    else:
        texts = args.text.split("|")

    runtime_loop_start = None
    if timing_enabled:
        synchronize_device(device)
        runtime_loop_start = time.perf_counter()
    for n, text_item in enumerate(texts):
        prompt_audio_encode_seconds = 0.0
        seed_tts_item = text_item if isinstance(text_item, dict) else None
        if seed_tts_item is not None:
            text = seed_tts_item["target_text"]
            current_text_prompt = seed_tts_item["prompt_text"]
            if fixed_seed_audio_prompts is not None:
                current_audio_prompts = fixed_seed_audio_prompts
            else:
                if timing_enabled:
                    prompt_frames, prompt_audio_encode_seconds = timed_call(
                        device,
                        lambda: tokenize_audio(
                            audio_tokenizer, seed_tts_item["prompt_wav"]
                        ),
                    )
                else:
                    prompt_frames = tokenize_audio(
                        audio_tokenizer, seed_tts_item["prompt_wav"]
                    )
                current_audio_prompts = (
                    prompt_frames[0][0].transpose(2, 1).to(device)
                )
            output_stem = seed_tts_item["utterance_id"]
            logging.info(
                f"Seed-TTS-Eval {n + 1}/{len(texts)}: {output_stem} | "
                f"prompt={seed_tts_item['prompt_wav']} | target={text}"
            )
        else:
            text = text_item
            current_text_prompt = text_prompts
            current_audio_prompts = audio_prompts
            output_stem = str(n)
            logging.info(f"synthesize text: {text}")

        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    text_tokenizer,
                    text=f"{current_text_prompt} {text}".strip(),
                )
            ]
        )

        # synthesis
        valle_token_generation_seconds = 0.0
        if args.continual:
            assert text == ""
            if timing_enabled:
                encoded_frames, valle_token_generation_seconds = timed_call(
                    device,
                    lambda: model.continual(
                        text_tokens.to(device),
                        text_tokens_lens.to(device),
                        current_audio_prompts,
                    ),
                )
            else:
                encoded_frames = model.continual(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    current_audio_prompts,
                )
        else:
            enroll_x_lens = None
            if current_text_prompt:
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer,
                            text=current_text_prompt.strip(),
                        )
                    ]
                )
            if timing_enabled:
                encoded_frames, valle_token_generation_seconds = timed_call(
                    device,
                    lambda: model.inference(
                        text_tokens.to(device),
                        text_tokens_lens.to(device),
                        current_audio_prompts,  # [B, 8, T/320]
                        enroll_x_lens=enroll_x_lens,
                        top_k=args.top_k,
                        temperature=args.temperature,
                    ),
                )
            else:
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    current_audio_prompts,  # [B, 8, T/320]
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )
        # import pdb; pdb.set_trace()
        if isinstance(current_audio_prompts, torch.Tensor):
            # Guard against empty AR output (early EOS). The time dimension is dim=1 for encoded_frames [B, T, n_q].
            time_len = encoded_frames.shape[1] if encoded_frames.dim() >= 2 else 0
            if time_len == 0 or encoded_frames.numel() == 0:
                logging.warning(f"Skip utt {n}: empty encoded frames (early EOS).")
                if timing_enabled:
                    runtime_skipped.append(
                        {
                            "index": n,
                            "utterance_id": output_stem,
                            "reason": "empty_encoded_frames",
                            "valle_token_generation_seconds": valle_token_generation_seconds,
                        }
                    )
                continue

            encoded_for_decode = [(encoded_frames.transpose(2, 1), None)]

            # Extra guard: after transpose we expect shape [B, n_q, T]; check T.
            seq_len = encoded_for_decode[0][0].shape[2] if encoded_for_decode[0][0].dim() >= 3 else 0
            if seq_len == 0:
                logging.warning(f"Skip utt {n}: zero-length sequence after transpose (early EOS).")
                if timing_enabled:
                    runtime_skipped.append(
                        {
                            "index": n,
                            "utterance_id": output_stem,
                            "reason": "zero_length_after_transpose",
                            "valle_token_generation_seconds": valle_token_generation_seconds,
                        }
                    )
                continue

            if timing_enabled:
                watermark_sign, watermark_message_seconds = timed_call(
                    device,
                    lambda: audio_tokenizer.random_watermark(
                        encoded_frames.size(0)
                    ),
                )
                decoded_clean, clean_decode_seconds = timed_call(
                    device,
                    lambda: audio_tokenizer.decode(
                        encoded_for_decode, watermark_sign=None
                    ),
                )
                decoded_wm = None
                watermarked_decode_seconds = 0.0
                watermark_extract_seconds = 0.0
                watermark_extract_available = False
                if watermark_sign is not None:
                    decoded_wm, watermarked_decode_seconds = timed_call(
                        device,
                        lambda: audio_tokenizer.decode(
                            encoded_for_decode, watermark_sign=watermark_sign
                        ),
                    )
                    detection_result, watermark_extract_seconds = timed_call(
                        device,
                        lambda: audio_tokenizer.detect_watermark(decoded_wm),
                    )
                    watermark_extract_available = detection_result is not None
            else:
                watermark_sign = audio_tokenizer.random_watermark(
                    encoded_frames.size(0)
                )
                # decode both reference (zero watermark) and watermarked in one call
                decoded_clean, decoded_wm = audio_tokenizer.decode_pair(
                    encoded_for_decode, watermark_sign=watermark_sign
                )

            clean_path = Path(args.output_dir) / f"{output_stem}_clean.wav"
            torchaudio.save(
                str(clean_path),
                decoded_clean[0].cpu(),
                audio_sample_rate,
                encoding="PCM_F",
                bits_per_sample=32,
            )

            if decoded_wm is not None:
                wm_path = Path(args.output_dir) / f"{output_stem}_wm.wav"
                torchaudio.save(
                    str(wm_path),
                    decoded_wm[0].cpu(),
                    audio_sample_rate,
                    encoding="PCM_F",
                    bits_per_sample=32,
                )
                wm_record = {
                    "index": n,
                    "utterance_id": output_stem,
                    "text": text,
                    "sample_rate": audio_sample_rate,
                    "clean": clean_path.name,
                    "watermarked": wm_path.name,
                }
                if timing_enabled:
                    audio_duration_seconds = (
                        decoded_clean.shape[-1] / audio_sample_rate
                    )
                    timing_row = {
                        "index": n,
                        "utterance_id": output_stem,
                        "clean": clean_path.name,
                        "watermarked": wm_path.name,
                        "audio_duration_seconds": audio_duration_seconds,
                        "prompt_audio_encode_seconds": prompt_audio_encode_seconds,
                        "valle_token_generation_seconds": valle_token_generation_seconds,
                        "clean_decode_seconds": clean_decode_seconds,
                        "clean_synthesis_seconds": (
                            prompt_audio_encode_seconds
                            + valle_token_generation_seconds
                            + clean_decode_seconds
                        ),
                        "watermark_message_seconds": watermark_message_seconds,
                        "watermarked_decode_seconds": watermarked_decode_seconds,
                        "watermark_embedding_path_seconds": (
                            watermark_message_seconds
                            + watermarked_decode_seconds
                        ),
                        "watermark_incremental_over_clean_seconds": (
                            watermark_message_seconds
                            + watermarked_decode_seconds
                            - clean_decode_seconds
                        ),
                        "watermarked_synthesis_seconds": (
                            prompt_audio_encode_seconds
                            + valle_token_generation_seconds
                            + watermark_message_seconds
                            + watermarked_decode_seconds
                        ),
                        "watermark_extract_seconds": watermark_extract_seconds,
                        "watermark_extract_available": (
                            watermark_extract_available
                        ),
                    }
                    runtime_rows.append(timing_row)
                    wm_record["runtime_timing"] = timing_row
                if seed_tts_item is not None:
                    primary_source = (
                        wm_path
                        if args.seed_tts_primary_output == "watermarked"
                        else clean_path
                    )
                    primary_path = Path(args.output_dir) / f"{output_stem}.wav"
                    _link_seed_tts_primary_output(primary_source, primary_path)
                    wm_record["seed_tts_eval"] = {
                        "prompt_text": seed_tts_item["prompt_text"],
                        "prompt_wav": seed_tts_item["prompt_wav"],
                        "manifest_prompt_text": seed_tts_item.get(
                            "manifest_prompt_text"
                        ),
                        "manifest_prompt_wav": seed_tts_item.get(
                            "manifest_prompt_wav"
                        ),
                        "target_text": seed_tts_item["target_text"],
                        "primary_output": primary_path.name,
                        "primary_variant": args.seed_tts_primary_output,
                    }
                if watermark_sign is not None:
                    native_message = watermark_sign.detach().long()
                    if watermark_backend == "traceablespeech":
                        wm_record["watermark_symbols"] = (
                            native_message.cpu().int().tolist()
                        )
                        wm_record["watermark_bits"] = (
                            traceable_symbols_to_bits(native_message)
                            .cpu()
                            .int()
                            .tolist()
                        )
                    else:
                        wm_record["watermark_bits"] = (
                            native_message.cpu().int().tolist()
                        )
                    wm_record["bit_accuracy_unit"] = "binary_bit"

                if args.skip_post_generation_eval:
                    wm_record["attacks"] = {}
                else:
                    quality = compute_quality(
                        decoded_clean, decoded_wm, audio_sample_rate
                    )
                    quality_metrics.append(quality)
                    wm_record.update(
                        {
                            "pesq_wb_clean_vs_wm": quality["pesq_wb"],
                            "stoi_clean_vs_wm": quality["stoi"],
                            "si_snr_clean_vs_wm": quality["si_snr"],
                            "attacks": {},
                        }
                    )
                    logging.info(
                        "Quality(clean reconstruction vs watermark): "
                        f"PESQ={quality['pesq_wb']:.4f}, "
                        f"STOI={quality['stoi']:.4f}, "
                        f"SI-SNR={quality['si_snr']:.4f} dB"
                    )
                wm_records.append(wm_record)
            else:
                logging.warning(
                    "Watermark backend unavailable; metrics skipped."
                )
        else:  # Transformer
            pass

    runtime_loop_seconds = None
    if timing_enabled:
        synchronize_device(device)
        runtime_loop_seconds = time.perf_counter() - runtime_loop_start
        runtime_summary = summarize_runtime_timing(
            args=args,
            watermark_backend=watermark_backend,
            audio_sample_rate=audio_sample_rate,
            rows=runtime_rows,
            skipped=runtime_skipped,
            setup=runtime_setup,
            loop_wall_seconds=runtime_loop_seconds,
            device=device,
        )
        timing_path = args.runtime_timing_json.expanduser().resolve()
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            json.dumps(runtime_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logging.info(f"Runtime timing JSON: {timing_path}")

    if args.skip_post_generation_eval:
        return

    # The synthesis model is no longer needed while codec attacks run. Releasing
    # it leaves room for the DAC/SNAC checkpoints without touching the detector.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if wm_records:
        logging.info(
            "Ori/Rec table columns both use the generated clean reconstruction; "
            "VALL-E inference has no separate pre-codec source waveform."
        )
        attack_stats = evaluate_voicemark_attacks(
            audio_tokenizer=audio_tokenizer,
            output_dir=args.output_dir,
            wm_records=wm_records,
            attack_specs=attack_specs,
            sample_rate=audio_sample_rate,
            device=device,
        )
    else:
        attack_stats = {
            name: _new_attack_stats(is_codec)
            for name, _, is_codec in attack_specs
        }

    summary_report = summarize_attacks(attack_stats)
    table = make_voicemark_table(
        args.checkpoint, summary_report, quality_metrics
    )
    print(f"\n{table}")
    table_path = Path(args.output_dir) / "watermark_validation_table.txt"
    table_path.write_text(table + "\n", encoding="utf-8")

    for record in wm_records:
        wm_path = Path(args.output_dir) / record["watermarked"]
        with open(wm_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    clean_stats = summary_report.get("Clean (Identity)", {})
    total_attack_correct = sum(
        stats["bits_correct"] for stats in summary_report.values()
    )
    total_attack_bits = sum(
        stats["bits_total"] for stats in summary_report.values()
    )
    avg_quality = {
        "pesq_wb": (
            float(np.mean([item["pesq_wb"] for item in quality_metrics]))
            if quality_metrics
            else None
        ),
        "stoi": (
            float(np.mean([item["stoi"] for item in quality_metrics]))
            if quality_metrics
            else None
        ),
        "si_snr": (
            float(np.mean([item["si_snr"] for item in quality_metrics]))
            if quality_metrics
            else None
        ),
    }
    summary = {
        "count": len(wm_records),
        "sample_rate": audio_sample_rate,
        "checkpoint": args.checkpoint,
        "watermark_backend": watermark_backend,
        "bit_accuracy_unit": "binary_bit",
        "bit_accuracy_version": 2,
        "bits_per_message": 16,
        "native_message_format": (
            "4 hexadecimal symbols expanded MSB-first to 16 binary bits"
            if watermark_backend == "traceablespeech"
            else "16 binary bits"
        ),
        "seed_tts_eval": (
            {
                "manifest": str(args.seed_tts_manifest.expanduser().resolve()),
                "requested_samples": args.seed_tts_num_samples,
                "loaded_samples": len(seed_tts_items),
                "primary_variant": args.seed_tts_primary_output,
                "primary_pattern": "<utterance_id>.wav",
                "fixed_prompt_audio": (
                    str(args.seed_tts_fixed_prompt_audio.expanduser().resolve())
                    if args.seed_tts_fixed_prompt_audio is not None
                    else None
                ),
                "fixed_prompt_text": args.seed_tts_fixed_prompt_text,
            }
            if seed_tts_items is not None
            else None
        ),
        "baseline_mapping": {
            "wm": "generated watermarked reconstruction (*_wm.wav)",
            "ori": "generated clean reconstruction (*_clean.wav)",
            "recon": "generated clean reconstruction (*_clean.wav)",
            "note": (
                "Ori and Rec intentionally share decoded_clean because infer.py "
                "has no independent pre-codec source waveform."
            ),
        },
        # Keep the earlier top-level quality keys for downstream scripts.
        "avg_pesq_wb_clean_vs_wm": avg_quality["pesq_wb"],
        "avg_stoi_clean_vs_wm": avg_quality["stoi"],
        "avg_si_snr_clean_vs_wm": avg_quality["si_snr"],
        "quality_clean_vs_wm": avg_quality,
        "wm_bit_accuracy": clean_stats.get("wm_bit_acc"),
        "wm_bits_correct": int(clean_stats.get("bits_correct", 0)),
        "wm_bits_total": int(clean_stats.get("bits_total", 0)),
        "attack_overall_accuracy": (
            total_attack_correct / total_attack_bits
            if total_attack_bits > 0
            else None
        ),
        "attack_bits_correct": int(total_attack_correct),
        "attack_bits_total": int(total_attack_bits),
        "attacks": summary_report,
        "table_file": table_path.name,
        "details": wm_records,
    }
    with open(
        Path(args.output_dir) / "watermark_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
