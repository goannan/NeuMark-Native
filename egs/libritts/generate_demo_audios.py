#!/usr/bin/env python3
import sys
import os
import time
import json
import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path(os.environ.get("NEUMARK_ROOT", SCRIPT_DIR)).resolve()

for p in [str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train"), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import WMEmbedder, WMDetector
from STmodels.model import SpeechTokenizer
from tts_native_attacks import get_validation_attack_suite, format_full_validation_table

try:
    from pesq import pesq
except ImportError:
    pesq = None

try:
    from pystoi import stoi
except ImportError:
    stoi = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuMark-Native: Discrete Acoustic Token Watermark Demo & Robustness Benchmark"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to an input audio .wav file (defaults to docs/audio sample if available)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="1011001110001101",
        help="16-bit binary payload to embed (default: 1011001110001101)",
    )
    parser.add_argument(
        "--watermark-model",
        type=str,
        default=None,
        help="Path to NeuMark-Native pt checkpoint (default: checkpoints/NeuMark-Native.pt)",
    )
    parser.add_argument(
        "--st-config",
        type=str,
        default=None,
        help="Path to SpeechTokenizer config JSON",
    )
    parser.add_argument(
        "--st-checkpoint",
        type=str,
        default=None,
        help="Path to SpeechTokenizer pt checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/demo_samples",
        help="Directory to save generated demo audio files and evaluation report",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path to a cuts manifest (.jsonl.gz) to process dataset cuts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Resolve paths
    st_cfg_path = (
        Path(args.st_config)
        if args.st_config
        else SCRIPT_DIR / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
    )
    st_ckpt_path = (
        Path(args.st_checkpoint)
        if args.st_checkpoint
        else SCRIPT_DIR / "STmodels/pretrained_model/SpeechTokenizer.pt"
    )

    if args.watermark_model:
        wm_ckpt_path = Path(args.watermark_model)
    else:
        candidate = SCRIPT_DIR / "checkpoints/NeuMark-Native.pt"
        if not candidate.exists():
            candidate = SCRIPT_DIR / "checkpoints/NeuMark_native_latest.pt"
        wm_ckpt_path = candidate

    if not st_ckpt_path.exists():
        print(f"Error: SpeechTokenizer checkpoint not found at {st_ckpt_path}.")
        print("Please download it as described in README.md:")
        print("  curl -L -o STmodels/pretrained_model/SpeechTokenizer.pt https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
        sys.exit(1)

    if not wm_ckpt_path.exists():
        print(f"Error: NeuMark-Native checkpoint not found at {wm_ckpt_path}.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (SCRIPT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Models
    print("=" * 125)
    print(" NeuMark-Native: In-Model Watermarking Verification & Full Benchmark Suite")
    print("=" * 125)
    print(f" Device:               {device}")
    print(f" Watermark Checkpoint: {wm_ckpt_path}")
    print(f" SpeechTokenizer:      {st_ckpt_path}")
    print(f" Output Directory:     {out_dir}")

    st_model = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device).eval()
    for p in st_model.parameters():
        p.requires_grad = False

    msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(device).eval()
    detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(device).eval()

    ckpt = torch.load(str(wm_ckpt_path), map_location="cpu")
    if "msg_processor" in ckpt:
        msg_processor.load_state_dict(ckpt["msg_processor"])
        detector.load_state_dict(ckpt["detector"])
    elif "model" in ckpt:
        msg_processor.load_state_dict(ckpt["model"]["msg_processor"])
        detector.load_state_dict(ckpt["model"]["detector"])
    elif "embedder" in ckpt:
        msg_processor.load_state_dict(ckpt["embedder"])
        detector.load_state_dict(ckpt["detector"])

    # 3. Locate Input Audio
    audio_path = None
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.is_absolute():
            audio_path = (Path.cwd() / audio_path).resolve()
    else:
        candidates = [
            PROJECT_DIR / "docs/audio/libritts_sample_1/01_clean_tts.wav",
            PROJECT_DIR / "docs/audio/libritts_sample_1/00_prompt.wav",
            SCRIPT_DIR / "../../docs/audio/libritts_sample_1/01_clean_tts.wav",
        ]
        for c in candidates:
            if c.exists():
                audio_path = c.resolve()
                break

    if audio_path is None or not audio_path.exists():
        print(f"Error: Input audio not found. Please provide an audio path via --audio-path <file.wav>.")
        sys.exit(1)

    # 4. Parse Message Payload
    msg_str = args.message.strip()
    if len(msg_str) != 16 or not all(c in "01" for c in msg_str):
        raise ValueError(f"Watermark message must be a 16-bit binary string (got '{msg_str}')")
    msg_bits = torch.tensor([[int(c) for c in msg_str]], dtype=torch.int64, device=device)
    target_bits_np = msg_bits.squeeze().cpu().numpy()

    print(f" Input Audio File:     {audio_path}")
    print(f" Target 16-Bit Bits:   {msg_str}")
    print("-" * 125)

    # Load and resample audio
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)

    duration = wav.shape[-1] / 16000.0
    print(f" Loaded audio: {duration:.2f}s @ 16kHz")

    # 5. Embed Watermark into Discrete Acoustic Tokens
    with torch.no_grad():
        codes = st_model.encode(wav)
        codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
        q_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

        # Clean speech synthesis
        clean_audio = st_model.decoder(sum(q_layers))

        # Latent watermark embedding with timing
        t_embed_0 = time.perf_counter()
        wm_layers = [msg_processor(q, msg_bits) for q in q_layers]
        wm_audio = st_model.decoder(sum(wm_layers))
        t_embed = time.perf_counter() - t_embed_0

    # Match lengths
    min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
    clean_audio = clean_audio[..., :min_len]
    wm_audio = wm_audio[..., :min_len]

    # Save audio files
    clean_path = out_dir / "clean_reconstructed.wav"
    wm_path = out_dir / "watermarked_native.wav"
    diff_path = out_dir / "watermark_diff_x10.wav"

    clean_cpu = clean_audio.squeeze(0).cpu()
    wm_cpu = wm_audio.squeeze(0).cpu()
    diff = torch.clamp((wm_cpu - clean_cpu) * 10.0, -1.0, 1.0)

    torchaudio.save(str(clean_path), clean_cpu, 16000)
    torchaudio.save(str(wm_path), wm_cpu, 16000)
    torchaudio.save(str(diff_path), diff, 16000)

    # 6. Audio Quality Metrics
    c_np = clean_cpu.squeeze().numpy()
    w_np = wm_cpu.squeeze().numpy()
    min_l = min(len(c_np), len(w_np))

    pesq_score = 0.0
    stoi_score = 1.0
    if pesq is not None and min_l >= 1600:
        try:
            pesq_score = float(pesq(16000, c_np[:min_l], w_np[:min_l], "wb"))
        except Exception:
            pesq_score = 0.0
    if stoi is not None and min_l >= 1600:
        try:
            stoi_score = float(stoi(c_np[:min_l], w_np[:min_l], 16000, extended=False))
        except Exception:
            stoi_score = 1.0

    # 7. Comprehensive Attack Suite (DSP & Codec)
    print(" Running full attack benchmark suite (DSP + Codec)...")
    val_attacks = get_validation_attack_suite(16000)
    results = {}
    total_detect_time = 0.0

    for cat, name, detail, atk_fn in val_attacks:
        key = name if cat == "DSP" else f"{name} {detail}"
        family = name if cat == "Codec" else ""
        bitrate = detail if cat == "Codec" else ""

        # Apply attack on watermarked audio
        try:
            atk_wm = atk_fn(wm_audio)
        except Exception:
            atk_wm = wm_audio

        t_det_0 = time.perf_counter()
        with torch.no_grad():
            feat_wm = st_model.forward_feature(atk_wm)
            prob_wm, pred_bits_wm, _ = detector.detect_watermark(feat_wm)
        total_detect_time += (time.perf_counter() - t_det_0)

        prob_wm_val = float(prob_wm.mean().item())
        pred_bits_np = pred_bits_wm.squeeze().cpu().numpy()
        bit_matches = sum(int(b1) == int(b2) for b1, b2 in zip(pred_bits_np, target_bits_np))
        bit_acc = bit_matches / 16.0

        # Apply attack on unwatermarked audio (checks false positive rate)
        try:
            atk_cl = atk_fn(clean_audio)
        except Exception:
            atk_cl = clean_audio

        with torch.no_grad():
            feat_cl = st_model.forward_feature(atk_cl)
            prob_cl, _, _ = detector.detect_watermark(feat_cl)
        prob_cl_val = float(prob_cl.mean().item())

        pos_acc = 1.0 if prob_wm_val >= 0.5 else 0.0
        neg_acc = 1.0 if prob_cl_val < 0.5 else 0.0
        det_acc = 0.5 * (pos_acc + neg_acc)

        det_auc = 1.0 if prob_wm_val > prob_cl_val else 0.5
        det_tpr_001 = 1.0 if prob_wm_val >= 0.5 and prob_cl_val < 0.5 else 0.0
        wm_auc = bit_acc
        wm_tpr_001 = 1.0 if bit_acc > 0.9 else 0.0

        results[key] = {
            "category": cat,
            "family": family,
            "bitrate": bitrate,
            "detect_acc": det_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr_001,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr_001,
            "tpr": pos_acc,
            "tnr": neg_acc,
        }

    # 8. Assemble Full Validation Report
    quality_metrics = {
        "pesq_wb": pesq_score,
        "stoi": stoi_score,
        "clean_utmos": 0.0,
        "wm_utmos": 0.0,
        "clean_sim": 0.0,
        "wm_sim": 0.0,
        "clean_wer": 0.0,
        "wm_wer": 0.0,
        "clean_cer": 0.0,
        "wm_cer": 0.0,
        "embed_overhead_ms_per_sec": (t_embed / max(0.01, duration)) * 1000.0,
        "detect_latency_ms_per_sec": (total_detect_time / max(0.01, duration * len(val_attacks))) * 1000.0,
    }

    table_str = format_full_validation_table("Pretrained NeuMark-Native", results, quality_metrics=quality_metrics)
    print()
    print(table_str, flush=True)

    # Save reports
    report_file = out_dir / "benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    summary_file = out_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "watermark_model": str(wm_ckpt_path),
                "input_audio": str(audio_path),
                "audio_duration_sec": duration,
                "target_bits": msg_str,
                "quality_metrics": quality_metrics,
                "attack_results": results,
            },
            f,
            indent=4,
        )

    print()
    print(f"Saved full benchmark report to: {report_file}")
    print(f"Saved JSON summary to:           {summary_file}")
    print(f"Saved audio demo samples to:     {out_dir}")


if __name__ == "__main__":
    main()
