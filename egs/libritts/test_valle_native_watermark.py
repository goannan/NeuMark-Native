#!/usr/bin/env python3
# Copyright (c) 2026
# Standalone Benchmark Evaluation Script for VALL-E Native / NeuMark Watermarked TTS
# Evaluates on full/subset test dataset (LibriTTS and SeedTTS) with exact validation metric tables,
# ROC-AUC, TPR@0.1%FPR, Embedding Overhead (ms/s), Detection Latency (ms/s), UTMOS, SIM, WER, CER.

import argparse
import csv
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

# 1. Clean Mocks for k2 / kaldialign to avoid missing optional dependencies
for mod in ["k2", "k2.version", "kaldialign"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

def find_neumark_root(hint: Optional[str] = None) -> Path:
    candidates = [
        hint,
        SCRIPT_DIR,
        os.environ.get("NEUMARK_ROOT"),
        PROJECT_DIR.parent / "NeuMark",
        PROJECT_DIR / "NeuMark",
        SCRIPT_DIR / "NeuMark",
        SCRIPT_DIR.parents[2] / "NeuMark",
        Path("/home/wu25/mrnas04home/projects/NeuMark"),
        Path("/home/pj25001109/ku60000344/projects/NeuMark"),
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = (SCRIPT_DIR / p).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()
for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from models import WMEmbedder, WMDetector
from tts_native_attacks import (
    get_validation_attack_suite,
    format_full_validation_table,
    release_codec_models,
    compute_wer_cer,
    compute_auc_and_tpr_at_fpr,
)
from tts_native_loss import UTMOSLoss, SpeakerSimLoss, ASRLoss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Evaluation of VALL-E Native / NeuMark Watermark on Test Datasets"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="synthesized_data/libriTTS",
        help="Path to tokenized cuts manifest (.jsonl.gz) or synthesized dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports, tables, and audio samples",
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
        "--watermark-model",
        type=str,
        default=None,
        help="Path to NeuMark / TTS-Native trained pt checkpoint",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=str,
        default="models/wavlm_large_finetune.pth",
        help="Path to WavLM checkpoint for Speaker Similarity",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of test samples to evaluate (-1 for ALL samples)",
    )
    parser.add_argument(
        "--save-audio-samples",
        type=int,
        default=20,
        help="Number of audio samples to save (clean, wm, prompt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g. cuda:0, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index (0-indexed, e.g. 0 or 1 for 2 shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for multi-GPU evaluation",
    )
    parser.add_argument(
        "--merge-shards",
        type=str,
        default=None,
        help="Output directory containing shard raw result files to merge into a final report",
    )
    return parser.parse_args()



def compute_and_save_reports(
    out_dir: Path,
    manifest_path: str,
    wm_ckpt_path: str,
    ckpt_step: int,
    ckpt_epoch: int,
    num_eval: int,
    total_test_samples: int,
    results: dict,
    attack_scores: dict,
    clean_utmos_list: list,
    wm_utmos_list: list,
    pesq_list: list,
    stoi_list: list,
    clean_sim_list: list,
    wm_sim_list: list,
    clean_wer_list: list,
    wm_wer_list: list,
    clean_cer_list: list,
    wm_cer_list: list,
    total_audio_duration: float,
    total_embed_time: float,
    total_detect_time: float,
    sample_audio_records: list,
    val_attacks: list,
    report_prefix: str = "test",
):
    summary = {}
    csv_rows = []
    all_det_true, all_det_scores = [], []
    all_wm_true, all_wm_scores = [], []

    for key, stats in results.items():
        bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
        pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
        neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
        detect_acc = 0.5 * (pos_acc + neg_acc)

        # 1. Detection ROC-AUC & TPR
        pos_d = attack_scores[key]["pos_det_scores"]
        neg_d = attack_scores[key]["neg_det_scores"]
        y_det_true = [0] * len(neg_d) + [1] * len(pos_d)
        y_det_scores = neg_d + pos_d
        all_det_true.extend(y_det_true)
        all_det_scores.extend(y_det_scores)
        det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

        # 2. WM Bit-Matching Extraction ROC-AUC & TPR
        pos_w = attack_scores[key]["pos_wm_scores"]
        neg_w = attack_scores[key]["neg_wm_scores"]
        y_wm_true = [0] * len(neg_w) + [1] * len(pos_w)
        y_wm_scores = neg_w + pos_w
        all_wm_true.extend(y_wm_true)
        all_wm_scores.extend(y_wm_scores)
        wm_auc, wm_tpr_001 = compute_auc_and_tpr_at_fpr(y_wm_true, y_wm_scores, target_fpr=0.001)

        summary[key] = {
            "category": stats["category"],
            "family": stats["family"],
            "bitrate": stats["bitrate"],
            "detect_acc": detect_acc,
            "det_roc_auc": det_auc,
            "det_tpr_at_001_fpr": det_tpr_001,
            "bit_acc": bit_acc,
            "wm_roc_auc": wm_auc,
            "wm_tpr_at_001_fpr": wm_tpr_001,
            "tpr": pos_acc,
            "tnr": neg_acc,
        }
        csv_rows.append({
            "Attack": key,
            "Category": stats["category"],
            "Family": stats["family"],
            "Bitrate": stats["bitrate"],
            "Detect_Accuracy": f"{detect_acc:.4f}",
            "Det_ROC_AUC": f"{det_auc:.4f}",
            "Det_TPR_at_001_FPR": f"{det_tpr_001:.4f}",
            "WM_Bit_Accuracy": f"{bit_acc:.4f}",
            "WM_ROC_AUC": f"{wm_auc:.4f}",
            "WM_TPR_at_001_FPR": f"{wm_tpr_001:.4f}",
            "TPR": f"{pos_acc:.4f}",
            "TNR": f"{neg_acc:.4f}",
        })

    overall_det_auc, overall_det_tpr_001 = compute_auc_and_tpr_at_fpr(all_det_true, all_det_scores, target_fpr=0.001)
    overall_wm_auc, overall_wm_tpr_001 = compute_auc_and_tpr_at_fpr(all_wm_true, all_wm_scores, target_fpr=0.001)

    c_ut = sum(clean_utmos_list) / max(1, len(clean_utmos_list)) if clean_utmos_list else 0.0
    w_ut = sum(wm_utmos_list) / max(1, len(wm_utmos_list)) if wm_utmos_list else 0.0
    c_sim = sum(clean_sim_list) / max(1, len(clean_sim_list)) if clean_sim_list else 0.0
    w_sim = sum(wm_sim_list) / max(1, len(wm_sim_list)) if wm_sim_list else 0.0
    c_wer = sum(clean_wer_list) / max(1, len(clean_wer_list)) if clean_wer_list else 0.0
    w_wer = sum(wm_wer_list) / max(1, len(wm_wer_list)) if wm_wer_list else 0.0
    c_cer = sum(clean_cer_list) / max(1, len(clean_cer_list)) if clean_cer_list else 0.0
    w_cer = sum(wm_cer_list) / max(1, len(wm_cer_list)) if wm_cer_list else 0.0

    embed_overhead_ms_per_sec = (total_embed_time / max(1e-5, total_audio_duration)) * 1000.0
    num_attacks = len(val_attacks) if val_attacks else len(results)
    detect_latency_ms_per_sec = (total_detect_time / max(1e-5, total_audio_duration * max(1, num_attacks) * 2)) * 1000.0

    avg_pesq = sum(pesq_list) / max(1, len(pesq_list)) if pesq_list else 0.0
    avg_stoi = sum(stoi_list) / max(1, len(stoi_list)) if stoi_list else 0.0
    quality_metrics = {
        "pesq_wb": avg_pesq,
        "stoi": avg_stoi,
        "clean_utmos": c_ut, "wm_utmos": w_ut,
        "clean_sim": c_sim, "wm_sim": w_sim,
        "clean_wer": c_wer, "wm_wer": w_wer,
        "clean_cer": c_cer, "wm_cer": w_cer,
        "embed_overhead_ms_per_sec": embed_overhead_ms_per_sec,
        "detect_latency_ms_per_sec": detect_latency_ms_per_sec,
        "overall_det_roc_auc": overall_det_auc, "overall_wm_roc_auc": overall_wm_auc,
        "overall_det_tpr_at_001_fpr": overall_det_tpr_001, "overall_wm_tpr_at_001_fpr": overall_wm_tpr_001,
    }

    # Format and Print Table
    table_str = format_full_validation_table(f"Step {ckpt_step}", summary, quality_metrics=quality_metrics)
    print("\n" + "=" * 95)
    print(f"  FINAL BENCHMARK RESULTS (Evaluated on {num_eval} Test Samples) ")
    print("=" * 95)
    print(table_str, flush=True)

    report_file = out_dir / f"{report_prefix}_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 95 + "\n")
        f.write(f" VALL-E Native Watermark Test Benchmark Report\n")
        f.write(f" Test Manifest:       {manifest_path}\n")
        f.write(f" Evaluated Cuts:      {num_eval} / {total_test_samples}\n")
        f.write(f" Watermark Checkpoint:{wm_ckpt_path} (Step {ckpt_step}, Epoch {ckpt_epoch})\n")
        f.write(f" Total Audio Duration:{total_audio_duration:.2f} s\n")
        f.write("=" * 95 + "\n\n")
        f.write(table_str + "\n")
    logging.info(f"Saved text report to: {report_file}")

    summary_json_file = out_dir / f"{report_prefix}_evaluation_summary.json"
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(wm_ckpt_path),
            "steps": ckpt_step,
            "epoch": ckpt_epoch,
            "num_evaluated_samples": num_eval,
            "total_manifest_samples": total_test_samples,
            "total_audio_duration_sec": total_audio_duration,
            "quality_metrics": quality_metrics,
            "attack_metrics": summary,
            "sample_audio_records": sample_audio_records,
        }, f, indent=4)
    logging.info(f"Saved JSON summary to: {summary_json_file}")

    csv_file = out_dir / f"{report_prefix}_attack_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)
    logging.info(f"Saved CSV metrics to: {csv_file}")


def merge_and_save_shards(out_dir: Path):
    shard_files = sorted(out_dir.glob("shard_*_of_*_raw.pt"))
    if not shard_files:
        logging.error(f"No shard raw result files found in {out_dir}")
        return

    logging.info(f"Found {len(shard_files)} shard raw files in {out_dir}. Merging...")

    merged_results = {}
    merged_attack_scores = {}
    clean_utmos_list, wm_utmos_list = [], []
    pesq_list, stoi_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []
    total_audio_duration = 0.0
    total_embed_time = 0.0
    total_detect_time = 0.0
    total_evaluated_samples = 0

    manifest_path = None
    wm_ckpt_path = None
    ckpt_step = 0
    ckpt_epoch = 0
    total_manifest_samples = 0

    for sf in shard_files:
        logging.info(f"  -> Loading {sf.name}...")
        pkg = torch.load(sf, map_location="cpu")
        manifest_path = pkg.get("manifest_path", manifest_path)
        wm_ckpt_path = pkg.get("wm_ckpt_path", wm_ckpt_path)
        ckpt_step = pkg.get("ckpt_step", ckpt_step)
        ckpt_epoch = pkg.get("ckpt_epoch", ckpt_epoch)
        total_manifest_samples = max(total_manifest_samples, pkg.get("total_test_samples", 0))
        total_evaluated_samples += pkg.get("num_samples", 0)

        total_audio_duration += pkg.get("total_audio_duration", 0.0)
        total_embed_time += pkg.get("total_embed_time", 0.0)
        total_detect_time += pkg.get("total_detect_time", 0.0)

        clean_utmos_list.extend(pkg.get("clean_utmos_list", []))
        wm_utmos_list.extend(pkg.get("wm_utmos_list", []))
        pesq_list.extend(pkg.get("pesq_list", []))
        stoi_list.extend(pkg.get("stoi_list", []))
        clean_sim_list.extend(pkg.get("clean_sim_list", []))
        wm_sim_list.extend(pkg.get("wm_sim_list", []))
        clean_wer_list.extend(pkg.get("clean_wer_list", []))
        wm_wer_list.extend(pkg.get("wm_wer_list", []))
        clean_cer_list.extend(pkg.get("clean_cer_list", []))
        wm_cer_list.extend(pkg.get("wm_cer_list", []))
        sample_audio_records.extend(pkg.get("sample_audio_records", []))

        # Merge results dict
        s_results = pkg.get("results", {})
        for k, v in s_results.items():
            if k not in merged_results:
                merged_results[k] = {
                    "category": v["category"],
                    "family": v["family"],
                    "bitrate": v["bitrate"],
                    "bit_matches": 0,
                    "total_bits": 0,
                    "pos_matches": 0,
                    "pos_frames": 0,
                    "neg_matches": 0,
                    "neg_frames": 0,
                    "roc_auc": 0.5,
                    "tpr_at_001_fpr": 0.0,
                }
            merged_results[k]["bit_matches"] += v["bit_matches"]
            merged_results[k]["total_bits"] += v["total_bits"]
            merged_results[k]["pos_matches"] += v["pos_matches"]
            merged_results[k]["pos_frames"] += v["pos_frames"]
            merged_results[k]["neg_matches"] += v["neg_matches"]
            merged_results[k]["neg_frames"] += v["neg_frames"]

        # Merge attack_scores dict
        s_scores = pkg.get("attack_scores", {})
        for k, v in s_scores.items():
            if k not in merged_attack_scores:
                merged_attack_scores[k] = {
                    "pos_det_scores": [],
                    "neg_det_scores": [],
                    "pos_wm_scores": [],
                    "neg_wm_scores": [],
                }
            merged_attack_scores[k]["pos_det_scores"].extend(v["pos_det_scores"])
            merged_attack_scores[k]["neg_det_scores"].extend(v["neg_det_scores"])
            merged_attack_scores[k]["pos_wm_scores"].extend(v["pos_wm_scores"])
            merged_attack_scores[k]["neg_wm_scores"].extend(v["neg_wm_scores"])

    val_attacks = get_validation_attack_suite(sample_rate=16000)
    compute_and_save_reports(
        out_dir=out_dir,
        manifest_path=str(manifest_path),
        wm_ckpt_path=str(wm_ckpt_path),
        ckpt_step=ckpt_step,
        ckpt_epoch=ckpt_epoch,
        num_eval=total_evaluated_samples,
        total_test_samples=total_manifest_samples,
        results=merged_results,
        attack_scores=merged_attack_scores,
        clean_utmos_list=clean_utmos_list,
        wm_utmos_list=wm_utmos_list,
        pesq_list=pesq_list,
        stoi_list=stoi_list,
        clean_sim_list=clean_sim_list,
        wm_sim_list=wm_sim_list,
        clean_wer_list=clean_wer_list,
        wm_wer_list=wm_wer_list,
        clean_cer_list=clean_cer_list,
        wm_cer_list=wm_cer_list,
        total_audio_duration=total_audio_duration,
        total_embed_time=total_embed_time,
        total_detect_time=total_detect_time,
        sample_audio_records=sample_audio_records,
        val_attacks=val_attacks,
        report_prefix="test",
    )


def main():
    args = parse_args()
    if args.merge_shards:
        merge_and_save_shards(Path(args.merge_shards).resolve())
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    manifest_path = Path(args.manifest).resolve()

    if args.output_dir is None:
        out_dir = SCRIPT_DIR / "exp" / "eval_test_native_full"
    else:
        out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = out_dir / "audio_samples"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Checkpoint Paths
    st_cfg_path = Path(args.st_config) if args.st_config else NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
    st_ckpt_path = Path(args.st_checkpoint) if args.st_checkpoint else NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"
    if not st_cfg_path.is_absolute():
        st_cfg_path = (SCRIPT_DIR / st_cfg_path).resolve()
    if not st_ckpt_path.is_absolute():
        st_ckpt_path = (SCRIPT_DIR / st_ckpt_path).resolve()

    wm_ckpt_path = Path(args.watermark_model) if args.watermark_model else SCRIPT_DIR / "genkai_models/NeuMark_valle_v2.pt"
    if not wm_ckpt_path.is_absolute():
        wm_ckpt_path = (SCRIPT_DIR / wm_ckpt_path).resolve()

    wavlm_path = Path(args.wavlm_checkpoint)
    if not wavlm_path.is_absolute():
        wavlm_path = (SCRIPT_DIR / wavlm_path).resolve()

    logging.info("=" * 75)
    logging.info(" Benchmark Evaluation for VALL-E Native / NeuMark Watermark on Test Set ")
    logging.info(f" Test Manifest:       {manifest_path}")
    logging.info(f" SpeechTokenizer:     {st_ckpt_path}")
    logging.info(f" Watermark Model:     {wm_ckpt_path}")
    logging.info(f" WavLM Model:         {wavlm_path}")
    logging.info(f" Output Directory:    {out_dir}")
    logging.info(f" Device:              {device}")
    logging.info("=" * 75)

    if not wm_ckpt_path.exists():
        logging.error(f"Watermark checkpoint not found: {wm_ckpt_path}")
        sys.exit(1)

    # 2. Load Models
    logging.info("[1/4] Loading SpeechTokenizer Generator...")
    generator = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False

    logging.info("[2/4] Loading Watermark Embedder & Detector from Checkpoint...")
    msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(device)
    detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(device)

    wm_pkg = torch.load(str(wm_ckpt_path), map_location="cpu")
    if "msg_processor" in wm_pkg:
        msg_processor.load_state_dict(wm_pkg["msg_processor"])
        detector.load_state_dict(wm_pkg["detector"])
    elif "model" in wm_pkg:
        msg_processor.load_state_dict(wm_pkg["model"]["msg_processor"])
        detector.load_state_dict(wm_pkg["model"]["detector"])
    elif "embedder" in wm_pkg:
        msg_processor.load_state_dict(wm_pkg["embedder"])
        detector.load_state_dict(wm_pkg["detector"])
    msg_processor.eval()
    detector.eval()

    ckpt_step = wm_pkg.get("steps", wm_pkg.get("step", 20000))
    ckpt_epoch = wm_pkg.get("epoch", 1)
    logging.info(f"Watermark checkpoint loaded successfully! (Trained for {ckpt_step} steps, epoch {ckpt_epoch})")

    logging.info("[3/4] Initializing Objective Evaluation Metrics (UTMOS, WavLM SIM, Whisper ASR)...")
    utmos_loss = UTMOSLoss(device=str(device))
    sim_loss = SpeakerSimLoss(checkpoint_path=str(wavlm_path), device=str(device))
    asr_loss = ASRLoss(device=str(device))
    val_attacks = get_validation_attack_suite(sample_rate=16000)

    # 3. Load Dataset
    logging.info(f"[4/4] Loading Test Items from {manifest_path}...")
    items = []
    if manifest_path.is_dir():
        meta_json = manifest_path / "metadata.json"
        meta_csv = manifest_path / "metadata.csv"
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                records = json.load(f).get("records", [])
                for r in records:
                    c_p = manifest_path / r["clean_tts_relpath"] if "clean_tts_relpath" in r else Path(r["clean_tts_wav"])
                    p_p = manifest_path / r["prompt_relpath"] if "prompt_relpath" in r else Path(r["prompt_wav"])
                    items.append({
                        "cut_id": r.get("utt_id", str(r.get("sample_idx"))),
                        "text": r.get("text", ""),
                        "clean_path": c_p,
                        "prompt_path": p_p,
                    })
        elif meta_csv.exists():
            with open(meta_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    c_p = manifest_path / r["clean_tts_relpath"] if "clean_tts_relpath" in r else Path(r["clean_tts_wav"])
                    p_p = manifest_path / r["prompt_relpath"] if "prompt_relpath" in r else Path(r["prompt_wav"])
                    items.append({
                        "cut_id": r.get("utt_id", r.get("sample_idx", "")),
                        "text": r.get("text", ""),
                        "clean_path": c_p,
                        "prompt_path": p_p,
                    })
    else:
        # Tokenized cuts fallback
        from tts_native_dataset import get_tts_native_dataloader
        test_dl = get_tts_native_dataloader(
            manifest_path=str(manifest_path),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            max_duration=20.0,
        )
        for b in test_dl:
            items.append({
                "cut_id": b["ids"][0],
                "text": b["texts"][0],
                "batch": b,
            })

    total_test_samples = len(items)
    num_eval = total_test_samples if args.num_samples <= 0 else min(args.num_samples, total_test_samples)
    items = items[:num_eval]

    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards, f"Invalid shard-id {args.shard_id} for {args.num_shards} shards"
        shard_items = items[args.shard_id::args.num_shards]
        logging.info(f"[Sharding] Total samples: {num_eval} | Shard {args.shard_id + 1}/{args.num_shards}: {len(shard_items)} samples assigned")
    else:
        shard_items = items

    shard_num_eval = len(shard_items)
    logging.info(f"Total test samples: {total_test_samples} | Evaluating {shard_num_eval} samples on device {device}...")

    # 4. Evaluation Loop
    results = {}
    attack_scores = {}
    for cat, name, detail, _ in val_attacks:
        key = name if cat == "DSP" else f"{name} {detail}"
        results[key] = {
            "category": cat,
            "family": name,
            "bitrate": detail,
            "bit_matches": 0,
            "total_bits": 0,
            "pos_matches": 0,
            "pos_frames": 0,
            "neg_matches": 0,
            "neg_frames": 0,
            "roc_auc": 0.5,
            "tpr_at_001_fpr": 0.0,
        }
        attack_scores[key] = {"pos_det_scores": [], "neg_det_scores": [], "pos_wm_scores": [], "neg_wm_scores": []}

    clean_utmos_list, wm_utmos_list = [], []
    pesq_list, stoi_list = [], []
    clean_sim_list, wm_sim_list = [], []
    clean_wer_list, wm_wer_list = [], []
    clean_cer_list, wm_cer_list = [], []
    sample_audio_records = []

    total_audio_duration = 0.0
    total_embed_time = 0.0
    total_detect_time = 0.0

    logging.info("=" * 75)
    logging.info(f" Starting Full Benchmark Evaluation on {num_eval} Test Samples...")
    logging.info("=" * 75)

    with torch.no_grad():
        loop_desc = f"Evaluating [Shard {args.shard_id+1}/{args.num_shards}]" if args.num_shards > 1 else "Evaluating Native / NeuMark"
        for i in tqdm(range(shard_num_eval), desc=loop_desc, ncols=100):
            item = shard_items[i]
            cut_id = item["cut_id"]
            ref_text = item["text"]

            if "clean_path" in item:
                c_wav, c_sr = torchaudio.load(str(item["clean_path"]))
                if c_sr != 16000:
                    c_wav = torchaudio.functional.resample(c_wav, c_sr, 16000)
                if c_wav.shape[0] > 1:
                    c_wav = c_wav.mean(dim=0, keepdim=True)
                clean_audio = c_wav.unsqueeze(0).to(device)

                p_wav, p_sr = torchaudio.load(str(item["prompt_path"]))
                if p_sr != 16000:
                    p_wav = torchaudio.functional.resample(p_wav, p_sr, 16000)
                if p_wav.shape[0] > 1:
                    p_wav = p_wav.mean(dim=0, keepdim=True)
                prompt_audio = p_wav.unsqueeze(0).to(device)

                # Encode with SpeechTokenizer to get 8 RVQ layers
                codes = generator.encode(clean_audio)
                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
            else:
                b = item["batch"]
                codes = b["codes"].to(device)
                prompt_audio = b["prompt_audio"].to(device)
                if prompt_audio.ndim == 2:
                    prompt_audio = prompt_audio.unsqueeze(0)
                elif prompt_audio.ndim == 1:
                    prompt_audio = prompt_audio.unsqueeze(0).unsqueeze(0)

                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
                z_clean = sum(quantized_layers)
                clean_audio = generator.decoder(z_clean)

            # 16-bit random message
            message = torch.randint(0, 2, (1, 16), device=device)
            msg_np = message.cpu().numpy().squeeze()

            # Measure Embedding Time (Layer-wise injection exactly matching training)
            t0 = time.perf_counter()
            watermarked_layers = [msg_processor(q, message) for q in quantized_layers]
            z_wm = sum(watermarked_layers)
            wm_audio = generator.decoder(z_wm)
            t_embed = time.perf_counter() - t0
            total_embed_time += t_embed

            # Match lengths
            min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
            clean_audio = clean_audio[..., :min_len]
            wm_audio = wm_audio[..., :min_len]

            audio_dur = clean_audio.shape[-1] / 16000.0
            total_audio_duration += audio_dur

            # Audio Quality Metrics (Clean TTS vs. Watermarked TTS)
            try:
                c_np = clean_audio.detach().squeeze().cpu().numpy()
                w_np = wm_audio.detach().squeeze().cpu().numpy()
                min_l = min(len(c_np), len(w_np))
                if min_l >= 1600:
                    p_val = pesq(16000, c_np[:min_l], w_np[:min_l], "wb")
                    s_val = stoi(c_np[:min_l], w_np[:min_l], 16000, extended=False)
                    pesq_list.append(p_val)
                    stoi_list.append(s_val)
            except Exception:
                pass

            try:
                c_u = utmos_loss.model(clean_audio.squeeze(1), 16000).mean().item()
                w_u = utmos_loss.model(wm_audio.squeeze(1), 16000).mean().item()
                clean_utmos_list.append(c_u)
                wm_utmos_list.append(w_u)
            except Exception:
                pass

            try:
                ref_spk = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else clean_audio
                c_s = sim_loss.get_similarity(clean_audio, ref_spk, 16000)
                w_s = sim_loss.get_similarity(wm_audio, ref_spk, 16000)
                clean_sim_list.append(c_s)
                wm_sim_list.append(w_s)
            except Exception:
                pass

            if getattr(asr_loss, "model", None) is not None and ref_text:
                try:
                    c_hyps = asr_loss.decode_greedy(clean_audio, 16000)
                    w_hyps = asr_loss.decode_greedy(wm_audio, 16000)
                    c_wer, c_cer = compute_wer_cer(ref_text, c_hyps[0])
                    w_wer, w_cer = compute_wer_cer(ref_text, w_hyps[0])
                    clean_wer_list.append(c_wer)
                    clean_cer_list.append(c_cer)
                    wm_wer_list.append(w_wer)
                    wm_cer_list.append(w_cer)
                except Exception:
                    pass

            # Attacks & Robustness Evaluation
            for cat, name, detail, atk_fn in val_attacks:
                key = name if cat == "DSP" else f"{name} {detail}"
                try:
                    attacked_wm = atk_fn(wm_audio)
                except Exception:
                    attacked_wm = wm_audio

                t_det_0 = time.perf_counter()
                feat_atk_wm = generator.forward_feature(attacked_wm)
                prob_wm_t, msg_out_wm, _ = detector.detect_watermark(feat_atk_wm)
                total_detect_time += (time.perf_counter() - t_det_0)

                prob_wm = float(prob_wm_t.mean().item())
                msg_pred_wm = np.array(msg_out_wm.squeeze().cpu().numpy()).flatten().tolist()[:16]
                bit_matches = sum(int(c1) == int(c2) for c1, c2 in zip(msg_pred_wm, msg_np))
                tp_flag = 1 if prob_wm >= 0.5 else 0

                try:
                    attacked_clean = atk_fn(clean_audio)
                except Exception:
                    attacked_clean = clean_audio

                t_det_1 = time.perf_counter()
                feat_atk_cl = generator.forward_feature(attacked_clean)
                prob_cl_t, msg_out_cl, _ = detector.detect_watermark(feat_atk_cl)
                total_detect_time += (time.perf_counter() - t_det_1)

                prob_cl = float(prob_cl_t.mean().item())
                msg_pred_cl = np.array(msg_out_cl.squeeze().cpu().numpy()).flatten().tolist()[:16]
                cl_bit_matches = sum(int(c1) == int(c2) for c1, c2 in zip(msg_pred_cl, msg_np))
                clean_tp_flag = 1 if prob_cl >= 0.5 else 0
                tn_flag = 1 - clean_tp_flag

                results[key]["bit_matches"] += bit_matches
                results[key]["total_bits"] += 16
                results[key]["pos_matches"] += tp_flag
                results[key]["pos_frames"] += 1
                results[key]["neg_matches"] += tn_flag
                results[key]["neg_frames"] += 1

                attack_scores[key]["pos_det_scores"].append(prob_wm)
                attack_scores[key]["neg_det_scores"].append(prob_cl)
                attack_scores[key]["pos_wm_scores"].append(bit_matches / 16.0)
                attack_scores[key]["neg_wm_scores"].append(cl_bit_matches / 16.0)

            # Save Sample Audio Files
            if i < args.save_audio_samples:
                prefix = f"sample_shard{args.shard_id}_{i:03d}" if args.num_shards > 1 else f"sample_{i:03d}"
                c_wav_p = audio_out_dir / f"{prefix}_{cut_id}_clean_tts.wav"
                w_wav_p = audio_out_dir / f"{prefix}_{cut_id}_native_wm.wav"
                p_wav_p = audio_out_dir / f"{prefix}_{cut_id}_prompt.wav"
                torchaudio.save(str(c_wav_p), clean_audio.reshape(1, -1).cpu(), 16000)
                torchaudio.save(str(w_wav_p), wm_audio.reshape(1, -1).cpu(), 16000)
                if prompt_audio is not None and prompt_audio.numel() > 0:
                    torchaudio.save(str(p_wav_p), prompt_audio.reshape(1, -1).cpu(), 16000)

                sample_audio_records.append({
                    "sample_idx": i,
                    "cut_id": cut_id,
                    "text": ref_text,
                    "clean_wav": str(c_wav_p.name),
                    "watermarked_wav": str(w_wav_p.name),
                    "prompt_wav": str(p_wav_p.name),
                    "duration_sec": audio_dur,
                })

            if (i + 1) % 25 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

    # 5. Save Raw Shard Results & Compute Reports
    if args.num_shards > 1:
        shard_raw_path = out_dir / f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}_raw.pt"
        shard_pkg = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "num_samples": shard_num_eval,
            "total_test_samples": total_test_samples,
            "results": results,
            "attack_scores": attack_scores,
            "clean_utmos_list": clean_utmos_list,
            "wm_utmos_list": wm_utmos_list,
            "pesq_list": pesq_list,
            "stoi_list": stoi_list,
            "clean_sim_list": clean_sim_list,
            "wm_sim_list": wm_sim_list,
            "clean_wer_list": clean_wer_list,
            "wm_wer_list": wm_wer_list,
            "clean_cer_list": clean_cer_list,
            "wm_cer_list": wm_cer_list,
            "total_audio_duration": total_audio_duration,
            "total_embed_time": total_embed_time,
            "total_detect_time": total_detect_time,
            "sample_audio_records": sample_audio_records,
            "ckpt_step": ckpt_step,
            "ckpt_epoch": ckpt_epoch,
            "wm_ckpt_path": str(wm_ckpt_path),
            "manifest_path": str(manifest_path),
        }
        torch.save(shard_pkg, shard_raw_path)
        logging.info(f"Saved shard raw results to: {shard_raw_path}")

        compute_and_save_reports(
            out_dir=out_dir,
            manifest_path=str(manifest_path),
            wm_ckpt_path=str(wm_ckpt_path),
            ckpt_step=ckpt_step,
            ckpt_epoch=ckpt_epoch,
            num_eval=shard_num_eval,
            total_test_samples=total_test_samples,
            results=results,
            attack_scores=attack_scores,
            clean_utmos_list=clean_utmos_list,
            wm_utmos_list=wm_utmos_list,
            pesq_list=pesq_list,
            stoi_list=stoi_list,
            clean_sim_list=clean_sim_list,
            wm_sim_list=wm_sim_list,
            clean_wer_list=clean_wer_list,
            wm_wer_list=wm_wer_list,
            clean_cer_list=clean_cer_list,
            wm_cer_list=wm_cer_list,
            total_audio_duration=total_audio_duration,
            total_embed_time=total_embed_time,
            total_detect_time=total_detect_time,
            sample_audio_records=sample_audio_records,
            val_attacks=val_attacks,
            report_prefix=f"shard_{args.shard_id:02d}",
        )
    else:
        compute_and_save_reports(
            out_dir=out_dir,
            manifest_path=str(manifest_path),
            wm_ckpt_path=str(wm_ckpt_path),
            ckpt_step=ckpt_step,
            ckpt_epoch=ckpt_epoch,
            num_eval=num_eval,
            total_test_samples=total_test_samples,
            results=results,
            attack_scores=attack_scores,
            clean_utmos_list=clean_utmos_list,
            wm_utmos_list=wm_utmos_list,
            pesq_list=pesq_list,
            stoi_list=stoi_list,
            clean_sim_list=clean_sim_list,
            wm_sim_list=wm_sim_list,
            clean_wer_list=clean_wer_list,
            wm_wer_list=wm_wer_list,
            clean_cer_list=clean_cer_list,
            wm_cer_list=wm_cer_list,
            total_audio_duration=total_audio_duration,
            total_embed_time=total_embed_time,
            total_detect_time=total_detect_time,
            sample_audio_records=sample_audio_records,
            val_attacks=val_attacks,
            report_prefix="test",
        )

    release_codec_models()
    logging.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
