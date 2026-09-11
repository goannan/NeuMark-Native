#!/usr/bin/env python3
# Copyright (c) 2026
# Generate VALL-E TTS-Native 8-layer acoustic tokens using exact VALL-E Inference Pipeline

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent

def find_neumark_root() -> Path:
    candidates = [
        SCRIPT_DIR,
        os.environ.get("NEUMARK_ROOT"),
        SCRIPT_DIR.parent.parent.parent / "NeuMark",
        PROJECT_DIR.parent / "NeuMark",
        Path.cwd() / "NeuMark",
        Path.cwd().parent / "NeuMark",
        Path.home() / "projects" / "NeuMark",
    ]
    for c in candidates:
        if c:
            p = Path(c).resolve()
            if p.is_dir():
                return p
    return (PROJECT_DIR.parent / "NeuMark").resolve()

NEUMARK_ROOT = find_neumark_root()

for p in [str(PROJECT_DIR), str(SCRIPT_DIR), str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from icefall.utils import AttributeDict
from lhotse import CutSet, MonoCut, SupervisionSegment, load_manifest_lazy
from lhotse.features import Features
from valle.data.tokenizer import AudioTokenizer, TextTokenizer, tokenize_audio, tokenize_text
from valle.data.collation import get_text_token_collater
from valle.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VALL-E native tokens conditioned on Prompt Audio + Target Text.")
    parser.add_argument(
        "--valle-checkpoint",
        type=str,
        default="exp/valle_voicemark/epoch-40.pt",
        help="Path to trained VALL-E checkpoint",
    )
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_dev.jsonl.gz",
        help="Path to input tokenized cuts manifest from database",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="data/tokenized_voicemark/cuts_dev_valle_native.jsonl.gz",
        help="Path to output manifest referencing the generated VALL-E tokens",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="data/tokenized_voicemark/libritts_valle_native_dev.h5",
        help="Path to H5 file where generated 8-layer tokens will be stored",
    )
    parser.add_argument(
        "--text-tokens",
        type=str,
        default="",
        help="Path to unique text tokens symbol table (defaults to model checkpoint args.text_tokens)",
    )
    parser.add_argument(
        "--speechtokenizer-config",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"),
    )
    parser.add_argument(
        "--speechtokenizer-checkpoint",
        type=str,
        default=str(NEUMARK_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"),
    )
    parser.add_argument("--min-duration", type=float, default=1.0, help="Min target cut duration (seconds)")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max target cut duration (strictly <= 10.0s)")
    parser.add_argument("--max-samples", type=int, default=-1, help="Max cuts to generate (-1 for all)")
    parser.add_argument("--top-k", type=int, default=-100, help="Top-K sampling for AR Decoder (-100 default)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_generation_tasks(cuts, min_dur: float, max_dur: float) -> List[Tuple[MonoCut, MonoCut]]:
    """
    Build 1:1 generation tasks:
    Each target cut (duration <= 10s) from the database is paired with an ideal, clean reference
    prompt audio (3.0s ~ 4.5s) from the same speaker to synthesize its exact target text.
    """
    spk_map = defaultdict(list)
    for c in cuts:
        if c.supervisions:
            spk_id = c.supervisions[0].speaker
            spk_map[spk_id].append(c)

    tasks = []
    for spk_id, spk_cuts in spk_map.items():
        if not spk_cuts:
            continue

        # Select ideal prompt cut (~3.5s) for this speaker
        short_candidates = [c for c in spk_cuts if 2.8 <= c.duration <= 4.8 and c.has_recording]
        if not short_candidates:
            valid_cuts = [c for c in spk_cuts if c.has_recording]
            short_candidates = sorted(valid_cuts if valid_cuts else spk_cuts, key=lambda c: abs(c.duration - 3.5))

        prompt_pool = short_candidates if short_candidates else spk_cuts

        # Filter target cuts within duration range (<= 10s)
        target_cuts = [
            c for c in spk_cuts 
            if min_dur <= c.duration <= max_dur and c.supervisions and c.supervisions[0].text.strip()
        ]

        for target_cut in target_cuts:
            # Pick a distinct prompt cut if possible
            chosen_prompt = None
            for p in prompt_pool:
                if p.id != target_cut.id:
                    chosen_prompt = p
                    break
            if chosen_prompt is None:
                chosen_prompt = prompt_pool[0]

            tasks.append((chosen_prompt, target_cut))

    return tasks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}, Rank: {args.rank}/{args.world_size}")

    # 1. Load VALL-E Model
    logging.info(f"Loading VALL-E model from {args.valle_checkpoint}...")
    ckpt_data = torch.load(args.valle_checkpoint, map_location="cpu", weights_only=False)
    model_args = AttributeDict(ckpt_data)
    valle_model = get_model(model_args)
    valle_model.load_state_dict(ckpt_data["model"], strict=True)
    valle_model.to(device)
    valle_model.eval()

    # 2. Text Collater & Text Tokenizer
    text_tokens_file = args.text_tokens if (args.text_tokens and os.path.exists(args.text_tokens)) else model_args.text_tokens
    text_collater = get_text_token_collater(text_tokens_file)
    text_tokenizer = TextTokenizer(backend="espeak")

    # 3. Audio Tokenizer (for exact waveform tokenization matching infer.py)
    logging.info("Loading AudioTokenizer (SpeechTokenizer)...")
    audio_tokenizer = AudioTokenizer(
        watermark_backend="neumark",
        voicemark_root=str(NEUMARK_ROOT),
        voicemark_config=args.speechtokenizer_config,
        voicemark_st_checkpoint=args.speechtokenizer_checkpoint,
        device=device,
    )

    # 4. Load Cuts from Database & Build Tasks
    logging.info(f"Loading input database cuts from {args.input_manifest} (Target duration: {args.min_duration}s - {args.max_duration}s)...")
    raw_cuts = load_manifest_lazy(args.input_manifest)
    tasks = build_generation_tasks(raw_cuts, args.min_duration, args.max_duration)
    logging.info(f"Total valid target cuts to synthesize: {len(tasks)}")

    # Shard for distributed multi-GPU generation
    if args.world_size > 1:
        tasks = tasks[args.rank :: args.world_size]
        logging.info(f"Rank {args.rank} assigned {len(tasks)} cuts.")

    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]

    out_h5_path = Path(args.output_h5)
    if args.world_size > 1:
        out_h5_path = out_h5_path.with_name(f"{out_h5_path.stem}_rank{args.rank}{out_h5_path.suffix}")
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    out_manifest_path = Path(args.output_manifest)
    if args.world_size > 1:
        stem = out_manifest_path.name
        if stem.endswith(".jsonl.gz"):
            base = stem[:-9]
        elif stem.endswith(".jsonl"):
            base = stem[:-6]
        else:
            base = out_manifest_path.stem
        out_manifest_path = out_manifest_path.with_name(f"{base}_rank{args.rank}.jsonl.gz")

    # 5. Open H5 file with Resume Support
    try:
        h5_file = h5py.File(str(out_h5_path), "a")
        existing_keys = set(h5_file.keys())
        logging.info(f"Found {len(existing_keys)} existing entries in {out_h5_path} (resuming)...")
    except Exception as ex:
        logging.warning(f"Warning: Recreating clean H5 file {out_h5_path} due to: {ex}")
        if out_h5_path.exists():
            out_h5_path.unlink()
        h5_file = h5py.File(str(out_h5_path), "w")
        existing_keys = set()

    generated_cuts = []

    # 6. Synthesis Loop (FP32 exact matching infer.py)
    with torch.no_grad():
        for prompt_cut, target_cut in tqdm(tasks, desc=f"VALL-E Native Gen [Rank {args.rank}]"):
            cut_key = f"{target_cut.id}_valle_native"

            prompt_text = prompt_cut.supervisions[0].text.strip()
            target_text = target_cut.supervisions[0].text.strip()

            if cut_key in existing_keys:
                gen_codes_np = h5_file[cut_key][:]
            else:
                try:
                    # Tokenize prompt wav using AudioTokenizer exactly as in infer.py
                    if prompt_cut.has_recording and os.path.exists(prompt_cut.recording.sources[0].source):
                        p_wav_path = prompt_cut.recording.sources[0].source
                        prompt_frames = tokenize_audio(audio_tokenizer, p_wav_path)
                        audio_prompt_tokens = prompt_frames[0][0].transpose(2, 1).to(device)
                    else:
                        p_codes_np = prompt_cut.load_features()
                        if p_codes_np is None or p_codes_np.shape[0] < 20:
                            continue
                        audio_prompt_tokens = torch.from_numpy(p_codes_np).long().unsqueeze(0).to(device)
                except Exception:
                    continue

                if audio_prompt_tokens.shape[1] < 20:
                    continue

                # Text conditioning: full_text = f"{prompt_text} {target_text}"
                full_text = f"{prompt_text} {target_text}".strip()
                text_tokens_idx, text_tokens_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=full_text)]
                )
                text_tokens_idx = text_tokens_idx.to(device)
                text_tokens_lens = text_tokens_lens.to(device)

                _, enroll_x_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=prompt_text)]
                )
                enroll_x_lens = enroll_x_lens.to(device)

                try:
                    # Pure FP32 inference (identical to bin/infer.py)
                    gen_tokens = valle_model.inference(
                        text_tokens_idx,
                        text_tokens_lens,
                        audio_prompt_tokens,
                        enroll_x_lens=enroll_x_lens,
                        top_k=args.top_k,
                        temperature=args.temperature,
                    )
                    # Extract target tokens: [T_gen, 8]
                    target_tokens = gen_tokens[0].cpu().numpy().astype(np.int16)
                    if target_tokens.shape[0] < 10:
                        continue

                    # Save to H5 file
                    h5_file.create_dataset(cut_key, data=target_tokens, compression="gzip")
                    h5_file.flush()
                    gen_codes_np = target_tokens
                except Exception as ex:
                    logging.warning(f"Error generating target {target_cut.id} with prompt {prompt_cut.id}: {ex}")
                    continue

            # SpeechTokenizer 16kHz with hop 320 -> 50 fps (frame_shift = 0.02s)
            frame_shift = 0.02
            gen_duration = float(gen_codes_np.shape[0] * frame_shift)
            storage_p = str(Path(args.output_h5) if args.world_size == 1 else Path(args.output_h5).with_name(f"{Path(args.output_h5).stem}_rank{args.rank}{Path(args.output_h5).suffix}"))

            feat = Features(
                type="valle_native",
                num_frames=gen_codes_np.shape[0],
                num_features=8,
                frame_shift=frame_shift,
                sampling_rate=16000,
                start=0.0,
                duration=gen_duration,
                storage_type="numpy_hdf5",
                storage_path=storage_p,
                storage_key=cut_key,
            )

            # Preserve exact target supervision (matching database text)
            new_supervision = target_cut.supervisions[0] if target_cut.supervisions else None

            # Construct Lhotse Cut with full metadata traceability
            new_cut = MonoCut(
                id=cut_key,
                start=0.0,
                duration=gen_duration,
                channel=0,
                features=feat,
                recording=target_cut.recording if target_cut.has_recording else None,
                supervisions=[new_supervision] if new_supervision else [],
                custom={
                    "target_cut_id": target_cut.id,
                    "target_text": target_text,
                    "prompt_cut_id": prompt_cut.id,
                    "prompt_text": prompt_text,
                    "prompt_duration": float(prompt_cut.duration),
                    "speaker": prompt_cut.supervisions[0].speaker if prompt_cut.supervisions else "unknown",
                }
            )
            generated_cuts.append(new_cut)

    h5_file.close()
    logging.info(f"Saving output manifest to {out_manifest_path} ({len(generated_cuts)} cuts)...")
    CutSet.from_cuts(generated_cuts).to_file(out_manifest_path)
    logging.info(f"Done! Successfully generated and saved {len(generated_cuts)} cuts to {out_manifest_path}")


if __name__ == "__main__":
    main()
