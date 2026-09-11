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
Clean version of VALL-E inference script.
Only focuses on generating speech from text and audio prompts.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

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
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/gen"),
        help="Path to the output directory.",
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

    return parser.parse_args()


def load_model(checkpoint, device):
    checkpoint = torch.load(checkpoint, map_location=device)
    args = AttributeDict(checkpoint)
    model = get_model(args)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, args.text_tokens


@torch.no_grad()
def main():
    args = get_args()
    logging.info(f"Arguments: {args}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    # 1. Load Tokenizers
    text_tokenizer = TextTokenizer(backend=args.text_extractor)
    
    watermark_backend = args.watermark_backend
    if args.ts_enable and watermark_backend == "encodec":
        watermark_backend = "traceablespeech"

    ts_config = Path(args.ts_checkpoint_file).parent / "config.json"
    audio_tokenizer = AudioTokenizer(
        device=device,
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
    audio_sample_rate = audio_tokenizer.sample_rate

    # 2. Load Model
    model, text_tokens = load_model(args.checkpoint, device)
    text_collater = get_text_token_collater(text_tokens)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 3. Prepare Audio Prompts
    text_prompts = " ".join(args.text_prompts.split("|"))
    audio_prompts_tokens = []
    if args.audio_prompts:
        for audio_file in args.audio_prompts.split("|"):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            audio_prompts_tokens.append(encoded_frames[0][0])
        
        audio_prompts_tokens = torch.concat(audio_prompts_tokens, dim=-1).transpose(2, 1)
        audio_prompts_tokens = audio_prompts_tokens.to(device)

    # 4. Prepare Texts
    if args.text_file:
        with open(args.text_file) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    else:
        texts = args.text.split("|")

    # 5. Inference and Save
    for n, text in enumerate(texts):
        logging.info(f"[{n+1}/{len(texts)}] Synthesizing: {text}")
        
        # Tokenize text
        full_text = f"{text_prompts} {text}".strip()
        text_tokens_idx, text_tokens_lens = text_collater(
            [tokenize_text(text_tokenizer, text=full_text)]
        )

        # Synthesis
        if args.continual:
            encoded_frames = model.continual(
                text_tokens_idx.to(device),
                text_tokens_lens.to(device),
                audio_prompts_tokens,
            )
        else:
            enroll_x_lens = None
            if text_prompts:
                _, enroll_x_lens = text_collater(
                    [tokenize_text(text_tokenizer, text=text_prompts.strip())]
                )
            
            encoded_frames = model.inference(
                text_tokens_idx.to(device),
                text_tokens_lens.to(device),
                audio_prompts_tokens,
                enroll_x_lens=enroll_x_lens,
                top_k=args.top_k,
                temperature=args.temperature,
            )

        if encoded_frames.numel() == 0:
            logging.warning(f"Skip utt {n}: empty output.")
            continue

        # Decode
        # model.inference returns [B, T, 8], need [B, 8, T]
        encoded_for_decode = [(encoded_frames.transpose(2, 1), None)]
        samples = audio_tokenizer.decode(encoded_for_decode)

        # Save
        out_path = args.output_dir / f"{n}.wav"
        torchaudio.save(str(out_path), samples[0].cpu(), audio_sample_rate)
        logging.info(f"Saved to {out_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
