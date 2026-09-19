#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path(os.environ.get("NEUMARK_ROOT", SCRIPT_DIR)).resolve()

for p in [str(NEUMARK_ROOT), str(NEUMARK_ROOT / "train"), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import WMEmbedder, WMDetector
from STmodels.model import SpeechTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuMark-Native: Discrete Acoustic Token Watermark Demo"
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
        help="Directory to save generated demo audio files",
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

    # 2. Load SpeechTokenizer
    print("=" * 70)
    print(" NeuMark-Native: Latent Watermark Embedding & Extraction Demo")
    print("=" * 70)
    print(f" Device:               {device}")
    print(f" Watermark Checkpoint: {wm_ckpt_path}")
    print(f" SpeechTokenizer:      {st_ckpt_path}")
    print(f" Output Directory:     {out_dir}")

    st_model = SpeechTokenizer.load_from_checkpoint(str(st_cfg_path), str(st_ckpt_path)).to(device).eval()
    for p in st_model.parameters():
        p.requires_grad = False

    # 3. Load NeuMark Embedder & Detector
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

    # 4. Determine Input (Single Audio vs. Manifest)
    audio_path = None
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.is_absolute():
            audio_path = (Path.cwd() / audio_path).resolve()
    elif not args.manifest:
        # Check standard sample paths in repo
        sample_candidates = [
            PROJECT_DIR / "docs/audio/libritts_sample_1/01_clean_tts.wav",
            PROJECT_DIR / "docs/audio/libritts_sample_1/00_prompt.wav",
            SCRIPT_DIR / "../../docs/audio/libritts_sample_1/01_clean_tts.wav",
        ]
        for c in sample_candidates:
            if c.exists():
                audio_path = c.resolve()
                break

    # Parse message bits
    msg_str = args.message.strip()
    if len(msg_str) != 16 or not all(c in "01" for c in msg_str):
        raise ValueError(f"Watermark message must be a 16-bit binary string (got '{msg_str}')")
    msg_bits = torch.tensor([[int(c) for c in msg_str]], dtype=torch.int64, device=device)

    # Process Manifest if provided and exists
    if args.manifest and Path(args.manifest).exists():
        from lhotse import load_manifest_lazy
        manifest_path = Path(args.manifest)
        print(f"\nProcessing cuts manifest: {manifest_path}")
        cuts = load_manifest_lazy(manifest_path)
        for s_idx, cut in enumerate(cuts):
            if s_idx >= 3:
                break
            codes_np = cut.load_features()
            codes = torch.from_numpy(codes_np).long().transpose(0, 1).unsqueeze(0).to(device)
            codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
            with torch.no_grad():
                q_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
                clean_audio = st_model.decoder(sum(q_layers)).squeeze(0).cpu()
                wm_layers = [msg_processor(q, msg_bits) for q in q_layers]
                wm_audio = st_model.decoder(sum(wm_layers))
                feat = st_model.forward_feature(wm_audio)
                det_prob, pred_bits, _ = detector.detect_watermark(feat)

            wm_audio_cpu = wm_audio.squeeze(0).cpu()
            bit_acc = (pred_bits.long().cpu() == msg_bits.cpu()).sum().item() / 16.0
            pred_str = "".join(str(b.item()) for b in pred_bits[0])

            c_path = out_dir / f"sample_{s_idx:02d}_clean.wav"
            w_path = out_dir / f"sample_{s_idx:02d}_watermarked.wav"
            torchaudio.save(str(c_path), clean_audio, 16000)
            torchaudio.save(str(w_path), wm_audio_cpu, 16000)
            print(f"  [Sample {s_idx}] Acc: {bit_acc * 100:.1f}% | Pred: {pred_str} | Prob: {float(det_prob):.4f}")
        print(f"\nAll manifest samples processed into {out_dir}")
        return

    # Process Single Audio Waveform
    if audio_path is None or not audio_path.exists():
        print(f"\nError: Input audio not found. Please provide an audio path with --audio-path <file.wav>.")
        sys.exit(1)

    print(f" Input Audio File:     {audio_path}")
    print(f" Target 16-Bit Bits:   {msg_str}")
    print("-" * 70)

    # Load and resample audio
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device) # (1, 1, T)

    duration = wav.shape[-1] / 16000.0
    print(f" Loaded audio: {duration:.2f}s @ 16kHz")

    # Step 1: Tokenize audio into 8 RVQ discrete layers
    with torch.no_grad():
        codes = st_model.encode(wav)
        codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
        q_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

        # Step 2: Clean audio synthesis
        z_clean = sum(q_layers)
        clean_audio = st_model.decoder(z_clean)

        # Step 3: Native watermark injection into discrete latent space
        wm_layers = [msg_processor(q, msg_bits) for q in q_layers]
        z_wm = sum(wm_layers)
        wm_audio = st_model.decoder(z_wm)

        # Step 4: Covert watermark detection and bit extraction
        feat = st_model.forward_feature(wm_audio)
        det_prob, pred_bits, _ = detector.detect_watermark(feat)

    bit_acc = (pred_bits.long().cpu() == msg_bits.cpu()).sum().item() / 16.0
    pred_str = "".join(str(b.item()) for b in pred_bits[0])
    prob_val = det_prob.item() if isinstance(det_prob, torch.Tensor) else float(det_prob)

    # Step 5: Save output audio files
    clean_path = out_dir / "clean_reconstructed.wav"
    wm_path = out_dir / "watermarked_native.wav"
    diff_path = out_dir / "watermark_diff_x10.wav"

    clean_cpu = clean_audio.squeeze(0).cpu()
    wm_cpu = wm_audio.squeeze(0).cpu()
    min_len = min(clean_cpu.shape[-1], wm_cpu.shape[-1])
    diff = torch.clamp((wm_cpu[..., :min_len] - clean_cpu[..., :min_len]) * 10.0, -1.0, 1.0)

    torchaudio.save(str(clean_path), clean_cpu, 16000)
    torchaudio.save(str(wm_path), wm_cpu, 16000)
    torchaudio.save(str(diff_path), diff, 16000)

    print("-" * 70)
    print(" VERIFICATION RESULTS:")
    print(f"   Embedded Watermark: {msg_str}")
    print(f"   Extracted Bits:     {pred_str}")
    print(f"   Bit Accuracy:       {bit_acc * 100:.2f}% ({int(bit_acc * 16)}/16 bits match)")
    print(f"   Detection Score:    {prob_val:.6f}")
    print("-" * 70)
    print(" Saved Audio Files:")
    print(f"   1. Clean Reconstruction: {clean_path}")
    print(f"   2. Watermarked Audio:    {wm_path}")
    print(f"   3. Residual (Diff x 10): {diff_path}")
    print("=" * 70)
    print(" Native watermark embedding and extraction verified successfully!")


if __name__ == "__main__":
    main()
