import argparse
import datetime
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import tensorboard
from tqdm import tqdm

from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)

# -------------------------------------------------------------
# Dynamic Path Setups
# -------------------------------------------------------------
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

for p in [
    str(SCRIPT_DIR),
    str(NEUMARK_ROOT / "train"),
    str(NEUMARK_ROOT),
]:
    if p in sys.path:
        sys.path.remove(p)
    if os.path.exists(p):
        sys.path.insert(0, p)

from STmodels.model import SpeechTokenizer
from STmodels.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from models import WMEmbedder, WMDetector

try:
    from optimizer import get_optimizer
except ImportError:
    from train.optimizer import get_optimizer

from tts_native_loss import (
    mel_loss,
    UTMOSLoss,
    SpeakerSimLoss,
    ASRLoss,
    latent_cosine_loss,
    adversarial_loss_d,
    adversarial_loss_g,
    bits_to_chunks,
    decoding_loss,
    feature_loss,
    vad_based_loss,
    margin_vad_loss,
)
from tts_native_dataset import get_tts_native_dataloader
from tts_native_attacks import (
    format_full_validation_table,
    compute_wer_cer,
    apply_train_augmentation,
    get_validation_attack_suite,
    compute_auc_and_tpr_at_fpr,
    release_codec_models,
)



def ensure_frozen_model_train_mode(model):
    """Ensures CuDNN RNNs can backward through activations without unfreezing parameters."""
    if model is None:
        return
    for m in model.modules():
        if isinstance(m, torch.nn.RNNBase):
            m.train()
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            m.eval()


class NeuMarkTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.batch_size = cfg.get("batch_size", 8)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("learning_rate", 5e-5)
        self.initial_lr = cfg.get("initial_learning_rate", 1e-6)
        self.num_warmup_steps = cfg.get("num_warmup_steps", 1000)
        self.val_steps = cfg.get("val_steps", 1000)
        self.save_steps = cfg.get("save_steps", 5000)
        self.num_val_samples = cfg.get("num_val_samples", 50)
        self.grad_accum_steps = max(1, cfg.get("gradient_accumulation_steps", 1))
        self.seed = cfg.get("seed", 1234)

        # Loss weights
        self.cos_loss_lambda = cfg.get("cos_loss_lambda", 2.0)
        self.adv_loss_lambda = cfg.get("adv_loss_lambda", 1.0)
        self.dec_loss_lambda = cfg.get("dec_loss_lambda", 10.0)
        self.vad_loss_lambda = cfg.get("vad_loss_lambda", 1.0)
        self.vad_margin = cfg.get("vad_margin", 2.0)
        self.utmos_loss_lambda = cfg.get("utmos_loss_lambda", 0.5)
        self.sim_loss_lambda = cfg.get("sim_loss_lambda", 1.0)
        self.asr_loss_lambda = cfg.get("asr_loss_lambda", 0.5)
        self.mel_loss_lambda = cfg.get("mel_loss_lambda", 0.0)
        self.multi_scale_mel_loss_lambdas = cfg.get("multi_scale_mel_loss_lambdas", [5, 1, 1, 1])
        self.multi_scale_mel_loss_kwargs_list = []
        mult = 1
        n_fft_base = cfg.get("n_fft", 1024)
        hop_size_base = cfg.get("hop_size", 256)
        win_size_base = cfg.get("win_size", 1024)
        num_mels_base = cfg.get("num_mels", 80)
        fmin_base = cfg.get("fmin", 0)
        fmax_base = cfg.get("fmax", 8000)
        for i in range(len(self.multi_scale_mel_loss_lambdas)):
            self.multi_scale_mel_loss_kwargs_list.append({
                "n_fft": n_fft_base // mult,
                "num_mels": num_mels_base,
                "sample_rate": self.sample_rate,
                "hop_size": hop_size_base // mult,
                "win_size": win_size_base // mult,
                "fmin": fmin_base,
                "fmax": fmax_base,
            })
            mult *= 2

        torch.manual_seed(self.seed)

        # Accelerate DDP setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        dl_config = DataLoaderConfiguration(split_batches=False)
        self.accelerator = Accelerator(
            dataloader_config=dl_config,
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.cufft_plan_cache.max_size = 4096
            except Exception:
                pass

        # Setup Logging
        base_results = Path(cfg.get("results_folder", "exp/tts_native_neumark"))
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_folder = base_results / run_name
        self.steps = torch.tensor(0)

        if self.is_main:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            with open(self.results_folder / "config.json", "w") as f:
                json.dump(cfg, f, indent=4)
            self.writer = tensorboard.SummaryWriter(str(self.results_folder / "logs"))

        # Resolve NeuMark root from config hint if provided
        global NEUMARK_ROOT
        if "neumark_root" in cfg:
            NEUMARK_ROOT = find_neumark_root(cfg["neumark_root"])

        # ---------------------------------------------------------
        # Models Init
        # ---------------------------------------------------------
        print(f"[Init] Using NeuMark Root: {NEUMARK_ROOT}")
        print("[Init] Loading SpeechTokenizer...")
        st_cfg_rel = cfg.get("neumark_config", "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json")
        st_ckpt_rel = cfg.get("neumark_st_checkpoint", "STmodels/pretrained_model/SpeechTokenizer.pt")
        st_cfg = str(st_cfg_rel if Path(st_cfg_rel).is_absolute() else NEUMARK_ROOT / st_cfg_rel)
        st_ckpt = str(st_ckpt_rel if Path(st_ckpt_rel).is_absolute() else NEUMARK_ROOT / st_ckpt_rel)
        self.generator = SpeechTokenizer.load_from_checkpoint(st_cfg, st_ckpt).to(self.device)
        self.generator.train()
        ensure_frozen_model_train_mode(self.generator)
        if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
            ensure_frozen_model_train_mode(self.utmos_loss.model)
        for p in self.generator.parameters():
            p.requires_grad = False
        for m in self.generator.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.eval()

        print("[Init] Creating Watermark Embedder & Detector...")
        self.msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(self.device)
        self.detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(self.device)

        print("[Init] Initializing Speech Realism Discriminators...")
        self.discriminators = {
            "mpd": MultiPeriodDiscriminator().to(self.device),
            "msd": MultiScaleDiscriminator().to(self.device),
            "mstftd": MultiScaleSTFTDiscriminator(32).to(self.device),
        }

        # Objective Loss Models
        print("[Init] Initializing Objective Loss Models (UTMOS, WavLM-SIM, Wav2Vec2-ASR)...")
        self.utmos_loss = UTMOSLoss(device=str(self.device))
        self.sim_loss = SpeakerSimLoss(
            checkpoint_path=str(SCRIPT_DIR / cfg.get("wavlm_checkpoint", "models/wavlm_large_finetune.pth")),
            device=str(self.device),
        )
        self.asr_loss = ASRLoss(device=str(self.device))

        # ---------------------------------------------------------
        # Dataset & Dataloader (Lhotse Pre-computed Tokens)
        # ---------------------------------------------------------
        print("[Init] Loading Lhotse Pre-computed Tokens Manifest...")
        train_manifest = str(SCRIPT_DIR / cfg.get("train_manifest"))
        valid_manifest = str(SCRIPT_DIR / cfg.get("valid_manifest"))
        max_dur = cfg.get("max_duration", 16.0)
        self.dl = get_tts_native_dataloader(
            manifest_path=train_manifest,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            max_duration=max_dur,
        )
        self.valid_dl = get_tts_native_dataloader(
            manifest_path=valid_manifest,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            max_duration=max_dur,
        )

        # ---------------------------------------------------------
        # Optimizers & Schedulers
        # ---------------------------------------------------------
        self.trainable_params = list(self.msg_processor.parameters()) + list(self.detector.parameters())
        self.optim_generator = get_optimizer(
            self.trainable_params, lr=self.lr, wd=cfg.get("wd", 0.0), betas=cfg.get("betas", [0.9, 0.99])
        )
        self.optim_discriminators = get_optimizer(
            itertools.chain(*[d.parameters() for d in self.discriminators.values()]),
            lr=self.lr,
            wd=cfg.get("wd", 0.0),
            betas=cfg.get("betas", [0.9, 0.99]),
        )

        num_train_steps = self.epochs * len(self.dl)
        self.scheduler_generator = CosineAnnealingLR(self.optim_generator, T_max=num_train_steps)
        self.scheduler_discriminator = CosineAnnealingLR(self.optim_discriminators, T_max=num_train_steps)

        # ---------------------------------------------------------
        # Checkpoint Resuming / Transfer Loading
        # ---------------------------------------------------------
        resume_ckpt_path = cfg.get("resume_checkpoint", None)
        if resume_ckpt_path:
            p = Path(resume_ckpt_path)
            if not p.is_absolute():
                p = SCRIPT_DIR / p
            if p.exists():
                print(f"[Init] Loading pretrained checkpoint weights from: {p}", flush=True)
                ckpt_dict = torch.load(str(p), map_location="cpu", weights_only=False)

                # 1. Load Embedder / MsgProcessor
                emb_sd = ckpt_dict.get("embedder", ckpt_dict.get("msg_processor", None))
                if emb_sd is not None:
                    clean_emb_sd = {k.replace("module.", ""): v for k, v in emb_sd.items()}
                    self.msg_processor.load_state_dict(clean_emb_sd, strict=False)
                    print("  -> MsgProcessor (Watermark Embedder) weights loaded.", flush=True)

                # 2. Load Detector
                det_sd = ckpt_dict.get("detector", None)
                if det_sd is not None:
                    clean_det_sd = {k.replace("module.", ""): v for k, v in det_sd.items()}
                    self.detector.load_state_dict(clean_det_sd, strict=False)
                    print("  -> Watermark Detector weights loaded.", flush=True)

                # 3. Load Discriminators
                disc_sd = ckpt_dict.get("discriminators", None)
                if disc_sd is not None:
                    for d_name, d_model in self.discriminators.items():
                        if d_name in disc_sd:
                            clean_d_sd = {k.replace("module.", ""): v for k, v in disc_sd[d_name].items()}
                            d_model.load_state_dict(clean_d_sd, strict=False)
                            print(f"  -> Discriminator '{d_name}' weights loaded.", flush=True)

                print(f"[Init] Successfully initialized base weights from {p.name}.", flush=True)
            else:
                print(f"[Warning] Specified resume_checkpoint does not exist: {p}", flush=True)

        # Accelerate prepare
        (
            self.msg_processor,
            self.detector,
            self.optim_generator,
            self.optim_discriminators,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.msg_processor,
            self.detector,
            self.optim_generator,
            self.optim_discriminators,
            self.scheduler_generator,
            self.scheduler_discriminator,
            self.dl,
            self.valid_dl,
        )
        self.discriminators = {
            name: self.accelerator.prepare(d) for name, d in self.discriminators.items()
        }

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def nchunk_size(self) -> int:
        return self.accelerator.unwrap_model(self.msg_processor).nchunk_size

    def warmup(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / max(1, self.num_warmup_steps)
        return self.lr

    def log(self, values: dict, step: int):
        if not self.is_main:
            return
        for k, v in values.items():
            self.writer.add_scalar(k, v, global_step=step)

    def detect_watermark(self, audio: torch.Tensor, return_logits: bool = True):
        generator = self.accelerator.unwrap_model(self.generator)
        embedding = generator.forward_feature(audio)
        if return_logits:
            return self.detector(embedding)
        detector = self.accelerator.unwrap_model(self.detector)
        return detector.detect_watermark(embedding)

    def validate(self, step: int):
        """Evaluate watermark extraction and detection across the full DSP + Neural Codec suite."""
        if not self.is_main:
            return

        print(f"\n[Validation @ Step {step:07d}] Running Full Robustness Suite ({self.num_val_samples} samples)...", flush=True)
        generator = self.accelerator.unwrap_model(self.generator)
        detector = self.accelerator.unwrap_model(self.detector)
        msg_proc = self.accelerator.unwrap_model(self.msg_processor)

        generator.eval()
        detector.eval()
        msg_proc.eval()

        val_attacks = get_validation_attack_suite(self.sample_rate)
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
            }
            attack_scores[key] = {
                "pos_det_scores": [],
                "neg_det_scores": [],
                "pos_wm_scores": [],
                "neg_wm_scores": [],
            }

        clean_utmos_list, wm_utmos_list = [], []
        clean_sim_list, wm_sim_list = [], []
        clean_wer_list, wm_wer_list = [], []
        clean_cer_list, wm_cer_list = [], []

        with torch.no_grad():
            count = 0
            for batch in self.valid_dl:
                if count >= self.num_val_samples:
                    break

                codes = batch["codes"].to(self.device)  # [1, 8, T]
                prompt_audio = batch["prompt_audio"].to(self.device)  # [1, 1, T_p]
                texts = batch["texts"]
                batch_size = codes.size(0)
                message = torch.randint(0, 2, (batch_size, 16), dtype=torch.int64, device=self.device)

                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [generator.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]
                
                # 1. Clean TTS Reconstruction (without watermark)
                z_clean = sum(quantized_layers)
                clean_audio = generator.decoder(z_clean)

                # 2. Watermarked TTS Synthesis
                watermarked_layers = [msg_proc(q, message) for q in quantized_layers]
                z_wm = sum(watermarked_layers)
                wm_audio = generator.decoder(z_wm)

                # 3. UTMOS Evaluation (Clean vs WM)
                if getattr(self.utmos_loss, "model", None) is not None:
                    try:
                        c_u = self.utmos_loss.model(clean_audio.squeeze(1), self.sample_rate).mean().item()
                        w_u = self.utmos_loss.model(wm_audio.squeeze(1), self.sample_rate).mean().item()
                        clean_utmos_list.append(c_u)
                        wm_utmos_list.append(w_u)
                    except Exception:
                        pass

                # 4. Speaker SIM Evaluation (Clean vs WM)
                if hasattr(self, "sim_loss"):
                    try:
                        ref_spk = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else clean_audio
                        c_s = self.sim_loss.get_similarity(clean_audio, ref_spk, self.sample_rate)
                        w_s = self.sim_loss.get_similarity(wm_audio, ref_spk, self.sample_rate)
                        clean_sim_list.append(c_s)
                        wm_sim_list.append(w_s)
                    except Exception:
                        pass

                # 5. ASR WER / CER Evaluation (Clean vs WM)
                if getattr(self.asr_loss, "model", None) is not None:
                    try:
                        c_hyps = self.asr_loss.decode_greedy(clean_audio, self.sample_rate)
                        w_hyps = self.asr_loss.decode_greedy(wm_audio, self.sample_rate)
                        for ref_t, c_h, w_h in zip(texts, c_hyps, w_hyps):
                            c_wer, c_cer = compute_wer_cer(ref_t, c_h)
                            w_wer, w_cer = compute_wer_cer(ref_t, w_h)
                            clean_wer_list.append(c_wer)
                            clean_cer_list.append(c_cer)
                            wm_wer_list.append(w_wer)
                            wm_cer_list.append(w_cer)
                    except Exception:
                        pass

                # Robustness Evaluation across DSP + Codec attacks (both positive and negative audio)
                for cat, name, detail, atk_fn in val_attacks:
                    key = name if cat == "DSP" else f"{name} {detail}"
                    
                    # 1. Positive Sample (Watermarked Audio)
                    try:
                        attacked_wm = atk_fn(wm_audio)
                    except Exception:
                        attacked_wm = wm_audio

                    embedding_wm = generator.forward_feature(attacked_wm)
                    prob_wm_t, pred_bits_wm, _ = detector.detect_watermark(embedding_wm)
                    prob_wm = float(prob_wm_t.mean().item())
                    bit_matches = (pred_bits_wm.long() == message.long()).sum().item()
                    tp_flag = 1 if prob_wm >= 0.5 else 0

                    # 2. Negative Sample (Clean / Unwatermarked Audio)
                    try:
                        attacked_cl = atk_fn(clean_audio)
                    except Exception:
                        attacked_cl = clean_audio

                    embedding_cl = generator.forward_feature(attacked_cl)
                    prob_cl_t, pred_bits_cl, _ = detector.detect_watermark(embedding_cl)
                    prob_cl = float(prob_cl_t.mean().item())
                    cl_bit_matches = (pred_bits_cl.long() == message.long()).sum().item()
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

                count += 1

        # Summary statistics & Dual ROC-AUC (Detection & Bit-Matching Extraction)
        summary = {}
        for key, stats in results.items():
            bit_acc = stats["bit_matches"] / max(1, stats["total_bits"])
            pos_acc = stats["pos_matches"] / max(1, stats["pos_frames"])
            neg_acc = stats["neg_matches"] / max(1, stats["neg_frames"])
            detect_acc = 0.5 * (pos_acc + neg_acc)

            # 1. Detection ROC-AUC & TPR@0.1% FPR
            pos_d = attack_scores[key]["pos_det_scores"]
            neg_d = attack_scores[key]["neg_det_scores"]
            y_det_true = [0] * len(neg_d) + [1] * len(pos_d)
            y_det_scores = neg_d + pos_d
            det_auc, det_tpr_001 = compute_auc_and_tpr_at_fpr(y_det_true, y_det_scores, target_fpr=0.001)

            # 2. Watermark Bit-Matching Extraction ROC-AUC & TPR@0.1% FPR
            pos_w = attack_scores[key]["pos_wm_scores"]
            neg_w = attack_scores[key]["neg_wm_scores"]
            y_wm_true = [0] * len(neg_w) + [1] * len(pos_w)
            y_wm_scores = neg_w + pos_w
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
            # TensorBoard logging
            clean_tag = key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_").replace("+", "p").replace("%", "pct").replace("-", "_")
            self.writer.add_scalar(f"val/{clean_tag}/bit_acc", bit_acc, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/detect_acc", detect_acc, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/det_roc_auc", det_auc, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/det_tpr_001", det_tpr_001, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/wm_roc_auc", wm_auc, global_step=step)
            self.writer.add_scalar(f"val/{clean_tag}/wm_tpr_001", wm_tpr_001, global_step=step)

        # Compute average quality metrics (Clean vs Watermarked)
        c_ut = sum(clean_utmos_list) / max(1, len(clean_utmos_list)) if clean_utmos_list else 0.0
        w_ut = sum(wm_utmos_list) / max(1, len(wm_utmos_list)) if wm_utmos_list else 0.0
        c_sim = sum(clean_sim_list) / max(1, len(clean_sim_list)) if clean_sim_list else 0.0
        w_sim = sum(wm_sim_list) / max(1, len(wm_sim_list)) if wm_sim_list else 0.0
        c_wer = sum(clean_wer_list) / max(1, len(clean_wer_list)) if clean_wer_list else 0.0
        w_wer = sum(wm_wer_list) / max(1, len(wm_wer_list)) if wm_wer_list else 0.0
        c_cer = sum(clean_cer_list) / max(1, len(clean_cer_list)) if clean_cer_list else 0.0
        w_cer = sum(wm_cer_list) / max(1, len(wm_cer_list)) if wm_cer_list else 0.0

        quality_metrics = {
            "clean_utmos": c_ut, "wm_utmos": w_ut,
            "clean_sim": c_sim, "wm_sim": w_sim,
            "clean_wer": c_wer, "wm_wer": w_wer,
            "clean_cer": c_cer, "wm_cer": w_cer,
        }

        # Log quality metrics & deltas to TensorBoard
        self.writer.add_scalar("val/quality/clean_utmos", c_ut, global_step=step)
        self.writer.add_scalar("val/quality/wm_utmos", w_ut, global_step=step)
        self.writer.add_scalar("val/quality/delta_utmos", w_ut - c_ut, global_step=step)

        self.writer.add_scalar("val/quality/clean_speaker_sim", c_sim, global_step=step)
        self.writer.add_scalar("val/quality/wm_speaker_sim", w_sim, global_step=step)
        self.writer.add_scalar("val/quality/delta_speaker_sim", w_sim - c_sim, global_step=step)

        self.writer.add_scalar("val/quality/clean_asr_wer", c_wer, global_step=step)
        self.writer.add_scalar("val/quality/wm_asr_wer", w_wer, global_step=step)
        self.writer.add_scalar("val/quality/delta_asr_wer", w_wer - c_wer, global_step=step)

        self.writer.add_scalar("val/quality/clean_asr_cer", c_cer, global_step=step)
        self.writer.add_scalar("val/quality/wm_asr_cer", w_cer, global_step=step)
        self.writer.add_scalar("val/quality/delta_asr_cer", w_cer - c_cer, global_step=step)

        # Print formatted table
        table_str = format_full_validation_table(step, summary, quality_metrics=quality_metrics)
        print(table_str, flush=True)

        # Release cached attack models from GPU memory
        release_codec_models()
        print(f"[Validation @ Step {step:07d}] Codec cache released. Returning to training.\n", flush=True)

    def train(self):
        print(f"[Training] Starting NeuMark Training for {self.epochs} epochs...")
        self.generator.train()
        ensure_frozen_model_train_mode(self.generator)
        if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
            ensure_frozen_model_train_mode(self.utmos_loss.model)
        for m in self.generator.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.eval()
        self.msg_processor.train()
        self.detector.train()
        for d in self.discriminators.values():
            d.train()

        steps = int(self.steps.item())
        for epoch in range(self.epochs):
            for i, batch in enumerate(tqdm(self.dl, desc=f"Epoch {epoch}", disable=not self.is_main, ncols=100)):
                codes = batch["codes"].to(self.device)  # [B, 8, T]
                real_audio = batch["audio"].to(self.device)  # [B, 1, T_samples]
                prompt_audio = batch["prompt_audio"].to(self.device)  # [B, 1, T_p]
                texts = batch["texts"]
                batch_size = codes.size(0)

                # Sample random 16-bit watermark message
                message = torch.randint(0, 2, (batch_size, 16), dtype=torch.int64, device=self.device)

                # -------------------------------------------------------------
                # 1. Decode Pre-computed Codes to Continuous Latents & Embed
                # -------------------------------------------------------------
                if i % self.grad_accum_steps == 0:
                    self.optim_generator.zero_grad()
                    if self.adv_loss_lambda > 0:
                        self.optim_discriminators.zero_grad()

                unwrapped_gen = self.accelerator.unwrap_model(self.generator)

                # Permute codes to [8, B, T] layout for SpeechTokenizer RVQ decode
                codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
                quantized_layers = [
                    unwrapped_gen.quantizer.decode(codes_qbt[k : k + 1], st=k)
                    for k in range(8)
                ]  # list of 8 tensors [B, 1024, T]
                z_q = sum(quantized_layers)

                # Layer-wise watermark injection
                watermarked_layers = [
                    self.msg_processor(q, message) for q in quantized_layers
                ]
                z_wm = sum(watermarked_layers)

                # Decode unwatermarked clean latent to waveform as transparent mel reference (no grad)
                with torch.no_grad():
                    recon_audio = unwrapped_gen.decoder(z_q)

                # Decode watermarked latent to waveform
                wm_audio = unwrapped_gen.decoder(z_wm)

                # Align temporal dimensions
                min_len = min(real_audio.shape[-1], wm_audio.shape[-1], recon_audio.shape[-1])
                real_audio_aligned = real_audio[..., :min_len]
                recon_audio_aligned = recon_audio[..., :min_len]
                wm_audio_aligned = wm_audio[..., :min_len]

                # -------------------------------------------------------------
                # 2. Compute TTS-Native Losses
                # -------------------------------------------------------------
                # A. Latent Space Cosine Loss (Soft Anchor)
                loss_cos = latent_cosine_loss(z_wm, z_q) * self.cos_loss_lambda

                # B. UTMOS Naturalness Loss (Absolute MOS Maximization)
                loss_utmos = (self.utmos_loss(wm_audio, self.sample_rate) * self.utmos_loss_lambda) if self.utmos_loss_lambda > 0 else torch.zeros((), device=self.device)

                # C. Speaker Similarity Loss (WavLM vs Prompt / Clean Speech Reference)
                ref_prompt = prompt_audio if (prompt_audio.numel() > 0 and prompt_audio.abs().max() > 1e-4) else recon_audio_aligned.detach()
                loss_sim = (self.sim_loss(wm_audio, ref_prompt, self.sample_rate) * self.sim_loss_lambda) if self.sim_loss_lambda > 0 else torch.zeros((), device=self.device)

                # D. ASR Pronunciation Loss (CTC vs Target Text)
                loss_asr = (self.asr_loss(wm_audio, texts, self.sample_rate) * self.asr_loss_lambda) if self.asr_loss_lambda > 0 else torch.zeros((), device=self.device)

                # Mel Loss (Multi-Scale Spectrogram L1 against unwatermarked clean audio)
                loss_mel = torch.zeros((), device=self.device)
                if self.mel_loss_lambda > 0:
                    loss_mel = sum(
                        mel_k[0] * mel_loss(recon_audio_aligned, wm_audio_aligned, **mel_k[1])
                        for mel_k in zip(self.multi_scale_mel_loss_lambdas, self.multi_scale_mel_loss_kwargs_list)
                    ) * self.mel_loss_lambda

                # E. GAN Adversarial & Feature Matching (Real = recon_audio, Fake = wm_audio)
                loss_adv = torch.zeros((), device=self.device)
                loss_fm = torch.zeros((), device=self.device)
                adversarial_components = {}
                discriminator_components = {}

                if self.adv_loss_lambda > 0:
                    for discriminator in self.discriminators.values():
                        discriminator.requires_grad_(False)

                    fm_components = []
                    for name, discriminator in self.discriminators.items():
                        d_unwrapped = self.accelerator.unwrap_model(discriminator)
                        _, fake_preds, real_fmaps, fake_fmaps = d_unwrapped(
                            recon_audio_aligned, wm_audio_aligned
                        )
                        comp = torch.stack([adversarial_loss_g(p) for p in fake_preds]).mean()
                        adversarial_components[name] = comp
                        fm_components.append(feature_loss(real_fmaps, fake_fmaps))

                    loss_fm = torch.stack(fm_components).mean()
                    loss_adv = (
                        torch.stack(list(adversarial_components.values())).mean() + 2.0 * loss_fm
                    ) * self.adv_loss_lambda

                # F. Watermark Decoding & Detection Loss (Strictly Encodec 3/6/12k, VC Masking, Clean)
                augmented_audio, vad_labels, attack_name = apply_train_augmentation(
                    wm_audio, sample_rate=self.sample_rate, orig_audio=real_audio
                )
                logits, chunk_logits = self.detect_watermark(augmented_audio, return_logits=True)

                target_chunks = bits_to_chunks(message.float(), chunk_size=self.nchunk_size)
                loss_dec = decoding_loss(chunk_logits, target_chunks) * self.dec_loss_lambda

                min_lens_pos = min(logits.shape[-1], vad_labels.shape[-1])
                loss_vad_pos = margin_vad_loss(logits[..., :min_lens_pos], vad_labels[..., :min_lens_pos], margin=self.vad_margin, from_logits=True) * self.vad_loss_lambda

                # Negative sample supervision (Alternating Real Audio and Clean Recon Audio matching NeuMark)
                if steps % 2 == 0:
                    augmented_neg, _, _ = apply_train_augmentation(
                        real_audio, sample_rate=self.sample_rate, orig_audio=None
                    )
                else:
                    augmented_neg, _, _ = apply_train_augmentation(
                        recon_audio, sample_rate=self.sample_rate, orig_audio=None
                    )
                neg_logits, _ = self.detect_watermark(augmented_neg, return_logits=True)
                vad_labels_neg = torch.zeros_like(neg_logits)
                min_lens_neg = min(neg_logits.shape[-1], vad_labels_neg.shape[-1])
                loss_vad_neg = margin_vad_loss(neg_logits[..., :min_lens_neg], vad_labels_neg[..., :min_lens_neg], margin=self.vad_margin, from_logits=True) * self.vad_loss_lambda

                # -------------------------------------------------------------
                # 3. Total Loss & Generator Backward
                # -------------------------------------------------------------
                total_loss = (
                    loss_dec
                    + loss_vad_pos
                    + loss_vad_neg
                    + loss_cos
                    + loss_adv
                    + loss_mel
                    + loss_utmos
                    + loss_sim
                    + loss_asr
                )

                # Normalize loss for gradient accumulation
                self.accelerator.backward(total_loss / self.grad_accum_steps)

                # -------------------------------------------------------------
                # 4. Train Discriminators
                # -------------------------------------------------------------
                loss_D = torch.zeros((), device=self.device)
                if self.adv_loss_lambda > 0:
                    for discriminator in self.discriminators.values():
                        discriminator.requires_grad_(True)

                    for name, discriminator in self.discriminators.items():
                        real_preds, fake_preds, _, _ = discriminator(
                            recon_audio_aligned, wm_audio_aligned.detach()
                        )
                        comp = torch.stack([adversarial_loss_d(r, f) for r, f in zip(real_preds, fake_preds)]).mean()
                        discriminator_components[name] = comp

                    loss_D = torch.stack(list(discriminator_components.values())).mean()
                    self.accelerator.backward(loss_D / self.grad_accum_steps)

                # Optimizer step on accumulation boundary
                if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.dl):
                    self.optim_generator.step()
                    self.optim_generator.zero_grad()
                    if self.adv_loss_lambda > 0:
                        self.optim_discriminators.step()
                        self.optim_discriminators.zero_grad()

                    if steps < self.num_warmup_steps:
                        lr = self.warmup(steps)
                        for pg in self.optim_generator.param_groups:
                            pg["lr"] = lr
                        for pg in self.optim_discriminators.param_groups:
                            pg["lr"] = lr
                    else:
                        self.scheduler_generator.step()
                        self.scheduler_discriminator.step()

                if steps % self.cfg.get("log_steps", 50) == 0 or steps in {1, 5, 10}:
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/dec_loss": loss_dec.item(),
                        "train/vad_pos": loss_vad_pos.item(),
                        "train/vad_neg": loss_vad_neg.item(),
                        "train/cos_loss": loss_cos.item(),
                        "train/utmos_loss": loss_utmos.item(),
                        "train/sim_loss": loss_sim.item(),
                        "train/asr_loss": loss_asr.item(),
                        "train/adv_loss": loss_adv.item(),
                        "train/fm_loss": loss_fm.item(),
                        "train/d_loss": loss_D.item(),
                    }
                    self.log(log_dict, step=steps)

                # Periodic Checkpoint Saving
                if steps > 0 and steps % self.save_steps == 0:
                    self.save(steps, epoch)

                # Validation Loop
                if steps > 0 and steps % self.val_steps == 0:
                    self.validate(steps)
                    self.generator.train()
                    ensure_frozen_model_train_mode(self.generator)
                    if hasattr(self, "utmos_loss") and getattr(self.utmos_loss, "model", None) is not None:
                        ensure_frozen_model_train_mode(self.utmos_loss.model)
                    self.msg_processor.train()
                    self.detector.train()
                    for d in self.discriminators.values():
                        d.train()

                # Periodic CUDA memory & cuFFT plan cache cleanup
                if steps > 0 and steps % 200 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.backends.cuda.cufft_plan_cache.clear()
                        except Exception:
                            pass

                self.steps += 1
                steps = int(self.steps.item())

            # Save checkpoint once every epoch
            self.save(steps, epoch)

        self.save(steps, self.epochs, final=True)
        print(f"[Done] Training complete after {steps} steps.")

    def save(self, steps: int, epoch: int, final: bool = False):
        if not self.is_main:
            return
        prefix = "NeuMark_final" if final else f"NeuMark_step_{steps:07d}_epoch_{epoch:03d}"
        ckpt_path = self.results_folder / f"{prefix}.pt"
        pkg = dict(
            msg_processor=self.accelerator.get_state_dict(self.msg_processor),
            detector=self.accelerator.get_state_dict(self.detector),
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},
            optim_generator=self.optim_generator.state_dict(),
            optim_discriminators=self.optim_discriminators.state_dict(),
            steps=steps,
            epoch=epoch,
        )
        torch.save(pkg, ckpt_path)
        print(f"[Checkpoint] Saved model checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config_tts_native.json"), help="Config path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    trainer = NeuMarkTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
