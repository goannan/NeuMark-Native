#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo)
# Copyright    2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
Usage:
python3 bin/trainer.py \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
    --max-duration 40 --model-name valle \
    --exp-dir exp/valle
    --dtype "bfloat16" \
"""

import argparse
import copy
import itertools
import json
import logging
import os
from contextlib import nullcontext

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from valle.data import TtsDataModule
from valle.models import add_model_arguments, get_model
from valle.modules.optim import Eden, Eve, ScaledAdam
from valle.modules.scheduler import get_scheduler

# TraceableSpeech imports
import sys
project_root = Path(__file__).resolve().parent.parent.parent
ts_dir = project_root / "traceableSpeech"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(ts_dir) not in sys.path:
    sys.path.append(str(ts_dir))

from traceableSpeech.env import AttrDict
from traceableSpeech.models import (
    Generator, Encoder, Quantizer,
    MultiPeriodDiscriminator, MultiScaleDiscriminator,
    feature_loss, generator_loss, discriminator_loss,
)
from traceableSpeech.msstftd import MultiScaleSTFTDiscriminator
from traceableSpeech.watermark import Watermark_Encoder, Watermark_Decoder, Random_watermark, sign_loss, attack
from traceableSpeech.train import reconstruction_loss
from traceableSpeech.meldataset import mel_spectrogram

LRSchedulerType = torch.optim.lr_scheduler._LRScheduler


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module

    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/valle_dev",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="ScaledAdam",
        help="The optimizer.",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="Eden",
        help="The scheduler.",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train %% save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=10000,
        help="""Run validation if batch_idx %% valid_interval is 0.""",
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=0,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--accumulate-grad-steps",
        type=int,
        default=1,
        help="""update gradient when batch_idx_train %% accumulate_grad_steps == 0.
        """,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Training dtype: float32 bfloat16 float16.",
    )

    parser.add_argument(
        "--filter-min-duration",
        type=float,
        default=0.0,
        help="Keep only utterances with duration > this.",
    )
    parser.add_argument(
        "--filter-max-duration",
        type=float,
        default=20.0,
        help="Keep only utterances with duration < this.",
    )

    parser.add_argument(
        "--train-stage",
        type=int,
        default=0,
        help="""0: train all modules, For VALL-E, support 1: AR Decoder 2: NAR Decoder(s)
        """,
    )

    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=False,
        help="visualize model results in eval step.",
    )

    parser.add_argument(
        "--oom-check",
        type=str2bool,
        default=True,
        help="perform OOM check on dataloader batches before starting training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,  # 10: debug 100: train
            "reset_interval": 200,
            "valid_interval": 10000,
            # parameters for TTS
            "env_info": get_env_info(),
        }
    )

    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    if isinstance(model, DDP):
        raise ValueError("load_checkpoint before DDP")

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    saved_stage = saved_params.get("train_stage", 0)
    if params.train_stage != saved_stage:
        # switch training stage
        if params.train_stage and saved_stage:  # switch between 1 and 2
            params.start_epoch = 1
            params.start_batch = 0
        else:
            # switch between 0 and 1/2
            assert params.num_epochs >= params.start_epoch
            params.batch_idx_train = saved_params["batch_idx_train"]

        for key in ["optimizer", "grad_scaler", "sampler"]:
            if key in saved_params:
                saved_params.pop(key)

        # when base on stage 0, we keep scheduler
        if saved_stage != 0:
            for key in ["scheduler"]:
                if key in saved_params:
                    saved_params.pop(key)

        best_train_filename = params.exp_dir / "best-train-loss.pt"
        if best_train_filename.is_file():
            copyfile(
                src=best_train_filename,
                dst=params.exp_dir / f"best-train-loss-stage{saved_stage}.pt",
            )

        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        if best_valid_filename.is_file():
            copyfile(
                src=best_valid_filename,
                dst=params.exp_dir / f"best-valid-loss-stage{saved_stage}.pt",
            )
    else:

        keys = [
            "best_train_epoch",
            "best_valid_epoch",
            "batch_idx_train",
            "best_train_loss",
            "best_valid_loss",
        ]
        for k in keys:
            params[k] = saved_params[k]

        if params.start_batch > 0:
            if "cur_epoch" in saved_params:
                params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_ts_checkpoint(
    filename: Path,
    ts_models: Dict[str, nn.Module],
    rank: int = 0,
) -> None:
    if rank != 0:
        return
    
    # Save in the same format as TraceableSpeech g_xxxxxx
    def get_state_dict(m):
        return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()

    checkpoint = {
        'encoder': get_state_dict(ts_models["encoder"]),
        'generator': get_state_dict(ts_models["generator"]),
        'quantizer_Audio': get_state_dict(ts_models["quantizer"]),
        'watermark_encoder': get_state_dict(ts_models["watermark_encoder"]),
        'watermark_decoder': get_state_dict(ts_models["watermark_decoder"]),
    }
    # Also save discriminator state dicts
    if "mpd" in ts_models:
        checkpoint['mpd'] = get_state_dict(ts_models["mpd"])
        checkpoint['msd'] = get_state_dict(ts_models["msd"])
        checkpoint['mstftd'] = get_state_dict(ts_models["mstftd"])
    torch.save(checkpoint, filename)
    logging.info(f"Saved TraceableSpeech checkpoint to {filename}")


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    ts_models: Dict[str, nn.Module],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    # Also save TS checkpoint
    ts_filename = params.exp_dir / f"ts_epoch-{params.cur_epoch}.pt"
    save_ts_checkpoint(ts_filename, ts_models, rank)

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    ts_models: Dict[str, nn.Module],
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute joint loss for VALL-E and TraceableSpeech.
    Returns (predicts, total_gen_loss, info).
    Discriminator step is handled separately in train_one_epoch.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )

    # VALL-E Inputs
    text_tokens = batch["text_tokens"].to(device)
    text_tokens_lens = batch["text_tokens_lens"].to(device)

    # Audio for TraceableSpeech
    audio = batch["audio"].to(device) # [B, T]
    audio_lens = batch["audio_lens"].to(device)

    # 1. Generate target tokens on-the-fly using frozen Encoder/Quantizer
    with torch.no_grad():
        # Ensure audio is [B, 1, T] for TS Encoder
        ts_audio = audio.unsqueeze(1)
        z = ts_models["encoder"](ts_audio)
        q_target, loss_q_ts, c_indices = ts_models["quantizer"](z)
        # c_indices is a list of [B*T/320] tensors
        # VALL-E expects y: [B, T_tokens, 8]
        y_target = torch.stack([code.reshape(audio.size(0), -1) for code in c_indices], -1)

    # 2. VALL-E Forward
    y_lens = audio_lens // 320
    y_lens = torch.clamp(y_lens, min=1, max=y_target.shape[1])

    with torch.set_grad_enabled(is_training):
        predicts, valle_loss, metrics = model(
            x=text_tokens,
            x_lens=text_tokens_lens,
            y=y_target,
            y_lens=y_lens,
            train_stage=params.train_stage,
        )

    # 3. TraceableSpeech Forward (Segmented for memory efficiency)
    # 3-second segment cropping (72000 samples @ 24kHz)
    segment_sample_size = 72000
    segment_frame_size = segment_sample_size // 320 # 225 frames

    sign = Random_watermark(audio.size(0)).to(device)
    sign_en = ts_models["watermark_encoder"](sign)

    if ts_audio.size(-1) > segment_sample_size:
        max_start_frame = (ts_audio.size(-1) - segment_sample_size) // 320
        start_frame = torch.randint(0, max_start_frame + 1, (1,)).item()
        start_sample = start_frame * 320
        
        ts_audio_seg = ts_audio[..., start_sample : start_sample + segment_sample_size]
        q_target_seg = q_target[..., start_frame : start_frame + segment_frame_size]
    else:
        # Pad if shorter than 3s
        ts_audio_seg = F.pad(ts_audio, (0, segment_sample_size - ts_audio.size(-1)))
        q_target_seg = F.pad(q_target, (0, segment_frame_size - q_target.size(-1)))

    # Generator forward using segmented quantized target
    # This significantly reduces memory usage as G/D only process 3s instead of full length
    y_g_hat_seg = ts_models["generator"](q_target_seg, sign_en)

    # 4. TraceableSpeech Losses (Using segments)

    # 4a. Watermark loss after attack
    y_g_hat_attacked_seg, _ = attack(y_g_hat_seg, [
        ("CLP", 0.4), ("RSP-90", 0.05), ("Noise-W35", 0.25),
        ("SS-01", 0.05), ("AS-90", 0.05), ("EA-0301", 0.15), ("LP5000", 0.05),
    ])
    
    y_g_hat_attacked_seg_mel = mel_spectrogram(
        y_g_hat_attacked_seg.squeeze(1), 1024, 80, 24000, 320, 1024, 0, 8000
    )
    sign_score, _ = ts_models["watermark_decoder"](y_g_hat_attacked_seg_mel)
    loss_watermark = sign_loss(sign_score, sign)

    # 4b. Reconstruction loss (wav L1 + Multi-scale Mel internal)
    loss_recon = reconstruction_loss(ts_audio_seg, y_g_hat_seg, device)

    # 4c. Main Mel-Spectrogram Loss (L1) and Multi-scale Mel
    ts_audio_seg_mel = mel_spectrogram(
        ts_audio_seg.squeeze(1), 1024, 80, 24000, 320, 1024, 0, 8000
    )
    y_g_hat_seg_mel = mel_spectrogram(
        y_g_hat_seg.squeeze(1), 1024, 80, 24000, 320, 1024, 0, 8000
    )
    
    y_r_mel_1 = mel_spectrogram(ts_audio_seg.squeeze(1), 512, 80, 24000, 120, 512, 0, 8000)
    y_g_mel_1 = mel_spectrogram(y_g_hat_seg.squeeze(1), 512, 80, 24000, 120, 512, 0, 8000)
    y_r_mel_2 = mel_spectrogram(ts_audio_seg.squeeze(1), 256, 80, 24000, 60, 256, 0, 8000)
    y_g_mel_2 = mel_spectrogram(y_g_hat_seg.squeeze(1), 256, 80, 24000, 60, 256, 0, 8000)
    
    loss_mel = F.l1_loss(ts_audio_seg_mel, y_g_hat_seg_mel) * 45 + \
               F.l1_loss(y_r_mel_1, y_g_mel_1) + \
               F.l1_loss(y_r_mel_2, y_g_mel_2)

    # 4d. GAN losses (Generator side)
    if "mpd" in ts_models:
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = ts_models["mpd"](ts_audio_seg, y_g_hat_seg)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = ts_models["msd"](ts_audio_seg, y_g_hat_seg)
        y_stftd_hat_r, fmap_stftd_r = ts_models["mstftd"](ts_audio_seg)
        y_stftd_hat_g, fmap_stftd_g = ts_models["mstftd"](y_g_hat_seg)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_stft, _ = generator_loss(y_stftd_hat_g)

        loss_gan = loss_gen_s + loss_gen_f + loss_gen_stft
        loss_fm = loss_fm_s + loss_fm_f + loss_fm_stft
    else:
        loss_gan = torch.tensor(0.0, device=device)
        loss_fm = torch.tensor(0.0, device=device)

    # 5. Combined Generator Total Loss
    loss_gen_all = loss_recon + loss_mel + loss_gan + loss_fm + loss_watermark * 5.0 + loss_q_ts * 10
    loss_v_gen = loss_gen_all + valle_loss


    info = MetricsTracker()
    # frames field is required for averaging
    info["frames"] = y_lens.sum().item()
    info["valle_loss"] = valle_loss.detach().cpu().item()
    info["recon_loss"] = loss_recon.detach().cpu().item()
    info["mel_loss"] = loss_mel.detach().cpu().item()
    info["gan_loss"] = loss_gan.detach().cpu().item()
    info["fm_loss"] = loss_fm.detach().cpu().item()
    info["wm_loss"] = loss_watermark.detach().cpu().item()
    info["loss"] = loss_v_gen.detach().cpu().item()

    return predicts, loss_v_gen, info


def compute_disc_loss(
    ts_models: Dict[str, nn.Module],
    batch: dict,
) -> Tensor:
    """
    Compute discriminator loss for TraceableSpeech GAN training.
    This runs the frozen Generator output through MPD/MSD/MSTFTD discriminators.
    """
    if "mpd" not in ts_models:
        return None

    device = next(ts_models["generator"].parameters()).device

    audio = batch["audio"].to(device)

    with torch.no_grad():
        ts_audio = audio.unsqueeze(1)
        z = ts_models["encoder"](ts_audio)
        q_target, _, _ = ts_models["quantizer"](z)
        sign = Random_watermark(audio.size(0)).to(device)
        sign_en = ts_models["watermark_encoder"](sign)

        # --- Segment Cropping for TS Efficiency ---
        segment_sample_size = 72000
        segment_frame_size = segment_sample_size // 320

        if ts_audio.size(-1) > segment_sample_size:
            max_start_frame = (ts_audio.size(-1) - segment_sample_size) // 320
            start_frame = torch.randint(0, max_start_frame + 1, (1,)).item()
            start_sample = start_frame * 320
            
            ts_audio_seg = ts_audio[..., start_sample : start_sample + segment_sample_size]
            q_target_seg = q_target[..., start_frame : start_frame + segment_frame_size]
        else:
            ts_audio_seg = F.pad(ts_audio, (0, segment_sample_size - ts_audio.size(-1)))
            q_target_seg = F.pad(q_target, (0, segment_frame_size - q_target.size(-1)))

        y_g_hat_seg = ts_models["generator"](q_target_seg, sign_en)

    # Discriminator losses (using segments)
    y_df_hat_r, y_df_hat_g, _, _ = ts_models["mpd"](ts_audio_seg, y_g_hat_seg.detach())
    loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

    y_ds_hat_r, y_ds_hat_g, _, _ = ts_models["msd"](ts_audio_seg, y_g_hat_seg.detach())
    loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

    y_disc_r, _ = ts_models["mstftd"](ts_audio_seg)
    y_disc_gen, _ = ts_models["mstftd"](y_g_hat_seg.detach())
    loss_disc_stft, _, _ = discriminator_loss(y_disc_r, y_disc_gen)

    return loss_disc_s + loss_disc_f + loss_disc_stft


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    ts_models: Dict[str, nn.Module],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        predicts, loss, loss_info = compute_loss(
            params=params,
            model=model,
            ts_models=ts_models,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
    if world_size > 1:
        tot_loss.reduce(loss.device)
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    if params.visualize:
        output_dir = Path(
            f"{params.exp_dir}/eval/step-{params.batch_idx_train:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(model, DDP):
            model.module.visualize(predicts, batch, output_dir=output_dir)
        else:
            model.visualize(predicts, batch, output_dir=output_dir)

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    ts_models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    rng: random.Random,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      rng:
        Random for selecting.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    # -------- FIX: define dtype and enabled --------
    if params.train_stage == 1:
        dtype = torch.bfloat16 if params.dtype == "bfloat16" else torch.float16
        enabled = True
    else:
        dtype = torch.float32
        enabled = False
    tot_loss = MetricsTracker()
    try:
        total_batches = len(train_dl)
    except Exception:
        total_batches = None

    batch_idx = 0
    # iterate batches with a visible progress bar
    # iterate batches with a visible progress bar
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total_batches, desc=f"Epoch {params.cur_epoch}", unit="batch")
    
    for batch in train_dl:
            batch_idx += 1
            params.batch_idx_train += 1
            batch_size = len(batch["text"])

            try:
                # --- Discriminator step ---
                if "mpd" in ts_models:
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                        disc_loss = compute_disc_loss(
                            ts_models=ts_models,
                            batch=batch,
                        )
                    if disc_loss is not None:
                        optim_d.zero_grad()
                        scaler.scale(disc_loss).backward()
                        scaler.step(optim_d)
                        scaler.update()

                # --- Generator + VALL-E step ---
                with torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                    _, loss, loss_info = compute_loss(
                        params=params,
                        model=model,
                        ts_models=ts_models,
                        batch=batch,
                        is_training=True,
                    )

                # summary stats
                tot_loss = (
                    tot_loss * (1 - 1 / params.reset_interval)
                ) + loss_info * (1 / params.reset_interval)

                scaler.scale(loss).backward()
                if params.batch_idx_train >= params.accumulate_grad_steps:
                    if (
                        params.batch_idx_train % params.accumulate_grad_steps
                        == 0
                    ):
                        if params.optimizer_name not in ["ScaledAdam", "Eve"]:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0
                            )

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        for k in range(params.accumulate_grad_steps):
                            if isinstance(scheduler, Eden):
                                scheduler.step_batch(params.batch_idx_train)
                            else:
                                scheduler.step()

                set_batch_count(model, params.batch_idx_train)
            except:  # noqa
                display_and_save_batch(batch, params=params)
                raise

            # update progress bar and optional postfix
            if rank == 0 and pbar is not None:
                try:
                    pbar.set_postfix(loss=float(loss_info["loss"]))
                except Exception:
                    pass
                pbar.update(1)

            if params.average_period > 0:
                if (
                    params.batch_idx_train > 0
                    and params.batch_idx_train % params.average_period == 0
                ):
                    if rank == 0:
                        update_averaged_model(
                            params=params,
                            model_cur=model,
                            model_avg=model_avg,
                        )

            if (
                params.batch_idx_train > 0
                and params.batch_idx_train % params.save_every_n == 0
            ):
                if rank == 0:
                    save_checkpoint_with_global_batch_idx(
                        out_dir=params.exp_dir,
                        global_batch_idx=params.batch_idx_train,
                        model=model,
                        model_avg=model_avg,
                        params=params,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=train_dl.sampler,
                        scaler=scaler,
                        rank=rank,
                    )
                    # Also save TS checkpoint
                    ts_filename = params.exp_dir / f"ts_iter-{params.batch_idx_train}.pt"
                    save_ts_checkpoint(ts_filename, ts_models, rank)
                    remove_checkpoints(
                        out_dir=params.exp_dir,
                        topk=params.keep_last_k,
                        rank=rank,
                    )

            if batch_idx % 100 == 0 and params.dtype in ["float16", "fp16"]:
                cur_grad_scale = scaler._scale.item()
                if cur_grad_scale < 1.0 or (
                    cur_grad_scale < 8.0 and batch_idx % 400 == 0
                ):
                    scaler.update(cur_grad_scale * 2.0)

                if cur_grad_scale < 0.01:
                    logging.warning(f"Grad scale is small: {cur_grad_scale}")
                if cur_grad_scale < 1.0e-05:
                    raise RuntimeError(
                        f"grad_scale is too small, exiting: {cur_grad_scale}"
                    )

            if batch_idx % params.log_interval == 0 or (batch_idx <= 100 and batch_idx % 10 == 0):
                cur_lr = scheduler.get_last_lr()[0]
                cur_grad_scale = (
                    scaler._scale.item()
                    if params.dtype in ["float16", "fp16"]
                    else 1.0
                )

                logging.info(
                    f"Epoch {params.cur_epoch}, "
                    f"batch {batch_idx}, train_loss[{loss_info}], "
                    f"tot_loss[{tot_loss}], "
                    f"batch size: {batch_size}, "
                    f"lr: {cur_lr:.2e}, "
                    f"accum_steps: {params.batch_idx_train % params.accumulate_grad_steps}"
                    + (
                        f", grad_scale: {cur_grad_scale}"
                        if params.dtype in ["float16", "fp16"]
                        else ""
                    )
                )

                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "train/learning_rate", cur_lr, params.batch_idx_train
                    )
                    loss_info.write_summary(
                        tb_writer,
                        "train/current_",
                        params.batch_idx_train,
                    )
                    tot_loss.write_summary(
                        tb_writer, "train/tot_", params.batch_idx_train
                    )
                    if params.dtype in ["float16", "fp16"]:
                        tb_writer.add_scalar(
                            "train/grad_scale",
                            cur_grad_scale,
                            params.batch_idx_train,
                        )

            if params.batch_idx_train % params.valid_interval == 0:
                # Calculate validation loss
                model.eval()
                for m in ts_models.values():
                    m.eval()
                logging.info("Computing validation loss")
                with torch.cuda.amp.autocast(dtype=dtype):
                    valid_info = compute_validation_loss(
                        params=params,
                        model=model,
                        ts_models=ts_models,
                        valid_dl=valid_dl,
                        world_size=world_size,
                    )
                logging.info(
                    f"Epoch {params.cur_epoch}, validation: {valid_info}"
                )
                logging.info(
                    f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
                )

                if tb_writer is not None:
                    valid_info.write_summary(
                        tb_writer, "train/valid_", params.batch_idx_train
                    )

                model.train()
                for name, m in ts_models.items():
                    if name not in ["encoder", "quantizer"]:
                        m.train()

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def filter_short_and_long_utterances(
    cuts: CutSet, min_duration: float, max_duration: float
) -> CutSet:
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 0.6 second and 20 seconds
        if c.duration < min_duration or c.duration > max_duration:
            # logging.warning(
            #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            # )
            return False
        return True

    cuts = cuts.filter(remove_short_and_long_utt)

    return cuts


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    rng = random.Random(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        if params.train_stage:
            tb_writer = SummaryWriter(
                log_dir=f"{params.exp_dir}/tensorboard_stage{params.train_stage}"
            )
        else:
            tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(f"Device: {device}")
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    # Initialize TraceableSpeech models
    logging.info("Initializing TraceableSpeech models")
    ts_config_path = project_root / "traceableSpeech" / "config.json"
    with open(ts_config_path) as f:
        ts_h = AttrDict(json.load(f))

    ts_encoder = Encoder(ts_h).to(device)
    ts_generator = Generator(ts_h).to(device)
    ts_quantizer = Quantizer(ts_h, 'Audio').to(device)
    ts_watermark_encoder = Watermark_Encoder(ts_h).to(device)
    ts_watermark_decoder = Watermark_Decoder(ts_h).to(device)

    # Load g_00150000 checkpoint
    ts_checkpoint_path = project_root / "traceableSpeech" / "g_00150000"
    if ts_checkpoint_path.exists():
        logging.info(f"Loading TraceableSpeech checkpoint: {ts_checkpoint_path}")
        ts_checkpoint = torch.load(ts_checkpoint_path, map_location=device)
        ts_encoder.load_state_dict(ts_checkpoint['encoder'])
        ts_generator.load_state_dict(ts_checkpoint['generator'])
        ts_quantizer.load_state_dict(ts_checkpoint['quantizer_Audio'])
        ts_watermark_encoder.load_state_dict(ts_checkpoint['watermark_encoder'])
        ts_watermark_decoder.load_state_dict(ts_checkpoint['watermark_decoder'])
    else:
        logging.warning(f"TraceableSpeech checkpoint NOT found at {ts_checkpoint_path}")

    # Freeze Audio Encoder and Quantizer (Phase 1)
    ts_encoder.eval(); ts_encoder.requires_grad_(False)
    ts_quantizer.eval(); ts_quantizer.requires_grad_(False)

    ts_models = {
        "encoder": ts_encoder,
        "generator": ts_generator,
        "quantizer": ts_quantizer,
        "watermark_encoder": ts_watermark_encoder,
        "watermark_decoder": ts_watermark_decoder,
    }

    # Initialize Discriminators
    logging.info("Initializing Discriminators (MPD, MSD, MSTFTD)")
    ts_mpd = MultiPeriodDiscriminator().to(device)
    ts_msd = MultiScaleDiscriminator().to(device)
    ts_mstftd = MultiScaleSTFTDiscriminator(32).to(device)

    # Load discriminator checkpoint if available
    ts_do_checkpoint_path = project_root / "traceableSpeech" / "do_00150000"
    if ts_do_checkpoint_path.exists():
        logging.info(f"Loading discriminator checkpoint: {ts_do_checkpoint_path}")
        ts_do_checkpoint = torch.load(ts_do_checkpoint_path, map_location=device)
        ts_mpd.load_state_dict(ts_do_checkpoint['mpd'])
        ts_msd.load_state_dict(ts_do_checkpoint['msd'])
        ts_mstftd.load_state_dict(ts_do_checkpoint['mstftd'])
    else:
        logging.warning(f"Discriminator checkpoint NOT found at {ts_do_checkpoint_path}, training from scratch")

    ts_models["mpd"] = ts_mpd
    ts_models["msd"] = ts_msd
    ts_models["mstftd"] = ts_mstftd

    with open(f"{params.exp_dir}/model.txt", "w") as f:
        print(model)
        print(model, file=f)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0 and params.average_period > 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # Wrap trainable TS models in DDP
        for name in ["generator", "watermark_encoder", "watermark_decoder", "mpd", "msd", "mstftd"]:
            ts_models[name] = DDP(ts_models[name], device_ids=[rank], find_unused_parameters=True)

    if params.train_stage:
        _model = model.module if isinstance(model, DDP) else model
        model_parameters = []
        model_parameter_names = []
        for name, param in _model.stage_named_parameters(params.train_stage):
            model_parameters.append(param)
            model_parameter_names.append(name)
    else:
        model_parameters = []
        model_parameter_names = []
        for name, param in model.named_parameters():
            model_parameters.append(param)
            model_parameter_names.append(name)

    # Add TraceableSpeech trainable parameters (generator side)
    ts_trainable_params = []
    ts_trainable_parameter_names = []
    for name in ["generator", "watermark_encoder", "watermark_decoder"]:
        for p_name, param in ts_models[name].named_parameters():
            ts_trainable_params.append(param)
            ts_trainable_parameter_names.append(f"ts_{name}.{p_name}")

    # Combine VALL-E and TraceableSpeech parameters
    all_parameters = model_parameters + ts_trainable_params
    all_parameter_names = [model_parameter_names + ts_trainable_parameter_names]

    if params.optimizer_name == "ScaledAdam":
        optimizer = ScaledAdam(
            all_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            show_dominant_parameters=False,
            clipping_update_period=1000,
            parameters_names=all_parameter_names,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(
            all_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.98),
            target_rms=0.1,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            all_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
            eps=1e-8,
        )
    elif params.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            all_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise NotImplementedError()

    # Discriminator optimizer (separate from generator/VALL-E optimizer)
    optim_d = torch.optim.Adam(
        itertools.chain(
            ts_models["mpd"].parameters(),
            ts_models["msd"].parameters(),
            ts_models["mstftd"].parameters(),
        ),
        lr=0.0002,
        betas=(0.5, 0.9),
    )

    scheduler = get_scheduler(params, optimizer)
    optimizer.zero_grad()

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.inf_check:
        register_inf_check_hooks(model)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    dataset = TtsDataModule(args)
    train_cuts = dataset.train_cuts()
    valid_cuts = dataset.dev_cuts()

    train_cuts = filter_short_and_long_utterances(
        train_cuts, params.filter_min_duration, params.filter_max_duration
    )
    valid_cuts = filter_short_and_long_utterances(
        valid_cuts, params.filter_min_duration, params.filter_max_duration
    )

    train_dl = dataset.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )
    valid_dl = dataset.valid_dataloaders(valid_cuts)

    if params.oom_check:
        scan_pessimistic_batches_for_oom(
            model=model,
            ts_models=ts_models,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    scaler = GradScaler(
        enabled=(params.dtype in ["fp16", "float16"]), init_scale=1.0
    )
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if isinstance(scheduler, Eden):
            scheduler.step_epoch(epoch - 1)

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            ts_models=ts_models,
            model_avg=model_avg,
            optimizer=optimizer,
            optim_d=optim_d,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            rng=rng,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        save_checkpoint(
            params=params,
            model=model,
            ts_models=ts_models,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    ts_models: Dict[str, nn.Module],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    logging.info(
        f"Sanity check -- see if any of the batches in epoch 1 would cause OOM. "
        f"Testing {len(batches)} pessimistic batches."
    )

    dtype = torch.float32
    if params.dtype in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif params.dtype in ["float16", "fp16"]:
        dtype = torch.float16

    for i, (criterion, cuts) in enumerate(batches.items()):
        logging.info(f"Checking pessimistic batch {i+1}/{len(batches)}: {criterion}")
        batch = train_dl.dataset[cuts]
        
        # BN fails with B=1 in train() mode.
        # Pessimistic batches often have B=1 (single longest file).
        is_single_sample = (len(batch["text"]) == 1)
        
        try:
            if is_single_sample:
                for m in ts_models.values():
                    m.eval()
                model.eval()
            else:
                for m in ts_models.values():
                    m.train()
                model.train()

            with torch.cuda.amp.autocast(dtype=dtype):
                _, loss, _ = compute_loss(
                    params=params,
                    model=model,
                    ts_models=ts_models,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Batch {i+1} passed. Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
