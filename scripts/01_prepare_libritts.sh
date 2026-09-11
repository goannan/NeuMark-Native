#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=3
dl_dir=$PWD/download
dataset_parts="--dataset-parts all"
audio_extractor="SpeechTokenizer"
audio_feats_dir="data/tokenized"
neumark_root="."
neumark_config="STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
neumark_st_checkpoint="STmodels/pretrained_model/SpeechTokenizer.pt"

. shared/parse_options.sh || exit 1

mkdir -p data

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download LibriTTS data into $dl_dir"
  mkdir -p "$dl_dir"
  if [ ! -d "$dl_dir/LibriTTS/dev-other" ] || [ ! -d "$dl_dir/LibriTTS/train-other-500" ]; then
    lhotse download libritts ${dataset_parts} "$dl_dir"
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LibriTTS Lhotse manifests"
  mkdir -p data/manifests
  if [ ! -e data/manifests/.libritts.done ]; then
    lhotse prepare libritts ${dataset_parts} -j $nj "$dl_dir/LibriTTS" data/manifests
    touch data/manifests/.libritts.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize LibriTTS acoustic codes using ${audio_extractor}"
  mkdir -p "${audio_feats_dir}"
  if [ ! -e "${audio_feats_dir}/.libritts.tokenize.done" ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --audio-extractor "${audio_extractor}" \
        --neumark-root "${neumark_root}" \
        --neumark-config "${neumark_config}" \
        --neumark-st-checkpoint "${neumark_st_checkpoint}" \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}" \
        --num-workers 4
    touch "${audio_feats_dir}/.libritts.tokenize.done"
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare train/dev/test combined cuts"
  if [ ! -e "${audio_feats_dir}/.libritts.train.done" ]; then
    if [ "${dataset_parts}" == "--dataset-parts all" ]; then
      lhotse combine \
        "${audio_feats_dir}/libritts_cuts_train-clean-100.jsonl.gz" \
        "${audio_feats_dir}/libritts_cuts_train-clean-360.jsonl.gz" \
        "${audio_feats_dir}/libritts_cuts_train-other-500.jsonl.gz" \
        "${audio_feats_dir}/cuts_train.jsonl.gz"

      lhotse copy \
        "${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz" \
        "${audio_feats_dir}/cuts_dev.jsonl.gz"
    else
      lhotse copy \
        "${audio_feats_dir}/libritts_cuts_dev-clean.jsonl.gz" \
        "${audio_feats_dir}/cuts_train.jsonl.gz"
      lhotse subset --first 400 \
        "${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz" \
        "${audio_feats_dir}/cuts_dev.jsonl.gz"
    fi

    lhotse copy \
      "${audio_feats_dir}/libritts_cuts_test-clean.jsonl.gz" \
      "${audio_feats_dir}/cuts_test.jsonl.gz"
    touch "${audio_feats_dir}/.libritts.train.done"
  fi
  log "Data preparation complete! Manifests saved to ${audio_feats_dir}"
fi
