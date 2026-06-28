#!/bin/bash

set -eo pipefail

PROJECT_ROOT="/mnt/workspace/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/accelerate}"

source /mnt/workspace/hwzhang/miniconda3/bin/activate dav3

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"


MANIFEST_PATH="${OMNIWORLD_MANIFEST_PATH:-/mnt/workspace/hwzhang/code/dataset/OmniWorld/manifests/omniworld_train_ti2v_81f.jsonl}"
DATA_ROOT="${OMNIWORLD_DATA_ROOT:-/mnt/workspace/hwzhang/code/dataset/OmniWorld}"
MODEL_ROOT="${WAN21_T2V_13B_MODEL_ROOT:-/mnt/workspace/common/models/Wan2.1-T2V-1.3B}"
OUTPUT_PATH="${WAN21_OMNIWORLD_MISTAKE_OUTPUT:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_sft_mistake}"

"${ACCELERATE_BIN}" launch --config_file examples/wanvideo/model_training/full/accelerate_config_zero3.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${MANIFEST_PATH}" \
  --omniworld_manifest_format \
  --max_data_items 100 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --dataset_num_workers 0 \
  --model_paths "[\"${MODEL_ROOT}/diffusion_pytorch_model.safetensors\",\"${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_ROOT}/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "${MODEL_ROOT}/xlm-roberta-large" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --task "sft:mistake_forcing" \
  --use_gradient_checkpointing
