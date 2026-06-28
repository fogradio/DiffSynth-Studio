#!/bin/bash

set -eo pipefail

PROJECT_ROOT="/mnt/workspace/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/python}"

source /mnt/workspace/hwzhang/miniconda3/bin/activate dav3

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"


PROMPTS_PATH="${WAN21_INFER_PROMPTS:-/mnt/workspace/hwzhang/code/mistake_forcing/inference_prompts_blink_vkitti_top5.jsonl}"
BASE_MODEL_ROOT="${WAN21_T2V_13B_MODEL_ROOT:-/mnt/workspace/common/models/Wan2.1-T2V-1.3B}"
#DIT_WEIGHTS="${WAN21_INFER_DIT_WEIGHTS:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_joint_copilot/step-1800.safetensors}"
DIT_WEIGHTS="${WAN21_INFER_DIT_WEIGHTS:-/mnt/workspace/common/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors}"

OUTPUT_ROOT="${WAN21_INFER_OUTPUT_ROOT:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/inference}"
TAG="${WAN21_INFER_TAG:-omniworld_mistake}"

HEIGHT="${WAN21_INFER_HEIGHT:-480}"
WIDTH="${WAN21_INFER_WIDTH:-832}"
NUM_FRAMES="${WAN21_INFER_NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${WAN21_INFER_STEPS:-50}"
CFG_SCALE="${WAN21_INFER_CFG_SCALE:-1.0}"
SIGMA_SHIFT="${WAN21_INFER_SIGMA_SHIFT:-5.0}"
SEED="${WAN21_INFER_SEED:-0}"
FPS="${WAN21_INFER_FPS:-16}"
LIMIT="${WAN21_INFER_LIMIT:-0}"

"${PYTHON_BIN}" examples/wanvideo/model_inference/Wan2.1-T2V-1.3B-OmniWorld-MistakeForcing.py \
  --prompts_path "${PROMPTS_PATH}" \
  --base_model_root "${BASE_MODEL_ROOT}" \
  --dit_weights "${DIT_WEIGHTS}" \
  --output_root "${OUTPUT_ROOT}" \
  --tag "${TAG}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --sigma_shift "${SIGMA_SHIFT}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
  --limit "${LIMIT}" \
  "$@"
