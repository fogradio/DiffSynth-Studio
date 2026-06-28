#!/bin/bash
# Wan2.1-T2V-1.3B inference with LoRA DiT + Video Copilot correction.
#
# Loads the BASE DiT weights, fuses LoRA on top, then runs copilot-corrected
# denoising.  Corresponds to the joint LoRA+copilot training script:
#   examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B-OmniWorld-Joint-Copilot.sh
#
# Usage:
#   bash Wan2.1-T2V-1.3B-OmniWorld-Joint-Copilot-LoRA-Inference.sh
#   COPILOT_SCALE=0.5 LORA_ALPHA=0.8 bash ...
#   bash ... --copilot_start_pct 0.1 --num_inference_steps 30

set -eo pipefail

PROJECT_ROOT="/mnt/workspace/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/python}"

source /mnt/workspace/hwzhang/miniconda3/bin/activate dav3

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# ---- Prompts / base model ----
PROMPTS_PATH="${WAN21_INFER_PROMPTS:-/mnt/workspace/hwzhang/code/mistake_forcing/inference_prompts.jsonl}"
BASE_MODEL_ROOT="${WAN21_T2V_13B_MODEL_ROOT:-/mnt/workspace/common/models/Wan2.1-T2V-1.3B}"
DIT_WEIGHTS="${WAN21_INFER_DIT_WEIGHTS:-${BASE_MODEL_ROOT}/diffusion_pytorch_model.safetensors}"

# ---- LoRA ----
LORA_WEIGHTS="${LORA_WEIGHTS:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_joint_copilot_lora/step-3000.safetensors}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"

# ---- Copilot ----
COPILOT_VARIANT="${COPILOT_VARIANT:-v2}"
COPILOT_CKPT="${COPILOT_CKPT:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_joint_copilot_lora/copilot_step-3000.pt}"
COPILOT_SCALE="${COPILOT_SCALE:-1.0}"
COPILOT_START_PCT="${COPILOT_START_PCT:-0.0}"
COPILOT_END_PCT="${COPILOT_END_PCT:-1.0}"

# ---- Output ----
OUTPUT_ROOT="${WAN21_INFER_OUTPUT_ROOT:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/inference}"
TAG="${WAN21_INFER_TAG:-lora_copilot_${COPILOT_VARIANT}_s${COPILOT_SCALE}}"

# ---- Generation ----
HEIGHT="${WAN21_INFER_HEIGHT:-480}"
WIDTH="${WAN21_INFER_WIDTH:-832}"
NUM_FRAMES="${WAN21_INFER_NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${WAN21_INFER_STEPS:-50}"
CFG_SCALE="${WAN21_INFER_CFG_SCALE:-5.0}"
SIGMA_SHIFT="${WAN21_INFER_SIGMA_SHIFT:-5.0}"
SEED="${WAN21_INFER_SEED:-0}"
FPS="${WAN21_INFER_FPS:-16}"
LIMIT="${WAN21_INFER_LIMIT:-0}"

"${PYTHON_BIN}" examples/wanvideo/model_inference/Wan2.1-T2V-1.3B-OmniWorld-Joint-Copilot-LoRA-Inference.py \
  --prompts_path "${PROMPTS_PATH}" \
  --base_model_root "${BASE_MODEL_ROOT}" \
  --dit_weights "${DIT_WEIGHTS}" \
  --lora_weights "${LORA_WEIGHTS}" \
  --lora_alpha "${LORA_ALPHA}" \
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
  --copilot_variant "${COPILOT_VARIANT}" \
  --copilot_ckpt "${COPILOT_CKPT}" \
  --copilot_scale "${COPILOT_SCALE}" \
  --copilot_start_pct "${COPILOT_START_PCT}" \
  --copilot_end_pct "${COPILOT_END_PCT}" \
  "$@"
