#!/bin/bash
# Wan2.1-T2V-1.3B inference with Video Copilot correction.
#
# Usage:
#   bash Wan2.1-T2V-1.3B-OmniWorld-CopilotInference.sh            # defaults (v2, scale=1.0)
#   COPILOT_VARIANT=v1 COPILOT_SCALE=0.5 bash ...                  # override
#   bash ... --copilot_scale 0.8 --copilot_start_pct 0.1           # extra CLI args

set -eo pipefail

PROJECT_ROOT="/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/home/hzhang093/.conda/envs/videogen/bin/python}"

module load anaconda/2025
eval "$(/usr/local/anaconda2025/bin/conda shell.bash hook)"
conda activate videogen

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- Prompts / base model ----
PROMPTS_PATH="${WAN21_INFER_PROMPTS:-/projects_vol/gp_chuanxia.zheng/hwzhang/datasets/WISA-80K/data/sample_videos_10/sample_10.jsonl}"
BASE_MODEL_ROOT="${WAN21_T2V_13B_MODEL_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/model/Wan2.1-T2V-1.3B}"
#DIT_WEIGHTS="${WAN21_INFER_DIT_WEIGHTS:-/projects_vol/gp_chuanxia.zheng/hwzhang/model/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors}"
DIT_WEIGHTS="${WAN21_INFER_DIT_WEIGHTS:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_wisa_joint_copilot_v2/step-2500.safetensors}"

# ---- Copilot ----
COPILOT_VARIANT="${COPILOT_VARIANT:-v2}"
if [ "${COPILOT_VARIANT}" = "v2" ]; then
    COPILOT_CKPT_DEFAULT="/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_wisa_joint_copilot_v2/copilot_final.pt"
else
    COPILOT_CKPT_DEFAULT="/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/video_copilot_v1/epoch-10/model.safetensors"
fi
COPILOT_CKPT="${COPILOT_CKPT:-${COPILOT_CKPT_DEFAULT}}"
COPILOT_SCALE="${COPILOT_SCALE:-1.0}"
COPILOT_START_PCT="${COPILOT_START_PCT:-0.0}"
COPILOT_END_PCT="${COPILOT_END_PCT:-1.0}"

# ---- Output ----
OUTPUT_ROOT="${WAN21_INFER_OUTPUT_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/inference}"
TAG="${WAN21_INFER_TAG:-copilot_${COPILOT_VARIANT}_s${COPILOT_SCALE}}"

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

"${PYTHON_BIN}" examples/wanvideo/model_inference/Wan2.1-T2V-1.3B-OmniWorld-CopilotInference.py \
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
  --copilot_variant "${COPILOT_VARIANT}" \
  --copilot_ckpt "${COPILOT_CKPT}" \
  --copilot_scale "${COPILOT_SCALE}" \
  --copilot_start_pct "${COPILOT_START_PCT}" \
  --copilot_end_pct "${COPILOT_END_PCT}" \
  "$@"
