#!/bin/bash

set -eo pipefail

PROJECT_ROOT="/mnt/workspace/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/workspace/hwzhang/miniconda3/envs/dav3/bin/accelerate}"

source /mnt/workspace/hwzhang/miniconda3/bin/activate dav3

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"


# ---------------- DiT SFT (mistake-forcing) inputs ----------------
MANIFEST_PATH="${OMNIWORLD_MANIFEST_PATH:-/mnt/workspace/hwzhang/code/dataset/OmniWorld/manifests/omniworld_train_ti2v_81f.jsonl}"
DATA_ROOT="${OMNIWORLD_DATA_ROOT:-/mnt/workspace/hwzhang/code/dataset/OmniWorld}"
MODEL_ROOT="${WAN21_T2V_13B_MODEL_ROOT:-/mnt/workspace/common/models/Wan2.1-T2V-1.3B}"
OUTPUT_PATH="${WAN21_OMNIWORLD_JOINT_OUTPUT:-/mnt/workspace/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_omniworld_joint_copilot_gap}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DIT_LR="${DIT_LR:-5e-6}"
# Default: save DiT + copilot every 200 global steps. Set SAVE_STEPS="" to fall
# back to per-epoch saving (then SAVE_COPILOT_EVERY_EPOCH takes effect).
SAVE_STEPS="${SAVE_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-1}"
MISTAKE_SELECTED_LAYERS="${MISTAKE_SELECTED_LAYERS:-3,11,19,29}"

# ---------------- Copilot inputs ----------------
COPILOT_VERSION="${COPILOT_VERSION:-v2}"
COPILOT_DIR="${COPILOT_DIR:-/mnt/workspace/hwzhang/code/mistake_forcing/video_copilot}"
COPILOT_DIM="${COPILOT_DIM:-1024}"
COPILOT_DEPTH="${COPILOT_DEPTH:-10}"
COPILOT_NUM_HEADS="${COPILOT_NUM_HEADS:-16}"
COPILOT_MLP_RATIO="${COPILOT_MLP_RATIO:-4.0}"
COPILOT_LR="${COPILOT_LR:-1e-4}"
COPILOT_LOSS_WEIGHT="${COPILOT_LOSS_WEIGHT:-1.0}"
# 0 = no epoch-end copilot save; we already save every SAVE_STEPS steps. Set to
# >0 only if you also want per-epoch copilot snapshots.
SAVE_COPILOT_EVERY_EPOCH="${SAVE_COPILOT_EVERY_EPOCH:-0}"
COPILOT_RESUME="${COPILOT_RESUME:-}"

# Optional MAX_DATA_ITEMS — leave unset to use the FULL manifest (the original
# mistake-forcing script hard-coded 100; we drop that cap by default).
MAX_DATA_ITEMS="${MAX_DATA_ITEMS:-}"

# --- Optimal Gap Sampling (default: off) ---
OPTIMAL_GAP_SAMPLING="${OPTIMAL_GAP_SAMPLING:-1}"
OGS_NUM_BINS="${OGS_NUM_BINS:-50}"
OGS_WARMUP_STEPS="${OGS_WARMUP_STEPS:-200}"
OGS_ALPHA="${OGS_ALPHA:-1.0}"
OGS_BETA="${OGS_BETA:-1.0}"

LOG_DIR="${LOG_DIR:-${OUTPUT_PATH}/logs}"
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
if [[ -n "${MAX_DATA_ITEMS}" ]]; then
  EXTRA_ARGS+=(--max_data_items "${MAX_DATA_ITEMS}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [[ "${COPILOT_USE_GRAD_CKPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--copilot_use_gradient_checkpointing)
fi
if [[ -n "${COPILOT_RESUME}" ]]; then
  EXTRA_ARGS+=(--copilot_resume "${COPILOT_RESUME}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${OPTIMAL_GAP_SAMPLING}" == "1" ]]; then
  EXTRA_ARGS+=(--optimal_gap_sampling)
  EXTRA_ARGS+=(--ogs_num_bins "${OGS_NUM_BINS}")
  EXTRA_ARGS+=(--ogs_warmup_steps "${OGS_WARMUP_STEPS}")
  EXTRA_ARGS+=(--ogs_alpha "${OGS_ALPHA}")
  EXTRA_ARGS+=(--ogs_beta "${OGS_BETA}")
fi

"${ACCELERATE_BIN}" launch \
  --config_file /mnt/workspace/hwzhang/code/mistake_forcing/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_zero2.yaml \
  examples/wanvideo/model_training/train_joint_copilot.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${MANIFEST_PATH}" \
  --omniworld_manifest_format \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers "${DATASET_NUM_WORKERS}" \
  --model_paths "[\"${MODEL_ROOT}/diffusion_pytorch_model.safetensors\",\"${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_ROOT}/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "${MODEL_ROOT}/xlm-roberta-large" \
  --learning_rate "${DIT_LR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --task "sft:mistake_forcing" \
  --use_gradient_checkpointing \
  --mistake_selected_layers "${MISTAKE_SELECTED_LAYERS}" \
  --copilot_dir "${COPILOT_DIR}" \
  --copilot_version "${COPILOT_VERSION}" \
  --copilot_dim "${COPILOT_DIM}" \
  --copilot_depth "${COPILOT_DEPTH}" \
  --copilot_num_heads "${COPILOT_NUM_HEADS}" \
  --copilot_mlp_ratio "${COPILOT_MLP_RATIO}" \
  --copilot_lr "${COPILOT_LR}" \
  --copilot_loss_weight "${COPILOT_LOSS_WEIGHT}" \
  --save_copilot_every_epoch "${SAVE_COPILOT_EVERY_EPOCH}" \
  --log_every "${LOG_EVERY}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/stdout.log"
