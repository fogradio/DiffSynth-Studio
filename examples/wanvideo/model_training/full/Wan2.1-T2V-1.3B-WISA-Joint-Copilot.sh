#!/bin/bash
#
# Wan2.1 T2V-1.3B joint-copilot SFT on WISA-80K (balanced_1k).
#
# 2-GPU data parallel (pure DDP, NO DeepSpeed) on NTU HPCC GaaS H200.
# This is a GPU job — submit it to a compute node, e.g.:
#   qsub -I -P gs_ccds_chuanxia.zheng -q gpu_as \
#        -l select=1:ncpus=8:ngpus=2 -l walltime=12:00:00
#   bash examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B-WISA-Joint-Copilot.sh
#
# Text  = captions + phys_law (prompt level B).
# Video = unified 480x832; per-clip num_frames is DYNAMIC:
#         align_4k1(min(NUM_FRAMES, total_frames)), capped at 81 (~5s@16fps),
#         full-span sparse sampling, never padded.

set -eo pipefail

# ===== Environment (videogen conda env) =====
module load anaconda/2025
eval "$(/usr/local/anaconda2025/bin/conda shell.bash hook)"
conda activate videogen

PROJECT_ROOT="/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/DiffSynth-Studio"
PYTHON_BIN="${PYTHON_BIN:-/home/hzhang093/.conda/envs/videogen/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/hzhang093/.conda/envs/videogen/bin/accelerate}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# 2-GPU data parallel.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

# ===== WISA-80K T2V dataset =====
WISA_DATA_ROOT="${WISA_DATA_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/datasets/WISA-80K/data}"
WISA_JSONL="${WISA_JSONL:-${WISA_DATA_ROOT}/balanced_1k.jsonl}"
WISA_VIDEO_ROOT="${WISA_VIDEO_ROOT:-${WISA_DATA_ROOT}/videos}"
WISA_PROMPT_LEVEL="${WISA_PROMPT_LEVEL:-B}"

# ===== Base model (DiffSynth-native Wan2.1-T2V-1.3B) =====
MODEL_ROOT="${MODEL_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/model/Wan2.1-T2V-1.3B}"
_TIMESTAMP="${_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_wisa_joint_copilot_${_TIMESTAMP}}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
# NUM_FRAMES is the CAP (max frames), not a fixed length. Each clip uses
# align_4k1(min(NUM_FRAMES, total_frames)); no padding.
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DIT_LR="${DIT_LR:-5e-6}"
# Save DiT + copilot every 500 global steps. Set SAVE_STEPS="" for per-epoch.
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_EVERY="${LOG_EVERY:-1}"
MISTAKE_SELECTED_LAYERS="${MISTAKE_SELECTED_LAYERS:-3,11,19,29}"

# ===== Copilot =====
COPILOT_VERSION="${COPILOT_VERSION:-v2}"
COPILOT_DIR="${COPILOT_DIR:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/video_copilot}"
COPILOT_DIM="${COPILOT_DIM:-1024}"
COPILOT_DEPTH="${COPILOT_DEPTH:-10}"
COPILOT_NUM_HEADS="${COPILOT_NUM_HEADS:-16}"
COPILOT_MLP_RATIO="${COPILOT_MLP_RATIO:-4.0}"
COPILOT_LR="${COPILOT_LR:-1e-4}"
COPILOT_LOSS_WEIGHT="${COPILOT_LOSS_WEIGHT:-1.0}"
# 0 = no epoch-end copilot save (already saved every SAVE_STEPS).
SAVE_COPILOT_EVERY_EPOCH="${SAVE_COPILOT_EVERY_EPOCH:-0}"
COPILOT_RESUME="${COPILOT_RESUME:-}"

# ===== Fused copilot->DiT loss (new mode, default off) =====
# When FUSE_COPILOT_INTO_DIT_LOSS=1, training optimizes
#   total = MSE(noise_pred + COPILOT_FUSE_SCALE * copilot_out, target)        # fused term
#         + COPILOT_LOSS_WEIGHT * MSE(copilot_out, velocity_residual)         # kept copilot term
# The fused term updates BOTH the base DiT and the copilot (mirrors the
# inference-time copilot correction); the kept residual term adds extra
# gradient to the copilot only. When 0, the original loss (separate DiT MSE
# + detached copilot MSE) is used.
# Tip: set COPILOT_RESUME to warm-start from a trained copilot.
FUSE_COPILOT_INTO_DIT_LOSS="${FUSE_COPILOT_INTO_DIT_LOSS:-0}"
COPILOT_FUSE_SCALE="${COPILOT_FUSE_SCALE:-1.0}"

# Optional cap; leave unset to use all 1000 balanced_1k items.
MAX_DATA_ITEMS="${MAX_DATA_ITEMS:-}"

# ===== Optimal Gap Sampling (default off, same as OmniWorld script) =====
OPTIMAL_GAP_SAMPLING="${OPTIMAL_GAP_SAMPLING:-0}"
OGS_NUM_BINS="${OGS_NUM_BINS:-50}"
OGS_WARMUP_STEPS="${OGS_WARMUP_STEPS:-200}"
OGS_ALPHA="${OGS_ALPHA:-1.0}"
OGS_BETA="${OGS_BETA:-1.0}"

LOG_DIR="${LOG_DIR:-${OUTPUT_PATH}/logs}"
mkdir -p "${LOG_DIR}"

# ===== Weights & Biases logging =====
# Logs total/dit/copilot losses (+ OGS entropy) per step. Set ENABLE_WANDB=0 to
# disable. You are NOT logged in yet — pick one:
#   1) wandb login                 # paste key from https://wandb.ai/authorize
#   2) export WANDB_API_KEY=<key>  # then run this script
#   3) do nothing -> auto-fallback to offline; upload later with:
#        wandb sync "${OUTPUT_PATH}/wandb_log"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-wan21-wisa-joint-copilot}"
export WANDB_NAME="${WANDB_NAME:-$(basename "${OUTPUT_PATH}")}"
# Leave WANDB_ENTITY unset to use your account's default entity (here:
# "fogradio-Beihang University"). Do NOT hard-code the username ("fogradio"),
# that is not a valid entity and triggers "entity ... not found". Only export
# if you intentionally target a specific team/entity.
[[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY
if [[ "${ENABLE_WANDB}" == "1" && -z "${WANDB_MODE:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  if ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[wandb] not logged in and no WANDB_API_KEY -> WANDB_MODE=offline (run 'wandb login' for online)."
    export WANDB_MODE=offline
  fi
fi

EXTRA_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_wandb_log)
  EXTRA_ARGS+=(--wandb_project "${WANDB_PROJECT}")
fi
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
if [[ "${FUSE_COPILOT_INTO_DIT_LOSS}" == "1" ]]; then
  EXTRA_ARGS+=(--fuse_copilot_into_dit_loss)
  EXTRA_ARGS+=(--copilot_fuse_scale "${COPILOT_FUSE_SCALE}")
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
  --config_file "${PROJECT_ROOT}/examples/wanvideo/model_training/full/accelerate_config_ddp_2gpu.yaml" \
  --num_processes "${NUM_PROCESSES}" \
  examples/wanvideo/model_training/train_joint_copilot.py \
  --wisa_manifest_format \
  --wisa_video_root "${WISA_VIDEO_ROOT}" \
  --wisa_prompt_level "${WISA_PROMPT_LEVEL}" \
  --dataset_base_path "${WISA_DATA_ROOT}" \
  --dataset_metadata_path "${WISA_JSONL}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers "${DATASET_NUM_WORKERS}" \
  --model_paths "[\"${MODEL_ROOT}/diffusion_pytorch_model.safetensors\",\"${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_ROOT}/Wan2.1_VAE.pth\"]" \
  --tokenizer_path "${MODEL_ROOT}/google/umt5-xxl" \
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
