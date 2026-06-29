#!/bin/bash
#
# Wan2.1 T2V-1.3B joint-copilot SFT on WISA-80K (balanced_1k).
#
# Single-GPU, BATCH_SIZE=4 (real batched DiT forward, NO DDP) on NTU HPCC GaaS H200.
# Submit to a compute node, e.g.:
#   qsub -I -P gs_ccds_chuanxia.zheng -q gpu_as \
#        -l select=1:ncpus=4:ngpus=1 -l walltime=12:00:00
#   bash examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B-WISA-Joint-Copilot-1GPU-bs4.sh
#
# vs. the bs=1 single-GPU script this only adds --batch_size 4. The DiT and the
# copilot head run ONE batched forward over 4 samples per step (per-sample
# VAE/T5 encoding is stacked into [B, ...] inside train_joint_copilot.py); the
# DiffSynth core is untouched.
#
# Text  = captions + phys_law (prompt level B).
# Video = unified 480x832. batch_size > 1 forces a CONSTANT frame count:
#         align_4k1(NUM_FRAMES) = 81 for every clip (full-span sparse sampling;
#         clips shorter than 81 freeze frames — 中间静默 padding — instead of
#         shrinking the tensor), so the batch can be stacked.
#
# Memory: 4x the bs=1 activation footprint. Gradient checkpointing on the DiT is
# always enabled. If you OOM, set COPILOT_USE_GRAD_CKPT=1 and/or lower BATCH_SIZE.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# ===== WISA-80K T2V dataset =====
WISA_DATA_ROOT="${WISA_DATA_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/datasets/WISA-80K/data}"
WISA_JSONL="${WISA_JSONL:-${WISA_DATA_ROOT}/balanced_1k.jsonl}"
WISA_VIDEO_ROOT="${WISA_VIDEO_ROOT:-${WISA_DATA_ROOT}/videos}"
WISA_PROMPT_LEVEL="${WISA_PROMPT_LEVEL:-B}"

# ===== Base model (DiffSynth-native Wan2.1-T2V-1.3B) =====
MODEL_ROOT="${MODEL_ROOT:-/projects_vol/gp_chuanxia.zheng/hwzhang/model/Wan2.1-T2V-1.3B}"
_TIMESTAMP="${_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/outputs/wan21_t2v_1_3b_wisa_joint_copilot_1gpu_bs4_${_TIMESTAMP}}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
# Single-GPU batched training. Effective batch = BATCH_SIZE * GRAD_ACCUM.
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DIT_LR="${DIT_LR:-5e-6}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_EVERY="${LOG_EVERY:-1}"
MISTAKE_SELECTED_LAYERS="${MISTAKE_SELECTED_LAYERS:-3,11,19,29}"

# ===== Copilot =====
COPILOT_VERSION="${COPILOT_VERSION:-v3}"
COPILOT_DIR="${COPILOT_DIR:-/projects_vol/gp_chuanxia.zheng/hwzhang/code/mistake_forcing/video_copilot}"
COPILOT_DIM="${COPILOT_DIM:-1024}"
COPILOT_DEPTH="${COPILOT_DEPTH:-10}"
COPILOT_NUM_HEADS="${COPILOT_NUM_HEADS:-16}"
COPILOT_MLP_RATIO="${COPILOT_MLP_RATIO:-4.0}"
COPILOT_LR="${COPILOT_LR:-1e-4}"
COPILOT_LOSS_WEIGHT="${COPILOT_LOSS_WEIGHT:-1.0}"
SAVE_COPILOT_EVERY_EPOCH="${SAVE_COPILOT_EVERY_EPOCH:-0}"
COPILOT_RESUME="${COPILOT_RESUME:-}"
# bs=4 raises activation memory; flip to 1 to also checkpoint the copilot head.
COPILOT_USE_GRAD_CKPT="${COPILOT_USE_GRAD_CKPT:-0}"

MAX_DATA_ITEMS="${MAX_DATA_ITEMS:-}"

# ===== Optimal Gap Sampling (default off) =====
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
if [[ "${COPILOT_USE_GRAD_CKPT}" == "1" ]]; then
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
  --config_file "${PROJECT_ROOT}/examples/wanvideo/model_training/full/accelerate_config_1gpu.yaml" \
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
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
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
