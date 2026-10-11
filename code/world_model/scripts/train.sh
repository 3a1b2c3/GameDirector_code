#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${HERE}")}"
REPOSITORY_ROOT="$(dirname "$(dirname "${PROJECT_ROOT}")")"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate || true)}"
BASE_MODEL="${WORLD_MODEL_BASE_MODEL:-${GAMEDIRECTOR_BASE_MODEL:-}}"
DATA_ROOT="${WORLD_MODEL_DATA_ROOT:-}"
METADATA_PATH="${WORLD_MODEL_METADATA:-${NRFTW_METADATA:-${DATA_ROOT}/jobs_train.jsonl}}"
OUTPUT_PATH="${WORLD_MODEL_OUTPUT:-${REPOSITORY_ROOT}/output/world_model}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Train the bidirectional GameDirector world model.

Required environment:
  GAMEDIRECTOR_BASE_MODEL  cache containing the Wan assets described in README
  WORLD_MODEL_DATA_ROOT    dataset root

Optional environment:
  WORLD_MODEL_METADATA     JSONL metadata (default: DATA_ROOT/jobs_train.jsonl)
  WORLD_MODEL_OUTPUT       checkpoint directory (default: repository output/)
  CUDA_VISIBLE_DEVICES     GPU list (default: 0,1,2,3)
  NUM_PROCESSES            must match the visible GPU count

Additional arguments are forwarded to training/train.py.
EOF
  exit 0
fi

if [[ -z "${BASE_MODEL}" || -z "${DATA_ROOT}" ]]; then
  echo "ERROR: set GAMEDIRECTOR_BASE_MODEL (or WORLD_MODEL_BASE_MODEL) and WORLD_MODEL_DATA_ROOT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29530}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-80}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
PROMPT_DROPOUT_PROB="${PROMPT_DROPOUT_PROB:-0.1}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"
RESERVE_CUDA_MEMORY_GB="${RESERVE_CUDA_MEMORY_GB:-0}"
RESUME_FROM_CKPT="${RESUME_FROM_CKPT:-}"
RESUME_STATE="${RESUME_STATE:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"

is_true() {
  case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}

required_files=(
  "${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"
  "${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors"
  "${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
  "${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
  "${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  "${BASE_MODEL}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
)
IFS=';' read -r -a METADATA_PATHS <<< "${METADATA_PATH}"
required_files+=("${METADATA_PATHS[@]}")
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file not found: ${required_file}" >&2
    exit 2
  fi
done
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "ERROR: accelerate not found: ${ACCELERATE_BIN}" >&2
  exit 2
fi
if [[ "${NUM_PROCESSES}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "ERROR: NUM_PROCESSES=${NUM_PROCESSES} but ${#GPU_IDS[@]} GPUs are visible" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM_CKPT}" && -n "${RESUME_STATE}" ]]; then
  echo "ERROR: RESUME_FROM_CKPT and RESUME_STATE are mutually exclusive" >&2
  exit 2
fi

OPTIONAL_ARGS=(--use_gradient_checkpointing)
if [[ -n "${RESUME_FROM_CKPT}" ]]; then OPTIONAL_ARGS+=(--resume_from_ckpt "${RESUME_FROM_CKPT}"); fi
if [[ -n "${RESUME_STATE}" ]]; then OPTIONAL_ARGS+=(--resume_state "${RESUME_STATE}"); fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then OPTIONAL_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}"); fi
if is_true "${SAVE_FULL_STATE:-false}"; then OPTIONAL_ARGS+=(--save_full_state); fi
if is_true "${SKIP_FINAL_SAVE:-false}"; then OPTIONAL_ARGS+=(--skip_final_save); fi
if is_true "${INITIALIZE_MODEL_ON_CPU:-false}"; then OPTIONAL_ARGS+=(--initialize_model_on_cpu); fi
if is_true "${FIND_UNUSED_PARAMETERS:-false}"; then OPTIONAL_ARGS+=(--find_unused_parameters); fi

LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}")
if (( NUM_PROCESSES > 1 )); then LAUNCH_ARGS+=(--multi_gpu); fi
MODEL_PATHS="[[\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE_MODEL}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth\"]"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "=== GameDirector World Model: first frame + 20 prompts -> 5s clip ==="
echo "GPUs=${CUDA_VISIBLE_DEVICES} DATA=${METADATA_PATH} OUTPUT=${OUTPUT_PATH}"
cd "${PROJECT_ROOT}"
exec "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}" training/train.py \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --dataset_num_workers "${DATASET_NUM_WORKERS}" \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${BASE_MODEL}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate "${LEARNING_RATE}" --num_epochs "${NUM_EPOCHS}" \
  --save_steps "${SAVE_STEPS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --output_path "${OUTPUT_PATH}" --trainable_models "${TRAINABLE_MODELS}" \
  --extra_inputs "${EXTRA_INPUTS}" \
  --prompt_dropout_prob "${PROMPT_DROPOUT_PROB}" \
  --reserve_cuda_memory_gb "${RESERVE_CUDA_MEMORY_GB}" \
  "${OPTIONAL_ARGS[@]}" "$@"
