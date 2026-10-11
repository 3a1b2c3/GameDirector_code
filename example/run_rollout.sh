#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(dirname "${EXAMPLE_ROOT}")"
RUNNER="${REPOSITORY_ROOT}/code/run.sh"
OUTPUT_ROOT="${EXAMPLE_OUTPUT_ROOT:-${REPOSITORY_ROOT}/output/ar_agentic}"
TICKS="${EXAMPLE_TICKS:-12}"
MODE="${1:-all}"

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  echo "Usage: $0 [all|nrftw|hk]"
  echo "Set GAMEDIRECTOR_LLM_MODEL and all GAMEDIRECTOR_* model paths."
  exit 0
fi
if [[ "${MODE}" != "all" && "${MODE}" != "nrftw" && "${MODE}" != "hk" ]]; then
  echo "Usage: $0 [all|nrftw|hk]" >&2
  exit 2
fi
ATTACK_CHECKPOINT="${GAMEDIRECTOR_ATTACK_CHECKPOINT:-${REPOSITORY_ROOT}/model/attacknet/attacknet.pt}"
SITUATION_CHECKPOINT="${GAMEDIRECTOR_SITUATION_CHECKPOINT:-${REPOSITORY_ROOT}/model/situationnet/situationnet.pt}"
GAMEDIRECTOR_BASE_MODEL="${GAMEDIRECTOR_BASE_MODEL:-${REPOSITORY_ROOT}/model}"
WORLD_MODEL_CHECKPOINT="${GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT:-${REPOSITORY_ROOT}/model/world_model/ar_world_model.safetensors}"
if [[ -z "${GAMEDIRECTOR_LLM_MODEL:-}" ]]; then
  echo "ERROR: set GAMEDIRECTOR_LLM_MODEL to the downloaded Gemma directory" >&2
  exit 2
fi
if [[ ! -f "${ATTACK_CHECKPOINT}" ]]; then
  echo "ERROR: AttackNet checkpoint not found: ${ATTACK_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${SITUATION_CHECKPOINT}" ]]; then
  echo "ERROR: SituationNet checkpoint not found: ${SITUATION_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${WORLD_MODEL_CHECKPOINT}" ]]; then
  echo "ERROR: world-model checkpoint not found: ${WORLD_MODEL_CHECKPOINT}" >&2
  echo "Set GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT to your downloaded checkpoint." >&2
  exit 2
fi

run_one() {
  local game="$1" name="$2" player_choice="$3" boss_choice="$4" seed="$5"
  local background="${6:-}"
  local default_ticks="${7:-12}"
  local player_attack="${8:-40}"
  local boss_attack="${9:-20}"
  local sample_ticks="${EXAMPLE_TICKS:-${default_ticks}}"
  local sample_dir="${EXAMPLE_ROOT}/${game}/${name}"
  local output_dir="${OUTPUT_ROOT}/${game}/${name}"
  local args=(
    infer
    --game "${game}"
    --image "${sample_dir}/first_frame.png"
    --player-prompts "${sample_dir}/player_track.json"
    --vocabulary-file "${sample_dir}/vocabulary.json"
    --player-choice "${player_choice}"
    --boss-choice "${boss_choice}"
    --ticks "${sample_ticks}"
    --player-attack "${player_attack}"
    --boss-attack "${boss_attack}"
    --seed "${seed}"
    --output-dir "${output_dir}"
    --attack-checkpoint "${ATTACK_CHECKPOINT}"
    --situation-checkpoint "${SITUATION_CHECKPOINT}"
    --base-model "${GAMEDIRECTOR_BASE_MODEL}"
    --world-model-mode ar
    --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}"
    --ar-stage 1
    --ar-sampler euler
    --world-model-steps 30
    --world-model-cfg 2.0
    --ar-attention-sink
    --no-ar-reanchor-between-ticks
    --llm-checkpoint "${GAMEDIRECTOR_LLM_MODEL}"
  )
  if [[ -n "${background}" ]]; then
    args+=(--background "${background}")
  fi
  echo "[$(date '+%F %T')] ${name}: starting ${sample_ticks}-tick rollout"
  "${RUNNER}" "${args[@]}"
  echo "[$(date '+%F %T')] ${name}: ${output_dir}/rollout.mp4"
}

if [[ "${MODE}" == "all" || "${MODE}" == "nrftw" ]]; then
  run_one nrftw nrftw_01 2 3 2 "Frozen Wasteland" 12 40 20
  run_one nrftw nrftw_02 3 2 2 "Frozen Wasteland" 12 40 20
  run_one nrftw nrftw_03 1 1 31 none 6 120 120
fi

if [[ "${MODE}" == "all" || "${MODE}" == "hk" ]]; then
  run_one hk hk_01 1 2 14
  run_one hk hk_02 1 1 11
  run_one hk hk_03 1 4 19
  run_one hk hk_04 1 1 3
  run_one hk hk_05 1 3 2
fi

echo "All requested rollouts completed under ${OUTPUT_ROOT}"
