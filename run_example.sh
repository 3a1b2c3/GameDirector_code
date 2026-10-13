#!/usr/bin/env bash
# Runs one of the bundled GameDirector example rollouts (same parameters as example/run_rollout.sh and run_example.bat).
#
# Usage:   bash run_example.sh <example> [ticks]
#   <example>  nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05
#   [ticks]    optional, overrides the example's default number of ticks (12, or 6 for nrftw_03)
#
# Settings (environment variables, all optional):
#   EXAMPLE_LLM_DEVICE    device of the Gemma planner (default cuda:0; use cpu if the GPU is too small)
#   EXAMPLE_LLM_TIMEOUT   planner request timeout in seconds (default: the program's own 90)
#   EXAMPLE_EXTRA_ARGS    extra run_closed_loop.py arguments, e.g. "--world-model-steps 20"
#   EXAMPLE_OUTPUT_ROOT   where results go (default output/local_runs, which git ignores; the repository's bundled sample
#                         results in output/ar_agentic are tracked and are cleared by a run that writes there)
#   GAMEDIRECTOR_LLM_MODEL  Gemma directory (default external_models/gemma-4-E2B-it)
#   EXAMPLE_DRY_RUN=1     print the command and exit
#
# The video is OUTPUT_ROOT/<game>/<example>/rollout.mp4. Needs setup_env.sh and download_models.sh to have completed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_example.sh <example> [ticks]   (nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05)" >&2
  exit 2
fi
name="$1"
background=""
case "${name}" in
  #            game  player boss seed ticks p_atk b_atk  background
  nrftw_01) game=nrftw; player=2; boss=3; seed=2;  ticks=12; p_atk=40;  b_atk=20;  background="Frozen Wasteland" ;;
  nrftw_02) game=nrftw; player=3; boss=2; seed=2;  ticks=12; p_atk=40;  b_atk=20;  background="Frozen Wasteland" ;;
  nrftw_03) game=nrftw; player=1; boss=1; seed=31; ticks=6;  p_atk=120; b_atk=120 ;;
  hk_01)    game=hk;    player=1; boss=2; seed=14; ticks=12; p_atk=40;  b_atk=20 ;;
  hk_02)    game=hk;    player=1; boss=1; seed=11; ticks=12; p_atk=40;  b_atk=20 ;;
  hk_03)    game=hk;    player=1; boss=4; seed=19; ticks=12; p_atk=40;  b_atk=20 ;;
  hk_04)    game=hk;    player=1; boss=1; seed=3;  ticks=12; p_atk=40;  b_atk=20 ;;
  hk_05)    game=hk;    player=1; boss=3; seed=2;  ticks=12; p_atk=40;  b_atk=20 ;;
  *) echo "Unknown example: ${name}" >&2; exit 2 ;;
esac
if [[ $# -ge 2 ]]; then
  ticks="$2"
fi

python_bin="code/.venv/bin/python"
llm_model="${GAMEDIRECTOR_LLM_MODEL:-external_models/gemma-4-E2B-it}"
sample="example/${game}/${name}"
output="${EXAMPLE_OUTPUT_ROOT:-output/local_runs}/${game}/${name}"

if [[ "${EXAMPLE_DRY_RUN:-0}" != "1" ]]; then
  [[ -x "${python_bin}" ]] || { echo "ERROR: ${python_bin} not found. Run setup_env.sh first." >&2; exit 1; }
  [[ -f model/attacknet/attacknet.pt ]] || { echo "ERROR: models not found in model/. Run download_models.sh first." >&2; exit 1; }
  [[ -f "${llm_model}/model.safetensors" ]] || { echo "ERROR: Gemma not found at ${llm_model}. Run download_models.sh or set GAMEDIRECTOR_LLM_MODEL." >&2; exit 1; }
  [[ -f "${sample}/first_frame.png" ]] || { echo "ERROR: ${sample}/first_frame.png not found." >&2; exit 1; }
fi

command=("${python_bin}" code/agentic_control/run_closed_loop.py
  --game "${game}" --image "${sample}/first_frame.png" --player-prompts "${sample}/player_track.json"
  --vocabulary-file "${sample}/vocabulary.json" --player-choice "${player}" --boss-choice "${boss}" --ticks "${ticks}"
  --player-attack "${p_atk}" --boss-attack "${b_atk}" --seed "${seed}" --output-dir "${output}"
  --attack-checkpoint model/attacknet/attacknet.pt --situation-checkpoint model/situationnet/situationnet.pt
  --base-model model --world-model-mode ar --world-model-checkpoint model/world_model/ar_world_model.safetensors
  --ar-stage 1 --ar-sampler euler --world-model-steps 30 --world-model-cfg 2.0 --ar-attention-sink
  --no-ar-reanchor-between-ticks --llm-checkpoint "${llm_model}" --llm-device "${EXAMPLE_LLM_DEVICE:-cuda:0}")
if [[ -n "${EXAMPLE_LLM_TIMEOUT:-}" ]]; then
  command+=(--llm-timeout "${EXAMPLE_LLM_TIMEOUT}")
fi
if [[ -n "${background}" ]]; then
  command+=(--background "${background}")
fi
if [[ -n "${EXAMPLE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<< "${EXAMPLE_EXTRA_ARGS}"
  command+=("${extra[@]}")
fi

echo "=========================================="
echo "GameDirector example ${name}: ${ticks} ticks, output ${output}/rollout.mp4"
echo "Gemma planner device: ${EXAMPLE_LLM_DEVICE:-cuda:0}"
echo "GPU memory used/total:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "=========================================="

if [[ "${EXAMPLE_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  echo
  exit 0
fi

PYTHONPATH="${PWD}/code" "${command[@]}"
echo "Done: ${output}/rollout.mp4"
