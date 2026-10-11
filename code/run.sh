#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(dirname "${CODE_ROOT}")"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -x "${CODE_ROOT}/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="${CODE_ROOT}/.venv/bin/python"
else
  DEFAULT_PYTHON="$(command -v python3)"
fi
PYTHON_RUN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"

usage() {
  cat <<'EOF'
Usage:
  ./code/run.sh train attacknet   [training arguments]
  ./code/run.sh train situationnet [training arguments]
  ./code/run.sh train world_model  [training arguments]
  ./code/run.sh infer              [inference arguments]
  ./code/run.sh validate           [--hash] [--world-model-checkpoint FILE]
  ./code/run.sh verify-rollouts    [--rollout-root DIR] [--report FILE]

Run a command with --help to see its component-specific arguments.
World-model training uses four GPUs by default (CUDA_VISIBLE_DEVICES=0,1,2,3).
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command_name="$1"
shift
case "${command_name}" in
  train)
    if [[ $# -lt 1 ]]; then
      usage
      exit 2
    fi
    component="$1"
    shift
    case "${component}" in
      attacknet)
        cd "${CODE_ROOT}/agentic_control"
        exec "${PYTHON_RUN}" train_attacknet.py "$@"
        ;;
      situationnet)
        cd "${CODE_ROOT}/agentic_control"
        exec "${PYTHON_RUN}" train_situationnet.py "$@"
        ;;
      world_model)
        export WORLD_MODEL_OUTPUT="${WORLD_MODEL_OUTPUT:-${REPOSITORY_ROOT}/output/world_model}"
        exec "${CODE_ROOT}/world_model/scripts/train.sh" "$@"
        ;;
      *)
        echo "Unknown training component: ${component}" >&2
        usage
        exit 2
        ;;
    esac
    ;;
  infer)
    cd "${CODE_ROOT}/agentic_control"
    exec "${PYTHON_RUN}" run_closed_loop.py "$@"
    ;;
  validate)
    exec "${PYTHON_RUN}" "${CODE_ROOT}/validate_bundle.py" "$@"
    ;;
  verify-rollouts)
    exec "${PYTHON_RUN}" "${CODE_ROOT}/verify_rollouts.py" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command_name}" >&2
    usage
    exit 2
    ;;
esac
