#!/usr/bin/env bash
# GameDirector inference runner for Linux: runs the closed-loop game world inference with the arguments you pass, the same
# way run_inference.bat does (see bash run_inference.sh --help for the arguments, and run_example.sh for ready-made examples).
#
# Usage:   bash run_inference.sh <run_closed_loop.py arguments>
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python_bin="code/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: venv not found at code/.venv. Run setup_env.sh first." >&2
  exit 1
fi
if [[ ! -f model/attacknet/attacknet.pt ]]; then
  echo "ERROR: models not found in model/. Run download_models.sh first." >&2
  exit 1
fi

echo "=========================================="
echo "GameDirector: Running Inference"
echo "=========================================="

PYTHONPATH="${PWD}/code" "${python_bin}" code/agentic_control/run_closed_loop.py "$@"

echo
echo "=========================================="
echo "Inference completed"
echo "=========================================="
