#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"
PYTHON_BIN="${GAMEDIRECTOR_GEMMA_PYTHON:-${PROJECT_ROOT}/.venv-gemma/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Gemma Python not found. Create ${PROJECT_ROOT}/.venv-gemma from llm/requirements.txt." >&2
  exit 2
fi

REPOSITORY_ROOT="$(dirname "${PROJECT_ROOT}")"
export GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-${GAMEDIRECTOR_LLM_MODEL:-${REPOSITORY_ROOT}/external_models/gemma-4-E2B-it}}"
export GEMMA_MODEL="${GEMMA_MODEL:-gemma-4-E2B-it}"
export GEMMA_DEVICE="${GEMMA_DEVICE:-cuda:0}"
export GEMMA_PORT="${GEMMA_PORT:-8018}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${REPOSITORY_ROOT}/output/llm"
echo "$$" > "${REPOSITORY_ROOT}/output/llm/gemma_server.pid"
exec "${PYTHON_BIN}" "${HERE}/gemma_server.py"
