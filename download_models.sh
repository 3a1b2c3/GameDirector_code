#!/usr/bin/env bash
# GameDirector model download for Linux: about 32 GB (GameDirector checkpoints 9.6 GB, Wan2.2 VAE + T5 13.2 GB, UMT5
# tokenizer 0.02 GB, Gemma planner 9.6 GB; sizes read from the Hugging Face API). Keep about 35 GB free.
# Resumable: running it again skips what is already complete. Needs setup_env.sh to have completed.
#
# The Gemma repo is gated: accept the license at https://huggingface.co/google/gemma-4-E2B-it, then log in once with
#     code/.venv/bin/hf auth login
# (or export HF_TOKEN). The token is never printed or stored by this script. Steps 1 to 3 run without a login.
#
# Usage:   bash download_models.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HF="code/.venv/bin/hf"
if [[ ! -x "${HF}" ]]; then
  echo "ERROR: ${HF} not found. Run setup_env.sh first." >&2
  exit 1
fi

echo "=========================================="
echo "GameDirector: Download Models (about 32 GB)"
echo "free on this drive: $(df -h . | awk 'NR==2 {print $4}')"
echo "=========================================="

echo "1/4: GameDirector checkpoints..."
"${HF}" download onepiece1999/GameDirector --include "attacknet/*" --include "situationnet/*" --include "world_model/*" --local-dir model

echo "2/4: Wan2.2 VAE + text encoder..."
"${HF}" download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --include models_t5_umt5-xxl-enc-bf16.pth --local-dir model/Wan-AI/Wan2.2-TI2V-5B

echo "3/4: UMT5 tokenizer..."
"${HF}" download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir model/Wan-AI/Wan2.1-T2V-1.3B

echo "4/4: Gemma planner..."
if [[ -z "${HF_TOKEN:-}" ]] && ! "${HF}" auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in to Hugging Face. Accept the license at https://huggingface.co/google/gemma-4-E2B-it, then run:" >&2
  echo "    ${HF} auth login" >&2
  echo "and run this script again (steps 1 to 3 are already done and will be skipped)." >&2
  exit 1
fi
"${HF}" download google/gemma-4-E2B-it --local-dir external_models/gemma-4-E2B-it

echo
echo "=========================================="
echo "All models downloaded. Next: bash run_example.sh nrftw_01"
echo "=========================================="
