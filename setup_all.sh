#!/usr/bin/env bash
# GameDirector complete setup for Linux: runs setup_env.sh, then download_models.sh in sequence.
# The same settings as setup_env.sh apply (CUDA_TAG, PYTHON_VERSION). The Gemma download needs a Hugging Face login, see
# download_models.sh.
#
# Usage:   bash setup_all.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo
echo "=========================================="
echo "GameDirector: Complete Setup"
echo "=========================================="
echo "This will:"
echo "1. Create the Python venv and install torch and all requirements (bash setup_env.sh)"
echo "2. Download the models, about 32 GB (bash download_models.sh)"
echo "=========================================="

echo
echo "Running: setup_env.sh"
bash setup_env.sh

echo
echo "Running: download_models.sh"
bash download_models.sh

echo
echo "=========================================="
echo "Setup complete. You can now run:"
echo "  bash run_inference.sh <run_closed_loop.py arguments>   or   bash run_example.sh nrftw_01"
echo "=========================================="
