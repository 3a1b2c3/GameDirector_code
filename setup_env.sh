#!/usr/bin/env bash
# GameDirector Linux setup (written for the DGX GB300, aarch64): creates code/.venv with uv on Python 3.11, installs CUDA
# torch first, then the requirements without their torch lines, and stops unless torch can run a bf16 matmul on the GPU.
#
# The requirements say only "torch>=2.0", which pip/uv may resolve to a build that does not match the machine's CUDA, so
# the torch build is chosen here. Windows uses setup_env.bat (cu130).
#
# Usage:   bash setup_env.sh                      (safe to run again)
#   CUDA_TAG=cu130 bash setup_env.sh              another PyTorch CUDA build; the default cu132 matches /usr/local/cuda-13.2
#   PYTHON_VERSION=3.12 bash setup_env.sh         another Python; the default is 3.11
#
# The Gemma planner runs from this same venv: code/llm/service.py uses code/.venv-gemma/bin/python only when it exists and
# otherwise the running interpreter, and this script does not create .venv-gemma.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CUDA_TAG="${CUDA_TAG:-cu132}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_PYTHON="code/.venv/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Abort unless torch is a CUDA build that can run a bf16 matmul on the GPU. Run after every install step: a later package
# must not be able to replace the CUDA torch with another build unnoticed.
check_cuda() {
  "${VENV_PYTHON}" -c '
import sys
import torch
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda, "| available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(1)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float()).all()
print("GPU:", torch.cuda.get_device_name(0), "| bf16 matmul ok")
' || { echo "ERROR: torch cannot use the GPU. Stopping." >&2; exit 1; }
}

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Creating venv with Python ${PYTHON_VERSION}..."
  uv venv --python "${PYTHON_VERSION}" code/.venv
fi

echo "===== Installing torch (${CUDA_TAG}) ====="
uv pip install --python "${VENV_PYTHON}" torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
check_cuda

echo "===== Installing requirements, without torch/torchvision ====="
requirements="$(mktemp)"
trap 'rm -f "${requirements}"' EXIT
for file in code/agentic_control/requirements.txt code/world_model/requirements.txt code/llm/requirements.txt; do
  grep -v '^torch' "${file}" >> "${requirements}" || true
  echo >> "${requirements}"
done
uv pip install --python "${VENV_PYTHON}" -r "${requirements}"
check_cuda

echo
echo "=========================================="
echo "Done. Venv: code/.venv  (Python ${PYTHON_VERSION}, torch ${CUDA_TAG})"
echo "Next: bash download_models.sh, then bash run_example.sh nrftw_01"
echo "=========================================="
