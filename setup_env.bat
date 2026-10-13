@echo off
REM GameDirector Windows setup: creates code\.venv with uv on Python 3.11, installs CUDA torch (cu130) first, then the
REM requirements without their torch lines, and aborts unless torch can see the GPU.
REM
REM Why this differs from a plain "python -m venv" + "pip install -r": the requirements say only "torch>=2.0", which pip
REM resolves to a CPU-only wheel on Windows (no RTX 5090 / Blackwell kernels, "Torch not compiled with CUDA enabled"),
REM and Python 3.12 is avoided on Windows because its thread pools can deadlock.
REM
REM The Gemma planner runs from this same venv on Windows (code\llm\service.py only looks for .venv-gemma\bin\python, the
REM Linux layout, then falls back to the running interpreter), so code\llm\requirements.txt is installed here too.
REM
REM Usage: setup_env.bat      (needs uv on PATH; safe to run again)

setlocal

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv not found on PATH. Install it first: https://docs.astral.sh/uv/
    exit /b 1
)

if not exist "code\.venv\Scripts\python.exe" (
    echo Creating venv with Python 3.11...
    uv venv --python 3.11 code\.venv
    if errorlevel 1 exit /b 1
)

echo ===== Installing CUDA torch (cu130) =====
uv pip install --python code\.venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1
call :check_cuda
if errorlevel 1 exit /b 1

echo ===== Installing requirements, without torch/torchvision =====
type nul > "%TEMP%\gamedirector_requirements.txt"
REM one findstr call per file: with several files findstr prefixes every line with the file name
for %%F in (code\agentic_control\requirements.txt code\world_model\requirements.txt code\llm\requirements.txt) do findstr /v /b /c:"torch" %%F >> "%TEMP%\gamedirector_requirements.txt"
uv pip install --python code\.venv\Scripts\python.exe -r "%TEMP%\gamedirector_requirements.txt"
if errorlevel 1 exit /b 1
call :check_cuda
if errorlevel 1 exit /b 1

echo.
echo ==========================================
echo Done. Venv: code\.venv  (Python 3.11, CUDA torch)
echo Next: download_models.bat, then run_example.bat nrftw_01
echo ==========================================
exit /b 0

REM Abort unless torch is a CUDA build that can see the GPU. Run after every install step: a later package must not
REM be able to replace the CUDA torch with a CPU build.
:check_cuda
code\.venv\Scripts\python.exe -c "import sys, torch; print(\"torch\", torch.__version__, \"CUDA build:\", torch.version.cuda, \"available:\", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ERROR: torch is not a CUDA build that sees the GPU. Stopping so it does not go unnoticed.
    exit /b 1
)
exit /b 0
