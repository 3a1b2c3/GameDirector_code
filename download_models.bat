@echo off
REM GameDirector model download script for Windows
REM Downloads all required checkpoints and weights (about 32 GB total: 9.6 + 13.2 + 0.02 + 9.6, sizes read from the Hugging Face API)
REM Requires: setup_env.bat completed first

setlocal enabledelayedexpansion

echo ==========================================
echo GameDirector: Download Models
echo ==========================================
echo Download size: about 32 GB (GameDirector 9.6 GB, Wan2.2 VAE + T5 13.2 GB, tokenizer 0.02 GB, Gemma 9.6 GB). Keep about 35 GB free.
echo.

cd /d "%~dp0"

REM Check venv exists (the Hugging Face CLI is the hf command; python -m huggingface_hub.commands.huggingface_cli no longer exists in huggingface_hub 1.x/2.x)
if not exist "code\.venv\Scripts\python.exe" (
    echo ERROR: venv not found at code\.venv
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

set PY=code\.venv\Scripts\python
set HF=code\.venv\Scripts\hf

REM Download GameDirector checkpoints
echo.
echo ==========================================
echo 1/4: Downloading GameDirector checkpoints...
echo ==========================================
%HF% download onepiece1999/GameDirector --include "attacknet/*" --include "situationnet/*" --include "world_model/*" --local-dir model
if errorlevel 1 (
    echo ERROR: Failed to download GameDirector checkpoints
    pause
    exit /b 1
)

REM Download Wan2.2 VAE + text encoder
echo.
echo ==========================================
echo 2/4: Downloading Wan2.2 VAE + text encoder...
echo ==========================================
%HF% download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --include models_t5_umt5-xxl-enc-bf16.pth --local-dir model\Wan-AI\Wan2.2-TI2V-5B
if errorlevel 1 (
    echo ERROR: Failed to download Wan2.2 assets
    pause
    exit /b 1
)

REM Download tokenizer
echo.
echo ==========================================
echo 3/4: Downloading UMT5 tokenizer...
echo ==========================================
%HF% download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir model\Wan-AI\Wan2.1-T2V-1.3B
if errorlevel 1 (
    echo ERROR: Failed to download tokenizer
    pause
    exit /b 1
)

REM Download Gemma planner
echo.
echo ==========================================
echo 4/4: Downloading Gemma planner...
echo ==========================================
echo Note: You must accept the Gemma license on HuggingFace first
echo Visit: https://huggingface.co/google/gemma-4-E2B-it
echo.
%HF% auth login
%HF% download google/gemma-4-E2B-it --local-dir external_models\gemma-4-E2B-it
if errorlevel 1 (
    echo ERROR: Failed to download Gemma (did you accept the license?)
    pause
    exit /b 1
)

echo.
echo ==========================================
echo All models downloaded successfully!
echo ==========================================
echo Next: edit config/config.example.env or run inference
echo ==========================================
pause
