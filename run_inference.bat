@echo off
REM GameDirector inference runner for Windows
REM Runs closed-loop game world inference

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Check venv exists
if not exist "code\.venv\Scripts\python.exe" (
    echo ERROR: venv not found at code\.venv
    echo Please run setup_env.bat first
    pause
    exit /b 1
)

REM Check models exist
if not exist "model\attacknet\attacknet.pt" (
    echo ERROR: Models not found in model\ directory
    echo Please run download_models.bat first
    pause
    exit /b 1
)

set PY=code\.venv\Scripts\python
set PYTHONPATH=%CD%\code

echo ==========================================
echo GameDirector: Running Inference
echo ==========================================
echo.

REM Run inference with passed arguments
%PY% code\agentic_control\run_closed_loop.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Inference failed
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Inference completed
echo ==========================================
pause
