@echo off
REM GameDirector complete setup for Windows
REM Runs setup_env.bat, then download_models.bat in sequence

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ==========================================
echo GameDirector: Complete Setup
echo ==========================================
echo This will:
echo 1. Create Python venv
echo 2. Install all requirements
echo 3. Download models (about 32 GB)
echo.
echo Total time: ~30-60 minutes (depending on network)
echo ==========================================
echo.
pause

REM Run setup_env
echo.
echo ==========================================
echo Running: setup_env.bat
echo ==========================================
call setup_env.bat
if errorlevel 1 (
    echo ERROR: setup_env.bat failed
    exit /b 1
)

REM Run download_models
echo.
echo ==========================================
echo Running: download_models.bat
echo ==========================================
call download_models.bat
if errorlevel 1 (
    echo ERROR: download_models.bat failed
    exit /b 1
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo You can now run:
echo   run_inference.bat
echo ==========================================
pause
