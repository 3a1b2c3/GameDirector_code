@echo off
REM Runs one of the bundled GameDirector example rollouts (the Windows version of example\run_rollout.sh, which needs bash).
REM The parameters per example are the ones in example\run_rollout.sh.
REM
REM Usage:  run_example.bat <example> [ticks]
REM   <example>  nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05
REM   [ticks]    optional, overrides the example's default number of ticks (12, or 6 for nrftw_03)
REM
REM Needs setup_env.bat and download_models.bat to have completed. The Gemma planner is the downloaded
REM external_models\gemma-4-E2B-it unless GAMEDIRECTOR_LLM_MODEL is already set. The video is written to
REM output\ar_agentic\<game>\<example>\rollout.mp4. The world model and Gemma both load onto the GPU, so make sure it has
REM room first (the line below prints the current GPU memory use).
REM
REM The Gemma planner runs on the CPU by default (EXAMPLE_LLM_DEVICE=cpu): the 5B world model needs about 31 GB of the
REM 32 GB GPU, and with Gemma (5 GB) on the GPU as well part of it spills into system memory and the run crawls.
REM The planner request timeout is EXAMPLE_LLM_TIMEOUT seconds (default 600; the program's own default of 90 is too short
REM for a CPU planner). Use  set EXAMPLE_LLM_DEVICE=cuda:0  to put Gemma back on the GPU. Extra run_closed_loop.py arguments can be passed
REM in EXAMPLE_EXTRA_ARGS, for example  set EXAMPLE_EXTRA_ARGS=--world-model-steps 20

setlocal

cd /d "%~dp0"

set NAME=%~1
if "%NAME%"=="" goto usage

set GAME=
if /i "%NAME%"=="nrftw_01" call :set_case nrftw 2 3 2 12 40 20 "Frozen Wasteland"
if /i "%NAME%"=="nrftw_02" call :set_case nrftw 3 2 2 12 40 20 "Frozen Wasteland"
if /i "%NAME%"=="nrftw_03" call :set_case nrftw 1 1 31 6 120 120 ""
if /i "%NAME%"=="hk_01" call :set_case hk 1 2 14 12 40 20 ""
if /i "%NAME%"=="hk_02" call :set_case hk 1 1 11 12 40 20 ""
if /i "%NAME%"=="hk_03" call :set_case hk 1 4 19 12 40 20 ""
if /i "%NAME%"=="hk_04" call :set_case hk 1 1 3 12 40 20 ""
if /i "%NAME%"=="hk_05" call :set_case hk 1 3 2 12 40 20 ""
if not defined GAME goto usage

if not "%~2"=="" set TICKS=%~2

if not exist "code\.venv\Scripts\python.exe" (
    echo ERROR: venv not found at code\.venv. Run setup_env.bat first.
    exit /b 1
)
if not exist "model\attacknet\attacknet.pt" (
    echo ERROR: models not found in model\. Run download_models.bat first.
    exit /b 1
)
if not defined GAMEDIRECTOR_LLM_MODEL set GAMEDIRECTOR_LLM_MODEL=external_models\gemma-4-E2B-it
if not exist "%GAMEDIRECTOR_LLM_MODEL%\model.safetensors" (
    echo ERROR: Gemma not found at %GAMEDIRECTOR_LLM_MODEL%. Run download_models.bat or set GAMEDIRECTOR_LLM_MODEL.
    exit /b 1
)

set SAMPLE=example\%GAME%\%NAME%
if not exist "%SAMPLE%\first_frame.png" (
    echo ERROR: %SAMPLE%\first_frame.png not found.
    exit /b 1
)
set OUT=output\ar_agentic\%GAME%\%NAME%
set PYTHONPATH=%CD%\code
if not defined EXAMPLE_LLM_DEVICE set EXAMPLE_LLM_DEVICE=cpu
if not defined EXAMPLE_LLM_TIMEOUT set EXAMPLE_LLM_TIMEOUT=600

echo ==========================================
echo GameDirector example %NAME%: %TICKS% ticks, output %OUT%\rollout.mp4
echo Gemma planner device: %EXAMPLE_LLM_DEVICE%
echo GPU memory used/total:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ==========================================

code\.venv\Scripts\python code\agentic_control\run_closed_loop.py --game %GAME% --image "%SAMPLE%\first_frame.png" --player-prompts "%SAMPLE%\player_track.json" --vocabulary-file "%SAMPLE%\vocabulary.json" --player-choice %PLAYER_CHOICE% --boss-choice %BOSS_CHOICE% --ticks %TICKS% --player-attack %PLAYER_ATTACK% --boss-attack %BOSS_ATTACK% --seed %SEED% --output-dir "%OUT%" --attack-checkpoint model\attacknet\attacknet.pt --situation-checkpoint model\situationnet\situationnet.pt --base-model model --world-model-mode ar --world-model-checkpoint model\world_model\ar_world_model.safetensors --ar-stage 1 --ar-sampler euler --world-model-steps 30 --world-model-cfg 2.0 --ar-attention-sink --no-ar-reanchor-between-ticks --llm-checkpoint "%GAMEDIRECTOR_LLM_MODEL%" --llm-device %EXAMPLE_LLM_DEVICE% --llm-timeout %EXAMPLE_LLM_TIMEOUT% %BACKGROUND_ARGS% %EXAMPLE_EXTRA_ARGS%
if errorlevel 1 (
    echo ERROR: the rollout failed.
    exit /b 1
)
echo Done: %OUT%\rollout.mp4
exit /b 0

:set_case
set GAME=%~1
set PLAYER_CHOICE=%~2
set BOSS_CHOICE=%~3
set SEED=%~4
set TICKS=%~5
set PLAYER_ATTACK=%~6
set BOSS_ATTACK=%~7
set BACKGROUND_ARGS=
if not "%~8"=="" set BACKGROUND_ARGS=--background "%~8"
exit /b 0

:usage
echo Usage: run_example.bat ^<example^> [ticks]
echo   example: nrftw_01 nrftw_02 nrftw_03 hk_01 hk_02 hk_03 hk_04 hk_05
exit /b 2
