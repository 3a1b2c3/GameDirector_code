# GameDirector on Linux (DGX GB300)

Bash scripts for setting up and running GameDirector on Linux. Written for the DGX GB300 workstation (aarch64, 256 GB GPU
memory, CUDA driver 610, toolkit 13.2). Windows: [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

**Status.** Verified on the DGX: `setup_all.sh` (torch 2.14.1+cu132, Python 3.11.16, transformers 5.19.0, all 32 GB of models)
and a 2-tick `bash run_example.sh nrftw_01 2` run, which produced a 10 s, 832x480, 16 fps `rollout.mp4` with the HUD
(HP bars, skill, keys). During that run the world model process peaked at 35.2 GB of GPU memory (PyTorch's allocator keeps what it can, so the true
minimum may be lower) and 86 GB of RAM, and Gemma used 11 GB of GPU memory.
The full 12-tick runs and the other examples have not been run yet.

## Why use the DGX

The world model's GPU memory peaked at 35 GB in the DGX test (about 31 GB in earlier samples on the 5090). On a 32 GB card the Gemma planner has to move to the CPU, which
is too slow. The GB300 has room for both on the GPU. Its connection to Hugging Face is also fast (about 94 MB/s, so the 32 GB of
models take minutes).

## Quick start

```bash
cd ~/GameDirector_code
bash setup_env.sh                    # venv + torch + requirements
code/.venv/bin/hf auth login         # once, for the gated Gemma repo (accept the license first)
bash download_models.sh              # about 32 GB
bash run_example.sh nrftw_01 2       # 2-tick test; drop the 2 for the full 12
```

`bash setup_all.sh` runs the first and third step in order.

## Scripts

| Script | What it does |
|---|---|
| `setup_env.sh` | Creates `code/.venv` with uv, installs torch from the PyTorch index first, runs a bf16 matmul on the GPU as a check, installs the requirements without their torch lines, then checks the GPU again. Safe to run again. |
| `download_models.sh` | The same 32 GB as on Windows with `hf download`; resumable. Steps 1 to 3 need no login; step 4 (Gemma) stops with a message if you are not logged in. |
| `run_example.sh <example> [ticks]` | Runs `nrftw_01` `nrftw_02` `nrftw_03` `hk_01` to `hk_05` with the parameters of `example/run_rollout.sh`. |
| `run_all_examples.sh [--ticks N] [example ...]` | Runs all eight examples (or the ones you name) one after another and prints a summary table (status, seconds, final combat state). One failure does not stop the rest. The Gemma service is shared: a running one is reused, and one the script starts is stopped at the end. Logs and `summary.txt` go to `$EXAMPLE_OUTPUT_ROOT/_logs/`. |
| `run_inference.sh <args>` | Passes your arguments to `code/agentic_control/run_closed_loop.py`. |
| `setup_all.sh` | Runs `setup_env.sh`, then `download_models.sh`. |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_TAG` | `cu132` | PyTorch CUDA build for `setup_env.sh`. `cu132` matches `/usr/local/cuda-13.2`; `cu130` also works on this machine (the driver supports CUDA up to 13.3). |
| `PYTHON_VERSION` | `3.11` | Python for the venv. |
| `EXAMPLE_LLM_DEVICE` | `cuda:0` | Device of the Gemma planner; `cpu` if the GPU is too small. |
| `EXAMPLE_LLM_TIMEOUT` | program default (90 s) | Seconds per planner request. |
| `EXAMPLE_EXTRA_ARGS` | empty | Extra `run_closed_loop.py` arguments. |
| `EXAMPLE_OUTPUT_ROOT` | `output/local_runs` | Where results go. git ignores it, and it leaves the tracked sample results in `output/ar_agentic` alone. |
| `EXAMPLE_DRY_RUN=1` | off | Print the command and exit. |

The video is `$EXAMPLE_OUTPUT_ROOT/<game>/<example>/rollout.mp4`, next to one clip per tick, the planner's decisions
(`tick_*.json`), `initial.json` and `summary.json`.

## Torch and CUDA on the DGX

Checked on the machine: the other venvs there run torch 2.13 to 2.14 builds for both cu130 and cu132, and a bf16 matmul works
with the cu130 build on the GB300 (compute capability 10.3). PyTorch publishes aarch64 cp311 wheels for torch 2.14.1 and
torchvision 0.29.1 for both tags. So no upgrade is required; `cu132` is the default because it matches the 13.2 toolkit, which
matters only if something compiles CUDA code against torch. GameDirector's requirements contain nothing like that.

## Notes

- **Line endings.** The `.sh` files must be LF. `.gitattributes` has `*.sh text eol=lf` for this. A CRLF script fails with
  errors like `set: pipefail: invalid option name`.
- **Shared GPU.** Other jobs use part of the DGX's GPU memory; check `nvidia-smi` before a run. The upstream code and
  `track_memory.py` are the same as on Windows (`track_memory.py` reads Windows GPU counters, so on Linux use `nvidia-smi`).
- **Copying results back** to a Windows machine: `scp -r kschmid@<DGX IP>:~/GameDirector_code/output/local_runs/... .`
  (use the IP from `hostname -I`, not the hostname).
- **Code changes** to `code/llm/gemma_server.py` and `code/distill/inference/ar_pipeline.py` are listed in
  `WINDOWS_SETUP.md`; they apply on Linux too (float32 only when Gemma runs on the CPU).
