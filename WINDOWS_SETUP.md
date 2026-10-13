# GameDirector on Windows

Scripts for setting up and running GameDirector on Windows (RTX 5090, 32 GB). For Linux and the DGX see
[LINUX_SETUP.md](LINUX_SETUP.md).

**Status.** Setup, the 32 GB model download and loading all weights (checked on the CPU) work on Windows. No example has
completed a full run on a 32 GB GPU yet: the world model alone needs about 31 GB at its peak, so Gemma has to run on the CPU,
which is slow. For real runs, use the DGX.

## Quick start

```cmd
cd C:\workspace\world\GameDirector_code
setup_all.bat
run_example.bat nrftw_01 2
```

`setup_all.bat` = `setup_env.bat` + `download_models.bat`. The last line is a 2-tick test; drop the `2` for the full 12 ticks.

## Scripts

| Script | What it does |
|---|---|
| `setup_env.bat` | Creates `code\.venv` with **uv on Python 3.11**, installs **CUDA torch (cu130) first**, then the requirements without their torch lines, and stops unless torch can see the GPU. Safe to run again. |
| `download_models.bat` | Downloads about **32 GB** with the `hf` command: GameDirector checkpoints 9.6 GB, Wan2.2 VAE + T5 encoder 13.2 GB, UMT5 tokenizer 0.02 GB, Gemma planner 9.6 GB. Resumable. Keep about 35 GB free. |
| `run_example.bat <example> [ticks]` | Runs a bundled example: `nrftw_01` `nrftw_02` `nrftw_03` `hk_01` to `hk_05`. Parameters match `example\run_rollout.sh`. |
| `run_inference.bat <args>` | Passes your arguments to `code\agentic_control\run_closed_loop.py` (`--help` lists them). |
| `setup_all.bat` | Runs the two setup scripts in order. |
| `track_memory.py` | Logs GPU and RAM use of a run to `output\memory_track.csv` (see Memory). |

Why `setup_env.bat` is not a plain `python -m venv` + `pip install`: the requirements only say `torch>=2.0`, which resolves to a
**CPU-only torch** on Windows. That fails with `Torch not compiled with CUDA enabled` and has no RTX 5090 (Blackwell) kernels.
Python 3.12 is avoided on Windows because its thread pools can deadlock. The Gemma planner runs from the same venv:
`code\llm\service.py` only looks for `.venv-gemma\bin\python` (the Linux layout) and otherwise uses the running interpreter,
so no separate Gemma venv is created.

## Running an example

```cmd
run_example.bat nrftw_01
```

The result is `output\ar_agentic\<game>\<example>\rollout.mp4`, with one clip per tick (`clip_0000.mp4`, ...), the planner's
decisions per tick (`tick_0000.json`, ...), `initial.json` (characters and starting state) and `summary.json`.

Settings (set before running, for example `set EXAMPLE_LLM_DEVICE=cuda:0`):

| Variable | Default | Meaning |
|---|---|---|
| `EXAMPLE_LLM_DEVICE` | `cpu` | Device of the Gemma planner. On a 32 GB GPU the world model and Gemma together do not fit and part spills into system RAM, which makes the run crawl. |
| `EXAMPLE_LLM_TIMEOUT` | `600` | Seconds per planner request. The program's own 90 s is too short for a CPU planner. |
| `EXAMPLE_EXTRA_ARGS` | empty | Extra `run_closed_loop.py` arguments, e.g. `--world-model-steps 20`. |
| `GAMEDIRECTOR_LLM_MODEL` | `external_models\gemma-4-E2B-it` | Gemma directory. |

**Output folder.** `run_example.bat` writes to `output\ar_agentic`, which also holds the repository's bundled sample results
(tracked in git). A run **clears the folder of that example first**, so `git status` then shows those samples as deleted.
`git restore output/ar_agentic/<game>/<example>` brings them back.

## Memory

Measured on the RTX 5090 (128 GB RAM) with Gemma on the CPU:

| What | Value |
|---|---|
| GPU, world model idle after loading | about 22 GB |
| GPU peak while sampling | about 31 to 32 GB of 32.6 GB |
| RAM while loading the world model | peak 54 GB (was 66 GB before the loading changes below), 16 GB afterwards |
| Gemma on the CPU (float32) | about 20 GB of RAM |

If the GPU is full, Windows borrows system RAM ("shared GPU memory") and everything slows down. `track_memory.py` shows it:

```cmd
code\.venv\Scripts\python track_memory.py --interval 5 --minutes 120
```

It writes `output\memory_track.csv` (GPU used, utilization, power, the world-model process's dedicated and **shared** GPU
memory, process RAM, Gemma CPU use, free RAM) until you stop it with Ctrl+C. It needs `psutil`.

## Changes to the upstream code

Backups sit next to each file as `*.before_*`.

- `code\llm\gemma_server.py`: Gemma is loaded in **float32 on the CPU** (bf16 matmuls are very slow on a CPU without native bf16
  support) and in bf16 on the GPU as before.
- `code\distill\inference\ar_pipeline.py`: `from_pretrained` loads the DiT first, builds the text encoder directly in bf16 and
  loads the `.pth` files with `mmap=True`. The loaded weights are identical to before (checked against the files). The DiT
  cannot be built in bf16: it creates float constants at construction and building it that way crashed natively in `nn.Linear`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `AssertionError: Torch not compiled with CUDA enabled` | The venv has a CPU torch. Run `setup_env.bat`; it installs the cu130 build and checks CUDA. |
| `Gemma special-skill planning failed; fallback is disabled`, or `ConnectionAbortedError` in `output\llm\gemma_server.log` | The planner request timed out (CPU Gemma). Use the defaults of `run_example.bat` (timeout 600), or free the GPU and use `set EXAMPLE_LLM_DEVICE=cuda:0`. |
| Run is very slow, GPU at 100% but low power | Shared GPU memory is in use (GPU full). Check `track_memory.py`; keep Gemma on the CPU. |
| `No module named 'huggingface_hub.commands'` | The old Hugging Face CLI module is gone. `download_models.bat` uses the `hf` command. Log in with `code\.venv\Scripts\hf auth login`. |
| Gemma download fails | Accept the license at https://huggingface.co/google/gemma-4-E2B-it and log in as above (never paste the token into scripts). |
| Out of disk during the download | Need about 35 GB free. The earlier "275 GB" figure was wrong. |
| `venv not found` / `Models not found` | Run `setup_env.bat` / `download_models.bat` first. |
| An old Gemma service keeps being reused | A service from an earlier run still listens on port 8018. End that python process, then run again. |

## See also

`README.md` (training, validation), `model/README.md`, `code\config.example.env` (all `GAMEDIRECTOR_*` variables).
