# GameDirector model setup

The GameDirector weights are published separately from the source code at
[onepiece1999/GameDirector](https://huggingface.co/onepiece1999/GameDirector).
The public model repository contains these three checkpoints:

| Checkpoint | Purpose |
|---|---|
| `attacknet/attacknet.pt` | detects player and boss hits after each generated cell |
| `situationnet/situationnet.pt` | estimates player/boss distance and orientation |
| `world_model/ar_world_model.safetensors` | causal AR video world model used by the agentic rollout |

`SHA256SUMS` is included in the public model repository for integrity checks.
The local `config.json` files document training settings but are not required
at runtime and are not distributed in the public model repository. Model
weights are intentionally excluded from the source-code repository.

## 0. Prepare the source environment

Run these commands from the GameDirector source repository root:

```bash
python3 -m venv code/.venv
code/.venv/bin/pip install -r code/requirements.txt
```

A CUDA-capable NVIDIA GPU is required for generation. The documented downloads
are about 55 GB in total before generated outputs and Hugging Face cache.
Install FFmpeg with
`ffmpeg` and `ffprobe` available on `PATH`; the latter is used by rollout
verification. Install the Hugging Face CLI if `hf` is not already available:

```bash
code/.venv/bin/pip install -U huggingface_hub
```

## 1. Download the GameDirector checkpoints

Choose your own local path. Do not copy a path from another user or machine.

```bash
export GAMEDIRECTOR_MODEL_DIR="/path/to/gamedirector-model"

hf download onepiece1999/GameDirector \
  --local-dir "$GAMEDIRECTOR_MODEL_DIR"

export GAMEDIRECTOR_ATTACK_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/attacknet/attacknet.pt"
export GAMEDIRECTOR_SITUATION_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/situationnet/situationnet.pt"
export GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/world_model/ar_world_model.safetensors"
```

Verify the download:

```bash
(cd "$GAMEDIRECTOR_MODEL_DIR" && sha256sum -c SHA256SUMS)
```

All three checkpoint lines should report `OK`.

## 2. Download Wan and the tokenizer

GameDirector does not redistribute the upstream Wan assets. Download them
from their original Hugging Face repositories into another user-selected
path. The directory layout below is the layout expected by the code.

```bash
export GAMEDIRECTOR_BASE_MODEL="/path/to/wan-assets"

hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$GAMEDIRECTOR_BASE_MODEL/Wan-AI/Wan2.2-TI2V-5B"

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local-dir "$GAMEDIRECTOR_BASE_MODEL/Wan-AI/Wan2.1-T2V-1.3B"
```

The first download supplies the Wan weights, VAE, and UMT5 text encoder. The
second supplies the `google/umt5-xxl` tokenizer used by the runtime.

## 3. Download Gemma

The local planner uses `google/gemma-4-E2B-it`. Accept the model license on
Hugging Face and authenticate before downloading it.

```bash
hf auth login
export GAMEDIRECTOR_LLM_MODEL="/path/to/google/gemma-4-E2B-it"
hf download google/gemma-4-E2B-it \
  --local-dir "$GAMEDIRECTOR_LLM_MODEL"
```

The source repository uses a separate Python environment for the Gemma
service:

```bash
python3 -m venv code/.venv-gemma
code/.venv-gemma/bin/pip install -r code/llm/requirements.txt
```

## 4. Generate the bundled examples

From the source repository root, keep the five variables above exported and
run:

```bash
./example/run_rollout.sh all
```

This runs the code-provided agentic workflow for all eight inputs and writes
results under `output/ar_agentic/`. To run only one game family:

```bash
./example/run_rollout.sh nrftw
./example/run_rollout.sh hk
```

Validate the installed models and the generated rollouts:

```bash
./code/run.sh validate
./code/run.sh verify-rollouts
```

`verify-rollouts` requires `ffprobe` from FFmpeg. See the root `README.md` for
environment setup, custom inputs, training, and output details.
