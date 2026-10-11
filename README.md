# GameDirector

GameDirector is a closed-loop agent for No Rest for the Wicked (NRFTW),
V Rising, and Hollow Knight. A causal autoregressive (AR) video world model
renders the fight one 0.25-second prompt cell at a time. After every cell,
AttackNet detects hits and updates the combat state. SituationNet then
estimates player/boss geometry, and a local Gemma planner chooses the boss's
special skills. The next cell's prompt is composed from the updated state.

## Layout

```text
code/
  agentic_control/   AttackNet / SituationNet training and the closed-loop agent
  world_model/       world-model training and shared Wan components
  distill/           causal AR world-model inference runtime
  llm/               local OpenAI-compatible Gemma service
  run.sh             single training / inference entry point
example/             eight self-contained rollout inputs + run_rollout.sh
model/               local model mount point (weights are published separately)
output/ar_agentic/   reference rollouts produced by example/run_rollout.sh
```

## Setup

```bash
python3 -m venv code/.venv
code/.venv/bin/pip install -r code/requirements.txt
```

Download the GameDirector checkpoints from the separate Hugging Face model
repository. Choose any local destination:

```bash
export GAMEDIRECTOR_MODEL_REPO="onepiece1999/GameDirector"
export GAMEDIRECTOR_MODEL_DIR="/path/to/gamedirector-model"
hf download "$GAMEDIRECTOR_MODEL_REPO" --local-dir "$GAMEDIRECTOR_MODEL_DIR"

export GAMEDIRECTOR_ATTACK_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/attacknet/attacknet.pt"
export GAMEDIRECTOR_SITUATION_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/situationnet/situationnet.pt"
export GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT="$GAMEDIRECTOR_MODEL_DIR/world_model/ar_world_model.safetensors"
```

Download the upstream Wan assets separately into another user-selected path:

```bash
export GAMEDIRECTOR_BASE_MODEL="/path/to/wan-assets"
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$GAMEDIRECTOR_BASE_MODEL/Wan-AI/Wan2.2-TI2V-5B"
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' \
  --local-dir "$GAMEDIRECTOR_BASE_MODEL/Wan-AI/Wan2.1-T2V-1.3B"
```

`model/SHA256SUMS` lists the expected GameDirector checkpoint hashes.

The planner uses `google/gemma-4-E2B-it`. Accept its license on Hugging Face,
then download it and create its environment:

```bash
hf auth login
export GAMEDIRECTOR_LLM_MODEL="/path/to/google/gemma-4-E2B-it"
hf download google/gemma-4-E2B-it --local-dir "$GAMEDIRECTOR_LLM_MODEL"
python3 -m venv code/.venv-gemma
code/.venv-gemma/bin/pip install -r code/llm/requirements.txt
```

The inference entry point starts the Gemma service automatically on
`127.0.0.1:8018`. To use an already-running compatible service instead, pass
`--no-llm-auto-start --llm-base-url URL`.

## One-click agentic AR rollouts

```bash
export GAMEDIRECTOR_LLM_MODEL="/path/to/google/gemma-4-E2B-it"
export GAMEDIRECTOR_ATTACK_CHECKPOINT="/path/to/gamedirector-model/attacknet/attacknet.pt"
export GAMEDIRECTOR_SITUATION_CHECKPOINT="/path/to/gamedirector-model/situationnet/situationnet.pt"
export GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT="/path/to/gamedirector-model/world_model/ar_world_model.safetensors"
export GAMEDIRECTOR_BASE_MODEL="/path/to/wan-assets"
./example/run_rollout.sh all       # all eight examples
./example/run_rollout.sh nrftw     # nrftw_01..03
./example/run_rollout.sh hk        # hk_01..05
```

Each rollout writes `output/ar_agentic/<game>/<sample>/`: `initial.json`,
per-tick `tick_XXXX.json` records and `clip_XXXX.mp4` HUD clips,
`summary.json`, and the combined `rollout.mp4`. The world model runs stage-1
causal AR with 30 Euler steps, CFG 2.0, and one continuous KV-cache session
with a 3+17 attention sink.

To check a finished run (from the repository root):

```bash
./code/run.sh verify-rollouts                     # all eight
./code/run.sh verify-rollouts --sample nrftw_01   # one sample
```

The bundled reference results in `output/ar_agentic/` are:

| Sample | Matchup | Outcome | Length |
|---|---|---|---|
| nrftw_01 | Red Knight vs Plague Leader | You Win | 43.5 s |
| nrftw_02 | Green Scout vs Mutant Soldier | You Win | 23.5 s |
| nrftw_03 | Grey Monk vs Golden Bird (120/120 attack, 6 ticks) | You Lose | 12.5 s |
| hk_01 | the Knight vs Broken Vessel | You Win | 14.75 s |
| hk_02 | the Knight vs Hornet | You Win | 14.5 s |
| hk_03 | the Knight vs Dung Defender | You Win | 12.75 s |
| hk_04 | the Knight vs Hornet | You Win | 16.5 s |
| hk_05 | the Knight vs Gruz Mother | You Win | 13.5 s |

See `example/README.md` for the input contract and termination rules.

## Custom inputs

For your own inputs, pass the initial frame, the player prompt track, and the
prompt vocabulary. The vocabulary can be a compact snapshot
(`--vocabulary-file`, as in `example/`) or a raw prompt corpus
(`--prompt-root`):

```bash
./code/run.sh infer \
  --game nrftw \
  --image /path/to/first_frame.png \
  --player-prompts /path/to/player_track.json \
  --prompt-root /path/to/nrftw_5s_prompts/prompts \
  --player-hp 800 --boss-hp 1600 \
  --player-attack 55 --boss-attack 35 \
  --boss-skill 30 --boss-skill-max 30 \
  --ticks 12
```

The HP, attack, and skill arguments set the real combat state: damage,
skill-meter readiness and cost, the state sent to Gemma, event logs, and the
HUD. Run `./code/run.sh infer --help` for character choices, cooldown
overrides, devices, and generation settings.

## Training

AttackNet and SituationNet take parallel lists of game names, dataset roots,
and train/validation manifests:

```bash
./code/run.sh train attacknet \
  --game_names nrftw vrising hk \
  --data_root /data/nrftw /data/vrising /data/hk \
  --train_jobs /data/splits/n_train.json /data/splits/v_train.json /data/splits/h_train.json \
  --val_jobs /data/splits/n_val.json /data/splits/v_val.json /data/splits/h_val.json

./code/run.sh train situationnet \
  --game_names nrftw vrising hk \
  --data_root /data/nrftw /data/vrising /data/hk \
  --train_jobs /data/splits/n_train.json /data/splits/v_train.json /data/splits/h_train.json \
  --val_jobs /data/splits/n_val.json /data/splits/v_val.json /data/splits/h_val.json
```

World-model training defaults to four visible GPUs:

```bash
export GAMEDIRECTOR_BASE_MODEL="/path/to/model-assets"
export WORLD_MODEL_DATA_ROOT=/data/world_model
export WORLD_MODEL_METADATA=/data/world_model/jobs_train.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 ./code/run.sh train world_model
```

Settings can be overridden by command-line arguments or the environment
variables in `code/config.example.env`.

## License

Third-party Wan/DiffSynth notices are in `code/world_model/LICENSE` and
`code/world_model/NOTICE`. Before publishing this repository as open source,
the project owner must add a root license covering the original GameDirector
code and confirm redistribution rights for examples and reference videos.
Checkpoint licensing belongs in the separate Hugging Face model repository.
