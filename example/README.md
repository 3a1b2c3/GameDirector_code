# Eight one-click agentic rollout examples

Three NRFTW and five Hollow Knight held-out inputs. Each sample directory is
self-contained:

- `first_frame.png`: the initial frame.
- `player_track.json`: a 240-cell player action/movement track (12 ticks).
- `vocabulary.json`: the compact prompt vocabulary for that matchup.

`manifest.json` lists each sample's characters, seed, and any non-default
settings.

Complete checkpoint, Wan, tokenizer, and Gemma download instructions are in
[`model/README.md`](../model/README.md).

The examples supply only the initial frame and the player's inputs. Every
other state starts from the default combat state: 500/500 HP, 40 player attack,
20 boss attack, and a full 20/20 boss skill meter. The one exception is
`nrftw_03`, which uses 120/120 attack and a 6-tick budget. The agent then
controls everything else:

- boss movement, actions, and skill choice;
- perception, damage, and HP;
- skill gain, spending, and cooldowns;
- death.

The causal AR world model generates four frames per 0.25-second cell. AttackNet
updates the combat state after every cell, and the next cell's prompt is
composed from the updated state. When either side dies, the game's
death/no-action prompt is held for five seconds and the rollout stops. The
tick budget is a hard ceiling, so no rollout exceeds 12 ticks (60 seconds).

```bash
export GAMEDIRECTOR_LLM_MODEL=/path/to/gemma-4-E2B-it
export GAMEDIRECTOR_ATTACK_CHECKPOINT=/path/to/gamedirector-model/attacknet/attacknet.pt
export GAMEDIRECTOR_SITUATION_CHECKPOINT=/path/to/gamedirector-model/situationnet/situationnet.pt
export GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT=/path/to/gamedirector-model/world_model/ar_world_model.safetensors
export GAMEDIRECTOR_BASE_MODEL=/path/to/wan-assets

./example/run_rollout.sh nrftw     # three NRFTW rollouts
./example/run_rollout.sh hk        # five Hollow Knight rollouts
./example/run_rollout.sh all       # all eight
```

All checkpoints and model assets are supplied through the variables shown
above. No model weight is required to live inside the public code repository.

Results go to `output/ar_agentic/<game>/<sample>/rollout.mp4`. Set
`EXAMPLE_OUTPUT_ROOT` to write them somewhere else, or `EXAMPLE_TICKS` to cap
the length for a quick smoke test.

To verify a run's state transitions, agent decisions, model identity, and
video frame counts, run from the repository root:

```bash
./code/run.sh verify-rollouts                          # output/ar_agentic
./code/run.sh verify-rollouts --rollout-root DIR       # relocated outputs
./code/run.sh verify-rollouts --report report.json     # save a JSON report
```
