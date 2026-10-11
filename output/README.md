# Reference outputs

`ar_agentic/<game>/<sample>/` holds the eight rollouts produced by
`example/run_rollout.sh`. Each sample directory contains:

- `initial.json`: the initial combat state, perception, and runtime settings;
- `tick_XXXX.json`: per-cell prompts, AttackNet signals, state transitions, and
  planner decisions;
- `clip_XXXX.mp4`: the HUD clip for each tick;
- `summary.json`: the outcome and final state;
- `rollout.mp4`: the full combined video.

New runs overwrite these directories. Gemma service logs go to `llm/`, which
git ignores.
