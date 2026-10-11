#!/usr/bin/env python3
"""Verify packaged causal-AR agentic rollouts and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_ROOT.parent
MANIFEST_PATH = REPOSITORY_ROOT / "example" / "manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    require(len(streams) == 1, f"expected one video stream: {path}")
    return streams[0]


def verify_input(sample: dict[str, Any]) -> None:
    sample_dir = REPOSITORY_ROOT / "example" / sample["game"] / sample["name"]
    require((sample_dir / "first_frame.png").is_file(), f"missing first frame: {sample['name']}")
    track = load_json(sample_dir / "player_track.json")
    prompts = track.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 240, f"invalid player track: {sample['name']}")
    allowed = {"player_action", "player_movement", "location"}
    for index, prompt in enumerate(prompts):
        require(isinstance(prompt, dict), f"invalid player prompt {sample['name']}:{index}")
        require(set(prompt) <= allowed, f"non-player field in {sample['name']}:{index}: {sorted(set(prompt) - allowed)}")
        require({"player_action", "player_movement"} <= set(prompt), f"missing player field in {sample['name']}:{index}")


def verify_sample(
    sample: dict[str, Any],
    rollout_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    name = sample["name"]
    output_dir = rollout_root / sample["game"] / name
    summary = load_json(output_dir / "summary.json")
    initial = load_json(output_dir / "initial.json")
    ticks = summary.get("ticks")
    generated_cells = summary.get("generated_cells")
    requested = sample.get("ticks", manifest["ticks_per_rollout"])
    frames_per_tick = manifest["frames_per_tick"]
    fps = manifest["fps"]
    require(isinstance(ticks, int) and 1 <= ticks <= requested, f"invalid AR tick count: {name}")
    require(
        isinstance(generated_cells, int) and 1 <= generated_cells <= requested * 20,
        f"invalid AR cell count: {name}",
    )
    require(summary.get("requested_ticks") == requested, f"wrong requested tick count: {name}")
    require(ticks == (generated_cells + 19) // 20, f"tick/cell count mismatch: {name}")
    require(summary.get("duration_s") == generated_cells * 0.25, f"wrong duration: {name}")
    require(summary.get("characters") == {"player": sample["player"], "boss": sample["boss"]}, f"wrong characters: {name}")
    expected_initial = sample.get(
        "initial_state", manifest["input_contract"]["default_state"],
    )
    require(initial.get("combat") == expected_initial, f"wrong initial state: {name}")
    termination = initial.get("termination_policy", {})
    require(termination.get("mode") == "post_death_window", f"wrong AR termination policy: {name}")
    require(termination.get("post_death_seconds") == manifest["world_model"]["post_death_seconds"], f"wrong death hold: {name}")
    require(termination.get("post_death_cells") == 20, f"wrong death-hold cell count: {name}")
    require(termination.get("max_cells") == requested * 20, f"wrong duration ceiling: {name}")

    expected_checkpoint = Path(os.environ.get(
        "GAMEDIRECTOR_WORLD_MODEL_CHECKPOINT",
        REPOSITORY_ROOT / manifest["world_model"]["checkpoint"],
    )).expanduser().resolve()
    checkpoint = Path(summary.get("world_model_checkpoint", "")).resolve()
    require(checkpoint == expected_checkpoint, f"wrong AR checkpoint: {name}: {checkpoint}")
    require(checkpoint.is_file(), f"missing world model checkpoint: {name}")
    runtime = summary.get("world_model_runtime", {})
    require(runtime.get("sampler") == manifest["world_model"]["sampler"], f"wrong AR sampler: {name}")
    require(runtime.get("steps") == manifest["world_model"]["steps"], f"wrong AR step count: {name}")
    require(runtime.get("cfg_scale") == manifest["world_model"]["cfg"], f"wrong AR CFG: {name}")
    require(runtime.get("attention_sink") is True, f"attention sink disabled: {name}")
    require(runtime.get("reanchor_between_ticks") is False, f"AR session is not continuous: {name}")

    tick_files = sorted(output_dir.glob("tick_*.json"))
    clip_files = sorted(output_dir.glob("clip_*.mp4"))
    require(len(tick_files) == ticks, f"tick-file count mismatch: {name}")
    require(len(clip_files) == ticks, f"clip-file count mismatch: {name}")
    require(summary.get("records") == [path.name for path in tick_files], f"summary record list mismatch: {name}")

    previous = initial["combat"]
    planner_sources: Counter[str] = Counter()
    llm_calls = 0
    repairs = 0
    death_prompt_cells = 0
    first_death_cell: int | None = None
    global_cell = 0
    for index, (tick_path, clip_path) in enumerate(zip(tick_files, clip_files)):
        tick = load_json(tick_path)
        cells = tick.get("cells")
        require(isinstance(cells, int) and 1 <= cells <= 20, f"invalid tick cell count: {name}:{index}")
        if index < ticks - 1:
            require(cells == 20, f"non-final partial tick: {name}:{index}")
        require(tick.get("partial_tick") is (cells != 20), f"wrong partial-tick flag: {name}:{index}")
        require(tick.get("tick") == index, f"non-sequential tick index: {name}:{index}")
        require(tick.get("combat_before") == previous, f"broken combat-state continuity: {name}:{index}")
        require(len(tick.get("prompt_cells", [])) == cells, f"wrong prompt-cell count: {name}:{index}")
        require(len(tick.get("attacknet_cells", [])) == cells, f"wrong AttackNet-cell count: {name}:{index}")
        require(len(tick.get("health_timeline", [])) == cells, f"wrong health-timeline count: {name}:{index}")
        require(Path(tick.get("world_model_checkpoint", "")).resolve() == expected_checkpoint, f"wrong model in tick: {name}:{index}")
        planner = tick.get("planner", {})
        source = str(planner.get("source", ""))
        require(source and "fallback" not in source.lower(), f"planner fallback used: {name}:{index}")
        planner_sources[source] += 1
        if source == "llm":
            llm_calls += 1
            require(planner.get("error") is None, f"LLM planner error: {name}:{index}")
        repairs += sum(cell.get("repair") is not None for cell in tick["prompt_cells"])
        state_cursor = dict(previous)
        tick_start_cost = sum(
            int(event.get("skill_cost", 0))
            for event in tick.get("plan_events", [])
            if event.get("onset_cell") == 0
            and not event.get("continued_from_previous_tick", False)
        )
        if tick_start_cost:
            require(state_cursor["boss_skill"] >= tick_start_cost, f"illegal tick-start skill spend: {name}:{index}")
            state_cursor["boss_skill"] -= tick_start_cost
        for cell_index, (cell, event) in enumerate(
            zip(tick["prompt_cells"], tick["health_timeline"])
        ):
            require(event.get("before") == state_cursor, f"broken cell-state input: {name}:{index}:{cell_index}")
            if state_cursor["boss_hp"] <= 0:
                require(cell.get("boss_requested", {}).get("action") == "dying", f"missing boss death prompt: {name}:{index}:{cell_index}")
                death_prompt_cells += 1
            elif state_cursor["player_hp"] <= 0:
                require(cell.get("boss_requested", {}).get("action") == "no_action", f"boss acts after player death: {name}:{index}:{cell_index}")
                death_prompt_cells += 1
            state_cursor = event.get("after")
            if first_death_cell is None and (
                state_cursor["player_hp"] <= 0 or state_cursor["boss_hp"] <= 0
            ):
                first_death_cell = global_cell
            global_cell += 1
        require(tick.get("combat_after") == state_cursor, f"wrong tick final state: {name}:{index}")
        previous = state_cursor

        clip_probe = probe_video(clip_path)
        require(int(clip_probe["nb_read_frames"]) == cells * 4, f"wrong clip frame count: {name}:{index}")
        require((int(clip_probe["width"]), int(clip_probe["height"])) == (832, 480), f"wrong clip dimensions: {name}:{index}")
        require(clip_probe["r_frame_rate"] == f"{fps}/1", f"wrong clip FPS: {name}:{index}")

    require(llm_calls >= 1, f"no successful Gemma planning call: {name}")
    require(summary.get("final_combat") == previous, f"wrong final combat state: {name}")
    final_player = previous["player_hp"]
    final_boss = previous["boss_hp"]
    expected_outcome = "You Lose" if final_player <= 0 else ("You Win" if final_boss <= 0 else None)
    require(summary.get("outcome") == expected_outcome, f"wrong outcome: {name}")
    require(summary.get("death_detected_cell") == first_death_cell, f"wrong death cell: {name}")
    if first_death_cell is None:
        require(ticks == requested, f"live rollout stopped before tick budget: {name}")
        require(generated_cells == requested * 20, f"live rollout has wrong cell budget: {name}")
        require(summary.get("termination_reason") == "tick_budget", f"wrong live termination: {name}")
        require(summary.get("stopped_after_death") is False, f"live rollout marked death stop: {name}")
    else:
        post_death_cells = int(manifest["world_model"]["post_death_seconds"] / 0.25)
        expected_cells = min(requested * 20, first_death_cell + 1 + post_death_cells)
        require(generated_cells == expected_cells, f"wrong post-death hold length: {name}")
        require(summary.get("post_death_generated_s") == (expected_cells - first_death_cell - 1) * 0.25, f"wrong reported death hold: {name}")
        require(summary.get("termination_reason") == "post_death_window", f"wrong death termination: {name}")
        require(summary.get("stopped_after_death") is True, f"death rollout not marked stopped: {name}")
    if first_death_cell is not None and first_death_cell + 1 < generated_cells:
        require(death_prompt_cells > 0, f"no reactive death prompt after death: {name}")

    rollout_probe = probe_video(output_dir / "rollout.mp4")
    expected_frames = generated_cells * 4
    require(int(rollout_probe["nb_read_frames"]) == expected_frames, f"wrong rollout frame count: {name}")
    require((int(rollout_probe["width"]), int(rollout_probe["height"])) == (832, 480), f"wrong rollout dimensions: {name}")
    require(rollout_probe["r_frame_rate"] == f"{fps}/1", f"wrong rollout FPS: {name}")

    return {
        "name": name,
        "game": sample["game"],
        "ticks": ticks,
        "duration_s": summary["duration_s"],
        "outcome": summary["outcome"],
        "final_player_hp": final_player,
        "final_boss_hp": final_boss,
        "frames": expected_frames,
        "llm_calls": llm_calls,
        "planner_sources": dict(sorted(planner_sources.items())),
        "prompt_repairs": repairs,
        "death_detected_cell": first_death_cell,
        "death_prompt_cells": death_prompt_cells,
        "video": str(output_dir / "rollout.mp4"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "ar_agentic",
        help="directory containing <game>/<sample>/summary.json",
    )
    parser.add_argument("--report", type=Path, help="optional path for the JSON verification report")
    parser.add_argument(
        "--sample", action="append",
        help="verify only this sample name; repeat for multiple samples",
    )
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    samples = manifest.get("samples", [])
    require(len(samples) == 8, "manifest must contain exactly eight samples")
    if args.sample:
        selected = set(args.sample)
        samples = [sample for sample in samples if sample["name"] in selected]
        require({sample["name"] for sample in samples} == selected, "unknown --sample name")
    results = []
    for sample in samples:
        verify_input(sample)
        results.append(verify_sample(sample, args.rollout_root.resolve(), manifest))

    report = {
        "ok": True,
        "rollout_root": str(args.rollout_root.resolve()),
        "verified_rollouts": len(results),
        "total_ticks": sum(row["ticks"] for row in results),
        "total_frames": sum(row["frames"] for row in results),
        "rollouts": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
