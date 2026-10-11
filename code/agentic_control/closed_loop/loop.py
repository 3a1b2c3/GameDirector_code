"""One shared closed-loop pipeline for three game-specific configurations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .combat import CombatState, DamageCalculator
from .components import save_mp4
from .decision import CELL_SECONDS, PlanItem, automatic_basic, expand_action_plan
from .games import GameSpec
from .prompts import IDLE_MOVEMENT, PlayerPromptTrack, PromptVocabulary
from .hud_renderer import concat_mp4, draw_combat_hud_cell


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


class _ClosedLoopCore:
    def __init__(
        self,
        *,
        game: GameSpec,
        vocabulary: PromptVocabulary,
        player_track: PlayerPromptTrack,
        planner,
        attacknet,
        situationnet,
        world_model,
        output_dir: str | Path,
        cards: dict[str, dict[str, Any]],
        cooldowns,
        filler_track,
        state: CombatState | None = None,
        seed: int = 2,
        plan_max: int = 3,
    ):
        self.game = game
        self.vocabulary = vocabulary
        self.player_track = player_track
        self.planner = planner
        self.attacknet = attacknet
        self.situationnet = situationnet
        self.world_model = world_model
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cards = cards
        self.cooldowns = cooldowns
        self.filler_track = filler_track
        self.state = state or CombatState()
        self.player_max_hp = max(1, self.state.player_hp)
        self.boss_max_hp = max(1, self.state.boss_hp)
        # Hollow Knight gains five meter points for either side's landed hit.
        # Other games retain the original one-point rule.
        self.calculator = DamageCalculator()
        self.skill_gain_per_hit = 5 if self.game.key == "hk" else 1
        self.seed = int(seed)
        self.plan_max = int(plan_max)
        self.previous_skill: str | None = None
        self.previous_lead: dict[str, Any] | None = None
        self.committed_item: PlanItem | None = None
        self.committed_cells_remaining = 0

    def _prepare_output_dir(self) -> None:
        """Remove only artifacts owned by a previous run in this output dir."""
        for name in ("initial.json", "summary.json", "rollout.mp4"):
            path = self.output_dir / name
            if path.is_file() or path.is_symlink():
                path.unlink()
        for pattern in ("tick_*.json", "clip_*.mp4", "contact_*.jpg"):
            for path in self.output_dir.glob(pattern):
                if path.is_file() or path.is_symlink():
                    path.unlink()

    def _cooldown_snapshot(self, now_s: float) -> dict[str, Any]:
        return {} if self.cooldowns is None else self.cooldowns.snapshot(now_s)

    def _cell_prompt(self, *, player_clause: str, boss_item: PlanItem):
        if self.state.player_dead:
            player_clause = self.vocabulary.player_death_clause
        elif self.state.boss_dead:
            player_clause = self.vocabulary.player_idle_clause

        if self.state.boss_dead:
            requested_action, requested_movement = self.game.death_action, IDLE_MOVEMENT
        elif self.state.player_dead:
            requested_action, requested_movement = self.game.no_action, IDLE_MOVEMENT
        else:
            requested_action, requested_movement = boss_item.action, boss_item.movement

        prompt, repair = self.vocabulary.compose(player_clause, requested_action, requested_movement)
        player_action, player_movement = self.vocabulary.player_clause_to_tuple[player_clause]
        return prompt, {
            "player_clause": player_clause,
            "player_action": player_action,
            "player_movement": player_movement,
            "boss_requested": {"action": requested_action, "movement": requested_movement},
            "boss_emitted": {
                "action": repair["emitted_action"] if repair else requested_action,
                "movement": repair["emitted_movement"] if repair else requested_movement,
            },
            "prompt": prompt,
            "repair": repair,
        }

    def _tick_schedule(self, *, tick: int, situation: dict[str, Any]):
        if self.state.player_dead or self.state.boss_dead:
            item = automatic_basic(self.vocabulary.basic_action, situation)
            return [item] * 20, [], {"source": "death_override", "error": None, "raw": None}

        if self.game.decision_mode == "plan":
            plan, planner_meta = self.planner.decide_plan(
                situation=situation,
                combat=self.state.as_dict(),
                previous_lead=self.previous_lead,
            )
            self.previous_lead = plan[0].as_dict() if plan else None
            cells, events = expand_action_plan(
                plan, self.game.action_cards(self.vocabulary.boss_name), cells=20
            )
            return cells, events, planner_meta

        now_s = tick * 5.0
        cells: list[PlanItem] = []
        events: list[dict[str, Any]] = []
        if self.committed_item is not None and self.committed_cells_remaining > 0:
            count = min(20, self.committed_cells_remaining)
            cells.extend(self.filler_track.next_special(count))
            events.append({
                "action": self.committed_item.action,
                "movement": "data_backed_sequence",
                "onset_cell": 0,
                "cells": count,
                "continued_from_previous_tick": True,
            })
            self.committed_cells_remaining -= count
            if self.committed_cells_remaining == 0:
                self.filler_track.end_special()
                self.committed_item = None
            planner_meta = {"source": "release_commitment", "error": None, "raw": None}
        elif not self.state.skill_ready:
            planner_meta = {
                "source": "automatic_melee" if self.game.key == "nrftw" else "automatic_basic",
                "error": f"boss skill {self.state.boss_skill}/{self.state.boss_skill_max}",
                "raw": None,
            }
        else:
            legal = self.cooldowns.menu(now_s)
            scheduler_state = self.cooldowns.snapshot(now_s)
            item, planner_meta = self.planner.decide(
                situation=situation,
                combat=self.state.as_dict(),
                legal_skills=legal,
                previous_skill=self.previous_skill,
            )
            planner_meta["cooldown_scheduler"] = {
                "time_s": now_s,
                "legal_skills": legal,
                "state": scheduler_state,
                "selected": item.action if item.action in self.cards else None,
            }
            if item.action in self.cards:
                if not self.cooldowns.ready(item.action, now_s):
                    raise AssertionError("planner selected a cooldown-illegal skill")
                self.state.spend_skill()
                self.cooldowns.commit(item.action, now_s)
                self.previous_skill = item.action
                total_cells = max(
                    1,
                    math.ceil(float(self.cards[item.action]["duration_s"]) / CELL_SECONDS - 1e-9),
                )
                count = min(20, total_cells)
                self.filler_track.begin_special(item)
                cells.extend(self.filler_track.next_special(count))
                self.committed_cells_remaining = total_cells - count
                self.committed_item = item if self.committed_cells_remaining else None
                if self.committed_item is None:
                    self.filler_track.end_special()
                events.append({
                    "action": item.action,
                    "movement": "data_backed_sequence",
                    "onset_cell": 0,
                    "cells": count,
                    "release_duration_s": float(self.cards[item.action]["duration_s"]),
                    "continued_cells": self.committed_cells_remaining,
                    "skill_cost": self.state.boss_skill_max,
                })
        if len(cells) < 20:
            cells.extend(self.filler_track.next(20 - len(cells)))
        return cells, events, planner_meta

    def _apply_cell(self, signal: dict[str, Any], info: dict[str, Any]):
        # NRFTW keeps its original attack-prompt gate untouched. Vampire and
        # Hollow Knight bosses spend most cells on non-"attack"-classified
        # skills (repositioning, wind-ups, etc.), so gating damage on that
        # classification left AttackNet's correctly-detected hits discarded
        # most of the time -- e.g. boss taking ~10x more hits than the player
        # across real rollouts. For those two games a detected hit lands on
        # its own, matching AttackNet's own precision/recall instead of an
        # extra classifier on top of it.
        if self.game.key == "nrftw":
            player_attack_prompt = info["player_action"] == self.vocabulary.player_melee_action
            boss_attack_prompt = info["boss_emitted"]["action"] in self.vocabulary.boss_attack_actions
        else:
            player_attack_prompt = True
            boss_attack_prompt = True
        return self.calculator.apply_cell(
            self.state,
            player_hit=bool(signal["player_hit"]),
            boss_hit=bool(signal["boss_hit"]),
            player_attack_prompt=player_attack_prompt,
            boss_attack_prompt=boss_attack_prompt,
            skill_gain_per_hit=self.skill_gain_per_hit,
        )

    def _initial_record(self, situation, warm_attack, world_model_runtime=None):
        return {
            "game": self.game.key,
            "characters": {"player": self.vocabulary.player_name, "boss": self.vocabulary.boss_name},
            "combat": self.state.as_dict(),
            "attacknet_warm_start_not_applied": warm_attack,
            "situation": situation,
            "visual_input": "initial RGB frame copied three times",
            "world_model_runtime": world_model_runtime,
        }

    def _hud(self, frames, event, info, **kwargs):
        return draw_combat_hud_cell(
            frames,
            event["after"],
            player_max_hp=self.player_max_hp,
            boss_max_hp=self.boss_max_hp,
            player_action=info["player_action"],
            player_movement=info["player_movement"],
            boss_name=self.vocabulary.boss_name,
            boss_action=info["boss_emitted"]["action"],
            boss_movement=info["boss_emitted"]["movement"],
            **kwargs,
        )


class ClosedLoopAgent(_ClosedLoopCore):
    """Bidirectional, 80-frame-per-tick implementation of the shared pipeline."""

    def run(self, initial_image: str | Path | Image.Image, ticks: int) -> dict[str, Any]:
        if ticks <= 0:
            raise ValueError("ticks must be positive")
        self._prepare_output_dir()
        image = initial_image.convert("RGB") if isinstance(initial_image, Image.Image) else Image.open(Path(initial_image).expanduser()).convert("RGB")
        warm_attack = self.attacknet.initial(image)
        situation = self.situationnet.initial(image)
        _atomic_json(self.output_dir / "initial.json", self._initial_record(situation, warm_attack))

        clip_paths: list[Path] = []
        record_names: list[str] = []
        outcome: str | None = None
        for tick in range(ticks):
            state_before = self.state.as_dict()
            cooldown_before = self._cooldown_snapshot(tick * 5.0)
            boss_cells, plan_events, planner_meta = self._tick_schedule(tick=tick, situation=situation)
            player_clauses, wrapped = self.player_track.chunk(tick)
            prompts: list[str] = []
            prompt_cells: list[dict[str, Any]] = []
            for cell in range(20):
                prompt, info = self._cell_prompt(player_clause=player_clauses[cell], boss_item=boss_cells[cell])
                prompts.append(prompt)
                prompt_cells.append({"cell": cell, **info})
            frames = self.world_model.generate(image, prompts, self.seed + tick)
            if len(frames) != 80:
                raise AssertionError(f"world model returned {len(frames)} frames")

            raw_frames: list[Image.Image] = []
            hud_frames: list[Image.Image] = []
            hit_signals = []
            health_timeline = []
            for cell in range(20):
                cell_frames = frames[4 * cell : 4 * cell + 4]
                signal = self.attacknet.predict_cell(cell_frames)
                event = self._apply_cell(signal, prompt_cells[cell])
                event.update({"cell": cell, "attacknet": signal})
                raw_frames.extend(cell_frames)
                hud_frames.extend(self._hud(cell_frames, event, prompt_cells[cell]))
                hit_signals.append(signal)
                health_timeline.append(event)
            next_situation = situation if self.state.player_dead or self.state.boss_dead else self.situationnet.clip(raw_frames)
            clip_path = self.output_dir / f"clip_{tick:04d}.mp4"
            save_mp4(hud_frames, clip_path, fps=16)
            record_name = f"tick_{tick:04d}.json"
            _atomic_json(self.output_dir / record_name, {
                "game": self.game.key,
                "tick": tick,
                "characters": {"player": self.vocabulary.player_name, "boss": self.vocabulary.boss_name},
                "clip": str(clip_path),
                "world_model_checkpoint": str(getattr(self.world_model, "checkpoint", "unknown")),
                "combat_before": state_before,
                "combat_after": self.state.as_dict(),
                "situation_input": situation,
                "situation_next": next_situation,
                "cooldowns_before": cooldown_before,
                "cooldowns_after": self._cooldown_snapshot((tick + 1) * 5.0),
                "plan_events": plan_events,
                "planner": planner_meta,
                "player_track_wrapped": wrapped,
                "prompt_cells": prompt_cells,
                "attacknet_cells": hit_signals,
                "health_timeline": health_timeline,
            })
            clip_paths.append(clip_path)
            record_names.append(record_name)
            image = raw_frames[-1].convert("RGB")
            situation = next_situation
            if self.state.player_dead or self.state.boss_dead:
                outcome = "You Lose" if self.state.player_dead else "You Win"
                break

        combined_video = self.output_dir / "rollout.mp4"
        concat_mp4(clip_paths, combined_video)
        completed_ticks = len(clip_paths)
        summary = {
            "game": self.game.key,
            "ticks": completed_ticks,
            "requested_ticks": ticks,
            "duration_s": completed_ticks * 5.0,
            "outcome": outcome,
            "termination_reason": "combat_finished" if outcome is not None else "tick_budget",
            "characters": {"player": self.vocabulary.player_name, "boss": self.vocabulary.boss_name},
            "final_combat": self.state.as_dict(),
            "world_model_checkpoint": str(getattr(self.world_model, "checkpoint", "unknown")),
            "combined_video": str(combined_video),
            "records": record_names,
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary


class ClosedLoopAgentAR(_ClosedLoopCore):
    """Causal four-frame-per-cell implementation of the same pipeline."""

    def __init__(
        self,
        *,
        outcome_ui: bool = True,
        outcome_fade_seconds: float = 2.0,
        post_death_seconds: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.outcome_ui = bool(outcome_ui)
        self.outcome_fade_seconds = float(outcome_fade_seconds)
        self.post_death_seconds = float(post_death_seconds)
        if self.outcome_fade_seconds < 0 or self.post_death_seconds < 0:
            raise ValueError("outcome/post-death seconds must be non-negative")

    def run(self, initial_image: str | Path | Image.Image, ticks: int) -> dict[str, Any]:
        if ticks <= 0:
            raise ValueError("ticks must be positive")
        self._prepare_output_dir()
        image = initial_image.convert("RGB") if isinstance(initial_image, Image.Image) else Image.open(Path(initial_image).expanduser()).convert("RGB")
        reset = getattr(self.world_model, "reset_rollout", None)
        if callable(reset):
            reset()
        runtime_fn = getattr(self.world_model, "runtime_config", None)
        runtime = runtime_fn() if callable(runtime_fn) else None
        post_death_cells = max(
            0, math.ceil(self.post_death_seconds / CELL_SECONDS - 1e-9)
        )
        max_cells = ticks * 20
        generated_cells = 0
        outcome: str | None = None
        outcome_frame_count = 0
        death_detected_cell: int | None = None
        stop_after: int | None = None
        stopped_after_death = False
        clip_paths: list[Path] = []
        record_names: list[str] = []
        warm_attack = self.attacknet.initial(image)
        situation = self.situationnet.initial(image)
        initial = self._initial_record(situation, warm_attack, runtime)
        initial["outcome_ui"] = {"enabled": self.outcome_ui, "fade_seconds": self.outcome_fade_seconds}
        initial["termination_policy"] = {
            "mode": "post_death_window",
            "post_death_seconds": self.post_death_seconds,
            "post_death_cells": post_death_cells,
            "max_cells": max_cells,
        }
        _atomic_json(self.output_dir / "initial.json", initial)

        tick = 0
        # The requested tick budget is an absolute duration ceiling. Even an
        # explicit diagnostic post-death window may not extend the rollout
        # beyond it (12 ticks / 60 seconds for the packaged examples).
        while tick < ticks:
            stop_rollout = False
            state_before = self.state.as_dict()
            cooldown_before = self._cooldown_snapshot(tick * 5.0)
            boss_cells, plan_events, planner_meta = self._tick_schedule(tick=tick, situation=situation)
            player_clauses, wrapped = self.player_track.chunk(tick)
            self.world_model.open_tick(image, self.seed + tick)
            raw_frames: list[Image.Image] = []
            hud_frames: list[Image.Image] = []
            prompt_cells = []
            hit_signals = []
            health_timeline = []
            for cell in range(20):
                prompt, info = self._cell_prompt(player_clause=player_clauses[cell], boss_item=boss_cells[cell])
                cell_frames = self.world_model.step_cell(prompt)
                if len(cell_frames) != 4:
                    raise AssertionError(f"AR world model returned {len(cell_frames)} frames")
                signal = self.attacknet.predict_cell(cell_frames)
                dead_before = self.state.player_dead or self.state.boss_dead
                event = self._apply_cell(signal, info)
                if not dead_before and (self.state.player_dead or self.state.boss_dead):
                    outcome = "You Lose" if self.state.player_dead else "You Win"
                    death_detected_cell = generated_cells
                    stop_after = min(
                        max_cells, generated_cells + 1 + post_death_cells
                    )
                event.update({"cell": cell, "global_cell": generated_cells, "attacknet": signal, "outcome": outcome})
                raw_frames.extend(cell_frames)
                hud_frames.extend(self._hud(
                    cell_frames, event, info,
                    outcome=outcome if self.outcome_ui else None,
                    outcome_frame_start=outcome_frame_count,
                    outcome_fade_seconds=self.outcome_fade_seconds,
                ))
                prompt_cells.append({"cell": cell, **info})
                hit_signals.append(signal)
                health_timeline.append(event)
                generated_cells += 1
                if outcome is not None:
                    outcome_frame_count += len(cell_frames)
                if stop_after is not None and generated_cells >= stop_after:
                    stop_rollout = True
                    stopped_after_death = True
                    break
            next_situation = situation if self.state.player_dead or self.state.boss_dead or len(raw_frames) != 80 else self.situationnet.clip(raw_frames)
            clip_path = self.output_dir / f"clip_{tick:04d}.mp4"
            save_mp4(hud_frames, clip_path, fps=16)
            tick_end_s = tick * 5.0 + len(hit_signals) * CELL_SECONDS
            record_name = f"tick_{tick:04d}.json"
            _atomic_json(self.output_dir / record_name, {
                "game": self.game.key,
                "tick": tick,
                "cells": len(hit_signals),
                "duration_s": len(hit_signals) * CELL_SECONDS,
                "partial_tick": len(hit_signals) != 20,
                "characters": {"player": self.vocabulary.player_name, "boss": self.vocabulary.boss_name},
                "clip": str(clip_path),
                "world_model_checkpoint": str(getattr(self.world_model, "checkpoint", "unknown")),
                "world_model_runtime": runtime,
                "combat_before": state_before,
                "combat_after": self.state.as_dict(),
                "situation_input": situation,
                "situation_next": next_situation,
                "cooldowns_before": cooldown_before,
                "cooldowns_after": self._cooldown_snapshot(tick_end_s),
                "plan_events": plan_events,
                "planner": planner_meta,
                "player_track_wrapped": wrapped,
                "prompt_cells": prompt_cells,
                "attacknet_cells": hit_signals,
                "health_timeline": health_timeline,
                "outcome": outcome,
                "death_detected_cell": death_detected_cell,
            })
            clip_paths.append(clip_path)
            record_names.append(record_name)
            image = raw_frames[-1].convert("RGB")
            situation = next_situation
            if stop_rollout:
                break
            tick += 1

        combined_video = self.output_dir / "rollout.mp4"
        concat_mp4(clip_paths, combined_video)
        summary = {
            "game": self.game.key,
            "ticks": len(record_names),
            "requested_ticks": ticks,
            "generated_cells": generated_cells,
            "duration_s": generated_cells * CELL_SECONDS,
            "characters": {"player": self.vocabulary.player_name, "boss": self.vocabulary.boss_name},
            "final_combat": self.state.as_dict(),
            "world_model_checkpoint": str(getattr(self.world_model, "checkpoint", "unknown")),
            "world_model_runtime": runtime,
            "outcome": outcome,
            "death_detected_cell": death_detected_cell,
            "post_death_generated_s": None if death_detected_cell is None else max(0, generated_cells - death_detected_cell - 1) * CELL_SECONDS,
            "stopped_after_death": stopped_after_death,
            "termination_reason": (
                "post_death_window" if stopped_after_death else "tick_budget"
            ),
            "combined_video": str(combined_video),
            "records": record_names,
        }
        _atomic_json(self.output_dir / "summary.json", summary)
        return summary


__all__ = ["ClosedLoopAgent", "ClosedLoopAgentAR"]
