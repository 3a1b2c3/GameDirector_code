"""Shared decision layer with game-specific action cards and behavior data."""

from __future__ import annotations

import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .games import GameSpec, MOVEMENTS
from .skills import DISTANCE_UNIT, skill_cards


CELL_SECONDS = 0.25
COOLDOWN_LOG_CAP_SECONDS = 60.0


@dataclass(frozen=True)
class PlanItem:
    action: str
    movement: str
    reason: str | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _confidence(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


class _HTTPPlanner:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 90.0,
        retries: int = 2,
        allow_fallback: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.allow_fallback = bool(allow_fallback)

    def _call(self, payload: dict[str, Any], max_tokens: int = 256) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON decision module for an action-game boss. "
                            "Output JSON only and select only listed actions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            }
        ).encode()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                return str(result["choices"][0]["message"]["content"])
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error


class CooldownTracker:
    def __init__(self, cards: Mapping[str, Mapping[str, Any]]):
        self.cards = dict(cards)
        self.last_use: dict[str, float] = {}
        self.log: list[tuple[float, str]] = []

    def _seconds_since_uncapped(self, skill: str, now_s: float) -> float | None:
        if skill not in self.last_use:
            return None
        return round(max(0.0, float(now_s) - self.last_use[skill]), 2)

    def seconds_since(self, skill: str, now_s: float) -> float:
        elapsed = self._seconds_since_uncapped(skill, now_s)
        return COOLDOWN_LOG_CAP_SECONDS if elapsed is None else min(elapsed, COOLDOWN_LOG_CAP_SECONDS)

    def ready(self, skill: str, now_s: float) -> bool:
        elapsed = self._seconds_since_uncapped(skill, now_s)
        return elapsed is None or elapsed >= float(self.cards[skill]["cooldown_s"])

    def menu(self, now_s: float) -> list[str]:
        return [skill for skill in self.cards if self.ready(skill, now_s)]

    def snapshot(self, now_s: float) -> dict[str, dict[str, object]]:
        return {
            skill: {
                "seconds_since_last_use": self.seconds_since(skill, now_s),
                "cooldown_s": float(card["cooldown_s"]),
                "ready": self.ready(skill, now_s),
            }
            for skill, card in self.cards.items()
        }

    def commit(self, skill: str, now_s: float) -> None:
        if skill not in self.cards or not self.ready(skill, now_s):
            raise ValueError(f"skill is unknown or on cooldown: {skill}")
        self.last_use[skill] = float(now_s)
        self.log.append((float(now_s), skill))


def automatic_basic(basic_action: str, situation: dict[str, Any]) -> PlanItem:
    del situation
    label = "melee" if basic_action == "melee_attack" else "basic attack"
    return PlanItem(
        basic_action,
        "none",
        f"automatic {label}: skill bar is not full or no special is cooldown-ready",
        1.0,
    )


def _percentile_cap(lengths: Iterable[int], percentile: float = 0.99) -> int:
    values = sorted(int(value) for value in lengths if value > 0)
    if not values:
        return 1
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


@dataclass
class BehaviorProfile:
    action_segment_counts: dict[str, dict[str, int]]
    action_segment_len_counts: dict[str, dict[str, dict[int, int]]]
    movement_segment_len_counts: dict[str, dict[str, dict[int, int]]]
    movement_segment_value_counts: dict[str, dict[str, dict[str, int]]]
    movement_cell_counts: dict[str, dict[str, dict[str, int]]]
    initial_movement_counts: dict[str, dict[str, dict[str, int]]]
    max_stationary_cells: dict[str, int]
    max_basic_action_cells: dict[str, dict[str, int]]
    action_cell_counts: dict[str, dict[str, int]]

    @classmethod
    def from_timelines(
        cls,
        boss_name: str,
        timelines: Mapping[str, list[dict[str, str]]],
        *,
        basic_actions: tuple[str, ...],
        special_actions: tuple[str, ...],
    ) -> "BehaviorProfile":
        all_actions = set(basic_actions) | set(special_actions)
        action_segments: dict[str, Counter[int]] = {action: Counter() for action in basic_actions}
        movement_lengths: dict[str, Counter[int]] = {action: Counter() for action in all_actions}
        movement_values: dict[str, Counter[str]] = {action: Counter() for action in all_actions}
        movement_cells: dict[str, Counter[str]] = {action: Counter() for action in all_actions}
        initial_movements: dict[str, Counter[str]] = {action: Counter() for action in all_actions}
        action_cells: Counter[str] = Counter()
        basic_run_lengths: dict[str, list[int]] = {action: [] for action in basic_actions}
        stationary_lengths: list[int] = []

        for rows in timelines.values():
            rows = [row for row in rows if row["action"] in all_actions]
            stationary = 0
            for row in rows:
                if row["movement"] == "none":
                    stationary += 1
                elif stationary:
                    stationary_lengths.append(stationary)
                    stationary = 0
            if stationary:
                stationary_lengths.append(stationary)

            start = 0
            while start < len(rows):
                action = rows[start]["action"]
                end = start + 1
                while end < len(rows) and rows[end]["action"] == action:
                    end += 1
                segment = rows[start:end]
                initial_movements[action][segment[0]["movement"]] += 1
                if action in action_segments:
                    action_segments[action][len(segment)] += 1
                    basic_run_lengths[action].append(len(segment))
                    action_cells[action] += len(segment)
                movement_start = 0
                while movement_start < len(segment):
                    movement = segment[movement_start]["movement"]
                    movement_end = movement_start + 1
                    while (
                        movement_end < len(segment)
                        and segment[movement_end]["movement"] == movement
                    ):
                        movement_end += 1
                    movement_values[action][movement] += 1
                    movement_lengths[action][movement_end - movement_start] += 1
                    movement_cells[action][movement] += movement_end - movement_start
                    movement_start = movement_end
                start = end

        for action in all_actions:
            if not movement_values[action]:
                raise ValueError(f"no dataset behavior for {boss_name}/{action}")
        return cls(
            action_segment_counts={boss_name: {a: sum(c.values()) for a, c in action_segments.items()}},
            action_segment_len_counts={boss_name: {a: dict(c) for a, c in action_segments.items()}},
            movement_segment_len_counts={boss_name: {a: dict(c) for a, c in movement_lengths.items()}},
            movement_segment_value_counts={boss_name: {a: dict(c) for a, c in movement_values.items()}},
            movement_cell_counts={boss_name: {a: dict(c) for a, c in movement_cells.items()}},
            initial_movement_counts={boss_name: {a: dict(c) for a, c in initial_movements.items()}},
            max_stationary_cells={boss_name: _percentile_cap(stationary_lengths)},
            max_basic_action_cells={
                boss_name: {a: _percentile_cap(lengths) for a, lengths in basic_run_lengths.items()}
            },
            action_cell_counts={boss_name: dict(action_cells)},
        )


class DatasetFillerTrack:
    """The NRFTW segment sampler, parameterized by another game's corpus."""

    ACTION_SHARE_TOLERANCE = 0.01
    MOVEMENT_SHARE_TOLERANCE = 0.005

    def __init__(self, boss_name: str, rng: random.Random, profile: BehaviorProfile):
        self.boss_name = boss_name
        self.rng = rng
        self.p = profile
        self._action: str | None = None
        self._action_remaining = 0
        self._last_emitted_action: str | None = None
        self._action_run_cells = 0
        self._basic_cells_emitted = 0
        self._basic_action_cells = {a: 0 for a in profile.action_cell_counts[boss_name]}
        self._movement: str | None = None
        self._movement_remaining = 0
        self._stationary_cells = 0
        self._movement_cells_emitted = {a: 0 for a in profile.movement_cell_counts[boss_name]}
        self._movement_cell_counts = {
            a: {movement: 0 for movement in counts}
            for a, counts in profile.movement_cell_counts[boss_name].items()
        }
        self._special_item: PlanItem | None = None
        self._special_movement: str | None = None
        self._special_movement_remaining = 0

    def _weighted_choice(self, weights: Mapping[Any, int], exclude: set[Any] | None = None) -> Any:
        keys = [key for key in weights if not exclude or key not in exclude]
        if not keys:
            raise ValueError("weighted choice has no candidates after exclusions")
        return self.rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]

    def _draw_movement(self, action: str, previous: str | None, *, initial: bool) -> tuple[str, int]:
        value_counts = self.p.movement_segment_value_counts[self.boss_name][action]
        length_counts = self.p.movement_segment_len_counts[self.boss_name][action]
        selection_counts = self.p.initial_movement_counts[self.boss_name][action] if initial else value_counts
        excluded: set[str] = set()
        if previous is not None:
            target_counts = self.p.movement_cell_counts[self.boss_name][action]
            emitted = self._movement_cells_emitted[action]
            observed_share = self._movement_cell_counts[action][previous] / emitted if emitted else 0.0
            target_share = target_counts[previous] / sum(target_counts.values())
            if not (previous == "none" and observed_share < target_share):
                excluded.add(previous)
        stationary_cap = self.p.max_stationary_cells[self.boss_name]
        if self._stationary_cells >= stationary_cap:
            excluded.add("none")
        emitted = self._movement_cells_emitted[action]
        if emitted >= 20:
            target_counts = self.p.movement_cell_counts[self.boss_name][action]
            target_total = sum(target_counts.values())
            share_excluded = {
                movement
                for movement, observed in self._movement_cell_counts[action].items()
                if observed / emitted > target_counts[movement] / target_total + self.MOVEMENT_SHARE_TOLERANCE
            } & set(selection_counts)
            if len(excluded | share_excluded) < len(selection_counts):
                excluded |= share_excluded
        if len(excluded) >= len(selection_counts):
            # Small/new game corpora may expose only one movement for an
            # action. Reusing it is preferable to inventing an unseen tuple.
            excluded = {"none"} if "none" in excluded and len(selection_counts) > 1 else set()
        movement = self._weighted_choice(selection_counts, excluded)
        raw_length = int(self._weighted_choice(length_counts))
        pooled_mean = sum(k * v for k, v in length_counts.items()) / sum(length_counts.values())
        target_mean = self.p.movement_cell_counts[self.boss_name][action][movement] / value_counts[movement]
        scaled = max(1.0, raw_length * target_mean / pooled_mean)
        length = math.floor(scaled)
        if self.rng.random() < scaled - length:
            length += 1
        if movement == "none":
            length = min(length, stationary_cap - self._stationary_cells)
        return movement, max(1, length)

    def _record_movement(self, action: str, movement: str) -> None:
        self._stationary_cells = self._stationary_cells + 1 if movement == "none" else 0
        self._movement_cells_emitted[action] += 1
        self._movement_cell_counts[action][movement] += 1

    def next(self, count: int) -> list[PlanItem]:
        items: list[PlanItem] = []
        for _ in range(count):
            new_action = False
            if self._action_remaining <= 0:
                hard_excluded: set[str] = set()
                if self._last_emitted_action is not None:
                    cap = self.p.max_basic_action_cells[self.boss_name][self._last_emitted_action]
                    if self._action_run_cells >= cap:
                        hard_excluded.add(self._last_emitted_action)
                share_excluded: set[str] = set()
                if self._basic_cells_emitted >= 20:
                    targets = self.p.action_cell_counts[self.boss_name]
                    total = sum(targets.values())
                    share_excluded = {
                        action for action, observed in self._basic_action_cells.items()
                        if observed / self._basic_cells_emitted > targets[action] / total + self.ACTION_SHARE_TOLERANCE
                    }
                excluded = hard_excluded | share_excluded
                counts = self.p.action_segment_counts[self.boss_name]
                if len(excluded) == len(counts):
                    excluded = hard_excluded
                self._action = self._weighted_choice(counts, excluded)
                sampled = int(self._weighted_choice(
                    self.p.action_segment_len_counts[self.boss_name][self._action]
                ))
                cap = self.p.max_basic_action_cells[self.boss_name][self._action]
                held = self._action_run_cells if self._action == self._last_emitted_action else 0
                self._action_remaining = max(1, min(sampled, max(1, cap - held)))
                new_action = True
                self._movement_remaining = 0
            if self._movement == "none" and self._stationary_cells >= self.p.max_stationary_cells[self.boss_name]:
                self._movement_remaining = 0
            if self._movement_remaining <= 0:
                self._movement, self._movement_remaining = self._draw_movement(
                    self._action, None if new_action else self._movement, initial=new_action
                )
            item = PlanItem(
                self._action, self._movement,
                "data-derived filler: sampled from selected game's segment statistics", None,
            )
            items.append(item)
            self._record_movement(item.action, item.movement)
            if item.action == self._last_emitted_action:
                self._action_run_cells += 1
            else:
                self._last_emitted_action = item.action
                self._action_run_cells = 1
            self._basic_cells_emitted += 1
            self._basic_action_cells[item.action] += 1
            self._action_remaining -= 1
            self._movement_remaining -= 1
        return items

    def begin_special(self, item: PlanItem) -> None:
        if item.action not in self.p.movement_segment_value_counts[self.boss_name]:
            raise ValueError(f"no data-backed special movement profile for {item.action!r}")
        self._action = None
        self._action_remaining = 0
        self._last_emitted_action = None
        self._action_run_cells = 0
        self._movement = None
        self._movement_remaining = 0
        self._special_item = item
        self._special_movement = None
        self._special_movement_remaining = 0

    def next_special(self, count: int) -> list[PlanItem]:
        if self._special_item is None:
            raise ValueError("begin_special must be called before next_special")
        items: list[PlanItem] = []
        for _ in range(count):
            if self._special_movement == "none" and self._stationary_cells >= self.p.max_stationary_cells[self.boss_name]:
                self._special_movement_remaining = 0
            if self._special_movement_remaining <= 0:
                self._special_movement, self._special_movement_remaining = self._draw_movement(
                    self._special_item.action, self._special_movement,
                    initial=self._special_movement is None,
                )
            item = PlanItem(
                self._special_item.action, self._special_movement,
                self._special_item.reason, self._special_item.confidence,
            )
            items.append(item)
            self._record_movement(item.action, item.movement)
            self._special_movement_remaining -= 1
        return items

    def end_special(self) -> None:
        self._special_item = None
        self._special_movement = None
        self._special_movement_remaining = 0


class SpecialBossPlanner(_HTTPPlanner):
    def __init__(self, *, game: GameSpec, boss_name: str, cards: dict[str, dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.game = game
        self.boss_name = boss_name
        self.cards = cards

    def decide(self, *, situation, combat, legal_skills, previous_skill):
        if not legal_skills:
            return automatic_basic(self.game.basic_action(self.boss_name), situation), {
                "source": "cooldown_scheduler", "error": "empty legal menu", "raw": None,
            }
        library = []
        for skill in legal_skills:
            card = self.cards[skill]
            library.append({
                "skill": skill,
                "description": card["phrase"],
                "function": card["function"],
                "effect_on_distance": card["effect_on_distance"],
                "role": card["role"],
                "release_duration_s": card["duration_s"],
                "effective_range": card["range_profile"],
                "skill_cost": card["cost"],
            })
        payload = {
            "task": "Choose exactly one available special skill from live geometry; movement is controlled locally.",
            "game": self.game.key,
            "boss": self.boss_name,
            "situation": situation,
            "distance_semantics": {
                "situation_fields": ["distance_value", "dp_value"],
                "skill_range_field": "effective_range.best_band",
                "unit": self.game.distance_unit,
            },
            "combat_state": combat,
            "previous_skill": previous_skill,
            "available_special_skill_library": library,
            "output_format": {"skill": "one listed skill", "reason": "short reason", "confidence": "0..1"},
        }
        request = payload
        raw_attempts: list[str] = []
        skill = None
        parsed: dict[str, Any] = {}
        for attempt in range(2):
            try:
                raw = self._call(request)
                raw_attempts.append(raw)
                parsed = _parse_json(raw) or {}
            except Exception as error:
                if not self.allow_fallback:
                    raise RuntimeError(
                        "Gemma special-skill planning failed; fallback is disabled"
                    ) from error
                return self.fallback(situation, legal_skills), {
                    "source": "fallback", "error": f"{type(error).__name__}: {error}",
                    "raw": raw_attempts[-1] if raw_attempts else None,
                    "raw_attempts": raw_attempts,
                    "request": request,
                }
            skill = parsed.get("skill")
            if skill in legal_skills:
                break
            request = {
                **payload,
                "validation_feedback": {
                    "attempt": attempt + 1,
                    "invalid_response": raw,
                    "error": f"skill must be exactly one of {list(legal_skills)}",
                    "instruction": "Return one JSON object only; do not return null or invent a skill.",
                },
            }
        if skill not in legal_skills:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Gemma returned off-menu special skill {skill!r} after one strict "
                    f"re-prompt; fallback is disabled; raw_attempts={raw_attempts!r}"
                )
            return self.fallback(situation, legal_skills), {
                "source": "fallback", "error": "LLM returned an off-menu skill",
                "raw": raw_attempts[-1], "raw_attempts": raw_attempts,
                "request": request,
            }
        return PlanItem(
            str(skill), "none", str(parsed.get("reason") or "")[:240] or None,
            _confidence(parsed.get("confidence")),
        ), {
            "source": "llm", "error": None, "raw": raw_attempts[-1],
            "raw_attempts": raw_attempts, "request": request,
        }

    def fallback(self, situation: dict[str, Any], legal_skills: list[str]) -> PlanItem:
        distance = float(situation.get("dp_value", situation.get("distance_value", 0.0)) or 0.0)
        def fit(skill: str) -> tuple[float, int]:
            low, high = self.cards[skill]["range_profile"]["best_band"]
            return max(float(low) - distance, 0.0, distance - float(high)), list(self.cards).index(skill)
        return PlanItem(min(legal_skills, key=fit), "none", "fallback: closest range fit", 0.3)


class ActionPlanner(_HTTPPlanner):
    """VRising's real situation-conditioned action/movement planner."""

    def __init__(self, *, game: GameSpec, boss_name: str, cards: dict[str, dict[str, Any]], max_plan: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.game = game
        self.boss_name = boss_name
        self.cards = cards
        self.max_plan = int(max_plan)

    def decide_plan(self, *, situation, combat, previous_lead):
        payload = {
            "task": f"Plan the next 1 to {self.max_plan} moves in order from the live situation.",
            "game": self.game.key,
            "boss": self.boss_name,
            "situation": situation,
            "combat_state": combat,
            "previous_lead": previous_lead,
            "action_library": [{"action": action, **card} for action, card in self.cards.items()],
            "allowed_movements": list(MOVEMENTS),
            "output_format": {"plan": [{"action": "listed action", "movement": "listed movement", "reason": "short reason", "confidence": "0..1"}]},
        }
        try:
            raw = self._call(payload, max_tokens=384)
            rows = (_parse_json(raw) or {}).get("plan", [])
        except Exception as error:
            if not self.allow_fallback:
                raise RuntimeError("Gemma action planning failed; fallback is disabled") from error
            return self.fallback(situation), {
                "source": "fallback", "error": f"{type(error).__name__}: {error}",
                "raw": None, "request": payload,
            }
        plan = [
            PlanItem(
                str(row["action"]), str(row["movement"]),
                str(row.get("reason") or "")[:200] or None,
                _confidence(row.get("confidence")),
            )
            for row in rows
            if isinstance(row, dict)
            and row.get("action") in self.cards
            and row.get("movement") in MOVEMENTS
        ][: self.max_plan]
        if not plan:
            if not self.allow_fallback:
                raise RuntimeError("Gemma returned no valid action plan; fallback is disabled")
            return self.fallback(situation), {
                "source": "fallback", "error": "unparseable or off-menu plan",
                "raw": raw, "request": payload,
            }
        return plan, {"source": "llm", "error": None, "raw": raw, "request": payload}

    @staticmethod
    def fallback(situation: dict[str, Any]) -> list[PlanItem]:
        distance = float(situation.get("dp_value", situation.get("distance_value", 0.0)) or 0.0)
        if distance <= 4.0:
            return [
                PlanItem("melee_attack", "none", "fallback: player is close", 0.3),
                PlanItem("no_action", "down", "fallback: create spacing", 0.3),
                PlanItem("melee_attack", "up", "fallback: re-engage", 0.3),
            ]
        return [
            PlanItem("no_action", "up", "fallback: close distance", 0.3),
            PlanItem("melee_attack", "up", "fallback: attack after approach", 0.3),
            PlanItem("no_action", "none", "fallback: recover", 0.3),
        ]


def expand_action_plan(plan: list[PlanItem], cards: Mapping[str, Mapping[str, Any]], cells: int = 20):
    expanded: list[PlanItem] = []
    events: list[dict[str, Any]] = []
    for item in plan[:3]:
        if len(expanded) >= cells:
            break
        count = min(
            cells - len(expanded),
            max(1, math.ceil(float(cards[item.action]["duration_s"]) / CELL_SECONDS - 1e-9)),
        )
        onset = len(expanded)
        expanded.extend([item] * count)
        events.append({"action": item.action, "movement": item.movement, "onset_cell": onset, "cells": count})
    movement = plan[-1].movement if plan else "none"
    expanded.extend([PlanItem("no_action", movement, "local filler", None)] * (cells - len(expanded)))
    return expanded, events


# ---------------------------------------------------------------------------
# Library-driven boss planner (per-boss skill cards + corpus segment statistics)
# ---------------------------------------------------------------------------

class BossCooldownTracker:
    """Local legal-action scheduler independent of the LLM.

    Cooldown legality never depends on the LLM.  The tracker owns onset time,
    derives the legal menu, and validates every commit.  The serialized
    elapsed time is capped at 60 seconds while readiness uses the uncapped
    elapsed time; a never-used skill is immediately legal.
    """

    def __init__(self, boss_name: str, *, cooldown_overrides: dict[str, float] | None = None, skill_cost: int = 20):
        self.cards = skill_cards(
            boss_name, cooldown_overrides=cooldown_overrides, skill_cost=skill_cost
        )
        self.last_use: dict[str, float] = {}
        self.log: list[tuple[float, str]] = []

    def _seconds_since_uncapped(self, skill: str, now_s: float) -> float | None:
        if skill not in self.last_use:
            return None
        return round(max(0.0, float(now_s) - self.last_use[skill]), 2)

    def seconds_since(self, skill: str, now_s: float) -> float:
        elapsed = self._seconds_since_uncapped(skill, now_s)
        return COOLDOWN_LOG_CAP_SECONDS if elapsed is None else min(
            elapsed, COOLDOWN_LOG_CAP_SECONDS
        )

    def ready(self, skill: str, now_s: float) -> bool:
        elapsed = self._seconds_since_uncapped(skill, now_s)
        return elapsed is None or elapsed >= float(self.cards[skill]["cooldown_s"])

    def menu(self, now_s: float) -> list[str]:
        """Return the complete cooldown-legal skill menu in library order."""
        return [skill for skill in self.cards if self.ready(skill, now_s)]

    def snapshot(self, now_s: float) -> dict[str, dict[str, object]]:
        return {
            skill: {
                "seconds_since_last_use": self.seconds_since(skill, now_s),
                "cooldown_s": float(card["cooldown_s"]),
                "ready": self.ready(skill, now_s),
            }
            for skill, card in self.cards.items()
        }

    def commit(self, skill: str, now_s: float) -> None:
        if skill not in self.cards:
            raise ValueError(f"not a special skill: {skill}")
        if self.log and float(now_s) < self.log[-1][0] - 1e-9:
            raise ValueError(f"cooldown commit time moved backwards: {now_s} < {self.log[-1][0]}")
        if not self.ready(skill, now_s):
            raise ValueError(f"skill is on cooldown: {skill}")
        self.last_use[skill] = float(now_s)
        self.log.append((float(now_s), skill))


def automatic_melee(situation: dict[str, Any]) -> PlanItem:
    """No-LLM behavior while the 20-point bar is not full.

    The boss is always mid-attack whenever this is called (melee filler or
    the movement basis for a committed special), so movement is always
    "none": the boss stays stationary for the whole duration of its attack.
    """
    return PlanItem(
        "melee_attack",
        "none",
        "automatic melee: skill bar is not full or no special is cooldown-ready",
        1.0,
    )


# Boss-specific *segment* statistics (not per-cell frequencies) for
# non-special-skill, non-death, non-post-death stretches of the NRFTW
# training corpus (25,029 clips / 449,904 pre-death cells, stitched across a
# fight's consecutive clips so a run is never artificially cut at a 5s clip
# boundary). A "segment" is a maximal run of consecutive cells sharing the
# same value.
# Drives FillerTrack, which reproduces both the real frequency AND the real
# duration of each behavior -- not just an independent per-cell draw -- so a
# non-attacking-with-a-special boss holds a believable, non-flickering
# action/movement for a while, the way real gameplay does.
#
# FILLER_ACTION_SEGMENT_COUNTS[boss][action] = how many action segments of
#   that type occur (used to draw which action a *new* segment is).
# FILLER_ACTION_SEGMENT_LEN_COUNTS[boss][action][length_in_cells] = how many
#   of those segments ran for exactly that many 0.25s cells.
# FILLER_MOVEMENT_SEGMENT_LEN_COUNTS[boss][action][length_in_cells] = same,
#   but for movement runs *nested inside* an action segment of that type
#   (movement can change several times within one action segment).
# FILLER_MOVEMENT_SEGMENT_VALUE_COUNTS[boss][action][movement] = how many
#   movement segments (not cells) of that value occur inside an action
#   segment of that type (used to draw which movement a *new* run is).
FILLER_ACTION_SEGMENT_COUNTS: dict[str, dict[str, int]] = {
    "Golden Bird": {"melee_attack": 1589, "no_action": 3247},
    "Mutant Soldier": {"melee_attack": 3269, "no_action": 2963},
    "Plague Leader": {"melee_attack": 2291, "no_action": 2392},
}

FILLER_ACTION_SEGMENT_LEN_COUNTS: dict[str, dict[str, dict[int, int]]] = {
    "Golden Bird": {
        "melee_attack": {1: 12, 2: 18, 3: 19, 4: 12, 5: 9, 6: 9, 7: 11, 8: 24, 9: 24, 10: 4, 11: 6, 12: 17, 13: 15, 14: 26, 15: 26, 16: 29, 17: 65, 18: 23, 19: 80, 20: 491, 21: 83, 22: 74, 23: 28, 24: 15, 25: 12, 26: 59, 27: 17, 28: 56, 29: 208, 30: 55, 31: 24, 32: 7, 33: 13, 34: 4, 35: 2, 42: 2, 43: 1, 50: 2, 51: 2, 52: 3, 53: 1, 55: 1},
        "no_action": {1: 286, 2: 35, 3: 68, 4: 263, 5: 304, 6: 97, 7: 14, 8: 4, 9: 28, 10: 33, 11: 7, 12: 1093, 13: 300, 14: 83, 15: 68, 16: 40, 17: 257, 18: 98, 19: 44, 20: 51, 21: 34, 22: 9, 23: 5, 24: 7, 25: 3, 26: 5, 27: 1, 28: 1, 30: 1, 31: 2, 33: 1, 40: 1, 41: 1, 57: 1, 64: 1, 69: 1},
    },
    "Mutant Soldier": {
        "melee_attack": {1: 24, 2: 27, 3: 20, 4: 30, 5: 11, 6: 7, 7: 9, 8: 14, 9: 29, 10: 117, 11: 186, 12: 309, 13: 608, 14: 106, 15: 50, 16: 50, 17: 29, 18: 21, 19: 119, 20: 345, 21: 82, 22: 51, 23: 14, 24: 4, 25: 7, 26: 6, 27: 7, 28: 12, 29: 12, 30: 19, 31: 13, 32: 5, 33: 6, 34: 5, 35: 8, 36: 4, 37: 50, 38: 61, 39: 49, 40: 31, 41: 13, 42: 3, 43: 10, 44: 11, 45: 8, 46: 3, 47: 2, 48: 22, 49: 20, 50: 158, 51: 170, 52: 75, 53: 25, 54: 10, 55: 4, 56: 10, 57: 17, 58: 24, 59: 12, 60: 11, 61: 33, 62: 26, 63: 12, 64: 6, 65: 2, 66: 2, 68: 3, 69: 10, 70: 6, 71: 1, 74: 1, 86: 1, 112: 1},
        "no_action": {1: 404, 2: 521, 3: 308, 4: 92, 5: 135, 6: 355, 7: 228, 8: 353, 9: 250, 10: 151, 11: 61, 12: 19, 13: 24, 14: 13, 15: 4, 16: 3, 17: 7, 18: 3, 19: 2, 20: 4, 21: 1, 22: 4, 23: 4, 24: 3, 25: 2, 26: 4, 27: 3, 28: 1, 29: 3, 32: 1},
    },
    "Plague Leader": {
        "melee_attack": {1: 23, 2: 29, 3: 29, 4: 12, 5: 12, 6: 17, 7: 23, 8: 355, 9: 964, 10: 94, 11: 36, 12: 4, 13: 3, 14: 13, 15: 17, 16: 8, 17: 9, 18: 9, 19: 2, 20: 1, 21: 2, 22: 11, 23: 356, 24: 187, 25: 53, 26: 11, 27: 3, 28: 2, 31: 1, 32: 1, 33: 1, 35: 2, 37: 1},
        "no_action": {1: 1070, 2: 279, 3: 134, 4: 156, 5: 254, 6: 235, 7: 173, 8: 31, 9: 7, 10: 6, 11: 2, 12: 5, 13: 19, 14: 7, 15: 4, 16: 4, 17: 2, 18: 1, 23: 1, 40: 1, 45: 1},
    },
}

FILLER_MOVEMENT_SEGMENT_LEN_COUNTS: dict[str, dict[str, dict[int, int]]] = {
    "Golden Bird": {
        "melee_attack": {1: 9489, 2: 2316, 3: 1318, 4: 629, 5: 553, 6: 503, 7: 353, 8: 167, 9: 83, 10: 145, 11: 13, 12: 16, 13: 9, 14: 12, 15: 2, 16: 8, 17: 4, 18: 2, 19: 2, 20: 3, 22: 1, 24: 1, 29: 1},
        "no_action": {1: 3411, 2: 260, 3: 571, 4: 327, 5: 535, 6: 429, 7: 122, 8: 52, 9: 73, 10: 504, 11: 536, 12: 377, 13: 98, 14: 29, 15: 28, 16: 12, 17: 43, 18: 6, 19: 11, 20: 19, 21: 13, 22: 3, 23: 2, 24: 2, 25: 6, 26: 1, 27: 1, 35: 1, 40: 1, 41: 1, 57: 1, 64: 1},
    },
    "Mutant Soldier": {
        "melee_attack": {1: 23617, 2: 9817, 3: 3934, 4: 1601, 5: 1666, 6: 549, 7: 419, 8: 188, 9: 77, 10: 46, 11: 22, 12: 19, 13: 13, 14: 7, 15: 6, 16: 2, 17: 3, 18: 2, 19: 3, 20: 4, 21: 4, 22: 1, 24: 1, 25: 1, 26: 1, 27: 1, 29: 1, 38: 1, 41: 1, 54: 1},
        "no_action": {1: 2505, 2: 1177, 3: 520, 4: 466, 5: 178, 6: 300, 7: 108, 8: 176, 9: 219, 10: 29, 11: 25, 12: 11, 13: 7, 14: 3, 16: 1, 18: 2, 20: 2, 21: 1, 22: 1, 23: 2, 24: 1, 27: 1},
    },
    "Plague Leader": {
        "melee_attack": {1: 6387, 2: 5592, 3: 2377, 4: 860, 5: 128, 6: 17, 7: 9, 8: 2, 10: 1, 11: 1, 12: 1, 14: 2, 15: 1, 16: 1, 17: 3, 18: 1, 19: 1, 20: 1, 21: 2, 22: 1, 24: 2, 25: 1, 27: 1},
        "no_action": {1: 1298, 2: 208, 3: 132, 4: 155, 5: 253, 6: 231, 7: 172, 8: 31, 9: 8, 10: 5, 11: 2, 12: 14, 13: 13, 14: 4, 15: 4, 16: 3, 17: 3, 22: 1, 31: 1, 45: 1},
    },
}

FILLER_MOVEMENT_SEGMENT_VALUE_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "Golden Bird": {
        "melee_attack": {"none": 4777, "up": 1679, "down_right": 1671, "up_left": 1588, "down": 1441, "down_left": 1242, "up_right": 1231, "right": 1189, "left": 812},
        "no_action": {"none": 4377, "up_right": 868, "up": 690, "up_left": 397, "left": 293, "down_right": 229, "down_left": 220, "right": 203, "down": 199},
    },
    "Mutant Soldier": {
        "melee_attack": {"none": 19107, "up_right": 3606, "down_left": 3364, "down": 2842, "up": 2731, "left": 2674, "up_left": 2599, "right": 2547, "down_right": 2538},
        "no_action": {"none": 2535, "up_right": 868, "up": 386, "down": 381, "down_left": 376, "right": 349, "left": 288, "down_right": 287, "up_left": 265},
    },
    "Plague Leader": {
        "melee_attack": {"none": 5812, "left": 1499, "down_left": 1410, "down": 1272, "up_left": 1223, "down_right": 1097, "up": 1081, "up_right": 1071, "right": 927},
        "no_action": {"none": 2086, "down": 81, "left": 73, "up_left": 58, "down_right": 52, "up_right": 51, "down_left": 49, "up": 48, "right": 41},
    },
}


# Special selection remains the LLM's only behavioral decision.  Once local
# code commits that skill, its movement comes from the same training-corpus
# segment process as ordinary behavior.  These are segment counts and pooled
# segment-length histograms mined with the same fight stitching/filtering as
# the filler tables above.
SPECIAL_MOVEMENT_SEGMENT_VALUE_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "Golden Bird": {
        "hammer_smash": {"up_left": 1968, "none": 10540, "down_right": 1919, "up": 1789, "up_right": 2145, "down": 1656, "left": 1716, "right": 1523, "down_left": 1954},
        "jump_slam": {"none": 4792, "down": 574, "left": 536, "up": 500, "right": 549, "up_left": 570, "up_right": 587, "down_left": 499, "down_right": 473},
        "spin_attack": {"none": 9123, "down_right": 1322, "up_left": 883, "down": 1307, "up": 752, "left": 1383, "down_left": 2067, "up_right": 1130, "right": 733},
    },
    "Mutant Soldier": {
        "back_spin": {"none": 6543, "up_right": 1147, "down_left": 420, "up_left": 723, "right": 688, "left": 499, "up": 929, "down": 403, "down_right": 485},
        "front_spin": {"none": 4633, "down_left": 562, "up_right": 692, "up": 716, "down": 541, "right": 459, "up_left": 565, "left": 526, "down_right": 449},
        "running_slash": {"right": 98, "none": 651, "up_right": 136, "down_left": 132, "up_left": 112, "up": 139, "down_right": 101, "left": 94, "down": 133},
    },
    "Plague Leader": {
        "left_spin": {"none": 5646, "down": 406, "up_right": 255, "up_left": 789, "right": 344, "down_left": 455, "up": 408, "left": 472, "down_right": 348},
        "right_spin": {"down_right": 786, "none": 4509, "up_left": 777, "up": 767, "down_left": 734, "down": 707, "left": 778, "up_right": 731, "right": 729},
        "shield_charge": {"none": 615, "down_left": 70, "up_left": 86, "up": 91, "right": 71, "down": 75, "up_right": 94, "down_right": 86, "left": 66},
    },
}

SPECIAL_MOVEMENT_SEGMENT_LEN_COUNTS: dict[str, dict[str, dict[int, int]]] = {
    "Golden Bird": {
        "hammer_smash": {1: 13074, 2: 6377, 3: 2358, 4: 910, 5: 655, 6: 195, 7: 229, 8: 605, 9: 282, 10: 129, 11: 72, 12: 65, 13: 61, 14: 22, 15: 22, 16: 29, 17: 28, 18: 9, 19: 11, 20: 11, 21: 11, 22: 9, 23: 5, 24: 2, 25: 6, 26: 6, 27: 4, 28: 16, 29: 5, 30: 1, 62: 1},
        "jump_slam": {1: 4798, 2: 1075, 3: 1252, 4: 656, 5: 461, 6: 90, 7: 113, 8: 207, 9: 135, 10: 76, 11: 52, 12: 38, 13: 31, 14: 51, 15: 14, 16: 13, 18: 10, 19: 4, 20: 1, 21: 1, 27: 1, 55: 1},
        "spin_attack": {1: 6356, 2: 5244, 3: 4305, 4: 986, 5: 164, 6: 231, 7: 58, 8: 490, 9: 69, 10: 74, 11: 111, 12: 431, 13: 56, 14: 13, 15: 13, 16: 14, 17: 7, 18: 4, 19: 4, 20: 12, 21: 8, 22: 2, 23: 4, 24: 14, 25: 3, 26: 1, 27: 3, 28: 3, 29: 5, 30: 2, 31: 3, 32: 2, 36: 4, 37: 1, 38: 1, 39: 1, 40: 1},
    },
    "Mutant Soldier": {
        "back_spin": {1: 4090, 2: 3400, 3: 1773, 4: 1579, 5: 848, 6: 60, 7: 7, 8: 34, 9: 12, 10: 2, 11: 2, 12: 4, 13: 2, 14: 16, 15: 6, 16: 1, 23: 1},
        "front_spin": {1: 4034, 2: 2126, 3: 1519, 4: 831, 5: 282, 6: 271, 7: 42, 8: 12, 9: 2, 10: 6, 11: 5, 12: 8, 13: 2, 16: 1, 17: 1, 36: 1},
        "running_slash": {1: 493, 2: 415, 3: 364, 4: 281, 5: 39, 7: 1, 9: 1, 17: 1, 26: 1},
    },
    "Plague Leader": {
        "left_spin": {1: 3524, 2: 679, 3: 1044, 4: 1633, 5: 89, 6: 44, 7: 1236, 8: 678, 9: 77, 10: 22, 11: 11, 12: 41, 13: 30, 14: 6, 15: 3, 17: 1, 18: 1, 22: 1, 24: 1, 51: 1, 52: 1},
        "right_spin": {1: 4857, 2: 2722, 3: 1720, 4: 548, 5: 67, 6: 34, 7: 19, 8: 301, 9: 205, 10: 25, 11: 11, 12: 1, 13: 3, 14: 1, 16: 1, 27: 1, 35: 1, 68: 1},
        "shield_charge": {1: 328, 2: 273, 3: 96, 4: 104, 5: 238, 6: 151, 7: 12, 8: 12, 9: 8, 10: 9, 11: 5, 12: 2, 13: 6, 14: 7, 15: 2, 16: 1},
    },
}

# 99th-percentile uninterrupted stationary runs (all pre-death Boss actions,
# stitched across real fight clips).  This is a data-derived guardrail against
# rare sampler concatenations making the generated Boss look stuck.
MAX_CONTINUOUS_STATIONARY_CELLS = {
    "Golden Bird": 21,      # 5.25 s
    "Mutant Soldier": 13,  # 3.25 s
    "Plague Leader": 15,   # 3.75 s
}

MAX_CONTINUOUS_BASIC_ACTION_CELLS = {
    "Golden Bird": {"melee_attack": 34, "no_action": 22},
    "Mutant Soldier": {"melee_attack": 63, "no_action": 18},
    "Plague Leader": {"melee_attack": 25, "no_action": 13},
}

# Exact pre-death filler-cell totals in the same training corpus.  A small
# feedback constraint keeps finite rollouts close to these Boss-specific
# attack/idle occupancies while segment values and lengths remain stochastic.
FILLER_ACTION_CELL_COUNTS = {
    "Golden Bird": {"melee_attack": 32285, "no_action": 33719},
    "Mutant Soldier": {"melee_attack": 78613, "no_action": 15930},
    "Plague Leader": {"melee_attack": 28401, "no_action": 7542},
}
ACTION_SHARE_TOLERANCE = 0.01

BOSS_MOVEMENT_CELL_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "Golden Bird": {
        "melee_attack": {"up": 2891, "up_left": 2039, "down_left": 1598, "down_right": 4541, "none": 13538, "up_right": 1999, "right": 2252, "left": 1162, "down": 2265},
        "no_action": {"none": 30419, "up_right": 902, "left": 318, "down_left": 232, "right": 217, "down": 200, "up": 751, "up_left": 437, "down_right": 243},
        "hammer_smash": {"up_left": 4030, "none": 30132, "down_right": 3937, "up": 3849, "up_right": 3953, "down": 3258, "left": 3533, "right": 2989, "down_left": 3879},
        "jump_slam": {"none": 15403, "down": 1393, "down_left": 1168, "left": 1116, "right": 1046, "down_right": 1035, "up_left": 1029, "up": 1015, "up_right": 956},
        "spin_attack": {"none": 35004, "down_left": 3774, "left": 2860, "down_right": 2398, "down": 2375, "up_right": 1717, "up_left": 1481, "up": 1348, "right": 1240},
    },
    "Mutant Soldier": {
        "melee_attack": {"up_right": 5025, "none": 43892, "down_left": 5464, "down": 4516, "up": 3846, "down_right": 3834, "up_left": 4008, "right": 3718, "left": 4310},
        "no_action": {"none": 10000, "up_right": 1136, "down_left": 903, "right": 617, "up_left": 526, "up": 653, "left": 657, "down_right": 596, "down": 842},
        "back_spin": {"none": 16781, "up_right": 2494, "up": 1997, "up_left": 1592, "right": 1440, "left": 1082, "down_right": 1020, "down_left": 917, "down": 860},
        "front_spin": {"none": 11655, "up": 1447, "up_right": 1314, "up_left": 1160, "down_left": 1082, "down": 1029, "left": 1024, "right": 888, "down_right": 850},
        "running_slash": {"none": 1310, "down": 382, "up": 374, "down_left": 354, "up_right": 351, "up_left": 331, "right": 269, "down_right": 261, "left": 239},
    },
    "Plague Leader": {
        "melee_attack": {"up_left": 2238, "left": 2845, "none": 11060, "right": 1654, "down_left": 2471, "up_right": 1902, "down_right": 1971, "down": 2397, "up": 1863},
        "no_action": {"none": 7064, "down": 101, "up_left": 60, "left": 73, "up_right": 49, "down_left": 49, "right": 41, "down_right": 52, "up": 53},
        "left_spin": {"none": 28260, "up_left": 869, "left": 532, "down_left": 522, "down": 486, "up": 454, "down_right": 413, "right": 392, "up_right": 284},
        "right_spin": {"none": 15765, "up_left": 1082, "left": 1081, "down_right": 1078, "up": 1035, "right": 1010, "up_right": 998, "down_left": 995, "down": 986},
        "shield_charge": {"none": 2658, "up": 260, "up_right": 252, "up_left": 234, "down_right": 218, "right": 197, "down_left": 196, "down": 195, "left": 169},
    },
}
MOVEMENT_SHARE_TOLERANCE = 0.005

BOSS_ACTION_INITIAL_MOVEMENT_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "Golden Bird": {
        "melee_attack": {"up": 409, "none": 407, "up_right": 316, "up_left": 142, "right": 58, "left": 44, "down_left": 44, "down_right": 48, "down": 50},
        "no_action": {"none": 2326, "left": 101, "up_right": 118, "down": 60, "up": 153, "down_left": 84, "up_left": 134, "right": 90, "down_right": 94},
        "hammer_smash": {"up_left": 244, "up": 247, "right": 192, "down_right": 198, "left": 201, "none": 433, "up_right": 326, "down": 198, "down_left": 197},
        "jump_slam": {"none": 1078, "up_left": 55, "right": 38, "down_right": 25, "down": 28, "left": 37, "up_right": 42, "down_left": 28, "up": 33},
        "spin_attack": {"none": 1509, "right": 2, "up": 1, "up_left": 1},
    },
    "Mutant Soldier": {
        "melee_attack": {"up_right": 503, "none": 976, "down": 232, "down_right": 191, "down_left": 204, "left": 231, "up": 305, "up_left": 260, "right": 310},
        "no_action": {"none": 1352, "down_left": 259, "right": 167, "up_right": 159, "up_left": 181, "up": 160, "left": 195, "down_right": 187, "down": 237},
        "back_spin": {"none": 1667, "left": 24, "up_left": 39, "right": 31, "down_left": 17, "up": 51, "up_right": 40, "down": 19, "down_right": 20},
        "front_spin": {"none": 1142, "right": 11, "up_right": 15, "up": 18, "down_right": 17, "left": 21, "up_left": 9, "down_left": 19, "down": 12},
        "running_slash": {"right": 63, "up_right": 85, "up_left": 79, "up": 97, "down_right": 69, "down": 94, "down_left": 90, "left": 55},
    },
    "Plague Leader": {
        "melee_attack": {"up_left": 361, "up_right": 178, "down_right": 206, "down": 301, "right": 212, "left": 442, "down_left": 274, "up": 178, "none": 65},
        "no_action": {"none": 2040, "up_left": 39, "up_right": 36, "left": 50, "right": 34, "up": 36, "down": 47, "down_right": 37, "down_left": 23},
        "left_spin": {"none": 2896, "up_left": 1},
        "right_spin": {"down_right": 292, "up_left": 291, "up": 297, "left": 252, "down": 226, "none": 18, "right": 271, "up_right": 288, "down_left": 261},
        "shield_charge": {"none": 204},
    },
}


class FillerTrack:
    """Stateful, seeded sampler for Boss action and movement.

    Basic action/movement and LLM-selected special movement reproduce the
    NRFTW training corpus's real *segment* structure instead of independent
    per-cell draws.  The LLM never selects movement or ordinary actions.

    One instance lives for the whole rollout (constructed once per agent);
    state persists across `next()` calls so a real multi-second action or
    movement run can span tick boundaries, exactly like it does in the
    training data, instead of re-rolling every 5 seconds. A committed special
    ends the current basic segment, receives its own data-derived movement
    segments, and the next basic stretch starts fresh after release.
    """

    def __init__(self, boss_name: str, rng: random.Random):
        if boss_name not in FILLER_ACTION_SEGMENT_COUNTS:
            raise ValueError(f"no filler statistics for boss: {boss_name}")
        self.boss_name = boss_name
        self.rng = rng
        self._action: str | None = None
        self._action_remaining = 0
        self._last_emitted_action: str | None = None
        self._action_run_cells = 0
        self._basic_cells_emitted = 0
        self._basic_action_cells = {action: 0 for action in FILLER_ACTION_CELL_COUNTS[boss_name]}
        self._movement: str | None = None
        self._movement_remaining = 0
        self._stationary_cells = 0
        self._movement_cells_emitted = {
            action: 0 for action in BOSS_MOVEMENT_CELL_COUNTS[boss_name]
        }
        self._movement_cell_counts = {
            action: {movement: 0 for movement in counts}
            for action, counts in BOSS_MOVEMENT_CELL_COUNTS[boss_name].items()
        }
        self._special_item: PlanItem | None = None
        self._special_movement: str | None = None
        self._special_movement_remaining = 0

    def _weighted_choice(self, weights: dict, *, exclude: set[Any] | None = None) -> Any:
        keys = [key for key in weights if not exclude or key not in exclude]
        if not keys:
            raise ValueError("weighted choice has no candidates after exclusions")
        return self.rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]

    def _draw_movement(
        self, action: str, previous: str | None, *, special: bool, initial: bool = False
    ) -> tuple[str, int]:
        if special:
            value_counts = SPECIAL_MOVEMENT_SEGMENT_VALUE_COUNTS[self.boss_name][action]
            length_counts = SPECIAL_MOVEMENT_SEGMENT_LEN_COUNTS[self.boss_name][action]
        else:
            value_counts = FILLER_MOVEMENT_SEGMENT_VALUE_COUNTS[self.boss_name][action]
            length_counts = FILLER_MOVEMENT_SEGMENT_LEN_COUNTS[self.boss_name][action]
        selection_counts = (
            BOSS_ACTION_INITIAL_MOVEMENT_COUNTS[self.boss_name][action]
            if initial
            else value_counts
        )
        excluded: set[str] = set()
        if previous is not None:
            # A still-underrepresented stationary run may renew until the
            # hard p99 cap. This preserves the corpus's high stationary share
            # for idle-heavy actions without permitting an unbounded hold.
            target_counts = BOSS_MOVEMENT_CELL_COUNTS[self.boss_name][action]
            target_total = sum(target_counts.values())
            emitted = self._movement_cells_emitted[action]
            observed_previous_share = (
                self._movement_cell_counts[action][previous] / emitted if emitted else 0.0
            )
            target_previous_share = target_counts[previous] / target_total
            may_renew_stationary = (
                previous == "none"
                and observed_previous_share < target_previous_share
            )
            if not may_renew_stationary:
                excluded.add(previous)
        stationary_cap = MAX_CONTINUOUS_STATIONARY_CELLS[self.boss_name]
        if self._stationary_cells >= stationary_cap:
            excluded.add("none")
        emitted = self._movement_cells_emitted[action]
        share_excluded: set[str] = set()
        if emitted >= 20:
            target_counts = BOSS_MOVEMENT_CELL_COUNTS[self.boss_name][action]
            target_total = sum(target_counts.values())
            for movement, observed_cells in self._movement_cell_counts[action].items():
                observed_share = observed_cells / emitted
                target_share = target_counts[movement] / target_total
                if observed_share > target_share + MOVEMENT_SHARE_TOLERANCE:
                    share_excluded.add(movement)
        share_excluded &= set(selection_counts)
        if len(excluded | share_excluded) < len(selection_counts):
            excluded |= share_excluded
        movement = self._weighted_choice(selection_counts, exclude=excluded)
        raw_length = int(self._weighted_choice(length_counts))
        pooled_mean = sum(
            segment_length * frequency
            for segment_length, frequency in length_counts.items()
        ) / sum(length_counts.values())
        target_cells = BOSS_MOVEMENT_CELL_COUNTS[self.boss_name][action][movement]
        target_mean = target_cells / value_counts[movement]
        scaled_length = max(1.0, raw_length * target_mean / pooled_mean)
        length = math.floor(scaled_length)
        if self.rng.random() < scaled_length - length:
            length += 1
        if movement == "none":
            length = min(length, stationary_cap - self._stationary_cells)
        return movement, max(1, length)

    def _record_movement(self, action: str, movement: str) -> None:
        self._stationary_cells = self._stationary_cells + 1 if movement == "none" else 0
        self._movement_cells_emitted[action] += 1
        self._movement_cell_counts[action][movement] += 1

    def next(self, count: int) -> list[PlanItem]:
        """Return exactly `count` more filler cells, continuing whatever
        action/movement segment is already in progress."""
        if count <= 0:
            raise ValueError("count must be positive")
        items: list[PlanItem] = []
        for _ in range(count):
            new_action = False
            if self._action_remaining <= 0:
                if self.boss_name == "Golden Bird":
                    # Golden Bird never idles while alive: every filler
                    # segment is melee_attack (specials still come from the
                    # skill scheduler, not FillerTrack). The cap/share
                    # exclusion machinery below exists to rotate between
                    # multiple candidate actions, which doesn't apply when
                    # there is only one allowed action -- skip straight to
                    # sampling a fresh melee_attack segment length. Other
                    # bosses keep their data-derived no_action share.
                    self._action = "melee_attack"
                else:
                    hard_excluded: set[str] = set()
                    if self._last_emitted_action is not None:
                        cap = MAX_CONTINUOUS_BASIC_ACTION_CELLS[self.boss_name][
                            self._last_emitted_action
                        ]
                        if self._action_run_cells >= cap:
                            hard_excluded.add(self._last_emitted_action)
                    share_excluded: set[str] = set()
                    if self._basic_cells_emitted >= 20:
                        target_counts = FILLER_ACTION_CELL_COUNTS[self.boss_name]
                        target_total = sum(target_counts.values())
                        for action, observed_cells in self._basic_action_cells.items():
                            observed_share = observed_cells / self._basic_cells_emitted
                            target_share = target_counts[action] / target_total
                            if observed_share > target_share + ACTION_SHARE_TOLERANCE:
                                share_excluded.add(action)
                    excluded = hard_excluded | share_excluded
                    if len(excluded) == len(FILLER_ACTION_SEGMENT_COUNTS[self.boss_name]):
                        excluded = hard_excluded
                    self._action = self._weighted_choice(
                        FILLER_ACTION_SEGMENT_COUNTS[self.boss_name], exclude=excluded
                    )
                sampled_length = int(self._weighted_choice(
                    FILLER_ACTION_SEGMENT_LEN_COUNTS[self.boss_name][self._action]
                ))
                cap = MAX_CONTINUOUS_BASIC_ACTION_CELLS[self.boss_name][self._action]
                already_held = self._action_run_cells if self._action == self._last_emitted_action else 0
                self._action_remaining = min(sampled_length, cap - already_held)
                if self.boss_name == "Golden Bird" and self._action_remaining <= 0:
                    # already_held caught up to the historical cap; ignore it
                    # for Golden Bird (there is no alternative action to
                    # rotate into) and just keep the sampled segment length.
                    self._action_remaining = sampled_length
                # Each sampled action segment gets the empirical action-onset
                # movement distribution, even when the same action is drawn
                # again; the global run caps prevent visual over-holds.
                new_action = True
                self._movement_remaining = 0
            if (
                self._movement == "none"
                and self._stationary_cells >= MAX_CONTINUOUS_STATIONARY_CELLS[self.boss_name]
            ):
                self._movement_remaining = 0
            if self._movement_remaining <= 0:
                self._movement, self._movement_remaining = self._draw_movement(
                    self._action,
                    None if new_action else self._movement,
                    special=False,
                    initial=new_action,
                )
            items.append(
                PlanItem(
                    self._action,
                    self._movement,
                    "data-derived filler: sampled from training segment frequency/duration stats",
                    None,
                )
            )
            self._record_movement(self._action, self._movement)
            if self._action == self._last_emitted_action:
                self._action_run_cells += 1
            else:
                self._last_emitted_action = self._action
                self._action_run_cells = 1
            self._basic_cells_emitted += 1
            self._basic_action_cells[self._action] += 1
            self._action_remaining -= 1
            self._movement_remaining -= 1
        return items

    def begin_special(self, item: PlanItem) -> None:
        """Commit a locally validated LLM-selected skill; movement stays local."""
        if item.action not in SPECIAL_MOVEMENT_SEGMENT_VALUE_COUNTS[self.boss_name]:
            raise ValueError(f"no data-backed special movement profile for {item.action!r}")
        # A special is a real action boundary, not a pause in an ordinary
        # melee/idle segment that should resume many seconds later.
        self._action = None
        self._action_remaining = 0
        self._last_emitted_action = None
        self._action_run_cells = 0
        self._movement = None
        self._movement_remaining = 0
        self._special_item = item
        self._special_movement = None
        self._special_movement_remaining = 0

    def next_special(self, count: int) -> list[PlanItem]:
        if count <= 0:
            raise ValueError("count must be positive")
        if self._special_item is None:
            raise ValueError("begin_special must be called before next_special")
        items: list[PlanItem] = []
        for _ in range(count):
            if (
                self._special_movement == "none"
                and self._stationary_cells >= MAX_CONTINUOUS_STATIONARY_CELLS[self.boss_name]
            ):
                self._special_movement_remaining = 0
            if self._special_movement_remaining <= 0:
                self._special_movement, self._special_movement_remaining = self._draw_movement(
                    self._special_item.action,
                    self._special_movement,
                    special=True,
                    initial=self._special_movement is None,
                )
            items.append(
                PlanItem(
                    self._special_item.action,
                    self._special_movement,
                    self._special_item.reason,
                    self._special_item.confidence,
                )
            )
            self._record_movement(self._special_item.action, self._special_movement)
            self._special_movement_remaining -= 1
        return items

    def end_special(self) -> None:
        self._special_item = None
        self._special_movement = None
        self._special_movement_remaining = 0


class BossPlanner:
    """One bounded LLM call at a 5-second boundary when the skill bar is full."""

    def __init__(
        self,
        *,
        boss_name: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 90.0,
        retries: int = 2,
        allow_fallback: bool = False,
        cooldown_overrides: dict[str, float] | None = None,
        skill_cost: int = 20,
    ):
        self.boss_name = boss_name
        self.cards = skill_cards(
            boss_name, cooldown_overrides=cooldown_overrides, skill_cost=skill_cost
        )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.allow_fallback = bool(allow_fallback)

    def _call(self, payload: dict[str, Any]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON decision module for a top-down action-game "
                            "boss. Output JSON only and only select a listed skill."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            }
        ).encode()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                return str(result["choices"][0]["message"]["content"])
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _payload(
        self,
        *,
        situation: dict[str, Any],
        combat: dict[str, int],
        legal_skills: list[str],
        previous_skill: str | None,
    ) -> dict[str, Any]:
        cards = []
        for skill in legal_skills:
            card = self.cards[skill]
            cards.append(
                {
                    "skill": skill,
                    "description": card["phrase"],
                    "function": card["function"],
                    "effect_on_distance": card["effect_on_distance"],
                    "role": card["role"],
                    "release_duration_s": card["duration_s"],
                    "effective_range": card["range_profile"],
                    "skill_cost": card["cost"],
                }
            )
        return {
            "task": (
                "Choose exactly one available special skill. Reason only from the live "
                "distance/orientation, release commitment, range and mechanics. Movement "
                "and action legality are controlled separately by local code."
            ),
            "boss": self.boss_name,
            "situation": situation,
            "distance_semantics": {
                "situation_fields": ["distance_value", "dp_value"],
                "skill_range_field": "effective_range.best_band",
                "unit": DISTANCE_UNIT,
                "rule": "Compare these numeric values directly; they use the same unit.",
            },
            "combat_state": combat,
            "previous_skill": previous_skill,
            "skill_resource_rule": {
                "current": combat["boss_skill"],
                "maximum": combat["boss_skill_max"],
                "cost_rule": "the bar must be 20/20 to release; every special costs exactly 20",
            },
            # This library is already masked by the local cooldown scheduler.
            # The LLM intentionally receives no cooldown values or rules.
            "available_special_skill_library": cards,
            "output_format": {
                "skill": "one available_special_skill_library skill",
                "reason": "one short sentence citing situation, range and mechanics",
                "confidence": "0..1",
            },
        }

    def decide(
        self,
        *,
        situation: dict[str, Any],
        combat: dict[str, int],
        legal_skills: list[str],
        previous_skill: str | None,
    ) -> tuple[PlanItem, dict[str, Any]]:
        unknown = sorted(set(legal_skills) - set(self.cards))
        if unknown:
            raise ValueError(f"local cooldown scheduler returned unknown skills: {unknown}")
        if not legal_skills:
            item = automatic_melee(situation)
            return item, {
                "source": "cooldown_scheduler",
                "error": "local cooldown scheduler returned an empty legal menu",
                "raw": None,
                "request": None,
            }
        payload = self._payload(
            situation=situation,
            combat=combat,
            legal_skills=legal_skills,
            previous_skill=previous_skill,
        )
        try:
            raw = self._call(payload)
            parsed = _parse_json(raw) or {}
        except Exception as error:
            if not self.allow_fallback:
                raise RuntimeError("Gemma special-skill planning failed; fallback is disabled") from error
            item = self.fallback(situation, legal_skills)
            return item, {
                "source": "fallback",
                "error": f"{type(error).__name__}: {error}",
                "raw": None,
                "request": payload,
            }
        skill = parsed.get("skill")
        if skill not in legal_skills:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Gemma returned off-menu special skill {skill!r}; fallback is disabled"
                )
            item = self.fallback(situation, legal_skills)
            return item, {
                "source": "fallback",
                "error": "LLM returned a skill outside the locally scheduled legal menu",
                "raw": raw,
                "request": payload,
            }
        return PlanItem(
            action=str(skill),
            # Movement is a local, data-backed sequence applied after selection.
            movement="none",
            reason=str(parsed.get("reason") or "")[:240] or None,
            confidence=_confidence(parsed.get("confidence")),
        ), {"source": "llm", "error": None, "raw": raw, "request": payload}

    def fallback(self, situation: dict[str, Any], legal_skills: list[str]) -> PlanItem:
        distance = float(situation.get("dp_value", situation.get("distance_value", 0.0)) or 0.0)

        def fit(skill: str) -> tuple[float, int]:
            low, high = self.cards[skill]["range_profile"]["best_band"]
            error = max(float(low) - distance, 0.0, distance - float(high))
            return error, list(self.cards).index(skill)

        skill = min(legal_skills, key=fit)
        # Movement is a local, data-backed sequence applied after selection.
        return PlanItem(skill, "none", "fallback: closest data-derived range fit", 0.3)


def make_runtime_decision_components(
    *,
    game: GameSpec,
    boss_name: str,
    timelines: Mapping[str, list[dict[str, str]]],
    seed: int,
    cooldown_overrides: dict[str, float],
    planner_kwargs: dict[str, Any],
    plan_max: int,
    skill_cost: int = 20,
):
    cards = game.skill_cards(boss_name, cooldown_overrides, skill_cost)
    if game.key == "nrftw":
        planner = BossPlanner(
            boss_name=boss_name,
            cooldown_overrides=cooldown_overrides,
            skill_cost=skill_cost,
            **planner_kwargs,
        )
        cooldowns = BossCooldownTracker(
            boss_name, cooldown_overrides=cooldown_overrides, skill_cost=skill_cost
        )
        filler = FillerTrack(boss_name, random.Random(seed))
        return cards, planner, cooldowns, filler
    if game.decision_mode == "plan":
        planner = ActionPlanner(
            game=game, boss_name=boss_name, cards=game.action_cards(boss_name),
            max_plan=plan_max, **planner_kwargs,
        )
        return {}, planner, None, None
    profile = BehaviorProfile.from_timelines(
        boss_name,
        timelines,
        basic_actions=(game.basic_action(boss_name), game.no_action),
        special_actions=tuple(cards),
    )
    planner = SpecialBossPlanner(game=game, boss_name=boss_name, cards=cards, **planner_kwargs)
    cooldowns = CooldownTracker(cards)
    filler = DatasetFillerTrack(boss_name, random.Random(seed), profile)
    return cards, planner, cooldowns, filler


__all__ = [
    "ActionPlanner",
    "BehaviorProfile",
    "CELL_SECONDS",
    "DatasetFillerTrack",
    "PlanItem",
    "SpecialBossPlanner",
    "automatic_basic",
    "expand_action_plan",
    "make_runtime_decision_components",
]
