"""Game-specific data injected into the otherwise shared agent pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .skills import (
    BOSS_MOVEMENTS,
    SKILL_LIBRARY as NRFTW_SKILL_LIBRARY,
)


MOVEMENTS = tuple(BOSS_MOVEMENTS)


def _card(
    phrase: str,
    function: str,
    effect: str,
    role: str,
    duration_s: float,
    cooldown_s: float,
    best_band: tuple[float, float],
    distance_unit: str,
) -> dict[str, Any]:
    return {
        "phrase": phrase,
        "function": function,
        "effect_on_distance": effect,
        "role": role,
        "duration_s": duration_s,
        "cooldown_s": cooldown_s,
        "range_profile": {
            "class": role,
            "best_band": list(best_band),
            "unit": distance_unit,
        },
    }


NRFTW_DISTANCE = "NRFTW world-coordinate units (horizontal_distance)"
HK_DISTANCE = "Hollow Knight world-coordinate units (distance_2d)"
VAMPIRE_DISTANCE = "Vampire world-coordinate units (horizontal_distance)"


# Modal uninterrupted release length, conservative observed onset gap, and
# central observed distance band are mined from hk_5s_prompts. These are
# the Hollow Knight equivalents of NRFTW's data-derived skill cards.
HK_SKILL_LIBRARY: dict[str, dict[str, dict[str, Any]]] = {
    "Hornet": {
        "aerial_dash": _card(
            "an aerial dash through the target", "airborne gap closer",
            "closes a short gap", "short gap-close", 0.75, 1.75,
            (0.5, 3.6), HK_DISTANCE,
        ),
        "needle_throw": _card(
            "a thrown needle attack", "ranged line attack",
            "pressures from mid range", "mid ranged", 2.5, 4.25,
            (2.8, 5.1), HK_DISTANCE,
        ),
        "orb_toss": _card(
            "an orb toss around the target area", "projectile area pressure",
            "covers close-to-mid spacing", "mid projectile", 1.75, 2.5,
            (2.0, 4.1), HK_DISTANCE,
        ),
    },
    "Broken Vessel": {
        "dive_stab": _card(
            "a diving stab onto the target", "airborne gap closer",
            "dives onto a nearby target", "short gap-close", 1.75, 2.5,
            (0.1, 3.3), HK_DISTANCE,
        ),
        "overhead_slash": _card(
            "an overhead slash", "committed melee strike",
            "punishes at close-to-mid spacing", "mid punish", 2.5, 3.0,
            (2.7, 5.5), HK_DISTANCE,
        ),
    },
    "Gruz Mother": {
        "shoulder_charge": _card(
            "a fast shoulder charge", "grounded gap closer",
            "closes a medium gap", "gap-close", 1.75, 3.25,
            (0.4, 5.0), HK_DISTANCE,
        ),
    },
    "Dung Defender": {
        "ground_burst": _card(
            "a ground burst beneath the target", "area strike",
            "covers close-to-mid spacing", "area punish", 1.75, 4.75,
            (1.3, 5.2), HK_DISTANCE,
        ),
        "rolling_dive": _card(
            "a rolling dive through the arena", "mobile body attack",
            "crosses close-to-mid spacing", "mobile pressure", 3.0, 0.75,
            (1.3, 6.2), HK_DISTANCE,
        ),
    },
}


HK_BASIC_ACTIONS = {
    "Hornet": "dash_strike",
    "Broken Vessel": "dash_flurry",
    "Gruz Mother": "body_slam",
    "Dung Defender": "ball_throw",
}


VAMPIRE_ACTION_CARDS: dict[str, dict[str, Any]] = {
    "no_action": {
        "phrase": "hold, recover, or reposition without attacking",
        "function": "spacing and recovery",
        "effect_on_distance": "movement controls spacing",
        "role": "reposition",
        "duration_s": 1.0,
        "range_profile": {"class": "any", "best_band": [0.0, 20.0], "unit": VAMPIRE_DISTANCE},
    },
    "melee_attack": {
        "phrase": "a close-range melee attack",
        "function": "basic melee pressure",
        "effect_on_distance": "best when the player is close",
        "role": "close punish",
        "duration_s": 1.0,
        "range_profile": {"class": "close", "best_band": [0.0, 4.0], "unit": VAMPIRE_DISTANCE},
    },
}


@dataclass(frozen=True)
class GameSpec:
    key: str
    display_name: str
    players: Mapping[int, str]
    bosses: Mapping[int, str]
    prompt_root: Path
    decision_mode: str
    distance_unit: str
    skill_library: Mapping[str, Mapping[str, Mapping[str, Any]]]
    basic_actions: Mapping[str, str]
    no_action: str = "no_action"
    death_action: str = "dying"
    player_melee_action: str = "melee_attack"
    default_player_hp: int = 500
    default_boss_hp: int = 500
    default_player_attack: int = 40
    default_boss_attack: int = 20

    def skill_cards(
        self,
        boss_name: str,
        cooldown_overrides: Mapping[str, float] | None = None,
        skill_cost: int = 20,
    ) -> dict[str, dict[str, Any]]:
        try:
            cards = deepcopy(dict(self.skill_library[boss_name]))
        except KeyError as error:
            raise ValueError(f"unknown {self.display_name} boss: {boss_name}") from error
        overrides = dict(cooldown_overrides or {})
        unknown = sorted(set(overrides).difference(cards))
        if unknown:
            raise ValueError(f"cooldown overrides do not belong to {boss_name}: {unknown}")
        for skill, card in cards.items():
            card["cooldown_s"] = min(float(overrides.get(skill, card["cooldown_s"])), 20.0)
            card["cost"] = int(skill_cost)
            card["range_profile"]["unit"] = self.distance_unit
        return cards

    def basic_action(self, boss_name: str) -> str:
        try:
            return self.basic_actions[boss_name]
        except KeyError as error:
            raise ValueError(f"no basic action configured for {boss_name}") from error

    def canonical_action(self, action: str) -> str:
        if self.key == "vampire":
            if action == "death":
                return self.death_action
            if action == "melee_attack":
                return action
            if action in {"idle", "walk", "run"} or action.startswith("move_"):
                return self.no_action
        return action

    def action_cards(self, boss_name: str) -> dict[str, dict[str, Any]]:
        if self.decision_mode == "plan":
            return deepcopy(VAMPIRE_ACTION_CARDS)
        return self.skill_cards(boss_name)

    def attack_actions(self, boss_name: str) -> set[str]:
        actions = {self.basic_action(boss_name)}
        actions.update(self.skill_library[boss_name])
        return actions


GAMES: dict[str, GameSpec] = {
    "nrftw": GameSpec(
        key="nrftw",
        display_name="No Rest for the Wicked",
        players={1: "Grey Monk", 2: "Red Knight", 3: "Green Scout"},
        bosses={1: "Golden Bird", 2: "Mutant Soldier", 3: "Plague Leader"},
        prompt_root=Path(),
        decision_mode="special",
        distance_unit=NRFTW_DISTANCE,
        skill_library=NRFTW_SKILL_LIBRARY,
        basic_actions={
            "Golden Bird": "melee_attack",
            "Mutant Soldier": "melee_attack",
            "Plague Leader": "melee_attack",
        },
    ),
    "vampire": GameSpec(
        key="vampire",
        display_name="Vampire",
        players={1: "Vampire Lord"},
        bosses={1: "Harpy Queen"},
        prompt_root=Path(),
        decision_mode="plan",
        distance_unit=VAMPIRE_DISTANCE,
        skill_library={"Harpy Queen": {}},
        basic_actions={"Harpy Queen": "melee_attack"},
        default_player_hp=1000,
        default_boss_hp=1000,
        default_player_attack=20,
    ),
    "hk": GameSpec(
        key="hk",
        display_name="Hollow Knight",
        players={1: "the Knight"},
        bosses={
            1: "Hornet",
            2: "Broken Vessel",
            3: "Gruz Mother",
            4: "Dung Defender",
        },
        prompt_root=Path(),
        decision_mode="special",
        distance_unit=HK_DISTANCE,
        skill_library=HK_SKILL_LIBRARY,
        basic_actions=HK_BASIC_ACTIONS,
    ),
}


ALL_SKILL_NAMES = tuple(
    dict.fromkeys(
        skill
        for spec in GAMES.values()
        for cards in spec.skill_library.values()
        for skill in cards
    )
)


def get_game(name: str) -> GameSpec:
    try:
        return GAMES[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown game {name!r}; choose from {tuple(GAMES)}") from error


__all__ = ["ALL_SKILL_NAMES", "GAMES", "GameSpec", "MOVEMENTS", "get_game"]
