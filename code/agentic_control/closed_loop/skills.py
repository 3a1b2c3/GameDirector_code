"""Data-derived boss skill cards for the NRFTW game."""

from __future__ import annotations

from copy import deepcopy

BOSS_MOVEMENTS = (
    "none",
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)

DEFAULT_SKILL_COST = 20
MAX_SKILL_COOLDOWN_S = 20.0
DISTANCE_UNIT = "NRFTW world-coordinate units (horizontal_distance)"

# duration/cooldown/range figures are mined from the aligned 0.25-second NRFTW
# state process. cooldown_s is the minimum observed onset-to-onset re-fire gap,
# using a conservative cooldown convention.
SKILL_LIBRARY = {
    "Golden Bird": {
        "hammer_smash": {
            "phrase": "a heavy hammer smash that bursts at the target area",
            "function": "heavy area strike",
            "effect_on_distance": "best at close-to-mid spacing",
            "role": "committed punish with a broad impact",
            "duration_s": 7.0,
            "cooldown_s": 14.5,
            "range_profile": {"class": "mid", "best_band": [1.1, 5.5], "unit": DISTANCE_UNIT},
        },
        "jump_slam": {
            "phrase": "a jumping slam that lands on a nearby target",
            "function": "gap-closing slam",
            "effect_on_distance": "closes a short gap before impact",
            "role": "engage or punish a nearby player",
            "duration_s": 4.5,
            "cooldown_s": 18.25,
            "range_profile": {"class": "short", "best_band": [0.0, 3.0], "unit": DISTANCE_UNIT},
        },
        "spin_attack": {
            "phrase": "a sustained spinning area attack around the boss",
            "function": "spinning area attack",
            "effect_on_distance": "pressures a wide close-to-mid area",
            "role": "area denial and surrounding pressure",
            "duration_s": 9.25,
            "cooldown_s": 20.0,
            "observed_refire_gap_s": 33.5,
            "range_profile": {"class": "wide", "best_band": [2.5, 7.0], "unit": DISTANCE_UNIT},
        },
    },
    "Mutant Soldier": {
        "back_spin": {
            "phrase": "a backward spinning slash around the boss",
            "function": "rear spinning sweep",
            "effect_on_distance": "covers close targets, especially behind",
            "role": "rear-space punish",
            "duration_s": 3.75,
            "cooldown_s": 5.5,
            "range_profile": {"class": "short", "best_band": [1.0, 4.0], "unit": DISTANCE_UNIT},
        },
        "front_spin": {
            "phrase": "a forward spinning slash that advances through the target",
            "function": "forward spinning sweep",
            "effect_on_distance": "pressures mid-range targets in front",
            "role": "frontal approach and sweep",
            "duration_s": 4.25,
            "cooldown_s": 4.75,
            "range_profile": {"class": "mid", "best_band": [2.0, 7.0], "unit": DISTANCE_UNIT},
        },
        "running_slash": {
            "phrase": "a fast running slash that closes a long gap",
            "function": "long gap-closing slash",
            "effect_on_distance": "rushes toward a distant player",
            "role": "far engage",
            "duration_s": 1.5,
            "cooldown_s": 5.25,
            "range_profile": {"class": "far", "best_band": [5.5, 9.0], "unit": DISTANCE_UNIT},
        },
    },
    "Plague Leader": {
        "left_spin": {
            "phrase": "a leftward spinning strike around the boss",
            "function": "left-side spinning sweep",
            "effect_on_distance": "covers nearby targets on the left flank",
            "role": "left-flank punish",
            "duration_s": 3.0,
            "cooldown_s": 3.25,
            "range_profile": {"class": "short", "best_band": [1.0, 3.5], "unit": DISTANCE_UNIT},
        },
        "right_spin": {
            "phrase": "a rightward spinning strike around the boss",
            "function": "right-side spinning sweep",
            "effect_on_distance": "covers nearby targets on the right flank",
            "role": "right-flank punish",
            "duration_s": 3.0,
            "cooldown_s": 3.0,
            "range_profile": {"class": "short", "best_band": [1.0, 4.0], "unit": DISTANCE_UNIT},
        },
        "shield_charge": {
            "phrase": "a shield-first charge that drives into a distant target",
            "function": "armored gap closer",
            "effect_on_distance": "closes a medium or long gap",
            "role": "far engage with a committed charge",
            "duration_s": 5.5,
            "cooldown_s": 16.25,
            "range_profile": {"class": "far", "best_band": [3.5, 8.5], "unit": DISTANCE_UNIT},
        },
    },
}


# The 9 special skills across all 3 bosses (3 each), in SKILL_LIBRARY order.
ALL_SKILL_NAMES = tuple(
    skill for boss_cards in SKILL_LIBRARY.values() for skill in boss_cards
)


def _resolve_overrides(overrides: dict[str, object] | None, label: str) -> dict[str, object]:
    if not overrides:
        return {}
    unknown = sorted(set(overrides) - set(ALL_SKILL_NAMES))
    if unknown:
        raise ValueError(f"unknown skill(s) in {label} overrides: {unknown}")
    return dict(overrides)


def skill_cards(
    boss_name: str,
    *,
    cooldown_overrides: dict[str, float] | None = None,
    skill_cost: int = DEFAULT_SKILL_COST,
) -> dict[str, dict[str, object]]:
    """Build one Boss's 3 skill cards.

    A special consumes the configured full boss meter. A cooldown override only
    replaces that skill's data-mined cooldown; the others keep their defaults.
    """
    try:
        cards = deepcopy(SKILL_LIBRARY[boss_name])
    except KeyError as error:
        raise ValueError(f"unknown NRFTW boss: {boss_name}") from error
    cooldown_overrides = _resolve_overrides(cooldown_overrides, "cooldown")
    for skill, card in cards.items():
        if skill in cooldown_overrides:
            card["cooldown_s"] = float(cooldown_overrides[skill])
        else:
            card["cooldown_s"] = min(float(card["cooldown_s"]), MAX_SKILL_COOLDOWN_S)
        card["cost"] = int(skill_cost)
        card["range_profile"]["unit"] = DISTANCE_UNIT
    return cards
