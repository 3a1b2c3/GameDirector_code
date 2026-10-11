"""Vampire dataset profile used by the world-model training entry point."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameProfile:
    name: str
    description: str
    fixed_prompt: str


VAMPIRE = GameProfile(
    name="vampire",
    description="Vampire: player-versus-boss 5-second clips",
    fixed_prompt="Player fights a boss in Vampire.",
)
