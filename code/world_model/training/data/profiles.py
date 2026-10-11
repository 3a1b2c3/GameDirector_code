"""VRising dataset profile used by the world-model training entry point."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameProfile:
    name: str
    description: str
    fixed_prompt: str


VRISING = GameProfile(
    name="vrising",
    description="V Rising: player-versus-boss 5-second clips",
    fixed_prompt="Player fights a boss in V Rising.",
)
