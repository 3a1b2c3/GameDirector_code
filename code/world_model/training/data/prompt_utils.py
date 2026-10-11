"""Temporal prompt resolver shared by training and cache precompute.

Any change to the empty-string fallback logic MUST be mirrored everywhere or
cache hash keys silently drift and training-time T5 lookups go to the wrong
row. The fixed-prompt fallback string lives on the `GameProfile`.
"""
from __future__ import annotations

from .profiles import GameProfile


def resolve_temporal_prompts(
    row,
    profile: GameProfile,
    expected_count: int = 20,
) -> list[str]:
    """Return the 20 already-rendered VRising prompt strings for one clip."""
    raw = row.get("prompts", None)
    if raw is None:
        raw = [profile.fixed_prompt] * expected_count
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_count:
        raise ValueError(
            f"expected {expected_count} temporal prompts, got "
            f"{None if raw is None else len(raw)}"
        )
    prompts = [str(value).strip() for value in raw]
    return [value if value else profile.fixed_prompt for value in prompts]
