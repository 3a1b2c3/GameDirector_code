"""One dataset-backed prompt vocabulary implementation for all three games."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .games import GameSpec, MOVEMENTS


PLAYER_IDLE_ACTION = "no_action"
PLAYER_DEATH_ACTION = "dying"
IDLE_MOVEMENT = "none"

_BACKGROUND_SUFFIX_RE = re.compile(r" in ([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)$")


def _row_background(prompt: str) -> str | None:
    match = _BACKGROUND_SUFFIX_RE.search(prompt)
    return match.group(1) if match else None


def detect_background(root: str | Path, session_id: str) -> str | None:
    root = Path(root).expanduser().resolve()
    for path in sorted(root.glob(f"*/{session_id}/**/clip_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("prompts", []):
            background = _row_background(str(row["prompt"]).strip())
            if background is not None:
                return background
    return None


class PromptVocabulary:
    """Load only exact strings observed for the selected game and characters.

    Action aliases are canonicalized while loading (not while composing), so
    VRising's locomotion labels become the exact dataset phrase ``no action``.
    The world model consequently never receives a synthetic or unseen skill.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        game: GameSpec,
        player_name: str,
        boss_name: str,
        background: str | None = None,
        strip_background: bool = False,
    ):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"prompt dataset directory not found: {self.root}")
        if player_name not in game.players.values():
            raise ValueError(f"unknown {game.display_name} player: {player_name}")
        if boss_name not in game.bosses.values():
            raise ValueError(f"unknown {game.display_name} boss: {boss_name}")
        if strip_background and background is not None:
            raise ValueError("strip_background and background are mutually exclusive")
        self.game = game
        self.game_name = game.key
        self.player_name = player_name
        self.boss_name = boss_name
        self.background = background
        self.strip_background = strip_background
        self.basic_action = game.basic_action(boss_name)
        self.no_action = game.no_action
        self.death_action = game.death_action
        self.player_melee_action = game.player_melee_action
        self.boss_attack_actions = game.attack_actions(boss_name)
        self.player_map: dict[tuple[str, str], str] = {}
        self.boss_map: dict[tuple[str, str], str] = {}
        self.full_map: dict[tuple[str, str, str, str], str] = {}
        self.player_clause_to_tuple: dict[str, tuple[str, str]] = {}
        self.timelines: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._load()

    @classmethod
    def from_snapshot(
        cls,
        path: str | Path,
        *,
        game: GameSpec,
        player_name: str,
        boss_name: str,
        background: str | None = None,
    ) -> "PromptVocabulary":
        """Load a compact, relocatable vocabulary bundled with rollout inputs."""
        snapshot_path = Path(path).expanduser().resolve()
        with snapshot_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        expected = {
            "game": game.key,
            "player_name": player_name,
            "boss_name": boss_name,
            "background": background,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"vocabulary snapshot metadata mismatch in {snapshot_path}: "
                f"expected {expected}, found {observed}"
            )

        self = cls.__new__(cls)
        self.root = snapshot_path
        self.game = game
        self.game_name = game.key
        self.player_name = player_name
        self.boss_name = boss_name
        self.background = background
        self.strip_background = False
        self.basic_action = game.basic_action(boss_name)
        self.no_action = game.no_action
        self.death_action = game.death_action
        self.player_melee_action = game.player_melee_action
        self.boss_attack_actions = game.attack_actions(boss_name)
        self.player_map = {
            (str(row["action"]), str(row["movement"])): str(row["clause"])
            for row in payload["player_map"]
        }
        self.boss_map = {
            (str(row["action"]), str(row["movement"])): str(row["clause"])
            for row in payload["boss_map"]
        }
        self.full_map = {
            (
                str(row["player_action"]),
                str(row["player_movement"]),
                str(row["boss_action"]),
                str(row["boss_movement"]),
            ): str(row["prompt"])
            for row in payload["full_map"]
        }
        self.player_clause_to_tuple = {
            clause: key for key, clause in self.player_map.items()
        }
        self.timelines = defaultdict(
            list,
            {
                f"snapshot_{index:05d}": [
                    {"action": str(row["action"]), "movement": str(row["movement"])}
                    for row in rows
                ]
                for index, rows in enumerate(payload["timelines"])
            },
        )
        self._validate_required()
        return self

    def snapshot_payload(self) -> dict[str, Any]:
        """Return a deterministic, relocatable representation of this vocabulary."""
        return {
            "version": 1,
            "game": self.game.key,
            "player_name": self.player_name,
            "boss_name": self.boss_name,
            "background": self.background,
            "player_map": [
                {"action": action, "movement": movement, "clause": clause}
                for (action, movement), clause in sorted(self.player_map.items())
            ],
            "boss_map": [
                {"action": action, "movement": movement, "clause": clause}
                for (action, movement), clause in sorted(self.boss_map.items())
            ],
            "full_map": [
                {
                    "player_action": player_action,
                    "player_movement": player_movement,
                    "boss_action": boss_action,
                    "boss_movement": boss_movement,
                    "prompt": prompt,
                }
                for (
                    player_action,
                    player_movement,
                    boss_action,
                    boss_movement,
                ), prompt in sorted(self.full_map.items())
            ],
            "timelines": [self.timelines[key] for key in sorted(self.timelines)],
        }

    def save_snapshot(self, path: str | Path) -> Path:
        """Write a compact vocabulary snapshot consumable by ``from_snapshot``."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.snapshot_payload(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def _put(table: dict, key: tuple, value: str) -> None:
        previous = table.setdefault(key, value)
        if previous != value:
            raise ValueError(f"conflicting prompt mapping for {key}: {previous!r} / {value!r}")

    @staticmethod
    def _split(full: str) -> tuple[str, str]:
        if ", " not in full:
            raise ValueError(f"not a two-character training prompt: {full!r}")
        player, boss = full.split(", ", 1)
        return player.strip(), boss.strip()

    def _selected(self, payload: dict[str, Any], player_clause: str, boss_clause: str) -> tuple[bool, bool]:
        row_player = payload.get("player_name")
        row_boss = payload.get("boss_name")
        has_player = row_player == self.player_name if row_player else player_clause.startswith(self.player_name + " ")
        has_boss = row_boss == self.boss_name if row_boss else boss_clause.startswith(self.boss_name + " ")
        return has_player, has_boss

    def _load(self) -> None:
        paths = sorted(self.root.rglob("clip_*.json"))
        if not paths:
            paths = sorted(self.root.rglob("*.json"))
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            timeline_rows: list[dict[str, str]] = []
            for row in payload.get("prompts", []):
                full = str(row["prompt"]).strip()
                if self.strip_background:
                    suffix = _row_background(full)
                    if suffix is not None:
                        full = full[: -(len(" in ") + len(suffix))]
                elif self.background is not None:
                    suffix = _row_background(full)
                    if suffix is not None and suffix != self.background:
                        continue
                player_clause, boss_clause = self._split(full)
                has_player, has_boss = self._selected(payload, player_clause, boss_clause)
                if not has_player and not has_boss:
                    continue
                player_action = self.game.canonical_action(str(row["player_action"]))
                boss_action = self.game.canonical_action(str(row["boss_action"]))
                player_key = (player_action, str(row["player_movement"]))
                boss_key = (boss_action, str(row["boss_movement"]))
                if has_player:
                    self._put(self.player_map, player_key, player_clause)
                    self.player_clause_to_tuple[player_clause] = player_key
                if has_boss:
                    self._put(self.boss_map, boss_key, boss_clause)
                if has_player and has_boss:
                    self._put(self.full_map, (*player_key, *boss_key), full)
                    if not row.get("post_death") and boss_action != self.death_action:
                        timeline_rows.append(
                            {"action": boss_action, "movement": str(row["boss_movement"])}
                        )
            if timeline_rows:
                self.timelines[str(path.parent)].extend(timeline_rows)

        self._validate_required()

    def _validate_required(self) -> None:
        """Reject incomplete raw corpora and snapshots in the same way."""

        required_player = {
            (PLAYER_IDLE_ACTION, IDLE_MOVEMENT),
            (PLAYER_DEATH_ACTION, IDLE_MOVEMENT),
        }
        required_boss_actions = {
            self.no_action,
            self.basic_action,
            self.death_action,
            *self.game.skill_library[self.boss_name].keys(),
        }
        missing_player = sorted(required_player.difference(self.player_map))
        missing_boss = sorted(
            action for action in required_boss_actions
            if not any(key[0] == action for key in self.boss_map)
        )
        if missing_player or missing_boss:
            raise ValueError(
                f"selected pair lacks required vocabulary; player={missing_player}, boss={missing_boss}"
            )

    @property
    def player_idle_clause(self) -> str:
        return self.player_map[(PLAYER_IDLE_ACTION, IDLE_MOVEMENT)]

    @property
    def player_death_clause(self) -> str:
        return self.player_map[(PLAYER_DEATH_ACTION, IDLE_MOVEMENT)]

    def player_clause(self, action: str, movement: str) -> str:
        action = self.game.canonical_action(action)
        try:
            return self.player_map[(action, movement)]
        except KeyError as error:
            raise ValueError(f"unseen selected-player action/movement: {(action, movement)}") from error

    def compose(
        self, player_clause: str, boss_action: str, boss_movement: str
    ) -> tuple[str, dict[str, str] | None]:
        try:
            player_action, player_movement = self.player_clause_to_tuple[player_clause]
        except KeyError as error:
            raise ValueError(f"unseen selected-player clause: {player_clause!r}") from error
        boss_action = self.game.canonical_action(boss_action)
        requested = (player_action, player_movement, boss_action, boss_movement)
        if requested in self.full_map:
            return self.full_map[requested], None

        movement_order = [boss_movement, IDLE_MOVEMENT, *MOVEMENTS]
        seen: set[str] = set()
        for candidate in movement_order:
            if candidate in seen:
                continue
            seen.add(candidate)
            key = (player_action, player_movement, boss_action, candidate)
            if key in self.full_map:
                return self.full_map[key], {
                    "requested_action": boss_action,
                    "requested_movement": boss_movement,
                    "emitted_action": boss_action,
                    "emitted_movement": candidate,
                    "reason": "requested full prompt was absent from selected-pair vocabulary",
                }
        for candidate in movement_order:
            boss_clause = self.boss_map.get((boss_action, candidate))
            if boss_clause is not None:
                return f"{player_clause}, {boss_clause}", {
                    "requested_action": boss_action,
                    "requested_movement": boss_movement,
                    "emitted_action": boss_action,
                    "emitted_movement": candidate,
                    "reason": "selected pair lacked this cross-product; composed exact clauses",
                }
        raise ValueError(f"selected boss lacks action in vocabulary: {boss_action!r}")


class PlayerPromptTrack:
    def __init__(self, clauses: Iterable[str], vocabulary: PromptVocabulary):
        self.clauses = [str(value).strip() for value in clauses]
        if not self.clauses or len(self.clauses) % 20:
            raise ValueError("player prompt track must contain a positive multiple of 20 cells")
        unseen = sorted(set(self.clauses).difference(vocabulary.player_clause_to_tuple))
        if unseen:
            raise ValueError(f"player track contains unseen selected-player clauses: {unseen[:3]}")

    @classmethod
    def from_path(cls, path: str | Path, vocabulary: PromptVocabulary) -> "PlayerPromptTrack":
        source = Path(path).expanduser()
        if source.is_dir():
            files = sorted(source.glob("clip_*.json"))
            if not files:
                raise ValueError(f"no clip_*.json player prompts under {source}")
            rows: list[Any] = []
            for prompt_file in files:
                with prompt_file.open(encoding="utf-8") as handle:
                    rows.extend(json.load(handle)["prompts"])
        else:
            with source.open(encoding="utf-8") as handle:
                payload: Any = json.load(handle)
            if isinstance(payload, dict) and "chunks" in payload:
                rows = [row for chunk in payload["chunks"] for row in chunk]
            elif isinstance(payload, dict) and "prompts" in payload:
                rows = payload["prompts"]
            elif isinstance(payload, list):
                rows = payload
            else:
                raise ValueError("player prompt JSON needs prompts/chunks or a list")
        return cls([cls._row_to_clause(row, vocabulary) for row in rows], vocabulary)

    @staticmethod
    def _row_to_clause(row: Any, vocabulary: PromptVocabulary) -> str:
        if isinstance(row, dict):
            if "player_action" in row and "player_movement" in row:
                return vocabulary.player_clause(
                    str(row["player_action"]), str(row["player_movement"])
                )
            row = row.get("player_prompt", row.get("prompt"))
        if not isinstance(row, str):
            raise ValueError(f"cannot read player prompt row: {row!r}")
        value = row.strip()
        if value in vocabulary.player_clause_to_tuple:
            return value
        if ", " in value:
            value = value.split(", ", 1)[0].strip()
        suffix = value.split(" ", 2)[-1]
        matches = [
            clause for clause in vocabulary.player_clause_to_tuple
            if clause.split(" ", 2)[-1] == suffix
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"cannot map player prompt to {vocabulary.player_name}: {row!r}")

    def chunk(self, tick: int) -> tuple[list[str], bool]:
        start = tick * 20
        if start + 20 <= len(self.clauses):
            return self.clauses[start : start + 20], False
        return self.clauses[-20:], True


__all__ = [
    "IDLE_MOVEMENT",
    "PlayerPromptTrack",
    "PromptVocabulary",
    "detect_background",
]
