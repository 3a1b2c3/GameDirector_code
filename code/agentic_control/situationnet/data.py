"""State-derived geometry targets and video decoding for SituationNet."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from attacknet.data import (
    CELLS_PER_CLIP,
    decode_clip_context,
    opencv_worker_init,
    select_records_by_manifest,
    split_records,
    split_records_from_train_jobs,
)


@dataclass(frozen=True)
class SituationClipRecord:
    clip_path: str
    state_path: str
    fight_key: tuple[str, str, int, int]
    distance: tuple[float, ...]
    angle_sincos: tuple[tuple[float, float], ...]
    valid: tuple[bool, ...]


def _fight_key(state: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(state.get("split", "nrftw")),
        str(state["session_id"]),
        int(state["fight_index"]),
        int(state["episode"]) if "episode" in state else int(state.get("round", 0)),
    )


def relative_angle_radians(sample: dict[str, Any]) -> float:
    """Player bearing in the boss frame.

    NRFTW state-process yaw is in radians with forward=(sin(yaw), cos(yaw)).
    Positive angles are on the boss's right and negative angles on its left.
    """
    def xz(position: Any) -> tuple[float, float]:
        if isinstance(position, dict):
            return float(position["x"]), float(position["z"])
        return float(position[0]), float(position[2])

    try:
        px, pz = xz(sample["player"]["position"])
        bx, bz = xz(sample["boss"]["position"])
        yaw = float(sample["boss"]["yaw"])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("state sample is missing player/boss position or boss.yaw") from error
    fx, fz = math.sin(yaw), math.cos(yaw)
    tox, toz = px - bx, pz - bz
    if math.hypot(tox, toz) <= 1e-8:
        return 0.0
    return math.atan2(tox * fz - toz * fx, tox * fx + toz * fz)


def horizontal_distance(sample: dict[str, Any]) -> float:
    try:
        return float(sample["important_state"]["horizontal_distance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("state sample is missing important_state.horizontal_distance") from error


def build_records(
    data_root: str | Path,
    distance_scale: float = 10.0,
    include_post_death: bool = False,
) -> list[SituationClipRecord]:
    """Read the 0.25-second state-process outputs as geometry supervision."""
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    root = Path(data_root)
    state_paths = sorted((root / "states").glob("**/*.json"))
    if not state_paths:
        raise FileNotFoundError(f"no state JSON files under {root / 'states'}")

    records: list[SituationClipRecord] = []
    skipped_missing_geometry = 0
    for state_path in state_paths:
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        samples = state.get("samples") or []
        if len(samples) != CELLS_PER_CLIP:
            raise ValueError(
                f"{state_path}: expected {CELLS_PER_CLIP} state samples, got {len(samples)}"
            )
        distances: list[float] = []
        angles: list[tuple[float, float]] = []
        valid: list[bool] = []
        try:
            for sample in samples:
                distance = horizontal_distance(sample)
                angle = relative_angle_radians(sample)
                distances.append(distance / distance_scale)
                angles.append((math.sin(angle), math.cos(angle)))
                valid.append(
                    not bool(sample.get("padded", False))
                    and (include_post_death or not bool(sample.get("post_death", False)))
                )
        except ValueError:
            # Some non-nrftw sources (e.g. Vampire sessions with a raw-log
            # gap) only have partial geometry coverage; skip those clips
            # instead of failing the whole multi-game build.
            skipped_missing_geometry += 1
            continue

        clip_path = root / state["clip"]
        if not clip_path.is_file():
            raise FileNotFoundError(f"missing clip referenced by {state_path}: {clip_path}")
        records.append(
            SituationClipRecord(
                clip_path=str(clip_path),
                state_path=str(state_path),
                fight_key=_fight_key(state),
                distance=tuple(distances),
                angle_sincos=tuple(angles),
                valid=tuple(valid),
            )
        )
    if skipped_missing_geometry:
        print(
            f"[data] {data_root}: skipped {skipped_missing_geometry} clip(s) missing geometry",
            flush=True,
        )
    return records


def target_statistics(
    records: Iterable[SituationClipRecord], distance_scale: float
) -> dict[str, Any]:
    distances: list[float] = []
    sector_counts = {"front": 0, "left_flank": 0, "right_flank": 0, "behind": 0}
    for record in records:
        for distance, angle, valid in zip(record.distance, record.angle_sincos, record.valid):
            if not valid:
                continue
            distances.append(distance * distance_scale)
            degrees = math.degrees(math.atan2(angle[0], angle[1]))
            if abs(degrees) <= 20.0:
                sector = "front"
            elif abs(degrees) >= 90.0:
                sector = "behind"
            else:
                sector = "right_flank" if degrees > 0 else "left_flank"
            sector_counts[sector] += 1
    values = np.asarray(distances, dtype=np.float64)
    return {
        "valid_cells": len(distances),
        "distance_units": {
            "min": float(values.min()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p95": float(np.quantile(values, 0.95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        },
        "tactical_sector_counts": sector_counts,
    }


class SituationClipDataset(Dataset):
    """Decode a clip once and return its twenty three-frame examples."""

    def __init__(
        self,
        records: list[SituationClipRecord],
        image_hw: tuple[int, int] = (192, 336),
        augment: bool = False,
        frame_mode: str = "short",
        cache_dir: str | Path | None = None,
        data_root: str | Path | None = None,
    ):
        self.records = records
        self.image_hw = image_hw
        self.augment = augment
        self.frame_mode = frame_mode
        # SituationNet selects frames with AttackNet's decode_clip_context, so the
        # two share one cache directory.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.data_root = Path(data_root).resolve() if data_root else None

    def _cached_frames(self, clip_path: str) -> np.ndarray | None:
        if self.cache_dir is None or self.data_root is None:
            return None
        try:
            relative = Path(clip_path).resolve().relative_to(self.data_root)
        except ValueError:
            return None
        path = (self.cache_dir / relative).with_suffix(".npy")
        return np.load(path) if path.is_file() else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = self._cached_frames(record.clip_path)
        if frames is None:
            frames = decode_clip_context(record.clip_path, self.image_hw, self.frame_mode)
        angles = np.asarray(record.angle_sincos, dtype=np.float32).copy()
        if self.augment:
            # One coherent transform is shared by all frames in the clip.
            if random.random() < 0.5:
                frames = frames[..., ::-1].copy()
                angles[:, 0] *= -1.0  # horizontal mirror swaps left and right
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-8.0, 8.0)
            frames = np.clip(frames.astype(np.float32) * contrast + brightness, 0, 255).astype(
                np.uint8
            )
        return {
            "frames": torch.from_numpy(frames),
            "distance": torch.tensor(record.distance, dtype=torch.float32),
            "angle_sincos": torch.from_numpy(angles),
            "valid": torch.tensor(record.valid, dtype=torch.bool),
            "clip_path": record.clip_path,
        }


__all__ = [
    "SituationClipDataset",
    "SituationClipRecord",
    "build_records",
    "horizontal_distance",
    "opencv_worker_init",
    "relative_angle_radians",
    "select_records_by_manifest",
    "split_records",
    "split_records_from_train_jobs",
    "target_statistics",
]
