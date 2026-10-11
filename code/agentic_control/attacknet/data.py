"""Dataset construction and video decoding for AttackNet."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


CELLS_PER_CLIP = 20
FRAMES_PER_EXAMPLE = 3
# Auxiliary damage-regression scale: divides raw HP loss so typical single hits are O(1).
DAMAGE_SCALE = 20.0
# "long" frame_mode cell look-back offsets for the temporal-context ablation
# (non-adjacent offsets to see real motion instead of 3 near-simultaneous frames).
LONG_CONTEXT_CELL_OFFSETS = (4, 1, 0)


@dataclass(frozen=True)
class ClipRecord:
    clip_path: str
    state_path: str
    fight_key: tuple[str, str, int, int]
    labels: tuple[tuple[float, float], ...]
    valid: tuple[bool, ...]
    # Continuous (player_damage, boss_damage) HP loss per cell, scaled by DAMAGE_SCALE.
    # Optional auxiliary regression target; zero wherever labels are also zero/invalid.
    damage: tuple[tuple[float, float], ...] = ()


def _fight_key(state: dict[str, Any]) -> tuple[str, str, int, int]:
    """Identify one continuous NRFTW fight without joining separate episodes."""
    return (
        str(state.get("split", "nrftw")),
        str(state["session_id"]),
        int(state["fight_index"]),
        int(state["episode"]) if "episode" in state else int(state.get("round", 0)),
    )


def _health(sample: dict[str, Any]) -> tuple[float, float]:
    """Return GT health ordered as player, boss. Supports both the nested NRFTW/hk
    schema (sample.player.hp / sample.boss.hp) and vrising's flat schema
    (sample.player_hp / sample.boss_hp)."""
    try:
        if "player" in sample and "boss" in sample:
            return float(sample["player"]["hp"]), float(sample["boss"]["hp"])
        return float(sample["player_hp"]), float(sample["boss_hp"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("state sample is missing numeric player/boss hp") from error


def build_records(data_root: str | Path, damage_epsilon: float = 1e-3) -> list[ClipRecord]:
    """Build labels from adjacent HP samples, including transitions across clip boundaries."""
    root = Path(data_root)
    state_paths = sorted((root / "states").glob("**/*.json"))
    if not state_paths:
        raise FileNotFoundError(f"no state JSON files under {root / 'states'}")

    docs: list[tuple[Path, dict[str, Any]]] = []
    timelines: dict[tuple[str, str, int, int], dict[float, tuple[float, float]]] = defaultdict(dict)
    for path in state_paths:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        samples = state.get("samples") or []
        if len(samples) != CELLS_PER_CLIP:
            raise ValueError(f"{path}: expected {CELLS_PER_CLIP} state samples, got {len(samples)}")
        key = _fight_key(state)
        for sample in samples:
            timelines[key][round(float(sample["source_t"]), 6)] = _health(sample)
        docs.append((path, state))

    previous: dict[tuple[tuple[str, str, int, int], float], tuple[float, float] | None] = {}
    for key, timeline in timelines.items():
        last: tuple[float, float] | None = None
        last_t: float | None = None
        for timestamp, hp in sorted(timeline.items()):
            # A discontinuity is not a valid attack transition.
            contiguous = last_t is not None and abs(timestamp - last_t - 0.25) <= 2e-3
            previous[(key, timestamp)] = last if contiguous else None
            last, last_t = hp, timestamp

    records: list[ClipRecord] = []
    for state_path, state in docs:
        key = _fight_key(state)
        labels: list[tuple[float, float]] = []
        damage: list[tuple[float, float]] = []
        valid: list[bool] = []
        for sample in state["samples"]:
            timestamp = round(float(sample["source_t"]), 6)
            current = _health(sample)
            prior = previous[(key, timestamp)]
            is_valid = prior is not None and not bool(sample.get("padded", False))
            if is_valid:
                player_loss = prior[0] - current[0]
                boss_loss = prior[1] - current[1]
                labels.append(
                    (
                        float(player_loss > damage_epsilon),
                        float(boss_loss > damage_epsilon),
                    )
                )
                damage.append(
                    (
                        max(player_loss, 0.0) / DAMAGE_SCALE,
                        max(boss_loss, 0.0) / DAMAGE_SCALE,
                    )
                )
            else:
                labels.append((0.0, 0.0))
                damage.append((0.0, 0.0))
            valid.append(is_valid)

        clip_path = root / state["clip"]
        if not clip_path.is_file():
            raise FileNotFoundError(f"missing clip referenced by {state_path}: {clip_path}")
        records.append(
            ClipRecord(
                clip_path=str(clip_path),
                state_path=str(state_path),
                fight_key=key,
                labels=tuple(labels),
                valid=tuple(valid),
                damage=tuple(damage),
            )
        )
    return records


def split_records(
    records: Iterable[ClipRecord], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Leakage-safe deterministic split: every clip from a fight stays in one split."""
    train: list[ClipRecord] = []
    val: list[ClipRecord] = []
    for record in records:
        token = f"{seed}|{'|'.join(map(str, record.fight_key))}".encode()
        value = int(hashlib.sha1(token).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if value < val_fraction else train).append(record)
    if not train or not val:
        raise RuntimeError("deterministic split produced an empty train or validation set")
    return train, val


def split_records_from_train_jobs(
    records: Iterable[ClipRecord], data_root: str | Path, train_jobs: str | Path
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Use an explicit training manifest; its complement is the evaluation set."""
    root = Path(data_root).resolve()
    manifest_path = Path(train_jobs)
    requested: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("clip"):
                raise ValueError(f"{manifest_path}:{line_number}: missing clip")
            requested.add(str((root / row["clip"]).resolve()))

    by_path = {str(Path(record.clip_path).resolve()): record for record in records}
    missing = requested - set(by_path)
    if missing:
        example = next(iter(missing))
        raise FileNotFoundError(
            f"{len(missing)} train-manifest clips are absent; e.g. {example}"
        )
    train = [by_path[path] for path in sorted(requested)]
    val = [record for path, record in sorted(by_path.items()) if path not in requested]
    if not train or not val:
        raise RuntimeError("explicit manifest produced an empty train or evaluation set")
    return train, val


def select_records_by_manifest(
    records: Iterable[ClipRecord], data_root: str | Path, manifest_path: str | Path
) -> list[ClipRecord]:
    """Return exactly the clips listed in ``manifest_path`` (order-independent)."""
    root = Path(data_root).resolve()
    wanted: set[str] = set()
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("clip"):
                raise ValueError(f"{manifest_path}:{line_number}: missing clip")
            wanted.add(str((root / row["clip"]).resolve()))
    by_path = {str(Path(record.clip_path).resolve()): record for record in records}
    missing = wanted - set(by_path)
    if missing:
        example = next(iter(missing))
        raise FileNotFoundError(f"{len(missing)} manifest clips are absent; e.g. {example}")
    return [by_path[path] for path in sorted(wanted)]


def label_statistics(records: Iterable[ClipRecord]) -> dict[str, Any]:
    positives = np.zeros(2, dtype=np.int64)
    count = 0
    joint = np.zeros(4, dtype=np.int64)
    for record in records:
        for label, valid in zip(record.labels, record.valid):
            if not valid:
                continue
            y = np.asarray(label, dtype=np.int64)
            positives += y
            count += 1
            joint[int(y[0]) * 2 + int(y[1])] += 1
    negatives = count - positives
    return {
        "valid_cells": int(count),
        "positives": positives.tolist(),
        "negatives": negatives.tolist(),
        "positive_rate": (positives / max(count, 1)).tolist(),
        "joint_00_01_10_11": joint.tolist(),
    }


def positive_weights(records: Iterable[ClipRecord], maximum: float = 20.0) -> torch.Tensor:
    stats = label_statistics(records)
    positives = np.asarray(stats["positives"], dtype=np.float64)
    negatives = np.asarray(stats["negatives"], dtype=np.float64)
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(np.clip(weights, 1.0, maximum), dtype=torch.float32)


def _cell_bounds(frame_count: int) -> list[tuple[int, int]]:
    # This supports both the real 80-frame/16-FPS files and old 100-frame metadata.
    return [
        (round(cell * frame_count / CELLS_PER_CLIP), round((cell + 1) * frame_count / CELLS_PER_CLIP))
        for cell in range(CELLS_PER_CLIP)
    ]


def _selected_frame_indices(frame_count: int, mode: str = "short") -> list[int]:
    if frame_count < CELLS_PER_CLIP * FRAMES_PER_EXAMPLE:
        raise ValueError(
            f"video has {frame_count} frames; need at least "
            f"{CELLS_PER_CLIP * FRAMES_PER_EXAMPLE}"
        )
    bounds = _cell_bounds(frame_count)
    selected: list[int] = []
    if mode == "short":
        for cell in range(CELLS_PER_CLIP):
            start, end = bounds[cell]
            if end - start < FRAMES_PER_EXAMPLE:
                raise ValueError(f"cell {cell} has only {end - start} frames")
            selected.extend(range(end - FRAMES_PER_EXAMPLE, end))
    elif mode == "long":
        # Look back over ~1s of cells (offsets 4,1,0) instead of 3 near-simultaneous
        # frames, trading fine-grained motion for a longer receptive field (the
        # L1 reader also uses non-adjacent frame offsets for the same reason).
        for cell in range(CELLS_PER_CLIP):
            for offset in LONG_CONTEXT_CELL_OFFSETS:
                source_cell = max(cell - offset, 0)
                selected.append(bounds[source_cell][1] - 1)
    else:
        raise ValueError(f"unknown frame_mode: {mode!r}")
    return selected


def decode_clip_context(
    clip_path: str, image_hw: tuple[int, int], mode: str = "short"
) -> np.ndarray:
    """Return uint8 RGB ``[20, 3, 3, H, W]`` from one five-second clip.

    ``mode="short"`` reproduces the original 3-consecutive-frame-per-cell behavior;
    ``mode="long"`` instead pulls one frame each from cells (c-4, c-1, c) for a
    longer temporal receptive field per example.
    """
    import cv2

    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {clip_path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = _selected_frame_indices(frame_count, mode)
    wanted = set(indices)
    decoded: dict[int, np.ndarray] = {}
    index = 0
    h, w = image_hw
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    missing = [i for i in indices if i not in decoded]
    if missing:
        raise RuntimeError(
            f"{clip_path}: decoder returned {index}/{frame_count} frames; "
            f"missing selected indices {missing[:5]}"
        )
    array = np.stack([decoded[i] for i in indices]).reshape(
        CELLS_PER_CLIP, FRAMES_PER_EXAMPLE, h, w, 3
    )
    return array.transpose(0, 1, 4, 2, 3).copy()


class AttackClipDataset(Dataset):
    """Decode one clip once and return all of its twenty training examples."""

    def __init__(
        self,
        records: list[ClipRecord],
        image_hw: tuple[int, int] = (192, 336),
        augment: bool = False,
        frame_mode: str = "short",
        augment_recipe: str = "baseline",
        grayscale: bool = False,
        cache_dir: str | Path | None = None,
        data_root: str | Path | None = None,
    ):
        self.records = records
        self.image_hw = image_hw
        self.augment = augment
        self.frame_mode = frame_mode
        self.augment_recipe = augment_recipe
        self.grayscale = grayscale
        # Pre-decoded uint8 frames (see build_frame_cache.py); decode is otherwise
        # the entire cost of an epoch.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.data_root = Path(data_root).resolve() if data_root else None
        self.cache_frame_mode = frame_mode
        if self.cache_dir is not None:
            meta_path = self.cache_dir / "cache_meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.cache_frame_mode = str(meta.get("frame_mode", frame_mode))
                if (meta.get("height"), meta.get("width")) != tuple(image_hw):
                    raise ValueError(
                        f"frame cache {self.cache_dir} is "
                        f"{meta.get('height')}x{meta.get('width')}, not {image_hw[0]}x{image_hw[1]}"
                    )
            if self.cache_frame_mode != frame_mode and not (
                self.cache_frame_mode == "short" and frame_mode == "long"
            ):
                raise ValueError(
                    f"frame cache holds {self.cache_frame_mode!r} frames, cannot serve {frame_mode!r}"
                )

    def _cached_frames(self, clip_path: str) -> np.ndarray | None:
        if self.cache_dir is None or self.data_root is None:
            return None
        try:
            relative = Path(clip_path).resolve().relative_to(self.data_root)
        except ValueError:
            return None
        path = (self.cache_dir / relative).with_suffix(".npy")
        if not path.is_file():
            return None
        frames = np.load(path)
        if self.cache_frame_mode == self.frame_mode:
            return frames
        # "long" mode wants the LAST frame of cells (c-4, c-1, c), and a "short" cache
        # already stores the last three frames of every cell -- so index 2 of cell s is
        # exactly the frame long mode asks for. Reconstruct it rather than re-decoding.
        return np.stack(
            [
                np.stack([frames[max(cell - offset, 0), 2] for offset in LONG_CONTEXT_CELL_OFFSETS])
                for cell in range(CELLS_PER_CLIP)
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from attacknet.augment import augment_clip, to_grayscale

        record = self.records[index]
        frames = self._cached_frames(record.clip_path)
        if frames is None:
            frames = decode_clip_context(record.clip_path, self.image_hw, self.frame_mode)
        if self.augment:
            # Every transform is clip-wide: a per-frame photometric jitter would fake
            # the very luminance step that marks a hit and corrupt the label.
            frames = augment_clip(frames, self.augment_recipe, random)
        if self.grayscale:
            # Input mode, not augmentation: applied to train and eval alike.
            frames = to_grayscale(frames)
        damage = record.damage if record.damage else ((0.0, 0.0),) * len(record.labels)
        return {
            "frames": torch.from_numpy(frames),
            "labels": torch.tensor(record.labels, dtype=torch.float32),
            "damage": torch.tensor(damage, dtype=torch.float32),
            "valid": torch.tensor(record.valid, dtype=torch.bool),
            "clip_path": record.clip_path,
        }


def opencv_worker_init(_: int) -> None:
    import cv2

    cv2.setNumThreads(0)
