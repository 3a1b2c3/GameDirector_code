"""VRising 5-second clips with 20 temporally aligned prompt cells."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from diffsynth.core import default_video_operator


DEFAULT_DATA_ROOT = None
EXPECTED_PROMPTS = 20
EXPECTED_FRAMES = 80
SOURCE_FPS = 16
FRAMES_PER_PROMPT = 4
PROMPT_DT = FRAMES_PER_PROMPT / SOURCE_FPS
PROMPT_FIRST_TIMESTAMP = PROMPT_DT / 2
TIMESTAMP_TOLERANCE = 1e-5


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        payload = _read_json(path)
        return payload if isinstance(payload, list) else [payload]
    if path.suffix != ".jsonl":
        raise ValueError(f"VRising metadata must be JSON or JSONL, got {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


class VRisingDataset(torch.utils.data.Dataset):
    """Return video targets, the first-frame condition, and 20 prompts only."""

    load_from_cache = False

    def __init__(
        self,
        base_path: str = DEFAULT_DATA_ROOT,
        metadata_path: str | None = None,
        repeat: int = 1,
        height: int | None = 480,
        width: int | None = 832,
        max_pixels: int = 1024 * 1024,
        num_frames: int = EXPECTED_FRAMES,
        expected_prompts: int = EXPECTED_PROMPTS,
    ):
        if base_path is None:
            raise ValueError("base_path is required")
        self.root = Path(base_path).expanduser().absolute()
        self.metadata_path = (
            Path(metadata_path).expanduser().absolute()
            if metadata_path
            else self.root / "jobs.jsonl"
        )
        if num_frames != EXPECTED_FRAMES:
            raise ValueError(
                f"dataset contains {EXPECTED_FRAMES} target frames; got {num_frames}"
            )
        if expected_prompts != EXPECTED_PROMPTS:
            raise ValueError(
                f"world model expects {EXPECTED_PROMPTS} prompts; got {expected_prompts}"
            )
        self.num_frames = num_frames
        self.expected_prompts = expected_prompts
        self.repeat = max(1, int(repeat))
        self.rows = _read_rows(self.metadata_path)
        if not self.rows:
            raise ValueError(f"VRising metadata is empty: {self.metadata_path}")
        self.video_operator = default_video_operator(
            base_path=str(self.root),
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=0,
            frame_rate=SOURCE_FPS,
            fix_frame_rate=True,
        )

    def __len__(self):
        return len(self.rows) * self.repeat

    def _prompt_cells(self, row: dict) -> list:
        cells = row.get("prompts")
        if cells is None:
            relative_path = row.get("sidecar") or row.get("prompt_file")
            if not relative_path:
                raise KeyError("VRising row has neither embedded prompts nor prompt_file")
            payload = _read_json(self.root / relative_path)
            cells = payload.get("prompts")
        if not isinstance(cells, list) or len(cells) != self.expected_prompts:
            raise ValueError(
                f"expected {self.expected_prompts} prompt cells, got "
                f"{None if cells is None else len(cells)}"
            )
        return cells

    @staticmethod
    def _validate_prompt_alignment(row: dict, cells: list, label: str) -> None:
        source_start = row.get("source_start")
        source_start = None if source_start is None else float(source_start)
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                # A raw list[str] is accepted when no timestamp metadata exists.
                continue
            expected_offset = PROMPT_FIRST_TIMESTAMP + PROMPT_DT * index
            if "offset" in cell:
                offset = float(cell["offset"])
                if abs(offset - expected_offset) > TIMESTAMP_TOLERANCE:
                    raise ValueError(
                        f"{label} prompt[{index}] offset={offset} is misaligned; "
                        f"expected {expected_offset:.6f}s"
                    )
            if source_start is not None and "source_t" in cell:
                expected_source_t = source_start + expected_offset
                source_t = float(cell["source_t"])
                if abs(source_t - expected_source_t) > TIMESTAMP_TOLERANCE:
                    raise ValueError(
                        f"{label} prompt[{index}] source_t={source_t} is misaligned; "
                        f"expected {expected_source_t:.6f}s"
                    )

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index % len(self.rows)]
        relative_clip = row.get("clip") or row.get("video")
        if not relative_clip:
            raise KeyError("VRising row is missing clip")
        video = self.video_operator(relative_clip)
        if len(video) != self.num_frames:
            raise ValueError(
                f"{relative_clip} decoded to {len(video)} frames; "
                f"expected {self.num_frames}"
            )
        cells = self._prompt_cells(row)
        self._validate_prompt_alignment(row, cells, str(relative_clip))
        prompts = [
            str(cell.get("prompt", "") if isinstance(cell, dict) else cell).strip()
            for cell in cells
        ]
        if any(not prompt for prompt in prompts):
            raise ValueError(f"{relative_clip} contains an empty temporal prompt")

        # Wan's causal 4k+1 layout: latent 0 is the clean first-frame
        # condition; latents 1..20 encode the original 80 target frames.
        video = [video[0].copy()] + video
        return {
            "video": video,
            "source_num_frames": self.num_frames,
            "prompts": prompts,
        }


__all__ = ["VRisingDataset"]
