"""Load and validate the 20 temporal VRising prompts."""
from __future__ import annotations

import json
from pathlib import Path


PROMPT_COUNT = 20
PROMPT_DT = 0.25
PROMPT_FIRST_TIMESTAMP = 0.125
TIMESTAMP_TOLERANCE = 1e-5


def load_prompts(path: str, expected_count: int = PROMPT_COUNT) -> list[str]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(cells, list) or len(cells) != expected_count:
        raise ValueError(f"{path} must contain exactly {expected_count} prompts")
    source_start = payload.get("source_start") if isinstance(payload, dict) else None
    source_start = None if source_start is None else float(source_start)
    prompts = []
    for index, cell in enumerate(cells):
        if isinstance(cell, dict):
            expected_offset = PROMPT_FIRST_TIMESTAMP + PROMPT_DT * index
            if "offset" in cell and abs(float(cell["offset"]) - expected_offset) > TIMESTAMP_TOLERANCE:
                raise ValueError(
                    f"{path} prompts[{index}] has a misaligned offset; "
                    f"expected {expected_offset:.6f}s"
                )
            if source_start is not None and "source_t" in cell:
                expected_source_t = source_start + expected_offset
                if abs(float(cell["source_t"]) - expected_source_t) > TIMESTAMP_TOLERANCE:
                    raise ValueError(
                        f"{path} prompts[{index}] has a misaligned source_t; "
                        f"expected {expected_source_t:.6f}s"
                    )
            prompt = str(cell.get("prompt", "")).strip()
        else:
            prompt = str(cell).strip()
        if not prompt:
            raise ValueError(f"{path} prompts[{index}] is empty")
        prompts.append(prompt)
    return prompts


__all__ = ["load_prompts"]
