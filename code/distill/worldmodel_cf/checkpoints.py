"""Resolve and validate teacher/student checkpoints without copying weights."""
from __future__ import annotations

import glob
import re
from pathlib import Path

from safetensors import safe_open


_STEP = re.compile(r"(?:step[-_]?|epoch[-_]?)(\d+)", re.IGNORECASE)


def checkpoint_step(path: str | Path) -> int:
    """Return the numeric step encoded in a supported checkpoint filename."""
    match = _STEP.search(Path(path).stem)
    return int(match.group(1)) if match else -1


def _candidates(spec: str | Path) -> list[Path]:
    raw = str(spec)
    path = Path(raw).expanduser()
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            candidate.resolve()
            for candidate in path.rglob("*.safetensors")
            if not candidate.name.startswith(".")
        )
    if any(character in raw for character in "*?["):
        return sorted(
            Path(candidate).expanduser().resolve()
            for candidate in glob.glob(raw, recursive=True)
            if Path(candidate).is_file() and candidate.endswith(".safetensors")
        )
    return []


def resolve_checkpoint(spec: str | Path) -> Path:
    """Resolve a file directly or select the highest-step file in a directory/glob."""
    candidates = _candidates(spec)
    if not candidates:
        raise FileNotFoundError(
            f"no .safetensors checkpoint found for {spec!s}; "
            "pass a file, directory, or glob"
        )
    return max(
        candidates,
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns, str(path)),
    )


def validate_checkpoint(
    spec: str | Path,
    *,
    expected_keys: set[str] | None = None,
    expected_key_count: int | None = None,
    require_bf16: bool = False,
) -> Path:
    """Resolve a checkpoint and validate its key contract without materializing tensors."""
    path = resolve_checkpoint(spec)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if expected_keys is not None and keys != expected_keys:
            missing = sorted(expected_keys - keys)[:8]
            unexpected = sorted(keys - expected_keys)[:8]
            raise RuntimeError(
                f"checkpoint keys do not match {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if expected_key_count is not None and len(keys) != expected_key_count:
            raise RuntimeError(
                f"unexpected key count for {path}: {len(keys)} "
                f"(expected {expected_key_count})"
            )
        if require_bf16:
            dtypes = {handle.get_slice(key).get_dtype() for key in keys}
            if dtypes != {"BF16"}:
                raise RuntimeError(f"unexpected dtypes for {path}: {sorted(dtypes)}")
    return path
