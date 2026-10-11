"""Threshold-aware AttackNet metrics.

A single macro-F1 at a hard-coded 0.5 threshold hides that positive-weighted
training puts the F1-optimal operating point nowhere near 0.5, so metrics are
reported both at a fixed threshold and at per-label tuned thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LABELS = ("player_hit", "boss_hit")


def _prf(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def metrics_at(probs: np.ndarray, truth: np.ndarray, thresholds: float | list[float]) -> dict[str, Any]:
    """Per-label and macro F1 at a scalar threshold or one threshold per label."""
    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)] * len(LABELS)
    predicted = np.stack([probs[:, i] >= thresholds[i] for i in range(len(LABELS))], axis=1)
    actual = truth.astype(bool)
    out: dict[str, Any] = {}
    f1s = []
    for index, name in enumerate(LABELS):
        out[name] = _prf(predicted[:, index], actual[:, index])
        f1s.append(out[name]["f1"])
    out["macro_f1"] = float(np.mean(f1s))
    out["exact_match"] = float((predicted == actual).all(1).mean()) if len(probs) else 0.0
    out["thresholds"] = list(map(float, thresholds))
    return out


def best_threshold(probs: np.ndarray, truth: np.ndarray, index: int) -> tuple[float, float]:
    """Grid-search the F1-optimal threshold for one label."""
    best_f1, best_t = -1.0, 0.5
    actual = truth[:, index].astype(bool)
    for threshold in np.arange(0.05, 0.96, 0.01):
        stats = _prf(probs[:, index] >= threshold, actual)
        if stats["f1"] > best_f1:
            best_f1, best_t = stats["f1"], float(threshold)
    return best_t, best_f1


def tuned_thresholds(probs: np.ndarray, truth: np.ndarray) -> list[float]:
    return [best_threshold(probs, truth, index)[0] for index in range(len(LABELS))]
