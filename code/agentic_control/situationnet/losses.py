"""Losses for the SituationNet geometry heads (plus an optional auxiliary sector head)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sector_index_from_sincos(angle_sincos: torch.Tensor) -> torch.Tensor:
    """0=front, 1=left_flank, 2=right_flank, 3=behind (matches train-script Metrics)."""
    degrees = torch.atan2(angle_sincos[:, 0], angle_sincos[:, 1]).rad2deg()
    absolute = degrees.abs()
    front = absolute <= 20.0
    behind = absolute >= 90.0
    return torch.where(front, 0, torch.where(behind, 3, torch.where(degrees > 0, 2, 1)))


def situation_loss(
    outputs: dict[str, torch.Tensor],
    distance_target: torch.Tensor,
    angle_target: torch.Tensor,
    distance_weight: float = 1.0,
    angle_weight: float = 1.0,
    sector_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    distance = F.smooth_l1_loss(outputs["distance_value"], distance_target)
    # Both vectors are unit normalized, making this circular and wrap-safe.
    angle = F.mse_loss(outputs["angle_sincos"], angle_target)
    total = distance_weight * distance + angle_weight * angle
    parts = {"distance": distance.detach(), "angle": angle.detach()}
    if "sector_logits" in outputs and sector_weight > 0.0:
        sector_target = sector_index_from_sincos(angle_target)
        sector = F.cross_entropy(outputs["sector_logits"], sector_target)
        total = total + sector_weight * sector
        parts["sector"] = sector.detach()
    return total, parts
