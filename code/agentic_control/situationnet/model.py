"""Minimal LLM-facing SituationNet: distance plus relative orientation."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SituationNet(nn.Module):
    """Estimate geometry from the last three frames of a 0.25-second cell.

    The network intentionally has only two supervised outputs:
    ``distance_value`` (normalized horizontal distance) and ``angle_sincos``.
    Every categorical field consumed by the LLM is deterministically derived
    from these values, avoiding contradictory redundant heads.
    """

    output_names = ("distance_value", "angle_sincos")

    def __init__(
        self,
        hidden: int = 384,
        pretrained: bool = False,
        dropout: float = 0.1,
        backbone: str = "resnet18",
        temporal_pool: str = "last",
        bidirectional: bool = False,
        aux_sector: bool = False,
        norm_layer: str = "batch",
    ):
        super().__init__()
        import torchvision.models as tvm

        from attacknet.model import _replace_batchnorm_with_groupnorm

        if backbone not in ("resnet18", "resnet34"):
            raise ValueError(f"unsupported backbone: {backbone!r}")
        if temporal_pool not in ("last", "mean"):
            raise ValueError(f"unsupported temporal_pool: {temporal_pool!r}")
        if norm_layer not in ("batch", "frozen", "group"):
            raise ValueError(f"unsupported norm_layer: {norm_layer!r}")
        weights_cls = tvm.ResNet18_Weights if backbone == "resnet18" else tvm.ResNet34_Weights
        weights = weights_cls.DEFAULT if pretrained else None
        backbone_net = getattr(tvm, backbone)(weights=weights)
        self.frame_encoder = nn.Sequential(*list(backbone_net.children())[:-1])
        self.norm_layer = norm_layer
        if norm_layer == "group":
            _replace_batchnorm_with_groupnorm(self.frame_encoder)
        frame_dim = int(backbone_net.fc.in_features)
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, hidden),
            nn.GELU(),
        )
        self.temporal_pool = temporal_pool
        self.bidirectional = bidirectional
        self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=bidirectional)
        temporal_out = hidden * (2 if bidirectional else 1)
        self.fuse = nn.Sequential(
            nn.LayerNorm(temporal_out),
            nn.Linear(temporal_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.distance_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.angle_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        self.aux_sector = aux_sector
        if aux_sector:
            # Direct 4-way front/left_flank/right_flank/behind classification head,
            # supervised jointly with the regression heads (see losses.situation_loss).
            self.sector_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 4),
            )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "SituationNet":
        """Keep frozen-BN backbones in inference mode so train and eval agree.

        A training microbatch holds ten cells of a single clip, so plain BatchNorm
        normalises that clip's appearance away and the running statistics it
        accumulates describe no real clip. Evaluating the shipped checkpoint with
        batch statistics instead halves both distance and angle error, which is how
        much this costs.
        """
        super().train(mode)
        if mode and self.norm_layer == "frozen":
            for module in self.frame_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (3, 3):
            raise ValueError(f"expected [B,3,3,H,W], got {tuple(rgb.shape)}")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        else:
            rgb = rgb.float()
        rgb = (rgb - self.image_mean) / self.image_std
        batch, frames = rgb.shape[:2]
        features = self.frame_encoder(rgb.reshape(batch * frames, *rgb.shape[2:])).flatten(1)
        features = self.frame_proj(features).reshape(batch, frames, -1)
        temporal, _ = self.temporal(features)
        pooled = temporal.mean(dim=1) if self.temporal_pool == "mean" else temporal[:, -1]
        fused = self.fuse(pooled)
        outputs = {
            "distance_value": self.distance_head(fused).squeeze(-1),
            "angle_sincos": F.normalize(self.angle_head(fused), dim=-1, eps=1e-6),
        }
        if self.aux_sector:
            outputs["sector_logits"] = self.sector_head(fused)
        return outputs


def _distance_bin(distance: float, edges: Sequence[float]) -> str:
    if len(edges) != 3 or not (edges[0] < edges[1] < edges[2]):
        raise ValueError("distance_edges must contain three increasing values")
    if distance < edges[0]:
        return "0-2"
    if distance < edges[1]:
        return "2-4"
    if distance < edges[2]:
        return "4-6"
    return "6+"


def geometry_from_outputs(
    outputs: dict[str, torch.Tensor],
    distance_scale: float = 10.0,
    distance_edges: Sequence[float] = (2.0, 4.0, 6.0),
    front_cone_degrees: float = 20.0,
) -> list[dict[str, Any]]:
    """Convert the two heads into the geometry dictionary consumed by the LLM planner."""
    distances = outputs["distance_value"].detach().float().clamp_min(0) * distance_scale
    angles = outputs["angle_sincos"].detach().float()
    if distances.ndim != 1 or angles.shape != (len(distances), 2):
        raise ValueError("outputs must have distance [B] and angle_sincos [B,2]")
    result: list[dict[str, Any]] = []
    for distance_tensor, angle_tensor in zip(distances.cpu(), angles.cpu()):
        distance = float(distance_tensor)
        angle = math.atan2(float(angle_tensor[0]), float(angle_tensor[1]))
        degrees = math.degrees(angle)
        front = abs(degrees) <= front_cone_degrees
        behind = abs(degrees) >= 90.0
        zone = "front" if front else "behind" if behind else "side"
        sector = (
            "front"
            if front
            else "behind"
            if behind
            else "right_flank"
            if degrees > 0
            else "left_flank"
        )
        # Coarse left/right/center angle bin derived from the regression output.
        # Deliberately distinct from the wider 20-degree front-cone decision zone.
        side = "left" if degrees < -0.5 else "right" if degrees > 0.5 else "center"
        distance_bin = _distance_bin(distance, distance_edges)
        result.append(
            {
                "distance_value": round(distance, 4),
                "distance_bin": distance_bin,
                # dp_* aliases preserve the visual-understanding -> action-planning handoff schema.
                "dp_value": round(distance, 4),
                "dp_bin": distance_bin,
                "angle_value": round(angle, 6),
                "angle_bin": side,
                "relative_side": side,
                "front_cone": front,
                "decision_zone": zone,
                "tactical_sector": sector,
                "behind": behind,
            }
        )
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
