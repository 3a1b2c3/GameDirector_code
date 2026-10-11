"""Three-frame RGB attack detector, following RGBSituationNet's CNN+GRU design."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttackNet(nn.Module):
    """Predict independent player-hit and boss-hit logits from three RGB frames.

    Input:  ``[B, 3, 3, H, W]`` uint8 in [0, 255] or float in [0, 1].
    Output: ``[B, 2]`` logits ordered as ``[player_hit, boss_hit]``.
    """

    output_names = ("player_hit", "boss_hit")

    def __init__(
        self,
        hidden: int = 384,
        pretrained: bool = False,
        dropout: float = 0.1,
        backbone: str = "resnet18",
        temporal_pool: str = "last",
        bidirectional: bool = False,
        aux_damage: bool = False,
        norm_layer: str = "batch",
        input_norm: str = "imagenet",
    ):
        super().__init__()
        import torchvision.models as tvm

        if backbone not in ("resnet18", "resnet34"):
            raise ValueError(f"unsupported backbone: {backbone!r}")
        if temporal_pool not in ("last", "mean"):
            raise ValueError(f"unsupported temporal_pool: {temporal_pool!r}")
        if norm_layer not in ("batch", "frozen", "group"):
            raise ValueError(f"unsupported norm_layer: {norm_layer!r}")
        if input_norm not in ("imagenet", "per_cell"):
            raise ValueError(f"unsupported input_norm: {input_norm!r}")
        self.input_norm = input_norm
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
        # Two independent logits are required because [1, 1] is a valid target.
        self.attack_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        self.aux_damage = aux_damage
        if aux_damage:
            # Auxiliary continuous (player_damage, boss_damage) regression head: richer
            # gradient than the binary label alone (a graze and a full combo both -> 1).
            self.damage_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 2),
                nn.Softplus(),
            )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (3, 3):
            raise ValueError(f"expected [B,3,3,H,W], got {tuple(rgb.shape)}")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        else:
            rgb = rgb.float()
        if self.input_norm == "per_cell":
            # Standardise each example against its own three frames. This makes
            # explicit -- and computable at batch size 1, which is how the closed-loop
            # runner calls the model -- the per-clip appearance normalisation that
            # BatchNorm was doing implicitly from single-clip microbatches. Statistics
            # are shared across the three frames, so a hit flash still shows up as a
            # relative change between them.
            mean = rgb.mean(dim=(1, 3, 4), keepdim=True)
            std = rgb.std(dim=(1, 3, 4), keepdim=True).clamp_min(1e-5)
            rgb = (rgb - mean) / std
        else:
            rgb = (rgb - self.image_mean) / self.image_std
        b, t = rgb.shape[:2]
        features = self.frame_encoder(rgb.reshape(b * t, *rgb.shape[2:])).flatten(1)
        features = self.frame_proj(features).reshape(b, t, -1)
        temporal, _ = self.temporal(features)
        pooled = temporal.mean(dim=1) if self.temporal_pool == "mean" else temporal[:, -1]
        fused = self.fuse(pooled)
        logits = self.attack_head(fused)
        if not self.aux_damage:
            return logits
        return {"logits": logits, "damage": self.damage_head(fused)}

    def train(self, mode: bool = True) -> "AttackNet":
        """Keep frozen-BN backbones in inference mode so train and eval agree.

        A training microbatch holds ten cells of a *single* clip, so ordinary
        BatchNorm normalises away that clip's palette and exposure. The model then
        depends on statistics no running average can reproduce, and evaluation with
        running stats collapses -- hardest on Temple Arena, whose warm/bright palette
        sits furthest from the snow-dominated global mean. ``norm_layer='frozen'``
        removes the discrepancy by always using the (ImageNet) running statistics.
        """
        super().train(mode)
        if mode and self.norm_layer == "frozen":
            for module in self.frame_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def _replace_batchnorm_with_groupnorm(module: nn.Module, max_groups: int = 32) -> None:
    """Swap every BatchNorm2d for GroupNorm, which has no train/eval discrepancy."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = max_groups
            while groups > 1 and channels % groups != 0:
                groups //= 2
            replacement = nn.GroupNorm(groups, channels)
            # Carry over the pretrained affine transform; it is still meaningful.
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _replace_batchnorm_with_groupnorm(child, max_groups)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
