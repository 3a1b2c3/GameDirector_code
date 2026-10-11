"""Losses for independent player-hit and boss-hit labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced BCE over two independent labels.

    Softmax cross-entropy is intentionally not used: neither-hit and both-hit are valid.
    ``pos_weight = negatives / positives`` compensates for the much rarer player-hit label.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-balanced focal loss (Lin et al. 2017) over the two independent labels.

    Down-weights already-easy examples (both classes are >85% correctly classified by
    the plain-BCE baseline) so gradient mass concentrates on the remaining hard cases,
    on top of the same pos_weight class balancing as ``balanced_multilabel_bce``.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = logits.sigmoid()
    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    focal = bce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
    return focal.mean()


def damage_regression_loss(damage_pred: torch.Tensor, damage_target: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 auxiliary loss on continuous (player_damage, boss_damage)."""
    return F.smooth_l1_loss(damage_pred, damage_target)
