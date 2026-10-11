#!/usr/bin/env python3
"""Train SituationNet on multiple games at once (nrftw + vampire + hk, ...).

Adapted from train_situationnet.py the same way train_attacknet_3games.py
adapts train_attacknet.py: one (data_root, train_jobs, val_jobs, game_name)
tuple per game, concatenated into one train/val set, with validation reported
per game in addition to pooled "all" numbers. Best-checkpoint selection uses
the worst-game selection_score (higher is worse) instead of the pooled score,
so a checkpoint that only works on one game cannot win.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from attacknet.data import select_records_by_manifest
from situationnet.data import SituationClipDataset, build_records, opencv_worker_init, target_statistics
from situationnet.losses import situation_loss
from situationnet.model import SituationNet, parameter_count


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "situationnet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game_names", nargs="+", required=True, help="e.g. nrftw vampire hk")
    parser.add_argument("--data_root", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--train_jobs", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--val_jobs", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--clip_batch_size", type=int, default=1)
    parser.add_argument("--cell_microbatch", type=int, default=10)
    parser.add_argument("--grad_accum_clips", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=14)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--temporal_pool", choices=["last", "mean"], default="last")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame_mode", choices=["short", "long"], default="short")
    parser.add_argument("--aux_sector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux_sector_weight", type=float, default=0.0)
    parser.add_argument("--distance_scale", type=float, default=10.0)
    parser.add_argument("--distance_loss_weight", type=float, default=1.0)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--norm_layer",
        choices=["batch", "frozen", "group"],
        default="frozen",
        help="'frozen' matches the currently-deployed nrftw-only recipe.",
    )
    parser.add_argument("--shuffle_cells", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_post_death", action="store_true")
    parser.add_argument("--max_train_clips", type=int, default=0)
    parser.add_argument("--max_val_clips", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means unlimited")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Metrics:
    """Pooled ("all") running metrics -- identical to train_situationnet.py's."""

    def __init__(self, distance_scale: float):
        self.distance_scale = distance_scale
        self.count = 0
        self.loss_sum = 0.0
        self.distance_loss_sum = 0.0
        self.angle_loss_sum = 0.0
        self.distance_error_sum = 0.0
        self.angle_error_sum = 0.0
        self.front_correct = 0
        self.zone_correct = 0
        self.sector_correct = 0

    @staticmethod
    def _derived(angle_radians: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degrees = angle_radians.rad2deg()
        absolute = degrees.abs()
        front = absolute <= 20.0
        behind = absolute >= 90.0
        zone = torch.where(front, 0, torch.where(behind, 2, 1))
        sector = torch.where(
            front,
            0,
            torch.where(behind, 3, torch.where(degrees > 0, 2, 1)),
        )
        return front, zone, sector

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        distance: torch.Tensor,
        angle: torch.Tensor,
        loss: float,
        parts: dict[str, torch.Tensor],
    ) -> None:
        predicted_angle = torch.atan2(outputs["angle_sincos"][:, 0], outputs["angle_sincos"][:, 1])
        target_angle = torch.atan2(angle[:, 0], angle[:, 1])
        delta = torch.atan2(
            torch.sin(predicted_angle - target_angle),
            torch.cos(predicted_angle - target_angle),
        ).abs()
        predicted_derived = self._derived(predicted_angle.detach().cpu())
        target_derived = self._derived(target_angle.detach().cpu())
        count = int(distance.shape[0])
        self.count += count
        self.loss_sum += float(loss) * count
        self.distance_loss_sum += float(parts["distance"]) * count
        self.angle_loss_sum += float(parts["angle"]) * count
        self.distance_error_sum += float(
            (outputs["distance_value"] - distance).abs().sum().detach().cpu()
        ) * self.distance_scale
        self.angle_error_sum += float(delta.rad2deg().sum().detach().cpu())
        self.front_correct += int((predicted_derived[0] == target_derived[0]).sum())
        self.zone_correct += int((predicted_derived[1] == target_derived[1]).sum())
        self.sector_correct += int((predicted_derived[2] == target_derived[2]).sum())

    def compute(self) -> dict[str, float]:
        count = max(self.count, 1)
        distance_mae = self.distance_error_sum / count
        angle_mae = self.angle_error_sum / count
        return {
            "loss": self.loss_sum / count,
            "distance_loss": self.distance_loss_sum / count,
            "angle_loss": self.angle_loss_sum / count,
            "distance_mae_units": distance_mae,
            "angle_mae_degrees": angle_mae,
            "front_cone_accuracy": self.front_correct / count,
            "decision_zone_accuracy": self.zone_correct / count,
            "tactical_sector_accuracy": self.sector_correct / count,
            "selection_score": distance_mae / self.distance_scale + angle_mae / 180.0,
        }


def _derived_np(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    absolute = np.abs(degrees)
    front = absolute <= 20.0
    behind = absolute >= 90.0
    zone = np.where(front, 0, np.where(behind, 2, 1))
    sector = np.where(front, 0, np.where(behind, 3, np.where(degrees > 0, 2, 1)))
    return front, zone, sector


def geometry_metrics_np(
    predicted_angle: np.ndarray,
    target_angle: np.ndarray,
    predicted_distance: np.ndarray,
    target_distance: np.ndarray,
    distance_scale: float,
) -> dict[str, float]:
    """Same arithmetic as Metrics, vectorised over a numpy subset (e.g. one game)."""
    count = max(len(predicted_angle), 1)
    delta = np.abs(np.arctan2(np.sin(predicted_angle - target_angle), np.cos(predicted_angle - target_angle)))
    predicted_derived = _derived_np(np.degrees(predicted_angle))
    target_derived = _derived_np(np.degrees(target_angle))
    distance_mae = float(np.abs(predicted_distance - target_distance).sum()) * distance_scale / count
    angle_mae = float(np.degrees(delta).sum()) / count
    return {
        "distance_mae_units": distance_mae,
        "angle_mae_degrees": angle_mae,
        "front_cone_accuracy": float((predicted_derived[0] == target_derived[0]).sum()) / count,
        "decision_zone_accuracy": float((predicted_derived[1] == target_derived[1]).sum()) / count,
        "tactical_sector_accuracy": float((predicted_derived[2] == target_derived[2]).sum()) / count,
        "selection_score": distance_mae / distance_scale + angle_mae / 180.0,
        "n_cells": int(len(predicted_angle)),
    }


def game_report(
    predicted_angle: np.ndarray,
    target_angle: np.ndarray,
    predicted_distance: np.ndarray,
    target_distance: np.ndarray,
    games_arr: np.ndarray,
    games: list[str],
    distance_scale: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "all": geometry_metrics_np(predicted_angle, target_angle, predicted_distance, target_distance, distance_scale)
    }
    for game in games:
        mask = games_arr == game
        if mask.any():
            out[game] = geometry_metrics_np(
                predicted_angle[mask], target_angle[mask], predicted_distance[mask], target_distance[mask], distance_scale
            )
    game_scores = [out[name]["selection_score"] for name in games if name in out]
    # Higher selection_score is worse, so the worst game is the max.
    out["worst_game_selection_score"] = float(max(game_scores)) if game_scores else float("inf")
    out["mean_game_selection_score"] = float(np.mean(game_scores)) if game_scores else float("inf")
    return out


def game_compact_line(report: dict[str, Any], games: list[str]) -> str:
    parts = [f"all={report['all']['selection_score']:.4f}"]
    for game in games:
        if game in report:
            parts.append(f"{game}={report[game]['selection_score']:.4f}")
    parts.append(f"worst={report['worst_game_selection_score']:.4f}")
    return " ".join(parts)


def flatten_valid(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = batch["frames"].flatten(0, 1)
    distance = batch["distance"].flatten(0, 1)
    angle = batch["angle_sincos"].flatten(0, 1)
    valid = batch["valid"].flatten(0, 1)
    return frames[valid], distance[valid], angle[valid]


def make_loader(dataset: SituationClipDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.clip_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=opencv_worker_init,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: SituationNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    game_of: dict[str, str],
    games: list[str],
) -> dict[str, float]:
    model.eval()
    metrics = Metrics(args.distance_scale)
    amp_enabled = args.amp and device.type == "cuda"
    pred_angle_chunks: list[np.ndarray] = []
    target_angle_chunks: list[np.ndarray] = []
    pred_distance_chunks: list[np.ndarray] = []
    target_distance_chunks: list[np.ndarray] = []
    cell_games: list[str] = []
    for batch in loader:
        frames, distance, angle = flatten_valid(batch)
        if not len(distance):
            continue
        for start in range(0, len(distance), args.cell_microbatch):
            x = frames[start : start + args.cell_microbatch].to(device, non_blocking=True)
            d = distance[start : start + args.cell_microbatch].to(device, non_blocking=True)
            a = angle[start : start + args.cell_microbatch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(x)
                loss, parts = situation_loss(
                    outputs, d, a, args.distance_loss_weight, args.angle_loss_weight, args.aux_sector_weight
                )
            metrics.update(outputs, d, a, float(loss), parts)
            predicted_angle = torch.atan2(outputs["angle_sincos"][:, 0], outputs["angle_sincos"][:, 1])
            target_angle = torch.atan2(a[:, 0], a[:, 1])
            pred_angle_chunks.append(predicted_angle.detach().float().cpu().numpy())
            target_angle_chunks.append(target_angle.detach().float().cpu().numpy())
            pred_distance_chunks.append(outputs["distance_value"].detach().float().cpu().numpy())
            target_distance_chunks.append(d.detach().float().cpu().numpy())
        for clip_path, valid_flags in zip(batch["clip_path"], batch["valid"]):
            game = game_of.get(str(Path(clip_path).resolve()), "UNKNOWN")
            cell_games.extend([game] * int(valid_flags.sum()))

    report = metrics.compute()
    if pred_angle_chunks:
        pred_angle = np.concatenate(pred_angle_chunks)
        target_angle = np.concatenate(target_angle_chunks)
        pred_distance = np.concatenate(pred_distance_chunks)
        target_distance = np.concatenate(target_distance_chunks)
        games_arr = (
            np.array(cell_games) if len(cell_games) == len(pred_angle) else np.array(["UNKNOWN"] * len(pred_angle))
        )
        report["by_game"] = game_report(
            pred_angle, target_angle, pred_distance, target_distance, games_arr, games, args.distance_scale
        )
    return report


def save_checkpoint(
    path: Path,
    model: SituationNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "nrftw-situationnet-v1-distance-angle-only",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
            "output_names": list(SituationNet.output_names),
            "geometry_config": {
                "distance_scale": args.distance_scale,
                "distance_edges": [2.0, 4.0, 6.0],
                "front_cone_degrees": 20.0,
                "angle_value_unit": "radians",
                "angle_frame": "player bearing relative to boss forward; positive is right",
            },
            "derived_geometry_fields": [
                "distance_value",
                "distance_bin",
                "dp_value",
                "dp_bin",
                "angle_value",
                "angle_bin",
                "relative_side",
                "front_cone",
                "decision_zone",
                "tactical_sector",
                "behind",
            ],
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.distance_scale <= 0:
        raise ValueError("--distance_scale must be positive")
    if not (len(args.game_names) == len(args.data_root) == len(args.train_jobs) == len(args.val_jobs)):
        raise ValueError("--game_names/--data_root/--train_jobs/--val_jobs must have the same length")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", vars(args))

    print(f"[data] reading geometry targets for games: {args.game_names}", flush=True)
    train_records = []
    val_records = []
    val_game_of: dict[str, str] = {}
    per_game_stats = []
    for game, root, tjobs, vjobs in zip(args.game_names, args.data_root, args.train_jobs, args.val_jobs):
        records = build_records(root, args.distance_scale, args.include_post_death)
        game_train = select_records_by_manifest(records, root, tjobs)
        game_val = select_records_by_manifest(records, root, vjobs)
        train_records.extend(game_train)
        val_records.extend(game_val)
        for record in game_val:
            val_game_of[str(Path(record.clip_path).resolve())] = game
        per_game_stats.append(
            {
                "game": game,
                "data_root": str(Path(root).resolve()),
                "train_jobs": str(Path(tjobs).resolve()),
                "val_jobs": str(Path(vjobs).resolve()),
                "all_clips": len(records),
                "train_clips": len(game_train),
                "val_clips": len(game_val),
            }
        )
        print(
            f"[data] {game}: all={len(records)} train={len(game_train)} val={len(game_val)}",
            flush=True,
        )

    if args.max_train_clips:
        train_records = train_records[: args.max_train_clips]
    if args.max_val_clips:
        val_records = val_records[: args.max_val_clips]
    if not train_records or not val_records:
        raise RuntimeError("train or validation subset is empty")
    split_summary = {
        "games": args.game_names,
        "per_game": per_game_stats,
        "all_clips": len(train_records) + len(val_records),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "split_unit": "explicit_train_and_val_manifests_per_game",
        "excluded_post_death": not args.include_post_death,
        "distance_scale": args.distance_scale,
        "train": target_statistics(train_records, args.distance_scale),
        "val": target_statistics(val_records, args.distance_scale),
    }
    atomic_json(output_dir / "split_summary.json", split_summary)
    print(json.dumps(split_summary, ensure_ascii=False), flush=True)

    train_data = SituationClipDataset(
        train_records,
        image_hw=(args.height, args.width),
        augment=args.augment,
        frame_mode=args.frame_mode,
        cache_dir=None,
        data_root=None,
    )
    val_data = SituationClipDataset(
        val_records,
        image_hw=(args.height, args.width),
        augment=False,
        frame_mode=args.frame_mode,
        cache_dir=None,
        data_root=None,
    )
    train_loader = make_loader(train_data, args, train=True)
    val_loader = make_loader(val_data, args, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SituationNet(
        args.hidden,
        args.pretrained,
        args.dropout,
        backbone=args.backbone,
        temporal_pool=args.temporal_pool,
        bidirectional=args.bidirectional,
        aux_sector=args.aux_sector,
        norm_layer=args.norm_layer,
    ).to(device)
    print(
        f"[model] SituationNet {args.backbone}+GRU norm={args.norm_layer} params={parameter_count(model):,} "
        f"heads={SituationNet.output_names} device={device}",
        flush=True,
    )
    backbone_params = list(model.frame_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_params}
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_clips)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = int(total_updates * args.warmup_fraction)

    def lr_factor(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(step, 1) / warmup_updates
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = args.amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_score = float("inf")
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = Metrics(args.distance_scale)
        epoch_start = time.time()
        for clip_step, batch in enumerate(train_loader, start=1):
            frames, distance, angle = flatten_valid(batch)
            if not len(distance):
                continue
            if args.shuffle_cells:
                order = torch.randperm(len(distance))
                frames, distance, angle = frames[order], distance[order], angle[order]
            micro_count = math.ceil(len(distance) / args.cell_microbatch)
            for start in range(0, len(distance), args.cell_microbatch):
                x = frames[start : start + args.cell_microbatch].to(device, non_blocking=True)
                d = distance[start : start + args.cell_microbatch].to(device, non_blocking=True)
                a = angle[start : start + args.cell_microbatch].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    outputs = model(x)
                    raw_loss, parts = situation_loss(
                        outputs, d, a, args.distance_loss_weight, args.angle_loss_weight, args.aux_sector_weight
                    )
                    loss = raw_loss / (micro_count * args.grad_accum_clips)
                loss.backward()
                metrics.update(outputs, d, a, float(raw_loss.detach()), parts)

            should_update = clip_step % args.grad_accum_clips == 0 or clip_step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduler.step()
                if global_step % 25 == 0 or global_step == 1:
                    current = metrics.compute()
                    print(
                        f"[train] epoch={epoch} step={global_step}/{total_updates} "
                        f"loss={current['loss']:.4f} "
                        f"distance_mae={current['distance_mae_units']:.3f} "
                        f"angle_mae={current['angle_mae_degrees']:.1f}deg "
                        f"lr={optimizer.param_groups[-1]['lr']:.2e}",
                        flush=True,
                    )
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        train_metrics = metrics.compute()
        val_metrics = evaluate(model, val_loader, device, args, val_game_of, args.game_names)
        report = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": round(time.time() - epoch_start, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        print("[epoch] " + json.dumps(report, ensure_ascii=False), flush=True)
        if "by_game" in val_metrics:
            print(f"[val] epoch={epoch} | " + game_compact_line(val_metrics["by_game"], args.game_names), flush=True)
        atomic_json(output_dir / "metrics_last.json", report)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, global_step, args, report)
        selection = (
            val_metrics["by_game"]["worst_game_selection_score"]
            if "by_game" in val_metrics
            else val_metrics["selection_score"]
        )
        if selection < best_score:
            best_score = selection
            save_checkpoint(output_dir / "situationnet.pt", model, optimizer, epoch, global_step, args, report)
        if stop:
            break

    print(f"[done] best_val_worst_game_selection_score={best_score:.5f} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
