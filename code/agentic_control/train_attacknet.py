#!/usr/bin/env python3
"""Train AttackNet on multiple games at once (nrftw + vrising + hk, ...).

Adapted from train_attacknet.py: instead of a single --data_root, this accepts
one (data_root, train_jobs, val_jobs, game_name) tuple per game, concatenates
their clip records into one training/validation set, and reports validation
metrics broken down per game (in addition to the pooled "all" numbers) so a
checkpoint that is only good at one game cannot silently win. Best-checkpoint
selection uses the worst-game macro-F1, mirroring the worst-location selection
already used for the single-game (nrftw) 2-arena training script.
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

from attacknet.data import (
    AttackClipDataset,
    build_records,
    label_statistics,
    opencv_worker_init,
    positive_weights,
    select_records_by_manifest,
)
from attacknet.losses import balanced_multilabel_bce, damage_regression_loss, focal_multilabel_bce
from attacknet.metrics import metrics_at, tuned_thresholds
from attacknet.model import AttackNet, parameter_count


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "attacknet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game_names", nargs="+", required=True, help="e.g. nrftw vrising hk")
    parser.add_argument("--data_root", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--train_jobs", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--val_jobs", nargs="+", required=True, help="one per game, same order")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--clip_batch_size", type=int, default=1)
    parser.add_argument("--cell_microbatch", type=int, default=10)
    parser.add_argument("--grad_accum_clips", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--temporal_pool", choices=["last", "mean"], default="last")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame_mode", choices=["short", "long"], default="short")
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--aux_damage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aux_damage_weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--damage_epsilon", type=float, default=1e-3)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augment_recipe",
        choices=["baseline", "medium", "strong", "extreme"],
        default="baseline",
    )
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--norm_layer",
        choices=["batch", "frozen", "group"],
        default="frozen",
        help="'frozen' matches the currently-deployed nrftw-only recipe (fixes the "
        "train-vs-eval BatchNorm statistics mismatch at batch=1 deploy time).",
    )
    parser.add_argument("--input_norm", choices=["imagenet", "per_cell"], default="imagenet")
    parser.add_argument("--shuffle_cells", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
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
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.loss_sum = 0.0
        self.count = 0
        self.tp = torch.zeros(2, dtype=torch.long)
        self.fp = torch.zeros(2, dtype=torch.long)
        self.fn = torch.zeros(2, dtype=torch.long)
        self.exact = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        predictions = logits.detach().sigmoid().cpu() >= self.threshold
        truth = targets.detach().cpu().bool()
        self.tp += (predictions & truth).sum(0)
        self.fp += (predictions & ~truth).sum(0)
        self.fn += (~predictions & truth).sum(0)
        self.exact += int((predictions == truth).all(1).sum())
        self.count += int(truth.shape[0])
        self.loss_sum += float(loss) * int(truth.shape[0])

    def compute(self) -> dict[str, Any]:
        eps = 1e-9
        precision = self.tp.float() / (self.tp + self.fp).float().clamp_min(eps)
        recall = self.tp.float() / (self.tp + self.fn).float().clamp_min(eps)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(eps)
        names = AttackNet.output_names
        return {
            "loss": self.loss_sum / max(self.count, 1),
            "exact_match": self.exact / max(self.count, 1),
            "macro_f1": float(f1.mean()),
            "per_label": {
                names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "tp": int(self.tp[i]),
                    "fp": int(self.fp[i]),
                    "fn": int(self.fn[i]),
                }
                for i in range(2)
            },
        }


def game_report(
    probs: np.ndarray,
    truth: np.ndarray,
    games_arr: np.ndarray,
    threshold: float,
    games: list[str],
) -> dict[str, Any]:
    """Per-game report at fixed and tuned thresholds, generalised to N games."""
    slices: dict[str, np.ndarray] = {"all": np.ones(len(probs), dtype=bool)}
    for game in games:
        mask = games_arr == game
        if mask.any():
            slices[game] = mask
    global_thresholds = tuned_thresholds(probs, truth) if len(probs) else [0.5, 0.5]
    out: dict[str, Any] = {"n_cells": int(len(probs)), "tuned_thresholds": global_thresholds}
    for name, mask in slices.items():
        out[name] = {
            "n_cells": int(mask.sum()),
            "positive_rate": truth[mask].mean(0).tolist() if mask.any() else [0.0, 0.0],
            "fixed": metrics_at(probs[mask], truth[mask], threshold),
            "tuned": metrics_at(probs[mask], truth[mask], global_thresholds),
        }
    game_scores = [out[name]["tuned"]["macro_f1"] for name in games if name in out]
    out["worst_game_macro_f1"] = float(min(game_scores)) if game_scores else 0.0
    out["mean_game_macro_f1"] = float(np.mean(game_scores)) if game_scores else 0.0
    return out


def game_compact_line(report: dict[str, Any], games: list[str]) -> str:
    parts = [f"all={report['all']['tuned']['macro_f1']:.4f}"]
    for game in games:
        if game in report:
            parts.append(f"{game}={report[game]['tuned']['macro_f1']:.4f}")
    parts.append(f"worst={report['worst_game_macro_f1']:.4f}")
    parts.append(f"thr={[round(t, 2) for t in report['tuned_thresholds']]}")
    return " ".join(parts)


def flatten_valid(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = batch["frames"].flatten(0, 1)
    labels = batch["labels"].flatten(0, 1)
    damage = batch["damage"].flatten(0, 1)
    valid = batch["valid"].flatten(0, 1)
    return frames[valid], labels[valid], damage[valid]


def compute_loss(
    output: torch.Tensor | dict[str, torch.Tensor],
    labels: torch.Tensor,
    damage: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = output["logits"] if isinstance(output, dict) else output
    if args.loss_type == "focal":
        loss = focal_multilabel_bce(logits, labels, pos_weight, args.focal_gamma)
    else:
        loss = balanced_multilabel_bce(logits, labels, pos_weight)
    if isinstance(output, dict) and args.aux_damage_weight > 0.0:
        loss = loss + args.aux_damage_weight * damage_regression_loss(output["damage"], damage)
    return loss, logits


def make_loader(dataset: AttackClipDataset, args: argparse.Namespace, train: bool) -> DataLoader:
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
    model: AttackNet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    game_of: dict[str, str],
    games: list[str],
) -> dict[str, Any]:
    model.eval()
    metrics = Metrics(args.threshold)
    amp_enabled = args.amp and device.type == "cuda"
    probs_chunks: list[np.ndarray] = []
    truth_chunks: list[np.ndarray] = []
    cell_games: list[str] = []
    for batch in loader:
        frames, labels, damage = flatten_valid(batch)
        if not len(labels):
            continue
        for start in range(0, len(labels), args.cell_microbatch):
            x = frames[start : start + args.cell_microbatch].to(device, non_blocking=True)
            y = labels[start : start + args.cell_microbatch].to(device, non_blocking=True)
            dmg = damage[start : start + args.cell_microbatch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(x)
                loss, logits = compute_loss(output, y, dmg, pos_weight, args)
            metrics.update(logits, y, float(loss))
            probs_chunks.append(logits.detach().float().sigmoid().cpu().numpy())
        truth_chunks.append(labels.numpy())
        for clip_path, valid_flags in zip(batch["clip_path"], batch["valid"]):
            game = game_of.get(str(Path(clip_path).resolve()), "UNKNOWN")
            cell_games.extend([game] * int(valid_flags.sum()))

    report = metrics.compute()
    if probs_chunks:
        probs = np.concatenate(probs_chunks)
        truth = np.concatenate(truth_chunks)
        games_arr = (
            np.array(cell_games) if len(cell_games) == len(probs) else np.array(["UNKNOWN"] * len(probs))
        )
        report["by_game"] = game_report(probs, truth, games_arr, args.threshold, games)
    return report


def save_checkpoint(
    path: Path,
    model: AttackNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
    pos_weight: torch.Tensor,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
            "output_names": list(AttackNet.output_names),
            "pos_weight": pos_weight.detach().cpu(),
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not (len(args.game_names) == len(args.data_root) == len(args.train_jobs) == len(args.val_jobs)):
        raise ValueError("--game_names/--data_root/--train_jobs/--val_jobs must have the same length")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", vars(args))

    print(f"[data] reading state sidecars for games: {args.game_names}", flush=True)
    train_records = []
    val_records = []
    val_game_of: dict[str, str] = {}
    per_game_stats = []
    for game, root, tjobs, vjobs in zip(args.game_names, args.data_root, args.train_jobs, args.val_jobs):
        records = build_records(root, args.damage_epsilon)
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
    train_stats = label_statistics(train_records)
    val_stats = label_statistics(val_records)
    pos_weight = positive_weights(train_records, args.max_pos_weight)
    split_summary = {
        "games": args.game_names,
        "per_game": per_game_stats,
        "all_clips": len(train_records) + len(val_records),
        "train_clips": len(train_records),
        "val_clips": len(val_records),
        "split_unit": "explicit_train_and_val_manifests_per_game",
        "train": train_stats,
        "val": val_stats,
        "pos_weight_player_boss": pos_weight.tolist(),
    }
    atomic_json(output_dir / "split_summary.json", split_summary)
    print(json.dumps(split_summary, ensure_ascii=False), flush=True)

    train_data = AttackClipDataset(
        train_records,
        image_hw=(args.height, args.width),
        augment=args.augment,
        frame_mode=args.frame_mode,
        augment_recipe=args.augment_recipe,
        grayscale=args.grayscale,
        cache_dir=None,
        data_root=None,
    )
    val_data = AttackClipDataset(
        val_records,
        image_hw=(args.height, args.width),
        augment=False,
        frame_mode=args.frame_mode,
        grayscale=args.grayscale,
        cache_dir=None,
        data_root=None,
    )
    train_loader = make_loader(train_data, args, train=True)
    val_loader = make_loader(val_data, args, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttackNet(
        args.hidden,
        args.pretrained,
        args.dropout,
        backbone=args.backbone,
        temporal_pool=args.temporal_pool,
        bidirectional=args.bidirectional,
        aux_damage=args.aux_damage,
        norm_layer=args.norm_layer,
        input_norm=args.input_norm,
    ).to(device)
    print(
        f"[model] AttackNet {args.backbone}+GRU norm={args.norm_layer} input_norm={args.input_norm} "
        f"params={parameter_count(model):,} device={device}",
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
    pos_weight = pos_weight.to(device)
    amp_enabled = args.amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_f1 = -1.0
    stop = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = Metrics(args.threshold)
        epoch_start = time.time()
        for clip_step, batch in enumerate(train_loader, start=1):
            frames, labels, damage = flatten_valid(batch)
            if not len(labels):
                continue
            if args.shuffle_cells:
                order = torch.randperm(len(labels))
                frames, labels, damage = frames[order], labels[order], damage[order]
            micro_count = math.ceil(len(labels) / args.cell_microbatch)
            for start in range(0, len(labels), args.cell_microbatch):
                x = frames[start : start + args.cell_microbatch].to(device, non_blocking=True)
                y = labels[start : start + args.cell_microbatch].to(device, non_blocking=True)
                dmg = damage[start : start + args.cell_microbatch].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    output = model(x)
                    raw_loss, logits = compute_loss(output, y, dmg, pos_weight, args)
                    loss = raw_loss / (micro_count * args.grad_accum_clips)
                loss.backward()
                metrics.update(logits, y, float(raw_loss.detach()))

            should_update = clip_step % args.grad_accum_clips == 0 or clip_step == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduler.step()
                if global_step % 25 == 0 or global_step == 1:
                    print(
                        f"[train] epoch={epoch} step={global_step}/{total_updates} "
                        f"loss={metrics.compute()['loss']:.4f} "
                        f"lr={optimizer.param_groups[-1]['lr']:.2e}",
                        flush=True,
                    )
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        train_metrics = metrics.compute()
        val_metrics = evaluate(model, val_loader, device, pos_weight, args, val_game_of, args.game_names)
        report = {
            "epoch": epoch,
            "global_step": global_step,
            "seconds": round(time.time() - epoch_start, 2),
            "train": train_metrics,
            "val": val_metrics,
        }
        print("[epoch] " + json.dumps(report, ensure_ascii=False), flush=True)
        if "by_game" in val_metrics:
            print(
                f"[val] epoch={epoch} fixed@{args.threshold}={val_metrics['macro_f1']:.4f} | "
                + game_compact_line(val_metrics["by_game"], args.game_names),
                flush=True,
            )
        atomic_json(output_dir / "metrics_last.json", report)
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch, global_step, args, pos_weight, report
        )
        selection = (
            val_metrics["by_game"]["worst_game_macro_f1"]
            if "by_game" in val_metrics
            else val_metrics["macro_f1"]
        )
        if selection > best_f1:
            best_f1 = selection
            save_checkpoint(
                output_dir / "attacknet.pt", model, optimizer, epoch, global_step, args, pos_weight, report
            )
        if stop:
            break

    print(f"[done] best_val_worst_game_macro_f1={best_f1:.4f} -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
