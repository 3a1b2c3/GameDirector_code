"""Real shared perception and world-model adapters; production exposes no mocks."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .skills import DISTANCE_UNIT


CLOSED_LOOP_ROOT = Path(__file__).resolve().parent
AGENTIC_CONTROL_ROOT = CLOSED_LOOP_ROOT.parent  # .../agentic_control
GAMEDIRECTOR_ROOT = AGENTIC_CONTROL_ROOT.parent
WORLD_MODEL_ROOT = GAMEDIRECTOR_ROOT / "world_model"
DISTILL_WORLDMODEL_ROOT = GAMEDIRECTOR_ROOT / "distill"


def choose_world_model_checkpoint(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"world-model checkpoint path not found: {candidate}")
    checkpoints = list(candidate.glob("step-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"no step-*.safetensors under {candidate}")

    def step(item: Path) -> int:
        match = re.fullmatch(r"step-(\d+)\.safetensors", item.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=step)


def _rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    value = np.asarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected uint8 RGB frame, got {value.shape} {value.dtype}")
    return value


class WorldModelRunner:
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        base_model: str | Path,
        device: str = "cuda",
        steps: int = 30,
        cfg: float = 5.0,
    ):
        import torch

        if str(WORLD_MODEL_ROOT) not in sys.path:
            sys.path.insert(0, str(WORLD_MODEL_ROOT))
        from world_model import WorldModelPipeline

        self.checkpoint = choose_world_model_checkpoint(checkpoint)
        self.pipeline = WorldModelPipeline.from_pretrained(
            str(Path(base_model).expanduser()),
            str(self.checkpoint),
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.steps = int(steps)
        self.cfg = float(cfg)

    def generate(self, image: Image.Image, prompts: Sequence[str], seed: int):
        result = self.pipeline(
            image=image.convert("RGB"),
            prompts=list(prompts),
            num_frames=80,
            num_inference_steps=self.steps,
            cfg_scale=self.cfg,
            seed=seed,
        )
        frames = result.frames[0]
        if len(frames) != 80:
            raise AssertionError(f"world model returned {len(frames)} frames, expected 80")
        return frames


class ARWorldModelRunner:
    """Adapter around `distill/worldmodel`'s causal, unidirectional (AR)
    world model -- the three-stage-distilled counterpart of `WorldModelRunner`
    above. Unlike the bidirectional runner's `generate(image, prompts, seed) ->
    80 frames`, the causal student only ever needs one prompt cell plus its own
    committed KV cache to produce the next 4 pixel frames, so this adapter
    exposes a per-tick session: `open_tick()` once per 5s window, then
    `step_cell()` once per 0.25s prompt cell -- matching how `closed_loop.
    loop.ClosedLoopAgentAR` needs to react to every cell's hit-detector
    result before composing the *next* cell's prompt, which the bidirectional
    runner's single batched call cannot support.

    Long-rollout cache behaviour is controlled independently through
    `attention_sink` and `reanchor_between_ticks`.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None,
        base_model: str | Path,
        stage: int,
        sampler: str | None = None,
        device: str = "cuda",
        steps: int | None = None,
        denoise_timesteps: Sequence[int] | None = None,
        cfg: float = 1.0,
        height: int = 480,
        width: int = 832,
        streaming_vae_decode: bool = True,
        carry_clean_latent_between_ticks: bool = False,
        attention_sink: bool = True,
        reanchor_between_ticks: bool = True,
    ):
        import torch

        if str(DISTILL_WORLDMODEL_ROOT) not in sys.path:
            sys.path.insert(0, str(DISTILL_WORLDMODEL_ROOT))
        from inference.ar_pipeline import ARWorldModelPipeline, stage_spec
        from worldmodel_cf.checkpoints import resolve_checkpoint

        spec = (
            stage_spec(stage, resolve_checkpoint(checkpoint))
            if checkpoint is not None
            else stage_spec(stage)
        )
        self.checkpoint = spec.checkpoint
        self.pipeline = ARWorldModelPipeline.from_pretrained(
            str(Path(base_model).expanduser()), str(self.checkpoint), torch_dtype=torch.bfloat16,
        ).to(device)
        self.sampler = sampler or spec.sampler
        self.steps = int(steps) if steps is not None else spec.steps
        self.denoise_timesteps = (
            tuple(denoise_timesteps) if denoise_timesteps is not None else spec.denoise_timesteps
        )
        self.cfg = float(cfg)
        self.height = int(height)
        self.width = int(width)
        self.streaming_vae_decode = bool(streaming_vae_decode)
        self.carry_clean_latent_between_ticks = bool(carry_clean_latent_between_ticks)
        if self.carry_clean_latent_between_ticks and not self.streaming_vae_decode:
            raise ValueError(
                "carry_clean_latent_between_ticks requires streaming_vae_decode"
            )
        self.attention_sink = bool(attention_sink)
        self.reanchor_between_ticks = bool(reanchor_between_ticks)
        self._session = None
        self._carried_anchor_latent = None

    def reset_rollout(self) -> None:
        """Prevent a carried anchor from leaking across independent samples."""
        self._session = None
        self._carried_anchor_latent = None

    def open_tick(self, image: Image.Image, seed: int) -> None:
        """Start a causal session anchored on `image`, bounded to the cells of
        one 5s tick (see `ARWorldModelSession` for why sessions are scoped this
        way rather than kept open for a whole episode)."""
        # Continuous mode deliberately ignores later tick anchors and seeds:
        # the original session, generator, streaming VAE state and KV cache
        # all remain live for the full rollout.
        if self._session is not None and not self.reanchor_between_ticks:
            return
        self._session = self.pipeline.open_session(
            image.convert("RGB"),
            sampler=self.sampler,
            steps=self.steps,
            denoise_timesteps=self.denoise_timesteps,
            cfg_scale=self.cfg,
            height=self.height,
            width=self.width,
            seed=seed,
            streaming_vae_decode=self.streaming_vae_decode,
            anchor_latent=(
                self._carried_anchor_latent
                if self.carry_clean_latent_between_ticks
                else None
            ),
            attention_sink=self.attention_sink,
        )

    def step_cell(self, prompt: str) -> list[Image.Image]:
        if self._session is None:
            raise RuntimeError("call open_tick() before step_cell()")
        frames = self._session.step(prompt)
        if self.carry_clean_latent_between_ticks:
            self._carried_anchor_latent = self._session.last_clean_latent
        return frames

    @property
    def last_frame(self) -> Image.Image:
        if self._session is None:
            raise RuntimeError("call open_tick() before last_frame")
        return self._session.last_frame

    def runtime_config(self) -> dict[str, object]:
        return {
            "cfg_scale": self.cfg,
            "sampler": self.sampler,
            "steps": self.steps,
            "attention_sink": self.attention_sink,
            "attention_sink_layout": (
                {"fixed_first_latents": 3, "sliding_latest_latents": 17, "max_latents": 20}
                if self.attention_sink
                else None
            ),
            "reanchor_between_ticks": self.reanchor_between_ticks,
            "kv_reset_between_ticks": self.reanchor_between_ticks,
            "rgb_vae_reencode_between_ticks": (
                self.reanchor_between_ticks and not self.carry_clean_latent_between_ticks
            ),
            "tick_cells": 20,
        }


class _ThreeFrameRunner:
    def _tensor(self, frames: Sequence[Image.Image | np.ndarray]):
        import cv2

        if len(frames) != 3:
            raise ValueError(f"model requires exactly 3 frames, got {len(frames)}")
        resized = [
            cv2.resize(_rgb_array(frame), (self.width, self.height), interpolation=cv2.INTER_AREA)
            for frame in frames
        ]
        array = np.stack(resized)[None].transpose(0, 1, 4, 2, 3).copy()
        return self.torch.from_numpy(array)

    def _autocast(self):
        return self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.bfloat16,
            enabled=self.device.type == "cuda",
        )


class AttackNetRunner(_ThreeFrameRunner):
    output_names = ("player_hit", "boss_hit")

    def __init__(self, checkpoint: str | Path, device: str = "auto", threshold: float | None = None):
        import torch

        from attacknet.model import AttackNet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        args = saved.get("args", {})
        self.height = int(args.get("height", 192))
        self.width = int(args.get("width", 336))
        self.threshold = float(args.get("threshold", 0.5) if threshold is None else threshold)
        self.model = AttackNet(
            hidden=int(args.get("hidden", 384)),
            pretrained=False,
            dropout=float(args.get("dropout", 0.1)),
            backbone=args.get("backbone", "resnet18"),
            temporal_pool=args.get("temporal_pool", "last"),
            bidirectional=bool(args.get("bidirectional", False)),
            aux_damage=bool(args.get("aux_damage", False)),
        )
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.eval().to(self.device)

    def predict_cell(self, frames: Sequence[Image.Image | np.ndarray]) -> dict[str, object]:
        if len(frames) != 4:
            raise ValueError(f"AttackNet cell needs 4 frames, got {len(frames)}")
        tensor = self._tensor(frames[1:4]).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            row = self.model(tensor).float().sigmoid().cpu().numpy()[0]
        return {
            "player_hit_probability": float(row[0]),
            "boss_hit_probability": float(row[1]),
            "player_hit": bool(row[0] >= self.threshold),
            "boss_hit": bool(row[1] >= self.threshold),
        }

    def initial(self, frame: Image.Image | np.ndarray) -> dict[str, object]:
        copied = [frame, frame, frame]
        tensor = self._tensor(copied).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            row = self.model(tensor).float().sigmoid().cpu().numpy()[0]
        return {
            "player_hit_probability": float(row[0]),
            "boss_hit_probability": float(row[1]),
            "player_hit": bool(row[0] >= self.threshold),
            "boss_hit": bool(row[1] >= self.threshold),
        }


class SituationNetRunner(_ThreeFrameRunner):
    """Native runner for the trained distance/orientation checkpoint, tagged with
    the selected game's coordinate semantics (`distance_unit`)."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", distance_unit: str | None = None):
        import torch

        from situationnet.model import SituationNet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.distance_unit = DISTANCE_UNIT if distance_unit is None else distance_unit
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        args = saved.get("args", {})
        self.height = int(args.get("height", 192))
        self.width = int(args.get("width", 336))
        self.distance_scale = float(args.get("distance_scale", 10.0))
        self.model = SituationNet(
            hidden=int(args.get("hidden", 384)),
            pretrained=False,
            dropout=float(args.get("dropout", 0.1)),
            backbone=args.get("backbone", "resnet18"),
            temporal_pool=args.get("temporal_pool", "last"),
            bidirectional=bool(args.get("bidirectional", False)),
            aux_sector=bool(args.get("aux_sector", False)),
        )
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.eval().to(self.device)

    def _run(self, frames: Sequence[Image.Image | np.ndarray]) -> dict[str, Any]:
        from situationnet.model import geometry_from_outputs

        tensor = self._tensor(frames).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            outputs = self.model(tensor)
        geometry = geometry_from_outputs(outputs, distance_scale=self.distance_scale)[0]
        geometry["distance_unit"] = self.distance_unit
        return geometry

    def initial(self, frame: Image.Image | np.ndarray) -> dict[str, Any]:
        return self._run([frame, frame, frame])

    def clip(self, frames: Sequence[Image.Image | np.ndarray]) -> dict[str, Any]:
        if len(frames) != 80:
            raise ValueError(f"SituationNet expects one 80-frame/5-second tick, got {len(frames)}")
        return self._run(frames[-3:])


def save_mp4(frames: Sequence[Image.Image | np.ndarray], path: str | Path, fps: int = 16) -> None:
    """Write H.264/yuv420p so the clips play in VSCode and browser previews."""
    import shutil
    import subprocess

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = _rgb_array(frames[0])
    height, width = first.shape[:2]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _save_mp4_opencv(frames, output, fps, width, height)
        return

    process = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            rgb = _rgb_array(frame)
            if rgb.shape[:2] != (height, width):
                raise ValueError("all generated frames must have one resolution")
            process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed to create video: {output}")


def _save_mp4_opencv(
    frames: Sequence[Image.Image | np.ndarray],
    output: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video: {output}")
    try:
        for frame in frames:
            rgb = _rgb_array(frame)
            if rgb.shape[:2] != (height, width):
                raise ValueError("all generated frames must have one resolution")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


__all__ = [
    "ARWorldModelRunner",
    "AttackNetRunner",
    "SituationNetRunner",
    "WorldModelRunner",
    "save_mp4",
]
