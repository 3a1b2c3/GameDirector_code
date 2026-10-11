"""Bidirectional whole-clip inference used by the GameDirector agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from .constants import DEFAULT_HEIGHT, DEFAULT_WIDTH, NEG_PROMPT
from .models import WorldModelDiT, WanTextEncoder, WanTokenizer, WanVideoVAE38
from .utils import (
    FlowMatchScheduler,
    load_prompts,
    preprocess_image,
    round_up,
    to_pil_video,
)


@dataclass
class WorldModelPipelineOutput:
    frames: List[List[Image.Image]] = field(default_factory=list)


class WorldModelPipeline(torch.nn.Module):
    """Generate an 80-frame/5-second clip from one frame and 20 prompts."""

    def __init__(self, dit, vae, text_encoder, tokenizer, scheduler, torch_dtype):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.torch_dtype = torch_dtype
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.dit = self.dit.to(self._device, dtype=self.torch_dtype)
        self.vae = self.vae.to(self._device, dtype=self.torch_dtype)
        self.text_encoder = self.text_encoder.to(self._device, dtype=self.torch_dtype)
        return self

    @classmethod
    def from_pretrained(
        cls,
        base_model_dir: str,
        checkpoint_path: str,
        torch_dtype=torch.bfloat16,
    ):
        base = Path(base_model_dir).expanduser().absolute()
        ti2v = base / "Wan-AI" / "Wan2.2-TI2V-5B"
        tokenizer_dir = (
            base / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"
        )
        required = [
            ti2v / "Wan2.2_VAE.pth",
            ti2v / "models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_dir / "spiece.model",
            Path(checkpoint_path).expanduser(),
        ]
        missing_paths = [str(path) for path in required if not path.exists()]
        if missing_paths:
            raise FileNotFoundError("missing model files: " + ", ".join(missing_paths))

        vae = WanVideoVAE38()
        vae_state = torch.load(
            ti2v / "Wan2.2_VAE.pth", map_location="cpu", weights_only=True,
        )
        vae.load_state_dict(
            {f"model.{key}": value for key, value in vae_state.items()}, strict=False,
        )
        vae = vae.to(torch_dtype).eval().requires_grad_(False)

        text_encoder = WanTextEncoder()
        text_state = torch.load(
            ti2v / "models_t5_umt5-xxl-enc-bf16.pth",
            map_location="cpu",
            weights_only=True,
        )
        text_encoder.load_state_dict(text_state, strict=False)
        text_encoder = text_encoder.to(torch_dtype).eval().requires_grad_(False)
        tokenizer = WanTokenizer(str(tokenizer_dir), seq_len=128)

        checkpoint = load_file(str(Path(checkpoint_path).expanduser()))
        dit = WorldModelDiT()
        missing, unexpected = dit.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint does not exactly match WorldModelDiT: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        dit = dit.to(torch_dtype).eval().requires_grad_(False)
        print(f"[WorldModel] loaded {len(checkpoint)} DiT tensors")
        return cls(
            dit, vae, text_encoder, tokenizer, FlowMatchScheduler(), torch_dtype,
        )

    @torch.no_grad()
    def encode_prompts(
        self, prompts: Sequence[str], allow_empty: bool = False,
    ) -> torch.Tensor:
        prompts = [str(prompt).strip() for prompt in prompts]
        if len(prompts) != 20 or (not allow_empty and any(not prompt for prompt in prompts)):
            raise ValueError("prompts must contain exactly 20 non-empty strings")
        unique = list(dict.fromkeys(prompts))
        lookup = {value: index for index, value in enumerate(unique)}
        inverse = torch.tensor([lookup[value] for value in prompts], device=self.device)
        ids, mask = self.tokenizer(unique)
        ids, mask = ids.to(self.device), mask.to(self.device)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        embeddings = self.text_encoder(ids, mask)
        embeddings = embeddings[:, : int(sequence_lengths.max().item())]
        for index, sequence_length in enumerate(sequence_lengths.tolist()):
            embeddings[index, sequence_length:] = 0
        return embeddings.index_select(0, inverse).unsqueeze(0).to(self.torch_dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompts: Optional[Sequence[str]] = None,
        prompt_file: Optional[str] = None,
        negative_prompt: str = NEG_PROMPT,
        num_frames: int = 80,
        num_inference_steps: int = 3,
        cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: int = 2,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: Sequence[int] = (30, 52),
        tile_stride: Sequence[int] = (15, 26),
    ) -> WorldModelPipelineOutput:
        if num_frames != 80:
            raise ValueError("VRising world-model checkpoints output exactly 80 frames")
        if (prompts is None) == (prompt_file is None):
            raise ValueError("provide exactly one of prompts or prompt_file")
        prompts = list(prompts) if prompts is not None else load_prompts(prompt_file)
        height = round_up(height or DEFAULT_HEIGHT, 32)
        width = round_up(width or DEFAULT_WIDTH, 32)
        context_positive = self.encode_prompts(prompts)
        context_negative = (
            self.encode_prompts([negative_prompt] * 20, allow_empty=True)
            if cfg_scale != 1 else None
        )

        image_tensor = preprocess_image(
            image, height, width, device=str(self.device), dtype=self.torch_dtype,
        )
        first_latent = self.vae.encode(
            [image_tensor],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(self.device, self.torch_dtype)
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(
            (1, 48, 21, height // 16, width // 16),
            generator=generator,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        latents[:, :, :1] = first_latent

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        for index, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            model_timestep = timestep.unsqueeze(0).to(self.device, self.torch_dtype)
            prediction_positive = self.dit(
                latents=latents,
                timestep=model_timestep,
                context=context_positive,
                prompt_num_frames=80,
                fuse_vae_embedding_in_latents=True,
            )
            if cfg_scale != 1:
                prediction_negative = self.dit(
                    latents=latents,
                    timestep=model_timestep,
                    context=context_negative,
                    prompt_num_frames=80,
                    fuse_vae_embedding_in_latents=True,
                )
                prediction = prediction_negative + cfg_scale * (
                    prediction_positive - prediction_negative
                )
            else:
                prediction = prediction_positive
            latents = self.scheduler.step(
                prediction, self.scheduler.timesteps[index], latents,
            )
            latents[:, :, :1] = first_latent

        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).clamp(-1, 1)
        if video.shape[2] != 81:
            raise AssertionError(f"Wan decoder returned {video.shape[2]} frames; expected 81")
        video = video[:, :, 1:]
        if video.shape[2] != 80:
            raise AssertionError(f"expected 80 output frames, got {video.shape[2]}")
        return WorldModelPipelineOutput(frames=[to_pil_video(video)])


__all__ = ["WorldModelPipeline", "WorldModelPipelineOutput"]
