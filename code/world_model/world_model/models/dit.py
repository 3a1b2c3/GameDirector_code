"""Standalone video-only DiT shared with the training implementation."""
from __future__ import annotations

from typing import Optional

import torch

from diffsynth.models.world_model_dit import WorldModel


WAN_MODEL_DEFAULTS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


class WorldModelDiT(WorldModel):
    def __init__(self, **kwargs):
        config = dict(WAN_MODEL_DEFAULTS)
        config.update(kwargs)
        super().__init__(**config)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        fuse_vae_embedding_in_latents: bool = True,
        prompt_num_frames: Optional[int] = 80,
        **kwargs,
    ):
        from diffsynth.pipelines.world_model import model_fn_world_model

        return model_fn_world_model(
            self,
            latents=latents,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            prompt_num_frames=prompt_num_frames,
            **kwargs,
        )


__all__ = ["WorldModelDiT"]
