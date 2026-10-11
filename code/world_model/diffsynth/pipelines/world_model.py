"""Training pipeline for the prompt-conditioned Vampire video world model."""
from __future__ import annotations

from typing import Union

import torch
from einops import rearrange

from ..core import ModelConfig, gradient_checkpoint_forward
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE


class WorldModelTrainingPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler()
        self.tokenizer: HuggingfaceTokenizer | None = None
        self.text_encoder: WanTextEncoder | None = None
        self.dit: WanModel | None = None
        self.vae: WanVideoVAE | None = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnitShapeChecker(),
            WanVideoUnitNoiseInitializer(),
            WanVideoUnitPromptEmbedder(),
            WanVideoUnitInputVideoEmbedder(),
            WanVideoUnitImageEmbedderFused(),
        ]
        self.model_fn = model_fn_world_model

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        vram_limit: float | None = None,
    ):
        # Local .pth paths are intentional: no Hugging Face lookup, so
        # one-command runs stay offline.
        pipe = WorldModelTrainingPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        pipe.dit = dit[0] if isinstance(dit, list) else dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=128, clean="whitespace",
            )
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


class WanVideoUnitShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames,
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnitNoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnitInputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if input_latents.shape[2] != noise.shape[2]:
            raise AssertionError(
                "Wan VAE/noise temporal mismatch: "
                f"encoded={input_latents.shape[2]}, noise={noise.shape[2]}"
            )
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(
            input_latents, noise, timestep=pipe.scheduler.timesteps[0],
        )
        return {"latents": latents}


class WanVideoUnitPromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe, prompt):
        temporal = isinstance(prompt, (list, tuple))
        prompts = list(prompt) if temporal else [prompt]
        if temporal and len(prompts) != 20:
            raise ValueError(f"expected exactly 20 temporal prompts, got {len(prompts)}")
        unique_prompts = list(dict.fromkeys(prompts))
        lookup = {value: index for index, value in enumerate(unique_prompts)}
        inverse = torch.tensor([lookup[value] for value in prompts], device=pipe.device)
        ids, mask = pipe.tokenizer(
            unique_prompts, return_mask=True, add_special_tokens=True,
        )
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        unique_embeddings = pipe.text_encoder(ids, mask)
        unique_embeddings = unique_embeddings[:, : int(seq_lens.max().item())]
        for index, seq_len in enumerate(seq_lens.tolist()):
            unique_embeddings[index, seq_len:] = 0
        embeddings = unique_embeddings.index_select(0, inverse)
        return embeddings.unsqueeze(0) if temporal else embeddings

    def process(self, pipe, prompt, positive):
        pipe.load_models_to_device(self.onload_model_names)
        return {"context": self.encode_prompt(pipe, prompt)}


class WanVideoUnitImageEmbedderFused(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image", "latents", "height", "width",
                "tiled", "tile_size", "tile_stride",
            ),
            output_params=(
                "latents", "fuse_vae_embedding_in_latents", "first_frame_latents",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self, pipe, input_image, latents, height, width,
        tiled, tile_size, tile_stride,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        first_frame_latents = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, :1] = first_frame_latents
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first_frame_latents,
        }


def model_fn_world_model(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Video-only forward with one clean latent followed by 20 predictions."""
    if context is None or context.ndim != 4 or context.shape[1] != 20:
        raise ValueError(
            "Vampire world model requires context [B,20,S,text_dim]; "
            f"got {None if context is None else tuple(context.shape)}"
        )
    context = dit.text_embedding(context)
    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    if f != 21:
        raise AssertionError(f"expected 21 video latents (1 clean + 20 target), got {f}")
    video_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    tokens_per_frame = h * w

    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        video_timestep = torch.cat([
            torch.zeros(tokens_per_frame, dtype=latents.dtype, device=latents.device),
            torch.ones(
                (f - 1) * tokens_per_frame,
                dtype=latents.dtype,
                device=latents.device,
            ) * timestep,
        ])
        t_video = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, video_timestep).unsqueeze(0),
        )
        t_mod_video = dit.time_projection(t_video).unflatten(2, (6, dit.dim))
    else:
        t_video = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod_video = dit.time_projection(t_video).unflatten(1, (6, dit.dim))

    video_freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(video_tokens.device)

    raw_num_frames = int(kwargs.get("prompt_num_frames") or 80)
    if raw_num_frames != 80 or raw_num_frames != context.shape[1] * 4:
        raise AssertionError(
            "temporal contract requires 80 frames and four frames per prompt; "
            f"frames={raw_num_frames}, prompts={context.shape[1]}"
        )
    for block in dit.blocks:
        video_tokens = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            video_tokens,
            context,
            t_mod_video,
            video_freqs,
            raw_num_frames,
            f,
            1,
        )

    prediction = dit.head(video_tokens, t_video)
    return dit.unpatchify(prediction, (f, h, w))


__all__ = ["WorldModelTrainingPipeline", "model_fn_world_model"]
