"""Stage-aware, frame-autoregressive WorldModel inference with exact KV caches.

Video-only counterpart of `distill/stateplay/inference/ar_pipeline.py`: same
`forward_frame` + committed joint KV-cache rollout, minus the StatePlay state
branch. The causal student has no notion of a fixed clip length — each KV-cache
step only needs the previous step's committed cache plus one new temporal
prompt cell, so this pipeline runs for as many prompt cells as it is given
instead of a hardcoded 20 (5s @ 4F-3/16fps). The only hard ceiling is the
model's precomputed frame-axis RoPE table (1024 latent frames, see
`worldmodel_cf/model.py` / `world_model_dit.py`'s `precompute_freqs_cis_3d`
default `end=1024`) — `MAX_LATENT_FRAMES` below fails fast instead of silently
indexing out of range. Default inference bounds KV memory with an attention
sink: the first 3 latent frames remain fixed while a 17-latent sliding window
retains the newest history, for a strict 20-latent cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORLD_MODEL_ROOT = ROOT.parent / "world_model"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT))

from worldmodel_cf.flow_match import FlowMatchScheduler as TrainingScheduler  # noqa: E402
from worldmodel_cf.model import CausalWorldModelDiT, load_checkpoint, worldmodel_config  # noqa: E402


from world_model.models.text_encoder import WanTextEncoder, WanTokenizer  # noqa: E402
from world_model.models.vae import WanVideoVAE38  # noqa: E402
from world_model.utils.image import preprocess_image, round_up, save_mp4, to_pil_video  # noqa: E402
from world_model.utils.scheduler import FlowMatchScheduler  # noqa: E402


# self.freqs[0] is precomputed with end=1024 in world_model_dit.py (no dynamic-fps
# override for this model config) -> frame_index must stay < 1024.
MAX_LATENT_FRAMES = 1024


@dataclass(frozen=True)
class StageSpec:
    stage: int
    checkpoint: Path
    sampler: str
    steps: int = 3
    denoise_timesteps: tuple[int, ...] = (1000, 500, 250)


STAGES = {
    1: StageSpec(1, ROOT / "outputs/ar_world_model.safetensors", "euler", steps=30),
    2: StageSpec(2, ROOT / "outputs/stage2_cd_teacher_cfg_1/ema/ema_step12000.safetensors", "distilled"),
    3: StageSpec(3, ROOT / "outputs/stage3_dmd_teacher_cfg_1/ema/ema_step5000.safetensors", "distilled"),
}


def stage_spec(stage: int, checkpoint: Optional[Path] = None) -> StageSpec:
    try:
        spec = STAGES[int(stage)]
    except KeyError as error:
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}") from error
    if checkpoint is None:
        return spec
    return StageSpec(
        stage=spec.stage, checkpoint=checkpoint.expanduser().resolve(), sampler=spec.sampler,
        steps=spec.steps, denoise_timesteps=spec.denoise_timesteps,
    )


class ARWorldModelPipeline(torch.nn.Module):
    """Generate len(prompts)+1 latent frames sequentially, one KV-cache step each."""

    def __init__(self, dit, vae, text_encoder, tokenizer, torch_dtype=torch.bfloat16):
        super().__init__()
        self.dit, self.vae = dit, vae
        self.text_encoder, self.tokenizer = text_encoder, tokenizer
        self.torch_dtype = torch_dtype
        self._device = torch.device("cpu")
        self.euler_scheduler = FlowMatchScheduler()
        self.training_scheduler = TrainingScheduler()

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.dit = self.dit.to(self._device, dtype=self.torch_dtype)
        self.vae = self.vae.to(self._device, dtype=self.torch_dtype)
        self.text_encoder = self.text_encoder.to(self._device, dtype=self.torch_dtype)
        self.training_scheduler.to(self._device)
        return self

    @classmethod
    def from_pretrained(cls, base_model_dir, checkpoint_path, torch_dtype=torch.bfloat16):
        base = Path(base_model_dir)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint is not available yet: {checkpoint}. "
                "Wait for the corresponding formal stage to complete."
            )
        ti2v = base / "Wan-AI/Wan2.2-TI2V-5B"
        tokenizer_dir = base / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
        vae = WanVideoVAE38()
        vae_state = torch.load(ti2v / "Wan2.2_VAE.pth", map_location="cpu", weights_only=True)
        vae.load_state_dict({f"model.{key}": value for key, value in vae_state.items()}, strict=False)
        vae = vae.to(torch_dtype).eval().requires_grad_(False)
        text_encoder = WanTextEncoder()
        text_state = torch.load(
            ti2v / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=True,
        )
        text_encoder.load_state_dict(text_state, strict=False)
        text_encoder = text_encoder.to(torch_dtype).eval().requires_grad_(False)
        tokenizer = WanTokenizer(str(tokenizer_dir), seq_len=128)
        dit = CausalWorldModelDiT(causal=True, **worldmodel_config())
        load_checkpoint(dit, str(checkpoint))
        dit = dit.to(torch_dtype).eval().requires_grad_(False)
        print(f"[WorldModel CF++] loaded causal AR checkpoint: {checkpoint}")
        return cls(dit, vae, text_encoder, tokenizer, torch_dtype)

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        """list[N] prompt strings -> [1, N, 128, 4096] UMT5 embeddings, one cell each.

        Unlike StatePlay's `encode_prompts`, N is not required to be 20 — the
        causal student conditions each KV-cache step on exactly one cell, so any
        number of cells (i.e. any generation length) works."""
        if not prompts:
            raise ValueError("need at least one temporal prompt")
        unique = list(dict.fromkeys(prompts))
        lookup = {value: index for index, value in enumerate(unique)}
        inverse = torch.tensor([lookup[value] for value in prompts], device=self.device)
        ids, mask = self.tokenizer(unique)
        ids, mask = ids.to(self.device), mask.to(self.device)
        lengths = mask.gt(0).sum(1).long()
        embeddings = self.text_encoder(ids, mask)
        embeddings = embeddings[:, :int(lengths.max().item())]
        for index, length in enumerate(lengths.tolist()):
            embeddings[index, length:] = 0
        return embeddings.index_select(0, inverse).unsqueeze(0).to(self.torch_dtype)

    def _predict(self, video, timestep, context_pos, context_neg, frame_index,
                 cache_pos, cache_neg, cfg_scale):
        video_pos = self.dit(
            video, timestep, context_pos, frame_index=frame_index, kv_cache=cache_pos, commit=False,
        )
        if context_neg is None:
            return video_pos
        video_neg = self.dit(
            video, timestep, context_neg, frame_index=frame_index, kv_cache=cache_neg, commit=False,
        )
        return video_neg + cfg_scale * (video_pos - video_neg)

    def _sample_euler(self, video, context_pos, context_neg, frame_index,
                      cache_pos, cache_neg, steps, sigma_shift, cfg_scale):
        self.euler_scheduler.set_timesteps(steps, shift=sigma_shift)
        for raw_timestep in self.euler_scheduler.timesteps:
            timestep = raw_timestep.to(self.device, self.torch_dtype).reshape(1)
            video_pred = self._predict(
                video, timestep, context_pos, context_neg, frame_index, cache_pos, cache_neg, cfg_scale,
            )
            video = self.euler_scheduler.step(video_pred, raw_timestep, video)
        return video

    def _sigma_for_timestep(self, raw_timestep: int) -> torch.Tensor:
        times = self.training_scheduler.timesteps
        index = (times - float(raw_timestep)).abs().argmin()
        return self.training_scheduler.sigmas[index].to(self.device, self.torch_dtype)

    def _sample_distilled(self, video, context_pos, context_neg, frame_index,
                          cache_pos, cache_neg, denoise_timesteps, generator, cfg_scale):
        clean_video = None
        for index, raw_timestep in enumerate(denoise_timesteps):
            timestep = torch.full(
                (1,), float(raw_timestep), device=self.device, dtype=self.torch_dtype,
            )
            video_pred = self._predict(
                video, timestep, context_pos, context_neg, frame_index, cache_pos, cache_neg, cfg_scale,
            )
            sigma = self._sigma_for_timestep(raw_timestep)
            clean_video = video - sigma * video_pred
            if index + 1 < len(denoise_timesteps):
                next_sigma = self._sigma_for_timestep(denoise_timesteps[index + 1])
                video = (1 - next_sigma) * clean_video + next_sigma * torch.randn(
                    clean_video.shape, generator=generator, device=self.device, dtype=self.torch_dtype,
                )
        return clean_video

    def _commit(self, video, context_pos, context_neg, frame_index, cache_pos, cache_neg):
        zero = torch.zeros(1, device=self.device, dtype=self.torch_dtype)
        self.dit(video, zero, context_pos, frame_index=frame_index, kv_cache=cache_pos, commit=True)
        if context_neg is not None:
            self.dit(video, zero, context_neg, frame_index=frame_index, kv_cache=cache_neg, commit=True)

    def _trim_attention_sink(self, cache_pos, cache_neg, tokens_per_latent):
        retained = self.dit.trim_kv_cache_attention_sink(
            cache_pos, tokens_per_latent, sink_latents=3, window_latents=17,
        )
        if cache_neg is not None:
            retained_neg = self.dit.trim_kv_cache_attention_sink(
                cache_neg, tokens_per_latent, sink_latents=3, window_latents=17,
            )
            if retained_neg != retained:
                raise AssertionError("positive and negative KV caches diverged")
        return retained

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompts: Sequence[str], sampler: str,
                 negative_prompt: str = "", steps: int = 3,
                 denoise_timesteps: Sequence[int] = (1000, 500, 250), cfg_scale: float = 1.0,
                 height: int = 480, width: int = 832, seed: int = 2,
                 sigma_shift: float = 5.0, tiled: bool = False,
                 progress_every: int = 1,
                 attention_sink: bool = True) -> list[Image.Image]:
        num_frames = len(prompts) + 1
        if num_frames > MAX_LATENT_FRAMES:
            raise ValueError(
                f"{len(prompts)} prompt cells -> {num_frames} latent frames exceeds "
                f"the model's RoPE table ({MAX_LATENT_FRAMES}); split into multiple runs "
                "or re-anchor from the tail frame."
            )
        height, width = round_up(height, 32), round_up(width, 32)
        context_pos = self.encode_prompts(list(prompts))
        needs_cfg = cfg_scale != 1.0
        context_neg = self.encode_prompts([negative_prompt] * len(prompts)) if needs_cfg else None
        image_tensor = preprocess_image(image, height, width, str(self.device), self.torch_dtype)
        first_video = self.vae.encode([image_tensor], device=self.device, tiled=tiled).to(
            self.device, self.torch_dtype,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        cache_pos = self.dit.init_kv_cache()
        cache_neg = self.dit.init_kv_cache() if needs_cfg else None
        self._commit(
            first_video, context_pos[:, :1], None if context_neg is None else context_neg[:, :1],
            0, cache_pos, cache_neg,
        )
        tokens_per_latent = cache_pos[0]["k"].shape[1]
        videos = [first_video]
        for frame_index in range(1, num_frames):
            video = torch.randn(
                (1, 48, 1, height // 16, width // 16), generator=generator,
                device=self.device, dtype=self.torch_dtype,
            )
            pos_cell = context_pos[:, frame_index - 1:frame_index]
            neg_cell = None if context_neg is None else context_neg[:, frame_index - 1:frame_index]
            if sampler == "euler":
                clean_video = self._sample_euler(
                    video, pos_cell, neg_cell, frame_index, cache_pos, cache_neg, steps, sigma_shift,
                    cfg_scale,
                )
            elif sampler == "distilled":
                clean_video = self._sample_distilled(
                    video, pos_cell, neg_cell, frame_index, cache_pos, cache_neg,
                    tuple(denoise_timesteps), generator, cfg_scale,
                )
            else:
                raise ValueError(f"unknown sampler: {sampler}")
            videos.append(clean_video)
            self._commit(clean_video, pos_cell, neg_cell, frame_index, cache_pos, cache_neg)
            if attention_sink:
                self._trim_attention_sink(cache_pos, cache_neg, tokens_per_latent)
            if progress_every and frame_index % progress_every == 0:
                print(f"  AR latent frame {frame_index}/{num_frames - 1}", flush=True)
        video_latents = torch.cat(videos, 2)
        video = self.vae.decode(video_latents, device=self.device, tiled=tiled).clamp(-1, 1)
        expected_pixel_frames = 4 * (num_frames - 1) + 1
        if video.shape[2] != expected_pixel_frames:
            raise AssertionError(
                f"expected {expected_pixel_frames} decoded frames, got {video.shape[2]}"
            )
        video = video[:, :, 1:]  # drop the clean anchor frame (matches StatePlay convention)
        return to_pil_video(video)

    def open_session(self, image: Image.Image, **kwargs) -> "ARWorldModelSession":
        """Start an incremental, cell-by-cell generation session.

        `__call__` above needs the whole prompt list up front and only decodes
        pixels once, at the very end. Closed-loop callers that must react to
        each new cell as it lands (e.g. running a hit detector every 4 frames
        and switching the *next* cell's prompt on a death) need decoded RGB
        frames after every single AR step instead. `ARWorldModelSession` reuses
        this pipeline's own `_sample_euler`/`_sample_distilled`/`_commit` step
        primitives -- the exact same math `__call__` uses -- one prompt cell at
        a time. Streaming VAE decode is enabled by default, so the anchor is
        decoded once and every `step()` decodes only its four new pixel frames.
        The explicit legacy opt-out redecodes the bounded session prefix. A new
        session also creates fresh DiT KV caches, which is why direct long
        rollouts use one 5s/20-cell session at a time and RGB-reencode the tail
        frame as the next anchor.
        """
        return ARWorldModelSession(self, image, **kwargs)


class ARWorldModelSession:
    """One bounded, incremental AR generation window: 4 new pixel frames per
    `step()` call, backed by a persistent causal KV cache for the session's
    lifetime. Streaming VAE decode is the default; set
    `streaming_vae_decode=False` to retain the legacy growing-prefix decoder.
    Sessions remain bounded so direct long rollouts can reset DiT KV state and
    VAE-reencode an RGB anchor every 5 seconds."""

    def __init__(
        self,
        pipeline: ARWorldModelPipeline,
        image: Image.Image,
        *,
        sampler: str,
        negative_prompt: str = "",
        steps: int = 3,
        denoise_timesteps: Sequence[int] = (1000, 500, 250),
        cfg_scale: float = 1.0,
        height: int = 480,
        width: int = 832,
        seed: int = 2,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        streaming_vae_decode: bool = True,
        anchor_latent: Optional[torch.Tensor] = None,
        attention_sink: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.pipeline = pipeline
        self.sampler = sampler
        self.negative_prompt = negative_prompt
        self.steps = int(steps)
        self.denoise_timesteps = tuple(denoise_timesteps)
        self.cfg_scale = float(cfg_scale)
        self.needs_cfg = self.cfg_scale != 1.0
        self.height, self.width = round_up(height, 32), round_up(width, 32)
        self.sigma_shift = float(sigma_shift)
        self.tiled = bool(tiled)
        self.streaming_vae_decode = bool(streaming_vae_decode)
        self.attention_sink = bool(attention_sink)
        self.frame_index = 0

        device = pipeline.device
        if anchor_latent is None:
            image_tensor = preprocess_image(
                image.convert("RGB"), self.height, self.width, str(device), pipeline.torch_dtype,
            )
            first_video = pipeline.vae.encode([image_tensor], device=device, tiled=tiled).to(
                device, pipeline.torch_dtype,
            )
        else:
            expected = (
                1, pipeline.vae.z_dim, 1, self.height // 16, self.width // 16,
            )
            if tuple(anchor_latent.shape) != expected:
                raise ValueError(
                    f"carried anchor latent has shape {tuple(anchor_latent.shape)}, "
                    f"expected {expected}"
                )
            first_video = anchor_latent.detach().to(device, pipeline.torch_dtype)
        self._last_clean_latent = first_video.detach()
        self.generator = (
            generator
            if generator is not None
            else torch.Generator(device=device).manual_seed(seed)
        )
        self.cache_pos = pipeline.dit.init_kv_cache()
        self.cache_neg = pipeline.dit.init_kv_cache() if self.needs_cfg else None
        self._context_cache: dict[str, torch.Tensor] = {}
        # frame_index 0 (the anchor) never attends to context (`forward_frame`
        # only runs cross-attention when frame_index > 0), so any single-cell
        # context is a safe placeholder for the commit call below.
        anchor_ctx_pos = self._encode(negative_prompt)
        anchor_ctx_neg = self._encode(negative_prompt) if self.needs_cfg else None
        pipeline._commit(
            first_video, anchor_ctx_pos, anchor_ctx_neg, 0, self.cache_pos, self.cache_neg,
        )
        self.tokens_per_latent = self.cache_pos[0]["k"].shape[1]
        self.cached_latents = 1
        self._vae_decode_stream = None
        if self.streaming_vae_decode:
            self._vae_decode_stream = pipeline.vae.open_decode_stream(device, tiled=tiled)
            anchor = self._vae_decode_stream.decode(first_video)
            if anchor.shape[2] != 1:
                raise AssertionError(
                    f"streaming VAE anchor must decode to 1 frame, got {anchor.shape[2]}"
                )
            self._latents = None
        else:
            self._latents = [first_video]
        self._decoded: list[Image.Image] = []
        self._cells_generated = 0

    @property
    def device(self):
        return self.pipeline.device

    def _encode(self, prompt: str) -> torch.Tensor:
        """One prompt -> [1, 1, seq_len, 4096], memoized per session.

        `ARWorldModelPipeline.encode_prompts` truncates its returned embeddings
        to `max(real token length)` *across whatever prompts happen to be
        batched together in that one call* -- fine for `__call__`, which
        always encodes an entire known prompt list in one shot, but wrong here:
        a session encodes one prompt at a time as each cell is decided, so
        reusing that method would make a given prompt's embedding (and hence
        the model's behavior) depend on which other prompts were encoded
        alongside it, breaking the "same prompt cell -> same conditioning"
        determinism the closed loop relies on. This instead always keeps the
        tokenizer's full, fixed sequence length -- no batch-dependent
        truncation -- so a prompt's embedding here never depends on anything
        outside itself."""
        cached = self._context_cache.get(prompt)
        if cached is not None:
            return cached
        pipeline = self.pipeline
        ids, mask = pipeline.tokenizer([prompt])
        ids, mask = ids.to(self.device), mask.to(self.device)
        length = int(mask.gt(0).sum(1).long().item())
        embeddings = pipeline.text_encoder(ids, mask).clone()
        embeddings[:, length:] = 0
        embeddings = embeddings.unsqueeze(0).to(pipeline.torch_dtype)
        self._context_cache[prompt] = embeddings
        return embeddings

    def step(self, prompt: str) -> list[Image.Image]:
        """Run exactly one AR step (one prompt cell) and return its 4 new
        decoded pixel frames, in order."""
        self.frame_index += 1
        frame_index = self.frame_index
        pipeline = self.pipeline
        context_pos = self._encode(prompt)
        context_neg = self._encode(self.negative_prompt) if self.needs_cfg else None
        video = torch.randn(
            (1, 48, 1, self.height // 16, self.width // 16),
            generator=self.generator, device=self.device, dtype=pipeline.torch_dtype,
        )
        if self.sampler == "euler":
            clean_video = pipeline._sample_euler(
                video, context_pos, context_neg, frame_index,
                self.cache_pos, self.cache_neg, self.steps, self.sigma_shift, self.cfg_scale,
            )
        elif self.sampler == "distilled":
            clean_video = pipeline._sample_distilled(
                video, context_pos, context_neg, frame_index,
                self.cache_pos, self.cache_neg, self.denoise_timesteps, self.generator, self.cfg_scale,
            )
        else:
            raise ValueError(f"unknown sampler: {self.sampler}")
        pipeline._commit(clean_video, context_pos, context_neg, frame_index, self.cache_pos, self.cache_neg)
        if self.attention_sink:
            self.cached_latents = pipeline._trim_attention_sink(
                self.cache_pos, self.cache_neg, self.tokens_per_latent,
            )
        else:
            self.cached_latents += 1
        self._last_clean_latent = clean_video.detach()
        if self.streaming_vae_decode:
            decoded = self._vae_decode_stream.decode(clean_video)
            pixel_frames = to_pil_video(decoded)
            if len(pixel_frames) != 4:
                raise AssertionError(
                    f"streaming VAE step must decode to 4 frames, got {len(pixel_frames)}"
                )
            new_frames = pixel_frames
        else:
            self._latents.append(clean_video)
            video_latents = torch.cat(self._latents, 2)
            decoded = pipeline.vae.decode(
                video_latents, device=self.device, tiled=self.tiled
            ).clamp(-1, 1)
            pixel_frames = to_pil_video(decoded)
            expected = 4 * (len(self._latents) - 1) + 1
            if len(pixel_frames) != expected:
                raise AssertionError(f"expected {expected} decoded frames, got {len(pixel_frames)}")
            new_frames = pixel_frames[-4:]
        self._cells_generated += 1
        self._decoded.extend(new_frames)
        return new_frames

    @property
    def last_frame(self) -> Image.Image:
        if not self._decoded:
            raise RuntimeError("no cells generated yet in this session")
        return self._decoded[-1]

    @property
    def cells_generated(self) -> int:
        return self._cells_generated

    @property
    def last_clean_latent(self) -> torch.Tensor:
        """The most recently committed clean latent, for an optional new session anchor."""
        return self._last_clean_latent
