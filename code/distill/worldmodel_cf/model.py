"""Weight-compatible causal form of the video-only Vampire world model.

The bidirectional teacher and causal student share an identical parameter state dict.
Only self-attention geometry changes; the StatePlay state branch is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


WORLD_MODEL_ROOT = Path(__file__).resolve().parents[2] / "world_model"
if str(WORLD_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT))

from diffsynth.models.wan_video_dit import (  # noqa: E402
    rope_apply,
    sinusoidal_embedding_1d,
)
from diffsynth.models.world_model_dit import (  # noqa: E402
    temporal_prompt_attention,
    zero_temporal_prefix,
)
from world_model.models.dit import (  # noqa: E402
    WAN_MODEL_DEFAULTS,
    WorldModelDiT,
)


def worldmodel_config(tiny: bool = False) -> dict:
    cfg = dict(WAN_MODEL_DEFAULTS)
    if tiny:
        cfg.update(dim=128, ffn_dim=256, num_heads=4, num_layers=2, text_dim=64)
    return cfg


def _sdpa(q, k, v, heads, mask=None):
    def split(value):
        batch, sequence, width = value.shape
        return value.view(batch, sequence, heads, width // heads).transpose(1, 2)

    output = F.scaled_dot_product_attention(
        split(q), split(k), split(v), attn_mask=mask,
    )
    return output.transpose(1, 2).flatten(2)


def _causal_mask(q_frames, k_frames, device, q_tokens=1, k_tokens=1):
    query_frame = torch.arange(q_frames, device=device).repeat_interleave(q_tokens)
    key_frame = torch.arange(k_frames, device=device).repeat_interleave(k_tokens)
    return (key_frame[None, :] <= query_frame[:, None])[None, None]


def _teacher_forcing_mask(frames, device, tokens_per_frame=1):
    """CF++ mask for the concatenated ``[clean F | noisy F]`` streams."""
    index = torch.arange(2 * frames, device=device).repeat_interleave(tokens_per_frame)
    is_clean = index < frames
    frame = index.remainder(frames)
    allowed = (
        (is_clean[:, None] & is_clean[None, :] & (frame[None, :] <= frame[:, None]))
        | (~is_clean[:, None] & ~is_clean[None, :] & (frame[None, :] == frame[:, None]))
        | (~is_clean[:, None] & is_clean[None, :] & (frame[None, :] < frame[:, None]))
    )
    return allowed[None, None]


def _self_attention(module, x, freqs, mask, teacher_forcing=False):
    query = module.norm_q(module.q(x))
    key = module.norm_k(module.k(x))
    value = module.v(x)
    if teacher_forcing:
        query_clean, query_noisy = query.chunk(2, dim=1)
        key_clean, key_noisy = key.chunk(2, dim=1)
        query = torch.cat([
            rope_apply(query_clean, freqs, module.num_heads),
            rope_apply(query_noisy, freqs, module.num_heads),
        ], dim=1)
        key = torch.cat([
            rope_apply(key_clean, freqs, module.num_heads),
            rope_apply(key_noisy, freqs, module.num_heads),
        ], dim=1)
    else:
        query = rope_apply(query, freqs, module.num_heads)
        key = rope_apply(key, freqs, module.num_heads)
    return module.o(_sdpa(query, key, value, module.num_heads, mask))


def _plain_attention(module, x, context):
    query = module.norm_q(module.q(x))
    key = module.norm_k(module.k(context))
    value = module.v(context)
    return module.o(_sdpa(query, key, value, module.num_heads))


def _temporal_cross(module, x, context, frames, teacher_forcing=False):
    query = module.norm_q(module.q(x))
    key = module.norm_k(module.k(context))
    value = module.v(context)

    def one(stream_query):
        output = temporal_prompt_attention(
            stream_query,
            key,
            value,
            module.num_heads,
            frames,
            num_frames=80,
            condition_prefix=1,
        )
        return zero_temporal_prefix(module.o(output), frames, 1)

    if teacher_forcing:
        clean, noisy = query.chunk(2, dim=1)
        return torch.cat([one(clean), one(noisy)], dim=1)
    return one(query)


class CausalWorldModelDiT(WorldModelDiT):
    """Bidirectional teacher or parameter-identical causal video student."""

    def __init__(self, causal=True, **kwargs):
        self.causal = bool(causal)
        super().__init__(**kwargs)
        self.use_gradient_checkpointing = False

    def _freqs(self, frames, height, width, device, frame_start=0):
        return torch.cat([
            self.freqs[0][frame_start:frame_start + frames]
            .view(frames, 1, 1, -1).expand(frames, height, width, -1),
            self.freqs[1][:height].view(1, height, 1, -1)
            .expand(frames, height, width, -1),
            self.freqs[2][:width].view(1, 1, width, -1)
            .expand(frames, height, width, -1),
        ], dim=-1).reshape(frames * height * width, 1, -1).to(device)

    def _time(self, timestep, frames, spatial, dtype, clean_timestep=None):
        batch = timestep.shape[0]
        if timestep.ndim == 1:
            noisy = timestep[:, None].expand(batch, frames)
        else:
            if timestep.shape[1] != frames:
                raise ValueError(
                    f"timestep has {timestep.shape[1]} frames, expected {frames}"
                )
            noisy = timestep
        if clean_timestep is not None:
            if clean_timestep.ndim == 1:
                clean_timestep = clean_timestep[:, None].expand_as(noisy)
            stream = torch.cat([clean_timestep, noisy], dim=1)
        else:
            stream = noisy
        token_timestep = stream[:, :, None].expand(
            batch, stream.shape[1], spatial,
        ).reshape(-1)
        embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timestep).to(dtype)
        )
        embedding = embedding.view(batch, stream.shape[1] * spatial, self.dim)
        modulation = self.time_projection(embedding).unflatten(2, (6, self.dim))
        return embedding, modulation

    def _block(
        self, block, video, context, time_mod, freqs, mask, frames,
        teacher_forcing,
    ):
        chunks = tuple(
            chunk.squeeze(2)
            for chunk in (
                block.modulation.unsqueeze(1).to(time_mod) + time_mod
            ).chunk(6, dim=2)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        normalized = block.norm1(video) * (1 + scale_msa) + shift_msa
        video = video + gate_msa * _self_attention(
            block.self_attn, normalized, freqs, mask,
            teacher_forcing=teacher_forcing,
        )
        video = video + _temporal_cross(
            block.cross_attn,
            block.norm3(video),
            context,
            frames,
            teacher_forcing=teacher_forcing,
        )
        video = video + gate_mlp * block.ffn(
            block.norm2(video) * (1 + scale_mlp) + shift_mlp
        )
        return video

    def forward(
        self,
        latents,
        timestep,
        context,
        clean_x=None,
        aug_t=None,
        frame_index=None,
        kv_cache=None,
        commit=True,
        **kwargs,
    ):
        if not self.causal:
            if clean_x is not None or kv_cache is not None:
                raise ValueError("bidirectional world model does not accept TF/KV inputs")
            if timestep.ndim == 2:
                target = timestep[:, 1:]
                if not torch.allclose(target, target[:, :1].expand_as(target)):
                    raise ValueError(
                        "bidirectional scoring requires one shared target timestep"
                    )
                timestep = target[:, 0]
            return super().forward(
                latents,
                timestep,
                context,
                fuse_vae_embedding_in_latents=True,
                prompt_num_frames=80,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                **kwargs,
            )
        if kv_cache is not None:
            return self.forward_frame(
                latents,
                timestep,
                context,
                int(frame_index),
                kv_cache,
                commit=commit,
            )

        teacher_forcing = clean_x is not None
        patched = self.patchify(latents)
        batch, _, frames, height, width = patched.shape
        spatial = height * width
        video = patched.flatten(2).transpose(1, 2)
        if teacher_forcing:
            clean_video = self.patchify(clean_x).flatten(2).transpose(1, 2)
            video = torch.cat([clean_video, video], dim=1)
        clean_timestep = (
            torch.zeros_like(timestep) if aug_t is None else aug_t
        ) if teacher_forcing else None
        time_embedding, time_mod = self._time(
            timestep, frames, spatial, video.dtype, clean_timestep,
        )
        context = self.text_embedding(context)
        freqs = self._freqs(frames, height, width, video.device)
        mask = (
            _teacher_forcing_mask(frames, video.device, spatial)
            if teacher_forcing
            else _causal_mask(frames, frames, video.device, spatial, spatial)
        )
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                def run(value, selected_block=block):
                    return self._block(
                        selected_block,
                        value,
                        context,
                        time_mod,
                        freqs,
                        mask,
                        frames,
                        teacher_forcing,
                    )

                video = checkpoint(run, video, use_reentrant=False)
            else:
                video = self._block(
                    block,
                    video,
                    context,
                    time_mod,
                    freqs,
                    mask,
                    frames,
                    teacher_forcing,
                )
        if teacher_forcing:
            video = video[:, frames * spatial:]
            time_embedding = time_embedding[:, frames * spatial:]
        prediction = self.head(video, time_embedding)
        return self.unpatchify(prediction, (frames, height, width))

    def init_kv_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    @staticmethod
    def trim_kv_cache_attention_sink(
        cache,
        tokens_per_latent,
        *,
        sink_latents=3,
        window_latents=17,
    ):
        """Keep the first sink latents and the newest sliding-window latents.

        KV entries are flattened spatial tokens, not one entry per video
        latent, so every slice is made in whole-latent token blocks. The
        method mutates the cache in place and returns the retained latent
        count. With the inference defaults this is a strict 20-latent cache:
        the first 3 latents never move and the newest 17 form the window.
        """
        tokens_per_latent = int(tokens_per_latent)
        sink_latents = int(sink_latents)
        window_latents = int(window_latents)
        if tokens_per_latent <= 0:
            raise ValueError("tokens_per_latent must be positive")
        if sink_latents < 0 or window_latents <= 0:
            raise ValueError("sink_latents must be >= 0 and window_latents must be > 0")

        max_latents = sink_latents + window_latents
        observed_latents = None
        for layer_index, slot in enumerate(cache):
            key, value = slot["k"], slot["v"]
            if key is None or value is None:
                if key is not None or value is not None:
                    raise ValueError(f"incomplete KV cache at layer {layer_index}")
                continue
            if key.shape[1] != value.shape[1]:
                raise ValueError(f"K/V token count mismatch at layer {layer_index}")
            if key.shape[1] % tokens_per_latent:
                raise ValueError(
                    f"layer {layer_index} has {key.shape[1]} tokens, not divisible by "
                    f"tokens_per_latent={tokens_per_latent}"
                )
            latent_count = key.shape[1] // tokens_per_latent
            if observed_latents is None:
                observed_latents = latent_count
            elif observed_latents != latent_count:
                raise ValueError("KV layers do not contain the same number of latents")
            if latent_count <= max_latents:
                continue

            sink_tokens = sink_latents * tokens_per_latent
            window_tokens = window_latents * tokens_per_latent
            for name in ("k", "v"):
                tensor = slot[name]
                if sink_tokens:
                    tensor = torch.cat(
                        (tensor[:, :sink_tokens], tensor[:, -window_tokens:]), dim=1,
                    )
                else:
                    tensor = tensor[:, -window_tokens:]
                slot[name] = tensor.detach()
        if observed_latents is None:
            return 0
        return min(observed_latents, max_latents)

    @staticmethod
    def _cached_attention(
        module, x, slot, *, rope=None, commit=False,
    ):
        query = module.norm_q(module.q(x))
        key = module.norm_k(module.k(x))
        value = module.v(x)
        if rope is not None:
            query = rope_apply(query, rope, module.num_heads)
            key = rope_apply(key, rope, module.num_heads)
        all_key = key if slot["k"] is None else torch.cat([slot["k"], key], dim=1)
        all_value = value if slot["v"] is None else torch.cat(
            [slot["v"], value], dim=1,
        )
        output = module.o(_sdpa(
            query, all_key, all_value, module.num_heads,
        ))
        if commit:
            slot["k"], slot["v"] = all_key.detach(), all_value.detach()
        return output

    def forward_frame(
        self, latent, timestep, context, frame_index, cache, commit=True,
    ):
        if latent.shape[2] != 1:
            raise ValueError("forward_frame expects exactly one video latent")
        patched = self.patchify(latent)
        batch, _, _, height, width = patched.shape
        spatial = height * width
        video = patched.flatten(2).transpose(1, 2)
        time_embedding, time_mod = self._time(
            timestep.reshape(batch), 1, spatial, video.dtype,
        )
        context = self.text_embedding(context)
        if context.ndim == 4:
            if context.shape[1] != 1:
                raise ValueError("forward_frame expects one temporal prompt cell")
            context = context[:, 0]
        freqs = self._freqs(
            1, height, width, video.device, frame_start=frame_index,
        )
        for layer_index, block in enumerate(self.blocks):
            # Later rollout frames replace entries in ``cache`` before DMD calls
            # backward. Checkpoint recomputation must retain this forward prefix.
            cache_snapshot = {
                "k": cache[layer_index]["k"],
                "v": cache[layer_index]["v"],
            }

            def run(
                value, selected_block=block, selected_cache=cache_snapshot,
            ):
                chunks = tuple(
                    chunk.squeeze(2)
                    for chunk in (
                        selected_block.modulation.unsqueeze(1).to(time_mod) + time_mod
                    ).chunk(6, dim=2)
                )
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
                normalized = selected_block.norm1(value) * (1 + scale_msa) + shift_msa
                value = value + gate_msa * self._cached_attention(
                    selected_block.self_attn,
                    normalized,
                    selected_cache,
                    rope=freqs,
                    commit=commit,
                )
                if frame_index > 0:
                    value = value + _plain_attention(
                        selected_block.cross_attn,
                        selected_block.norm3(value),
                        context,
                    )
                return value + gate_mlp * selected_block.ffn(
                    selected_block.norm2(value) * (1 + scale_mlp) + shift_mlp
                )

            if (
                self.use_gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                video = checkpoint(run, video, use_reentrant=False)
            else:
                video = run(video)
                if commit:
                    cache[layer_index]["k"] = cache_snapshot["k"]
                    cache[layer_index]["v"] = cache_snapshot["v"]
        prediction = self.head(video, time_embedding)
        return self.unpatchify(prediction, (1, height, width))


def load_checkpoint(model, spec):
    """Resolve a checkpoint spec, then enforce exact key compatibility."""
    from safetensors.torch import load_file

    from .checkpoints import validate_checkpoint

    expected = set(model.state_dict())
    path = validate_checkpoint(spec, expected_keys=expected)
    result = model.load_state_dict(load_file(path), strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {result}")
    return model, path
