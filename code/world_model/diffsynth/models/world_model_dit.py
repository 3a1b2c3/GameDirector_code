"""Temporally prompted Wan video world model."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import (
    GateModule,
    RMSNorm,
    SelfAttention,
    WanModel,
    modulate,
)


def temporal_prompt_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    temporal_size: int,
    num_frames: Optional[int] = None,
    condition_prefix: int = 0,
) -> torch.Tensor:
    """Attend each predicted video latent slice only to overlapping prompts."""
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "temporal_prompt_attention expects q=[B,L,D], k/v=[B,P,S,D]; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    batch, query_len, inner_dim = q.shape
    if temporal_size <= 0 or query_len % temporal_size:
        raise ValueError(
            f"query length {query_len} must be divisible by temporal_size={temporal_size}"
        )
    if condition_prefix < 0 or condition_prefix >= temporal_size:
        raise ValueError(
            f"condition_prefix={condition_prefix} is invalid for temporal_size={temporal_size}"
        )
    if k.shape != v.shape or k.shape[0] != batch:
        raise ValueError(
            f"temporal prompt K/V or batch mismatch: q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if inner_dim % num_heads:
        raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")

    num_prompts, text_tokens = k.shape[1:3]
    predicted_size = temporal_size - condition_prefix
    raw_frames = int(num_frames or num_prompts)
    if raw_frames <= 0 or num_prompts <= 0:
        raise ValueError(f"invalid temporal layout: frames={raw_frames}, prompts={num_prompts}")

    predicted_edges = torch.linspace(
        0, raw_frames, predicted_size + 1, device=q.device, dtype=torch.float32,
    )
    prompt_edges = torch.linspace(
        0, raw_frames, num_prompts + 1, device=q.device, dtype=torch.float32,
    )
    predicted_start, predicted_end = predicted_edges[:-1, None], predicted_edges[1:, None]
    prompt_start, prompt_end = prompt_edges[None, :-1], prompt_edges[None, 1:]
    overlap = (
        torch.minimum(predicted_end, prompt_end)
        - torch.maximum(predicted_start, prompt_start)
    ).clamp_min(0)
    max_overlaps = int((overlap > 0).sum(dim=1).max().item())
    if max_overlaps <= 0:
        raise RuntimeError("no temporal overlap between video slices and prompt cells")
    overlap_values, prompt_indices = overlap.topk(
        max_overlaps, dim=1, largest=True, sorted=True,
    )

    spatial_tokens = query_len // temporal_size
    head_dim = inner_dim // num_heads
    q_time = q.view(batch, temporal_size, spatial_tokens, inner_dim)
    q_pred = q_time[:, condition_prefix:]
    qh = q_pred.view(
        batch, predicted_size, spatial_tokens, num_heads, head_dim,
    ).permute(0, 1, 3, 2, 4).reshape(
        batch * predicted_size, num_heads, spatial_tokens, head_dim,
    )
    kh = k.view(batch, num_prompts, text_tokens, num_heads, head_dim)
    vh = v.view(batch, num_prompts, text_tokens, num_heads, head_dim)
    kh = kh[:, prompt_indices].permute(0, 1, 4, 2, 3, 5).reshape(
        batch * predicted_size, num_heads, max_overlaps * text_tokens, head_dim,
    )
    vh = vh[:, prompt_indices].permute(0, 1, 4, 2, 3, 5).reshape(
        batch * predicted_size, num_heads, max_overlaps * text_tokens, head_dim,
    )

    predicted_width = (predicted_end - predicted_start).clamp_min(1e-6)
    overlap_prior = overlap_values / predicted_width
    overlap_bias = torch.where(
        overlap_values > 0,
        overlap_prior.clamp_min(1e-6).log(),
        torch.full_like(overlap_prior, torch.finfo(q.dtype).min),
    ).to(dtype=q.dtype)
    overlap_bias = overlap_bias.repeat_interleave(text_tokens, dim=1)
    overlap_bias = overlap_bias.unsqueeze(0).expand(batch, -1, -1).reshape(
        batch * predicted_size, 1, 1, max_overlaps * text_tokens,
    )
    out_pred = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=overlap_bias)
    out_pred = out_pred.reshape(
        batch, predicted_size, num_heads, spatial_tokens, head_dim,
    ).permute(0, 1, 3, 2, 4).reshape(
        batch, predicted_size, spatial_tokens, inner_dim,
    )
    prefix = torch.zeros_like(q_time[:, :condition_prefix])
    return torch.cat([prefix, out_pred], dim=1).reshape(batch, query_len, inner_dim)


def zero_temporal_prefix(
    x: torch.Tensor, temporal_size: int, condition_prefix: int,
) -> torch.Tensor:
    """Remove output-projection bias from clean conditioning slices."""
    if condition_prefix <= 0:
        return x
    if x.shape[1] % temporal_size:
        raise ValueError(
            f"sequence length {x.shape[1]} is not divisible by temporal_size={temporal_size}"
        )
    prefix_tokens = condition_prefix * (x.shape[1] // temporal_size)
    return torch.cat([torch.zeros_like(x[:, :prefix_tokens]), x[:, prefix_tokens:]], dim=1)


class TemporalCrossAttention(nn.Module):
    """Wan-compatible cross-attention with local temporal prompt routing."""

    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6,
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        temporal_size: int,
        num_frames: int,
        condition_prefix: int = 1,
    ) -> torch.Tensor:
        if context.ndim != 4:
            raise ValueError(
                f"temporal context must be [B,P,S,C], got {tuple(context.shape)}"
            )
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        out = temporal_prompt_attention(
            q, k, v, self.num_heads, temporal_size,
            num_frames=num_frames, condition_prefix=condition_prefix,
        )
        out = self.o(out)
        return zero_temporal_prefix(out, temporal_size, condition_prefix)


class TemporalVideoDiTBlock(nn.Module):
    """Wan DiT block whose cross-attention routes temporal prompts locally."""

    def __init__(
        self, has_image_input: bool, dim: int, num_heads: int,
        ffn_dim: int, eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = TemporalCrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input,
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        t_mod_video: torch.Tensor,
        video_freqs: torch.Tensor,
        num_frames: int,
        video_temporal_size: int,
        condition_prefix: int = 1,
    ) -> torch.Tensor:
        has_sequence_timestep = t_mod_video.ndim == 4
        chunk_dim = 2 if has_sequence_timestep else 1
        chunks = (
            self.modulation.to(dtype=t_mod_video.dtype, device=t_mod_video.device)
            + t_mod_video
        ).chunk(6, dim=chunk_dim)
        if has_sequence_timestep:
            chunks = tuple(chunk.squeeze(2) for chunk in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        input_video = modulate(self.norm1(video), shift_msa, scale_msa)
        video = self.gate(video, gate_msa, self.self_attn(input_video, video_freqs))
        video = video + self.cross_attn(
            self.norm3(video), context,
            temporal_size=video_temporal_size,
            num_frames=num_frames,
            condition_prefix=condition_prefix,
        )
        input_video = modulate(self.norm2(video), shift_mlp, scale_mlp)
        return self.gate(video, gate_mlp, self.ffn(input_video))


class WorldModel(WanModel):
    """Wan2.2 video world model conditioned by a first frame and 20 prompts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_layers = len(self.blocks)
        ffn_dim = self.blocks[0].ffn_dim
        eps = self.blocks[0].norm1.eps
        num_heads = self.blocks[0].num_heads
        self.blocks = nn.ModuleList([
            TemporalVideoDiTBlock(
                self.has_image_input, self.dim, num_heads, ffn_dim, eps,
            )
            for _ in range(num_layers)
        ])


__all__ = [
    "TemporalCrossAttention", "TemporalVideoDiTBlock", "WorldModel",
    "temporal_prompt_attention", "zero_temporal_prefix",
]
