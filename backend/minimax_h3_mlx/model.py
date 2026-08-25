"""MLX-native MiniMax H3 transformer primitives.

The mathematics and layouts follow MiniMax's released configuration and the
ComfyUI reference implementation. Arrays use H3's native channel-first latent
layouts at the boundary and sequence-last-feature layouts inside the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


@dataclass(frozen=True)
class H3Config:
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0
    # Exact-BF16 layout optimization. Long full-attention kernels benefit from
    # head-major Q/K/V on M4; short token-refiner attention does not.
    head_major_sdpa: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "H3Config":
        raw = json.loads(Path(path).read_text())
        aliases = {
            "token_refiner_num_layers": "num_refiner_layers",
            "ffn_hidden_size": "ffn_dim",
            "latents_dim": "in_channels",
            "audio_latents_dim": "audio_in_channels",
            "timestep_input_dim": "freq_dim",
            "time_embed_hidden_size": "time_embed_hidden_dim",
            "rope_inv_freq_len": "rope_freq_dim",
        }
        for source, target in aliases.items():
            if source in raw:
                raw[target] = raw[source]
        keys = cls.__dataclass_fields__.keys()
        values = {k: raw[k] for k in keys if k in raw}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        return cls(**values)


def time_shift_sigma(sigma: mx.array, from_shift: float, to_shift: float) -> mx.array:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_sigma_float(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(x: mx.array, patch_size: Sequence[int] = (1, 2, 2)) -> mx.array:
    """[B,C,T,H,W] -> [B*T'*H'*W', C*pt*ph*pw]."""
    b, c, tf, hf, wf = x.shape
    pt, ph, pw = patch_size
    if tf % pt or hf % ph or wf % pw:
        raise ValueError("video latent dimensions must be divisible by patch_size")
    t, h, w = tf // pt, hf // ph, wf // pw
    x = x.reshape(b, c, t, pt, h, ph, w, pw)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(
    rows: mx.array,
    t: int,
    h: int,
    w: int,
    channels: int = 24,
    patch_size: Sequence[int] = (1, 2, 2),
) -> mx.array:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, channels, pt, ph, pw)
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, channels, t * pt, h * ph, w * pw)


def pack_audio(x: mx.array) -> mx.array:
    """[1,32,2,T] -> [2*T,32], preserving channel-major ordering."""
    if x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    _, channels, stereo, steps = x.shape
    return x[0].transpose(1, 2, 0).reshape(stereo * steps, channels)


def unpack_audio(rows: mx.array, stereo: int = 2) -> mx.array:
    steps = rows.shape[0] // stereo
    return rows.reshape(stereo, steps, rows.shape[-1]).transpose(2, 0, 1)[None]


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def apply_split_half_rope(x: mx.array, angles: mx.array, rot_dim: int) -> mx.array:
    """Apply H3 partial split-half RoPE to [S,H,D] q or k arrays."""
    if rot_dim % 2:
        raise ValueError("rot_dim must be even")
    half = rot_dim // 2
    rotated, tail = x[..., :rot_dim], x[..., rot_dim:]
    left, right = rotated[..., :half], rotated[..., half:]
    c = mx.cos(angles[:, None, :half]).astype(x.dtype)
    s = mx.sin(angles[:, None, :half]).astype(x.dtype)
    rotated = mx.concatenate((left * c - right * s, right * c + left * s), axis=-1)
    return mx.concatenate((rotated, tail), axis=-1)


def reorder_per_head_qkv(weight: mx.array, heads: int, head_dim: int) -> mx.array:
    """Per-head ``[q,k,v]`` rows -> global ``[all-q;all-k;all-v]`` rows."""
    if weight.shape[0] != heads * 3 * head_dim:
        raise ValueError(f"unexpected QKV shape: {weight.shape}")
    rows = weight.reshape(heads, 3, head_dim, weight.shape[1])
    return mx.concatenate(
        [rows[:, index].reshape(heads * head_dim, weight.shape[1]) for index in range(3)],
        axis=0,
    )


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float,
                 head_major_sdpa: bool = True):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.head_major_sdpa = head_major_sdpa
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
        profiler=None,
    ) -> mx.array:
        seq = x.shape[0]
        project = lambda: self.qkv_proj(x)
        qkv = profiler.measure("attention_qkv_projection", project) if profiler else project()
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(seq, self.heads, self.head_dim)
        k = k.reshape(seq, self.heads, self.head_dim)
        v = v.reshape(seq, self.heads, self.head_dim)

        def normalize_and_rotate():
            q_local = _rms_norm(q, self.q_norm.weight, self.q_norm.eps)
            k_local = _rms_norm(k, self.k_norm.weight, self.k_norm.eps)
            if rope is not None:
                cos, sin = rope
                rot_dim = cos.shape[-1] * 2
                half = rot_dim // 2
                values = []
                for value in (q_local, k_local):
                    rotated, tail = value[..., :rot_dim], value[..., rot_dim:]
                    left, right = rotated[..., :half], rotated[..., half:]
                    rotated = mx.concatenate(
                        (left * cos - right * sin, right * cos + left * sin), axis=-1
                    )
                    values.append(mx.concatenate((rotated, tail), axis=-1))
                q_local, k_local = values
            return q_local, k_local

        q, k = (profiler.measure("qk_norm_and_rope", normalize_and_rotate)
                if profiler else normalize_and_rotate())
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        # For H3's 15k-token full attention, sequence-major views make each
        # head stride across interleaved storage (and V also retains the fused
        # QKV row gap). A one-time exact BF16 pack costs ~4 ms but saves ~22 ms
        # in the same MLX fused SDPA call. Short refiner attention is faster
        # without packing, so it deliberately stays on the original layout.
        if self.head_major_sdpa and seq >= 1024:
            q, k, v = mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)
        attend = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=None)
        y = profiler.measure("attention_sdpa", attend) if profiler else attend()
        output = lambda: self.out_proj(y[0].transpose(1, 0, 2).reshape(seq, -1))
        return (profiler.measure("attention_output_projection", output)
                if profiler else output())


class MLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x: mx.array, profiler=None) -> mx.array:
        expand = lambda: self.fc1(x)
        expanded = profiler.measure("mlp_input_projection", expand) if profiler else expand()
        gate, value = mx.split(expanded, 2, axis=-1)
        activate = lambda: nn.silu(gate) * value
        activated = profiler.measure("mlp_activation", activate) if profiler else activate()
        contract = lambda: self.fc2(activated)
        return profiler.measure("mlp_output_projection", contract) if profiler else contract()


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim: int, hidden: int, out: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden)
        self.proj_out = nn.Linear(hidden, out)

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t.astype(mx.float32)[:, None] * freqs[None]
        emb = mx.concatenate((mx.cos(args), mx.sin(args)), axis=-1)
        return self.proj_out(nn.silu(self.proj_in(emb)))


class AdalnProj(nn.Module):
    def __init__(
        self, t_dim: int, hidden: int, expand: int, modalities: int, apply_silu: bool = True
    ):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities)

    def __call__(self, t_emb: mx.array) -> list[mx.array]:
        # The released mixed-precision model keeps the shared timestep
        # embedding in FP32 through SiLU.  Only the input to each BF16 AdaLN
        # projection is rounded.  Rounding before SiLU introduces the same
        # bias in every block and compounds over the denoising trajectory.
        x = nn.silu(t_emb) if self.apply_silu else t_emb
        x = x.astype(self.linear.weight.dtype)
        x = self.linear(x).reshape(t_emb.shape[0] * self.modalities, self.expand * self.hidden)
        return list(mx.split(x, self.expand, axis=-1))


def _replace_segments(base: mx.array, replacements: Iterable[tuple[int, int, mx.array]]) -> mx.array:
    """Functional replacement for MLX arrays (MLX has no PyTorch-style in-place ops)."""
    parts: list[mx.array] = []
    cursor = 0
    for start, stop, value in replacements:
        if start != cursor:
            raise ValueError("segments must be sorted, contiguous, and cover the sequence")
        parts.append(value)
        cursor = stop
    if cursor != base.shape[0]:
        raise ValueError("segments do not cover the sequence")
    return mx.concatenate(parts, axis=0)


def mod_scale_shift(
    h: mx.array,
    shift: mx.array,
    scale: mx.array,
    segments: Sequence[tuple[int, int, int]],
) -> mx.array:
    return _replace_segments(
        h,
        ((a, b, h[a:b] * (1 + scale[row].astype(h.dtype)) + shift[row].astype(h.dtype))
         for a, b, row in segments),
    )


_FUSED_BF16_ADDCMUL_KERNELS = {}


def _fused_bf16_addcmul(base: mx.array, branch: mx.array, gate: mx.array) -> mx.array:
    """Match PyTorch/MPS BF16 ``base.addcmul_(branch, gate)`` rounding."""
    if base.dtype != mx.bfloat16 or branch.dtype != mx.bfloat16 or gate.dtype != mx.bfloat16:
        return base + branch * gate
    width = base.shape[-1]
    kernel = _FUSED_BF16_ADDCMUL_KERNELS.get(width)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"h3_bf16_addcmul_{width}",
            input_names=["base", "branch", "gate"],
            output_names=["out"],
            source=f"""
                uint elem = thread_position_in_grid.x;
                out[elem] = metal::fma(branch[elem], gate[elem % {width}], base[elem]);
            """,
        )
        _FUSED_BF16_ADDCMUL_KERNELS[width] = kernel
    return kernel(
        inputs=[base, branch, gate],
        template=[("T", mx.bfloat16)],
        grid=(base.size, 1, 1),
        threadgroup=(min(256, base.size), 1, 1),
        output_shapes=[base.shape],
        output_dtypes=[base.dtype],
    )[0]


def mod_gate(
    x: mx.array,
    gate: mx.array,
    other: mx.array,
    segments: Sequence[tuple[int, int, int]],
) -> mx.array:
    return _replace_segments(
        x,
        ((_a, _b, _fused_bf16_addcmul(
            x[_a:_b], other[_a:_b], gate[_row].astype(x.dtype)
        ))
         for _a, _b, _row in segments),
    )


class RefinerBlock(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = Attention(cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim,
                              cfg.qk_norm_eps, cfg.head_major_sdpa)
        self.mlp = MLP(cfg.hidden_size, cfg.ffn_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TokenRefiner(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.blocks = [RefinerBlock(cfg) for _ in range(cfg.num_refiner_layers)]
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attn = Attention(cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim,
                              cfg.qk_norm_eps, cfg.head_major_sdpa)
        self.mlp = MLP(cfg.hidden_size, cfg.ffn_dim)
        self.adaln_proj = AdalnProj(cfg.time_embed_dim, cfg.hidden_size, 6, 3)

    def __call__(self, x, t_emb, segments, rope, profiler=None):
        adaln = lambda: self.adaln_proj(t_emb)
        params = profiler.measure("adaln_projection", adaln) if profiler else adaln()
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params
        norm_attn = lambda: mod_scale_shift(self.norm1(x), shift_a, scale_a, segments)
        h = profiler.measure("normalization_and_modulation", norm_attn) if profiler else norm_attn()
        attn = self.attn(h, rope, profiler=profiler)
        residual_attn = lambda: mod_gate(x, gate_a, attn, segments)
        x = profiler.measure("residual_and_gating", residual_attn) if profiler else residual_attn()
        norm_mlp = lambda: mod_scale_shift(self.norm2(x), shift_m, scale_m, segments)
        h = profiler.measure("normalization_and_modulation", norm_mlp) if profiler else norm_mlp()
        mlp = self.mlp(h, profiler=profiler)
        residual_mlp = lambda: mod_gate(x, gate_m, mlp, segments)
        return profiler.measure("residual_and_gating", residual_mlp) if profiler else residual_mlp()


class FinalLayer(nn.Module):
    def __init__(self, cfg: H3Config):
        super().__init__()
        patch_volume = math.prod(cfg.patch_size)
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.final_norm_eps)
        self.adaln_proj = AdalnProj(cfg.time_embed_dim, cfg.hidden_size, 2, 1)
        self.video_out = nn.Linear(cfg.hidden_size, cfg.in_channels * patch_volume)
        self.audio_out = nn.Linear(cfg.hidden_size, cfg.audio_in_channels)

    def __call__(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vr = video_seg
        aa, ab, ar = audio_seg
        normed = self.norm(x)
        hv = (normed[va:vb] * (1 + scale[vr]) + shift[vr]).astype(mx.float32)
        ha = (normed[aa:ab] * (1 + scale[ar]) + shift[ar]).astype(mx.float32)
        return self.video_out(hv), self.audio_out(ha)


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> list[float]:
    ratio = dim / sqrt_area
    count = dim // patch
    return [(i * ratio / count + (1.0 - ratio) / 2.0) * 32.0 for i in range(count)]


def _video_t_spans(count: int) -> list[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(count)]


class T2VLayout:
    """Static H3 packed layout for the shortest text-to-video path.

    Packing is exactly ``[text | target audio | target video]``. Position grids
    are built once on the CPU as Python values and converted to one MLX array,
    so sampling steps do not synchronize the GPU to inspect coordinates.
    """

    def __init__(self, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int):
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even for 2x2 patching")
        area = math.sqrt(latent_h * latent_w)
        hs = _axis_from_sqrt_area(latent_h, 2, area)
        ws = _axis_from_sqrt_area(latent_w, 2, area)
        frame = [(h, w) for h in hs for w in ws]
        cursor = float(text_len)

        positions: list[tuple[float, float, float]] = [
            (float(i), 0.0, 0.0) for i in range(text_len)
        ]
        w_low, w_high = ws[0], ws[-1]
        positions.extend((cursor + i, 0.0, w_low) for i in range(audio_t))
        positions.extend((cursor + i, 0.0, w_high) for i in range(audio_t))

        spans = _video_t_spans(latent_t)
        video_times: list[float] = []
        running = cursor
        for span in spans:
            video_times.append(running)
            running += span
        positions.extend((t, h, w) for t in video_times for h, w in frame)

        self.text = (0, text_len, "text")
        self.audio = (text_len, text_len + audio_t * 2, "audio")
        self.video = (
            self.audio[1],
            self.audio[1] + latent_t * len(frame),
            "video",
        )
        self.segments = (self.text, self.audio, self.video)
        self.position_ids = mx.array(positions, dtype=mx.float32)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.seq_len = len(positions)

    def modulation_segments(
        self,
        timestep_rows: dict[str, int],
        text_token_tags: Sequence[int] | None = None,
    ) -> list[tuple[int, int, int]]:
        result: list[tuple[int, int, int]] = []
        ta, tb, _ = self.text
        if text_token_tags is None:
            result.append((ta, tb, timestep_rows["text"] * 3 + 1))
        else:
            if len(text_token_tags) != tb - ta:
                raise ValueError("text_token_tags length must equal context length")
            start = 0
            for i in range(1, len(text_token_tags) + 1):
                if i == len(text_token_tags) or text_token_tags[i] != text_token_tags[start]:
                    result.append((ta + start, ta + i, timestep_rows["text"] * 3 + int(text_token_tags[start])))
                    start = i
        aa, ab, _ = self.audio
        va, vb, _ = self.video
        result.append((aa, ab, timestep_rows["audio"] * 3 + 2))
        result.append((va, vb, timestep_rows["video"] * 3))
        return result


class FL2VLayout:
    """Official ``[text | keyframes | audio | video]`` first/last-frame layout."""

    def __init__(self, text_len: int, latent_t: int, latent_h: int, latent_w: int,
                 audio_t: int, keyframe_anchors: tuple[str, ...] = ("first",)):
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even for 2x2 patching")
        area = math.sqrt(latent_h * latent_w)
        hs = _axis_from_sqrt_area(latent_h, 2, area)
        ws = _axis_from_sqrt_area(latent_w, 2, area)
        frame = [(h, w) for h in hs for w in ws]
        cursor = float(text_len)
        positions = [(float(i), 0.0, 0.0) for i in range(text_len)]
        spans = _video_t_spans(latent_t)
        # Official Diffusers reference uses NumPy's pairwise float64 sum for
        # the last-frame anchor (distinct from the sequential target clock).
        last_time = cursor + float(np.asarray(spans, dtype=np.float64).sum()) - FRAME_RESCALE
        for anchor in keyframe_anchors:
            if anchor == "first":
                anchor_time = cursor
            elif anchor == "last":
                anchor_time = last_time
            else:
                raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")
            positions.extend((anchor_time, h, w) for h, w in frame)
        w_low, w_high = ws[0], ws[-1]
        positions.extend((cursor + i, 0.0, w_low) for i in range(audio_t))
        positions.extend((cursor + i, 0.0, w_high) for i in range(audio_t))
        running = cursor
        for span in spans:
            positions.extend((running, h, w) for h, w in frame)
            running += span

        n_frame = len(frame)
        self.text = (0, text_len, "text")
        self.condition = (text_len, text_len + len(keyframe_anchors) * n_frame, "cond")
        self.audio = (self.condition[1], self.condition[1] + audio_t * 2, "audio")
        self.video = (self.audio[1], self.audio[1] + latent_t * n_frame, "video")
        self.segments = (self.text, self.condition, self.audio, self.video)
        self.position_ids = mx.array(positions, dtype=mx.float32)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.keyframe_anchors = keyframe_anchors
        self.seq_len = len(positions)

    def modulation_segments(self, timestep_rows, text_token_tags=None):
        result = []
        ta, tb, _ = self.text
        if text_token_tags is None:
            result.append((ta, tb, timestep_rows["text"] * 3 + 1))
        else:
            start = 0
            for i in range(1, len(text_token_tags) + 1):
                if i == len(text_token_tags) or text_token_tags[i] != text_token_tags[start]:
                    result.append((ta + start, ta + i, timestep_rows["text"] * 3 + int(text_token_tags[start])))
                    start = i
        ca, cb, _ = self.condition
        aa, ab, _ = self.audio
        va, vb, _ = self.video
        result.extend(((ca, cb, timestep_rows["cond"] * 3),
                       (aa, ab, timestep_rows["audio"] * 3 + 2),
                       (va, vb, timestep_rows["video"] * 3)))
        return result


class Ref2VImageLayout:
    """Exact image-only Ref2VA layout with references kept on their own grids."""

    def __init__(self, text_len: int, latent_t: int, latent_h: int, latent_w: int,
                 audio_t: int, reference_shapes: Sequence[tuple[int, int, int]]):
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 target latent grid must be even")
        positions = [(float(i), 0.0, 0.0) for i in range(text_len)]
        cursor_time = float(text_len)
        condition_rows = 0
        for frames, height, width in reference_shapes:
            if frames != 1 or height % 2 or width % 2:
                raise ValueError("image references must be one latent frame on an even spatial grid")
            area = math.sqrt(height * width)
            hs = _axis_from_sqrt_area(height, 2, area)
            ws = _axis_from_sqrt_area(width, 2, area)
            frame = [(h, w) for h in hs for w in ws]
            positions.extend((cursor_time, h, w) for h, w in frame)
            condition_rows += len(frame)
            cursor_time += 1.0

        target_area = math.sqrt(latent_h * latent_w)
        target_hs = _axis_from_sqrt_area(latent_h, 2, target_area)
        target_ws = _axis_from_sqrt_area(latent_w, 2, target_area)
        target_frame = [(h, w) for h in target_hs for w in target_ws]
        positions.extend((cursor_time + i, 0.0, target_ws[0]) for i in range(audio_t))
        positions.extend((cursor_time + i, 0.0, target_ws[-1]) for i in range(audio_t))
        running = cursor_time
        for span in _video_t_spans(latent_t):
            positions.extend((running, h, w) for h, w in target_frame)
            running += span

        self.text = (0, text_len, "text")
        self.condition = (text_len, text_len + condition_rows, "cond")
        self.audio = (self.condition[1], self.condition[1] + audio_t * 2, "audio")
        self.video = (self.audio[1], self.audio[1] + latent_t * len(target_frame), "video")
        self.segments = (self.text, self.condition, self.audio, self.video)
        self.position_ids = mx.array(positions, dtype=mx.float32)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.reference_shapes = tuple(reference_shapes)
        self.seq_len = len(positions)

    def modulation_segments(self, timestep_rows, text_token_tags=None):
        result = []
        ta, tb, _ = self.text
        tags = [1] * (tb - ta) if text_token_tags is None else text_token_tags
        start = 0
        for i in range(1, len(tags) + 1):
            if i == len(tags) or tags[i] != tags[start]:
                result.append((ta + start, ta + i, timestep_rows["text"] * 3 + int(tags[start])))
                start = i
        ca, cb, _ = self.condition
        aa, ab, _ = self.audio
        va, vb, _ = self.video
        result.extend(((ca, cb, timestep_rows["cond"] * 3),
                       (aa, ab, timestep_rows["audio"] * 3 + 2),
                       (va, vb, timestep_rows["video"] * 3)))
        return result


class MiniMaxH3Transformer(nn.Module):
    """Full-width H3 transformer module using the released architecture."""

    def __init__(self, cfg: H3Config = H3Config()):
        super().__init__()
        self.cfg = cfg
        video_patch_dim = cfg.in_channels * math.prod(cfg.patch_size)
        self.video_patch_proj = nn.Linear(video_patch_dim, cfg.hidden_size)
        self.audio_patch_proj = nn.Linear(cfg.audio_in_channels, cfg.hidden_size)
        self.condition_proj = nn.Linear(cfg.text_dim, cfg.hidden_size)
        self.time_embedder = TimeEmbedder(cfg.freq_dim, cfg.time_embed_hidden_dim, cfg.time_embed_dim)
        self.rope = nn.Module()
        self.rope.inv_freq = mx.empty((cfg.rope_freq_dim,), dtype=mx.float32)
        self.token_refiner = TokenRefiner(cfg)
        self.blocks = [DiTBlock(cfg) for _ in range(cfg.num_layers)]
        self.final_layer = FinalLayer(cfg)

    def rope_angles(self, position_ids: mx.array) -> mx.array:
        per_axis = position_ids.astype(mx.float32)[..., None] * self.rope.inv_freq[None, None]
        t, h, w = per_axis[:, 0], per_axis[:, 1], per_axis[:, 2]
        half = mx.concatenate((t, h, w), axis=-1)
        return mx.concatenate((half, half), axis=-1)

    def refine_text(self, text: mx.array) -> mx.array:
        if text.shape[-1] == self.cfg.hidden_size:
            return text
        return self.token_refiner(self.condition_proj(text))

    def run_blocks(self, h, t_emb, mod_segments, position_ids, profiler=None):
        def build_rope():
            angles = self.rope_angles(position_ids)
            half = angles.shape[-1] // 2
            # One rotary table is shared by all 50 blocks. Computing trig inside
            # every Attention call would repeat identical work 50 times per step.
            return (mx.cos(angles[:, None, :half]).astype(h.dtype),
                    mx.sin(angles[:, None, :half]).astype(h.dtype))

        rope = profiler.measure("position_table", build_rope) if profiler else build_rope()
        for block in self.blocks:
            h = block(h, t_emb, mod_segments, rope, profiler=profiler)
        return h

    def __call__(
        self,
        video: mx.array,
        audio: mx.array,
        context: mx.array,
        sigma_video: float,
        *,
        layout: T2VLayout | None = None,
        text_token_tags: Sequence[int] | None = None,
        condition_video: mx.array | Sequence[mx.array] | None = None,
        condition_noise: mx.array | Sequence[mx.array] | None = None,
        visual_cond_noise_aug: float = 0.999,
        _profiler=None,
    ) -> tuple[mx.array, mx.array]:
        """Run one native H3 T2VA velocity prediction.

        Args:
            video: ``[1,24,T,H/16,W/16]`` latent on its own video schedule.
            audio: ``[1,32,2,T40]`` latent on its own audio schedule.
            context: Qwen layer-50 states, ``[1,L,5120]`` or pre-refined
                ``[1,L,5376]``.
            sigma_video: shifted video sigma as a Python float in ``(0,1]``.
        """
        if video.shape[0] != 1 or audio.shape[0] != 1 or context.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        if not 0.0 < sigma_video <= 1.0:
            raise ValueError("sigma_video must be in (0, 1]")
        _, _, latent_t, latent_h, latent_w = video.shape
        audio_t = audio.shape[-1]
        text_len = context.shape[1]
        signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        if layout is None:
            layout = FL2VLayout(*signature) if condition_video is not None else T2VLayout(*signature)
        elif layout.signature != signature:
            raise ValueError(f"layout signature {layout.signature} does not match {signature}")

        sigma_audio = time_shift_sigma_float(
            sigma_video, self.cfg.sigma_shift_video, self.cfg.sigma_shift_audio
        )
        t_video = 1.0 - sigma_video
        t_audio = 1.0 - sigma_audio
        t_cond = max(t_video, visual_cond_noise_aug) if condition_video is not None else None
        unique_t = sorted({t_video, t_audio} | ({t_cond} if t_cond is not None else set()))
        rows = {value: index for index, value in enumerate(unique_t)}
        timestep_rows = {"text": rows[t_video], "video": rows[t_video], "audio": rows[t_audio]}
        if t_cond is not None:
            timestep_rows["cond"] = rows[t_cond]
        mod_segments = layout.modulation_segments(timestep_rows, text_token_tags)

        video_embed = self.video_patch_proj(patchify_video(video.astype(mx.float32), self.cfg.patch_size))
        condition_embed = None
        if condition_video is not None:
            if isinstance(layout, Ref2VImageLayout):
                conditions = tuple(condition_video)
                noises = tuple(condition_noise) if condition_noise is not None else ()
                if tuple(tuple(item.shape[2:5]) for item in conditions) != layout.reference_shapes:
                    raise ValueError("reference tensors do not match the Ref2VA layout")
                if visual_cond_noise_aug < 1.0 and len(noises) != len(conditions):
                    raise ValueError("each reference requires matching conditioning noise")
                rows = []
                for index, condition in enumerate(conditions):
                    value = patchify_video(condition.astype(mx.float32), self.cfg.patch_size)
                    if visual_cond_noise_aug < 1.0:
                        noise = patchify_video(noises[index].astype(mx.float32), self.cfg.patch_size)
                        value = visual_cond_noise_aug * value + (1.0 - visual_cond_noise_aug) * noise
                    rows.append(value)
                condition_rows = mx.concatenate(rows)
            else:
                expected_condition_shape = (1, self.cfg.in_channels,
                                            len(layout.keyframe_anchors), latent_h, latent_w)
                if condition_video.shape != expected_condition_shape:
                    raise ValueError(
                        f"condition_video must be {expected_condition_shape} on the target latent grid"
                    )
                condition_rows = patchify_video(condition_video.astype(mx.float32), self.cfg.patch_size)
                if visual_cond_noise_aug < 1.0:
                    if condition_noise is None or condition_noise.shape != condition_video.shape:
                        raise ValueError("condition_noise must match condition_video when augmentation is enabled")
                    noise_rows = patchify_video(condition_noise.astype(mx.float32), self.cfg.patch_size)
                    condition_rows = visual_cond_noise_aug * condition_rows + (1.0 - visual_cond_noise_aug) * noise_rows
            condition_embed = self.video_patch_proj(condition_rows)
        audio_embed = self.audio_patch_proj(pack_audio(audio.astype(mx.float32)))
        text_states = self.refine_text(context[0])
        packed = [text_states.astype(context.dtype)]
        if condition_embed is not None:
            packed.append(condition_embed.astype(context.dtype))
        packed.extend((audio_embed.astype(context.dtype), video_embed.astype(context.dtype)))
        h = mx.concatenate(packed, axis=0)
        t_values = mx.array(unique_t, dtype=mx.float32)
        # Official H3 mixed precision: the shared time embedding and its SiLU
        # stay FP32; AdalnProj casts only immediately before its own linear.
        t_emb = self.time_embedder(t_values)
        if _profiler:
            h, t_emb = _profiler.measure("input_and_conditioning", lambda: (h, t_emb))
        h = self.run_blocks(h, t_emb, mod_segments, layout.position_ids, profiler=_profiler)

        va, vb, _ = layout.video
        aa, ab, _ = layout.audio
        final = lambda: self.final_layer(
            h, t_emb, (va, vb, timestep_rows["video"]),
            (aa, ab, timestep_rows["audio"]),
        )
        video_rows, audio_rows = (_profiler.measure("final_output_projection", final)
                                  if _profiler else final())
        video_velocity = unpatchify_video(
            video_rows,
            latent_t,
            latent_h // self.cfg.patch_size[1],
            latent_w // self.cfg.patch_size[2],
            self.cfg.in_channels,
            self.cfg.patch_size,
        )
        audio_velocity = unpack_audio(audio_rows)
        return -video_velocity.astype(video.dtype), -audio_velocity.astype(audio.dtype)


def load_sharded_safetensors(
    model: nn.Module,
    index_path: str | Path,
    strict: bool = True,
    metrics=None,
    progress_callback=None,
) -> None:
    """Load an HF sharded safetensors checkpoint using MLX's native loader."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if strict:
        from mlx.utils import tree_flatten

        expected = {key for key, _ in tree_flatten(model.parameters())}
        actual = set(weight_map)
        if expected != actual:
            raise ValueError(
                f"checkpoint mismatch: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
    for shard_index, shard in enumerate(shards):
        # Loading/reordering is setup work.  Keep it off the Metal command
        # queue: multi-GB concatenations can trip the macOS GPU watchdog before
        # inference even begins.
        with mx.stream(mx.cpu):
            raw = mx.load(str(index_path.parent / shard), stream=mx.cpu)
            weights: list[tuple[str, mx.array]] = []
            for key, value in raw.items():
                if ((key.startswith("token_refiner.blocks.")
                        or key.startswith("blocks."))
                        and key.endswith(".attn.qkv_proj.weight")):
                    expected_shape = (
                        model.cfg.num_attention_heads * 3 * model.cfg.attention_head_dim,
                        model.cfg.hidden_size,
                    )
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"unexpected token-refiner QKV shape for {key}: "
                            f"{value.shape}, expected {expected_shape}"
                        )
                    value = reorder_per_head_qkv(
                        value,
                        model.cfg.num_attention_heads,
                        model.cfg.attention_head_dim,
                    )
                weights.append((key, value))
            # Materialize each shard before installing it so installed weights
            # do not retain the source shard tensors through MLX lazy graphs.
            mx.eval([value for _, value in weights])
        if metrics is not None:
            metrics.explicit_sync += 1
            metrics.cpu_materializations += 1
        model.load_weights(weights, strict=False)
        del weights, raw
        if progress_callback is not None:
            progress_callback(shard_index + 1, len(shards))
    if metrics is not None:
        # Persistent CPU-prepared weights are consumed by the Metal execution
        # zone through unified memory. This is one setup boundary, never per step.
        metrics.device_transitions += 1
