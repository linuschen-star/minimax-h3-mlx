"""MLX-native inference decoder for the released H3 Visual VAE."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def _reflect_pad_one(x: mx.array, axis: int) -> mx.array:
    """PyTorch reflect pad by one without leaving MLX (mx.pad has no reflect mode)."""
    before = [slice(None)] * x.ndim
    after = [slice(None)] * x.ndim
    before[axis] = slice(1, 2)
    after[axis] = slice(-2, -1)
    return mx.concatenate((x[tuple(before)], x, x[tuple(after)]), axis=axis)


def _reflect_pad_after_one(x: mx.array, axis: int) -> mx.array:
    """PyTorch ``F.pad(..., (0, 1), mode='reflect')`` for one axis."""
    after = [slice(None)] * x.ndim
    after[axis] = slice(-2, -1)
    return mx.concatenate((x, x[tuple(after)]), axis=axis)


class CausalConv3d(nn.Module):
    """NCTHW-facing causal Conv3d with official spatial reflection padding."""

    def __init__(self, in_channels, out_channels, kernel=3, stride=(1, 1, 1), downsample=False):
        super().__init__()
        self.kernel, self.downsample = kernel, downsample
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=0)

    def __call__(self, x):
        # MLX Conv3d consumes NDHWC.  The released H3 encoder uses constant
        # causal temporal padding: prepend kernel_t-1 *zero* frames.  Replicating
        # the image frame here produces a completely different posterior.
        x = x.transpose(0, 2, 3, 4, 1)
        if self.kernel == 3:
            x = mx.concatenate((mx.zeros_like(mx.repeat(x[:, :1], 2, axis=1)), x), axis=1)
            if self.downsample:
                # Official Downsample3D adds only right/bottom using the VAE's
                # configured spatial_padding_mode="reflect".  Zero padding here
                # changes every successive conditioning pyramid level.
                x = _reflect_pad_after_one(_reflect_pad_after_one(x, 2), 3)
            else:
                x = _reflect_pad_one(_reflect_pad_one(x, 2), 3)
        return self.conv(x).transpose(0, 4, 1, 2, 3)


class EncoderResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = CausalConv3d(in_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv2 = CausalConv3d(out_channels, out_channels)
        self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel=1) if in_channels != out_channels else None

    @staticmethod
    def _norm(norm, x):
        return norm(x.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)

    def __call__(self, x):
        h = self.conv1(nn.silu(self._norm(self.norm1, x)))
        h = self.conv2(nn.silu(self._norm(self.norm2, h)))
        return (self.conv_shortcut(x) if self.conv_shortcut is not None else x) + h


class EncoderDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, space_stride, time_stride, add_downsample):
        super().__init__()
        self.resnets = [EncoderResnetBlock3D(in_channels, out_channels), EncoderResnetBlock3D(out_channels, out_channels)]
        self.downsamplers = ([CausalConv3d(out_channels, out_channels, stride=(time_stride, space_stride, space_stride), downsample=True)]
                             if add_downsample else [])

    def __call__(self, x):
        for block in self.resnets:
            x = block(x)
        for conv in self.downsamplers:
            x = conv(x)
        return x


class H3VideoEncoder(nn.Module):
    """Released causal CNN image encoder; tensors remain NCTHW at its boundary."""

    def __init__(self):
        super().__init__()
        channels = [128, 256, 256, 512, 512, 1024]
        space = [2, 2, 2, 2, 1, 1]
        time = [1, 2, 2, 1, 1, 1]
        self.conv_in = CausalConv3d(3, channels[0])
        self.down_blocks = []
        previous = channels[0]
        for index, current in enumerate(channels):
            self.down_blocks.append(EncoderDownBlock3D(
                previous, current, space[index], time[index], space[index] * time[index] > 1,
            ))
            previous = current
        self.norm_out = nn.GroupNorm(32, channels[-1], eps=1e-6, pytorch_compatible=True)
        self.conv_out = CausalConv3d(channels[-1], 48)

    def __call__(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.norm_out(x.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
        return self.conv_out(nn.silu(x))


def _rms_no_affine(x: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps).astype(x.dtype)


def _apply_rope(x: mx.array, angles: mx.array) -> mx.array:
    # x [B,S,H,D], angles [B,S,pairs]; rotate the first 2*pairs split-half dims.
    pairs = angles.shape[-1]
    left, right, tail = x[..., :pairs], x[..., pairs:2 * pairs], x[..., 2 * pairs:]
    c, s = mx.cos(angles)[:, :, None].astype(x.dtype), mx.sin(angles)[:, :, None].astype(x.dtype)
    return mx.concatenate((left * c - right * s, right * c + left * s, tail), axis=-1)


class VAEAttention(nn.Module):
    def __init__(self, dim: int = 2048, heads: int = 32, head_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        self.heads, self.head_dim, self.eps = heads, head_dim, eps
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = [nn.Linear(dim, dim)]

    def __call__(self, x: mx.array, angles: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = _rms_no_affine(self.to_q(x).reshape(b, s, self.heads, self.head_dim), self.eps)
        k = _rms_no_affine(self.to_k(x).reshape(b, s, self.heads, self.head_dim), self.eps)
        v = self.to_v(x).reshape(b, s, self.heads, self.head_dim)
        q, k = _apply_rope(q, angles), _apply_rope(k, angles)
        y = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5, mask=None,
        )
        y = mx.nan_to_num(y).transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.to_out[0](y)


class VAEFeedForward(nn.Module):
    def __init__(self, dim: int = 2048, mult: int = 4):
        super().__init__()
        self.net = [nn.Module(), nn.Module(), nn.Linear(dim * mult, dim)]
        self.net[0].proj = nn.Linear(dim, dim * mult * 2)

    def __call__(self, x: mx.array) -> mx.array:
        # The released diffusers VAE converts SwiGLU to [value, gate]. This is
        # intentionally opposite to the original FL2VA transformer's fused
        # [gate, value] checkpoint layout.
        value, gate = mx.split(self.net[0].proj(x), 2, axis=-1)
        return self.net[2](nn.silu(gate) * value)


class VAEBlock(nn.Module):
    def __init__(self, dim=2048, heads=32, head_dim=64, eps=1e-5):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = VAEAttention(dim, heads, head_dim, eps)
        self.scale1 = mx.empty((dim,))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = VAEFeedForward(dim)
        self.scale2 = mx.empty((dim,))

    def __call__(self, x, angles):
        x = x + self.attn(self.norm1(x), angles) * self.scale1.astype(x.dtype)
        return x + self.ff(self.norm2(x)) * self.scale2.astype(x.dtype)


def _token_ids(t: int, h: int, w: int, suffix: int, dtype) -> mx.array:
    axes = []
    for size in (t, h, w):
        axes.append(2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0)
    tt, hh, ww = mx.meshgrid(*axes, indexing="ij")
    ids = mx.stack((tt, hh, ww), axis=-1).reshape(1, t * h * w, 3)
    return mx.concatenate((ids, mx.zeros((1, suffix, 3), dtype=mx.float32)), axis=1).astype(dtype)


class H3VideoDecoder(nn.Module):
    def __init__(self, layers=36, heads=32, head_dim=64, registers=4, eps=1e-5):
        super().__init__()
        self.heads, self.head_dim, self.registers = heads, head_dim, registers
        dim = heads * head_dim
        self.proj_in = nn.Linear(24, dim)
        self.register_tokens = mx.empty((1, registers, dim))
        self.transformer_blocks = [VAEBlock(dim, heads, head_dim, eps) for _ in range(layers)]
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(dim, 3 * 4 * 16 * 16)
        # Official formula: rotary dim 48 split across 3 axes -> 8 frequencies/axis.
        self.inv_freq = 1.0 / (100.0 ** mx.arange(0.0, 1.0, 1.0 / 8.0, dtype=mx.float32))

    def __call__(self, z: mx.array) -> mx.array:
        b, channels, t, h, w = z.shape
        tokens = z.transpose(0, 2, 3, 4, 1).reshape(b, t * h * w, channels)
        x = self.proj_in(tokens)
        suffix = self.registers + 1
        x = mx.concatenate((x, mx.broadcast_to(self.register_tokens, (b, self.registers, x.shape[-1])), mx.zeros((b, 1, x.shape[-1]), dtype=x.dtype)), axis=1)
        ids = mx.broadcast_to(_token_ids(t, h, w, suffix, z.dtype), (b, t * h * w + suffix, 3))
        angles = (2.0 * math.pi * ids[..., None].astype(mx.float32) * self.inv_freq[None, None, None]).reshape(b, ids.shape[1], -1)
        for block in self.transformer_blocks:
            x = block(x, angles)
        x = self.proj_out(self.norm_out(x))[:, :t * h * w]
        x = x.reshape(b, t, h, w, 3, 4, 16, 16).transpose(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(b, 3, t * 4, h * 16, w * 16)


class H3VideoVAE(nn.Module):
    """Decode normalized ``[B,24,T,H,W]`` H3 latents to ``[B,3,F,H*16,W*16]``."""
    def __init__(self, tile_size=256, overlap=64):
        super().__init__()
        self.encoder = H3VideoEncoder()
        self.quant_conv = nn.Conv3d(48, 48, 1)
        self.post_quant_conv = nn.Conv3d(24, 24, 1)
        self.decoder = H3VideoDecoder()
        self.latents_mean = mx.array(LATENTS_MEAN, dtype=mx.float32)
        self.latents_std = mx.array(LATENTS_STD, dtype=mx.float32)
        self.tile_size, self.overlap = tile_size, overlap
        self.tokens_chunk_size, self.token_overlap = 5, 2
        self.frame_pre_padding, self.frame_overlap = 3, 5

    def encode_image(self, pixels: mx.array, seed: int = 42) -> mx.array:
        """Encode RGB ``[B,H,W,3]`` [0,1] into normalized FL2VA keyframe latent."""
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("H3 image input must have shape [B,H,W,3].")
        if pixels.shape[1] % 16 or pixels.shape[2] % 16:
            raise ValueError("H3 image height and width must be divisible by 16.")
        mean = mx.array(PIXEL_MEAN).reshape(1, 1, 1, 3)
        std = mx.array(PIXEL_STD).reshape(1, 1, 1, 3)
        x = ((pixels.astype(mx.float32) - mean) / std).transpose(0, 3, 1, 2)[:, :, None]
        moments = self._encode_tiled(x.astype(mx.bfloat16))
        posterior_mean, logvar = mx.split(moments, 2, axis=1)
        logvar = mx.clip(logvar.astype(mx.float32), -30.0, 20.0)
        mx.random.seed(seed)
        latent = posterior_mean + mx.exp(0.5 * logvar).astype(posterior_mean.dtype) * mx.random.normal(posterior_mean.shape).astype(posterior_mean.dtype)
        # Official encode_vae_condition deliberately rounds the sampled latent
        # through FP16 and then returns FP32 before channel normalization.  BF16
        # here loses three additional mantissa bits and is visibly amplified by
        # the visual decoder as a 16-pixel block grid in the anchor frames.
        latent = latent.astype(mx.float16).astype(mx.float32)
        latent_mean = self.latents_mean.reshape(1, -1, 1, 1, 1)
        latent_std = self.latents_std.reshape(1, -1, 1, 1, 1)
        return (latent - latent_mean) / latent_std

    def _encode_tiled(self, x: mx.array) -> mx.array:
        """Official 256px/64px-overlap spatial VAE encode and latent stitch."""
        ys, yl, yo = self._tile_plan(x.shape[-2])
        xs, xl, xo = self._tile_plan(x.shape[-1])
        raw = []
        for py, ly in zip(ys, yl):
            row = []
            for px, lx in zip(xs, xl):
                tile = x[..., py:py + ly, px:px + lx]
                encoded = self.encoder(tile).transpose(0, 2, 3, 4, 1)
                row.append(self.quant_conv(encoded).transpose(0, 4, 1, 2, 3))
            raw.append(row)
        latent_yo = [overlap // 16 for overlap in yo]
        latent_xo = [overlap // 16 for overlap in xo]
        rows = []
        for i, row in enumerate(raw):
            parts = []
            for j, tile in enumerate(row):
                if i:
                    tile = self._blend(raw[i - 1][j], tile, latent_yo[i - 1], -2)
                if j:
                    tile = self._blend(row[j - 1], tile, latent_xo[j - 1], -1)
                if i < len(raw) - 1:
                    tile = tile[..., :-latent_yo[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_xo[j]]
                parts.append(tile)
            rows.append(mx.concatenate(parts, axis=-1))
        return mx.concatenate(rows, axis=-2)

    def _pixels(self, z):
        # Conv3d is NDHWC in official MLX; boundary remains H3's NCTHW.
        x = z.transpose(0, 2, 3, 4, 1)
        x = self.post_quant_conv(x).transpose(0, 4, 1, 2, 3)
        return self.decoder(x)

    @staticmethod
    def _blend(a, b, extent, axis):
        extent = min(a.shape[axis], b.shape[axis], extent)
        weights = mx.arange(extent, dtype=mx.float32) / extent
        shape = [1] * a.ndim
        shape[axis] = extent
        wb, wa = weights.reshape(shape).astype(b.dtype), (1 - weights).reshape(shape).astype(a.dtype)
        sa, sb = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        sa[axis], sb[axis] = slice(-extent, None), slice(0, extent)
        blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
        sb[axis] = slice(extent, None)
        return mx.concatenate((blended, b[tuple(sb)]), axis=axis) if extent < b.shape[axis] else blended

    @staticmethod
    def _split_temporal_chunk(decoded):
        """Return the official 17-frame body and 5-frame overlap from 28 frames."""
        if decoded.shape[2] != 28:
            raise ValueError(f"H3 VAE temporal chunk must decode to 28 frames, got {decoded.shape[2]}.")
        return decoded[:, :, 3:20], decoded[:, :, 23:]

    def _tile_plan(self, pixels):
        if self.tile_size >= pixels:
            return [0], [pixels], []
        count = math.ceil(pixels / self.tile_size)
        while self.tile_size * count - self.overlap * (count - 1) - pixels < 0:
            count += 1
        overlaps = [self.overlap] * (count - 1)
        remaining = self.tile_size * count - sum(overlaps) - pixels
        for i in range(remaining // 16):
            overlaps[i % (count - 1)] += 16
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + self.tile_size - overlap)
        return starts, [self.tile_size] * count, overlaps

    def _decode_tiled(self, z):
        ys, yl, yo = self._tile_plan(z.shape[-2] * 16)
        xs, xl, xo = self._tile_plan(z.shape[-1] * 16)
        raw = []
        for py, ly in zip(ys, yl):
            row = []
            for px, lx in zip(xs, xl):
                row.append(self._pixels(z[..., py // 16:(py + ly) // 16, px // 16:(px + lx) // 16]))
            raw.append(row)
        rows = []
        for i, row in enumerate(raw):
            parts = []
            for j, tile in enumerate(row):
                if i:
                    tile = self._blend(raw[i - 1][j], tile, yo[i - 1], -2)
                if j:
                    tile = self._blend(row[j - 1], tile, xo[j - 1], -1)
                if i < len(raw) - 1:
                    tile = tile[..., :-yo[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-xo[j]]
                parts.append(tile)
            rows.append(mx.concatenate(parts, axis=-1))
        return mx.concatenate(rows, axis=-2)

    def decode(self, z: mx.array) -> mx.array:
        mean = self.latents_mean.reshape(1, -1, 1, 1, 1).astype(z.dtype)
        std = self.latents_std.reshape(1, -1, 1, 1, 1).astype(z.dtype)
        z = z * std + mean
        pseudo = z.shape[2] + 3
        pad = (-pseudo) % self.tokens_chunk_size
        padded = mx.concatenate((z, mx.repeat(z[:, :, -1:], pad, axis=2)), axis=2) if pad else z
        chunks = max(1, (pseudo + pad) // self.tokens_chunk_size - 1)
        output, overlap = [], None
        for i in range(chunks):
            clip = padded[:, :, i * 5:i * 5 + 7]
            decoded = self._decode_tiled(clip)
            # Official token_drop=3 temporal split: the first 20 decoded
            # frames become 17 usable frames after frame_pre_padding=3.
            # Frames 20:23 belong to neither segment; including them caused
            # a three-frame luminance discontinuity every 20 output frames.
            first, tail = self._split_temporal_chunk(decoded)
            if overlap is not None:
                first = self._blend(overlap, first, self.frame_overlap, 2)
            output.append(first)
            overlap = tail
            mx.eval(output[-1])
        if overlap is not None:
            output.append(overlap)
        frames = mx.concatenate(output, axis=2)
        # For the supported 17k+5 grid: latent T=5k+2 -> output F=17k+5.
        target_frames = ((z.shape[2] - 2) // 5) * 17 + 5
        frames = frames[:, :, :target_frames].astype(mx.float32)
        pmean = mx.array(PIXEL_MEAN).reshape(1, 3, 1, 1, 1)
        pstd = mx.array(PIXEL_STD).reshape(1, 3, 1, 1, 1)
        return mx.clip(frames * pstd + pmean, 0.0, 1.0)


def load_video_vae_decoder(vae: H3VideoVAE, index_path: str | Path) -> None:
    """Load decoder-only weights from the official diffusers VAE checkpoint."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    wanted = {k: v for k, v in index["weight_map"].items() if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
    weights = []
    for shard in sorted(set(wanted.values())):
        data = mx.load(str(index_path.parent / shard))
        for key, shard_name in wanted.items():
            if shard_name != shard:
                continue
            value = data[key]
            if key == "post_quant_conv.weight":
                value = value.transpose(0, 2, 3, 4, 1)
            weights.append((key, value))
    # Decoder-only deployment intentionally omits the released encoder tensors.
    vae.load_weights(weights, strict=False)


def load_video_vae_encoder(vae: H3VideoVAE, index_path: str | Path) -> None:
    """Load only native image-encoder weights; decoder tensors remain unloaded."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    wanted = {k: v for k, v in index["weight_map"].items()
              if k.startswith("encoder.") or k.startswith("quant_conv.")}
    weights = []
    for shard in sorted(set(wanted.values())):
        data = mx.load(str(index_path.parent / shard))
        for key, shard_name in wanted.items():
            if shard_name != shard:
                continue
            mapped = key
            needs_wrapper = (key.startswith(("encoder.conv_in.", "encoder.conv_out."))
                             or any(part in key for part in (".conv1.", ".conv2.", ".conv_shortcut.")))
            if needs_wrapper:
                stem, leaf = mapped.rsplit(".", 1)
                mapped = f"{stem}.conv.{leaf}"
            # Diffusers OI(DHW) -> MLX O(DHW)I.
            value = data[key]
            if key.endswith(".weight") and value.ndim == 5:
                value = value.transpose(0, 2, 3, 4, 1)
            weights.append((mapped, value))
    vae.load_weights(weights, strict=False)
