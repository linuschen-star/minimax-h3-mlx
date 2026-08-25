"""MLX-native decoder for the released MiniMax H3 32 kHz Audio VAE."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class Conv1dCF(nn.Module):
    """Conv1d with an H3-compatible [B,C,T] boundary."""

    def __init__(self, cin, cout, kernel, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.weight = mx.empty((cout, kernel, cin))
        if bias:
            self.bias = mx.empty((cout,))
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def __call__(self, x):
        y = mx.conv1d(x.transpose(0, 2, 1), self.weight, self.stride, self.padding, self.dilation)
        if hasattr(self, "bias"):
            y = y + self.bias
        return y.transpose(0, 2, 1)


class ConvTranspose1dCF(nn.Module):
    def __init__(self, cin, cout, kernel, stride, padding):
        super().__init__()
        self.weight = mx.empty((cout, kernel, cin))
        self.bias = mx.empty((cout,))
        self.stride, self.padding = stride, padding

    def __call__(self, x):
        y = mx.conv_transpose1d(
            x.transpose(0, 2, 1), self.weight,
            stride=self.stride, padding=self.padding,
        )
        return (y + self.bias).transpose(0, 2, 1)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.filter = mx.empty((1, 1, kernel_size))

    def __call__(self, x):
        channels = x.shape[1]
        x = mx.pad(x, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")
        filt = mx.broadcast_to(self.filter, (channels, 1, self.filter.shape[-1])).transpose(0, 2, 1)
        y = mx.conv_transpose1d(x.transpose(0, 2, 1), filt, stride=self.ratio, groups=channels)
        y = (y * self.ratio).transpose(0, 2, 1)
        return y[..., self.pad_left:-self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.stride = ratio
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.filter = mx.empty((1, 1, kernel_size))

    def __call__(self, x):
        channels = x.shape[1]
        x = mx.pad(x, ((0, 0), (0, 0), (self.pad_left, self.pad_right)), mode="edge")
        filt = mx.broadcast_to(self.filter, (channels, 1, self.filter.shape[-1])).transpose(0, 2, 1)
        return mx.conv1d(x.transpose(0, 2, 1), filt, stride=self.stride, groups=channels).transpose(0, 2, 1)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, kernel_size)

    def __call__(self, x):
        return self.lowpass(x)


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.empty((channels,))
        self.beta = mx.empty((channels,))

    def __call__(self, x):
        alpha = mx.exp(self.alpha).reshape(1, -1, 1).astype(x.dtype)
        beta = mx.exp(self.beta).reshape(1, -1, 1).astype(x.dtype)
        return x + mx.sin(alpha * x) ** 2 / (beta + 1e-9)


class Activation1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d()
        self.downsample = DownSample1d()

    def __call__(self, x):
        return self.downsample(self.act(self.upsample(x)))


def _padding(kernel, dilation=1):
    return (kernel * dilation - dilation) // 2


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = [Conv1dCF(channels, channels, kernel, dilation=d, padding=_padding(kernel, d)) for d in dilations]
        self.convs2 = [Conv1dCF(channels, channels, kernel, padding=_padding(kernel)) for _ in dilations]
        self.activations = [Activation1d(channels) for _ in range(6)]

    def __call__(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            x = x + c2(self.activations[2 * i + 1](c1(self.activations[2 * i](x))))
        return x


class BigVGAN(nn.Module):
    def __init__(self):
        super().__init__()
        rates = (5, 5, 2, 2, 2, 2, 2)
        kernels = (9, 9, 4, 4, 4, 4, 4)
        self.conv_pre = Conv1dCF(2048, 1024, 7, padding=3)
        self.ups = []
        self.resblocks = []
        for i, (rate, kernel) in enumerate(zip(rates, kernels)):
            cin, cout = 1024 // (2**i), 1024 // (2 ** (i + 1))
            self.ups.append([ConvTranspose1dCF(cin, cout, kernel, rate, (kernel - rate) // 2)])
            self.resblocks.extend(AMPBlock1(cout, k) for k in (3, 7, 11))
        self.activation_post = Activation1d(8)
        self.conv_post = Conv1dCF(8, 1, 7, padding=3, bias=False)

    def __call__(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up[0](x)
            branches = [self.resblocks[i * 3 + j](x) for j in range(3)]
            x = sum(branches[1:], branches[0]) / 3
        return mx.clip(self.conv_post(self.activation_post(x)), -1.0, 1.0)


class H3AudioVAE(nn.Module):
    """Decode normalized [B,32,2,T] latents to [B,2,T*800] waveform."""

    def __init__(self, latents_mean, latents_std):
        super().__init__()
        self.dec_in_proj = Conv1dCF(32, 2048, 1)
        self.decoder = BigVGAN()
        self.latents_mean = mx.array(latents_mean, dtype=mx.float32)
        self.latents_std = mx.array(latents_std, dtype=mx.float32)
        self.sample_rate = 32000
        self.samples_per_latent = 800

    @classmethod
    def from_config(cls, config_path: str | Path):
        config = json.loads(Path(config_path).read_text())
        return cls(config["latents_mean"], config["latents_std"])

    def decode(self, z):
        b, channels, stereo, length = z.shape
        z = z.transpose(0, 2, 1, 3).reshape(b * stereo, channels, length)
        mean = self.latents_mean.reshape(1, -1, 1).astype(z.dtype)
        std = self.latents_std.reshape(1, -1, 1).astype(z.dtype)
        x = self.decoder(self.dec_in_proj(z * std + mean))
        return x.reshape(b, stereo, -1)


def _fold_weight_norm(v, g):
    # torch.nn.utils.weight_norm default dim=0: normalize over all other axes.
    denom = mx.sqrt(mx.sum(v.astype(mx.float32) ** 2, axis=tuple(range(1, v.ndim)), keepdims=True))
    return v * (g.astype(mx.float32) / denom).astype(v.dtype)


def load_audio_vae_decoder(vae: H3AudioVAE, checkpoint: str | Path) -> None:
    """Load decoder-only weights, folding the official weight-norm tensors."""
    raw = mx.load(str(checkpoint))
    # These two normalization arrays come from the official config rather than
    # the safetensors file, but include them in the strict parameter load.
    wanted = {"latents_mean": vae.latents_mean, "latents_std": vae.latents_std}
    prefixes = ("dec_in_proj.", "decoder.")
    for key, value in raw.items():
        if not key.startswith(prefixes) or key.endswith(".weight_g") or key.endswith(".weight_v"):
            continue
        if key == "dec_in_proj.weight":
            value = value.transpose(0, 2, 1)
        wanted[key] = value
    for key, v in raw.items():
        if not key.startswith(prefixes) or not key.endswith(".weight_v"):
            continue
        base = key[:-9]
        folded = _fold_weight_norm(v, raw[base + ".weight_g"])
        # PyTorch ConvTranspose1d is [in,out,k]; Conv1d is [out,in,k].
        if base.startswith("decoder.ups."):
            folded = folded.transpose(1, 2, 0)
        else:
            folded = folded.transpose(0, 2, 1)
        wanted[base + ".weight"] = folded
    vae.load_weights(list(wanted.items()), strict=True)
