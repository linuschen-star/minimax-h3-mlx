#!/usr/bin/env python3
"""First real H3 Audio VAE numerical validation for Apple Silicon."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.audio_vae import H3AudioVAE, load_audio_vae_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_vae_dir", type=Path)
    parser.add_argument("--latent-frames", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    vae = H3AudioVAE.from_config(args.audio_vae_dir / "config.json")
    load_audio_vae_decoder(vae, args.audio_vae_dir / "model.safetensors")
    mx.eval(vae.parameters())
    print(f"PASS strict official decoder load: {time.perf_counter() - started:.3f}s")

    z = mx.zeros((1, 32, 2, args.latent_frames), dtype=mx.float32)
    started = time.perf_counter()
    waveform = vae.decode(z)
    mx.eval(waveform)
    expected = args.latent_frames * vae.samples_per_latent
    assert waveform.shape == (1, 2, expected), (waveform.shape, expected)
    assert mx.all(mx.isfinite(waveform)).item()
    print(
        f"PASS MLX Metal decode: {time.perf_counter() - started:.3f}s, "
        f"shape={waveform.shape}, range=[{mx.min(waveform).item():.6f}, "
        f"{mx.max(waveform).item():.6f}]"
    )


if __name__ == "__main__":
    main()
