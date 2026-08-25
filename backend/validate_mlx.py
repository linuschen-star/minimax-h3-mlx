#!/usr/bin/env python3
"""First on-device MLX numerical validation for the MiniMax H3 port.

Run without arguments for a small deterministic Metal test. Optionally point
at the original FL2VA transformer directory for a real official-weight smoke
test on the smallest meaningful latent shape.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.model import (
    H3Config,
    MiniMaxH3Transformer,
    T2VLayout,
    load_sharded_safetensors,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)
from minimax_h3_mlx.sampling import sigma_schedule


def assert_finite(name: str, value: mx.array) -> None:
    mx.eval(value)
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise AssertionError(f"{name} contains NaN or Inf")


def max_error(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def tiny_test() -> None:
    cfg = H3Config(
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_dim=64,
        in_channels=4,
        audio_in_channels=4,
        text_dim=24,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    mx.random.seed(7)
    model = MiniMaxH3Transformer(cfg)
    video = mx.random.normal((1, 4, 2, 4, 6)).astype(mx.float32)
    audio = mx.random.normal((1, 4, 2, 5)).astype(mx.float32)
    context = mx.random.normal((1, 3, 24)).astype(mx.float32)

    video_rt = unpatchify_video(patchify_video(video), 2, 2, 3, channels=4)
    audio_rt = unpack_audio(pack_audio(audio))
    if max_error(video, video_rt) != 0.0:
        raise AssertionError("video patch round-trip is not exact")
    if max_error(audio, audio_rt) != 0.0:
        raise AssertionError("audio pack round-trip is not exact")

    layout = T2VLayout(3, 2, 4, 6, 5)
    t0 = time.perf_counter()
    out_v1, out_a1 = model(video, audio, context, 0.8, layout=layout)
    mx.eval(out_v1, out_a1)
    elapsed = time.perf_counter() - t0
    out_v2, out_a2 = model(video, audio, context, 0.8, layout=layout)
    mx.eval(out_v2, out_a2)

    assert out_v1.shape == video.shape
    assert out_a1.shape == audio.shape
    assert_finite("tiny video output", out_v1)
    assert_finite("tiny audio output", out_a1)
    if max_error(out_v1, out_v2) != 0.0 or max_error(out_a1, out_a2) != 0.0:
        raise AssertionError("repeated forward is not deterministic")
    print(f"PASS tiny MLX forward: {elapsed:.3f}s, seq={layout.seq_len}")
    schedule = sigma_schedule(4, 12.0)
    if schedule[0] != 1.0 or schedule[-1] != 0.0 or any(a <= b for a, b in zip(schedule, schedule[1:])):
        raise AssertionError("shifted flow schedule is invalid")
    print("PASS shifted flow schedule")


def official_weight_test(transformer_dir: Path) -> None:
    config_path = transformer_dir / "config.json"
    index_path = transformer_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "Use the original FL2VA/transformer directory containing "
            "model.safetensors.index.json and every shard"
        )
    cfg = H3Config.from_json(config_path)
    model = MiniMaxH3Transformer(cfg)
    started = time.perf_counter()
    load_sharded_safetensors(model, index_path, strict=True)
    mx.eval(model.parameters())
    print(f"Loaded official transformer in {time.perf_counter() - started:.1f}s")

    # Minimal real graph: 1 text token, two video latent frames, one 2x2
    # spatial patch, and one latent audio step. This verifies all 50 official
    # blocks and exact checkpoint naming without attempting a costly 480p step.
    video = mx.zeros((1, cfg.in_channels, 2, 2, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, cfg.audio_in_channels, 2, 1), dtype=mx.bfloat16)
    context = mx.zeros((1, 1, cfg.text_dim), dtype=mx.bfloat16)
    started = time.perf_counter()
    out_v, out_a = model(video, audio, context, 0.8)
    mx.eval(out_v, out_a)
    assert_finite("official video output", out_v)
    assert_finite("official audio output", out_a)
    print(
        f"PASS official 50-layer forward: {time.perf_counter() - started:.3f}s, "
        f"video={out_v.shape}, audio={out_a.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-dir",
        type=Path,
        help="MiniMax-H3/FL2VA/transformer with downloaded original BF16 shards",
    )
    args = parser.parse_args()
    print(f"Python {platform.python_version()} | machine={platform.machine()}")
    print(f"MLX device: {mx.device_info()}")
    tiny_test()
    if args.transformer_dir:
        official_weight_test(args.transformer_dir)


if __name__ == "__main__":
    main()
