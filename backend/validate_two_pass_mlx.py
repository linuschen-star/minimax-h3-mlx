#!/usr/bin/env python3
"""Fast M4 validation of native H3 schedule/noising/spatial-resize primitives."""

import platform
import time

import mlx.core as mx

from minimax_h3_mlx.runtime import ExactH3Runtime, H3GenerationSpec, H3LatentState
from minimax_h3_mlx.sampling import flow_add_noise, resume_schedule, sigma_schedule


def main():
    print(f"Python {platform.python_version()} | machine={platform.machine()}")
    print(f"MLX {mx.__version__} | device={mx.device_info()}")
    suffix, index, evaluations = resume_schedule(8, 0.5, 12.0)
    assert len(suffix) == 9 and (index, evaluations) == (0, 8)

    spec = H3GenerationSpec("validation", 864, 480, 124, 20, 0)
    video = mx.random.normal((1, 24, 37, 30, 54)).astype(mx.bfloat16)
    audio = mx.random.normal((1, 32, 2, 207)).astype(mx.bfloat16)
    context = mx.zeros((1, 8, 5120), dtype=mx.bfloat16)
    state = H3LatentState(video, audio, context, (1,) * 8, spec)
    start = time.perf_counter()
    resized = ExactH3Runtime.resize_latent(state, 1.5)
    noise = mx.random.normal(resized.video.shape).astype(mx.bfloat16)
    noised = flow_add_noise(resized.video, noise, suffix[0])
    mx.eval(noised)
    elapsed = time.perf_counter() - start

    assert resized.video.shape[:3] == video.shape[:3]
    assert resized.audio is audio and resized.context is context
    assert resized.spec.frames == 124
    assert noised.dtype == mx.bfloat16
    assert flow_add_noise(resized.video, noise, 0.0).shape == resized.video.shape
    print(f"PASS schedule: strength=0.5 sigma={suffix[0]:.9f} evaluations={evaluations}")
    print(f"PASS BCTHW spatial-only resize: {video.shape} -> {resized.video.shape}")
    print(f"PASS exact BF16 flow interpolation: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
