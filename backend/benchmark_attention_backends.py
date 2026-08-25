#!/usr/bin/env python3
"""Exact-BF16 attention A/B for real H3 token geometries on Apple MLX.

``baseline`` is deliberately the *current* production implementation: packed
head-major Q/K/V passed to MLX fused SDPA.  ``mlx_fused`` repeats that official
primitive as an explicit control; it is not presented as a new optimization.
``chunked_exact`` slices only the query rows and calls the same exact MLX SDPA
primitive for every slice, avoiding a materialized score matrix in all modes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class Workload:
    name: str
    width: int
    height: int
    frames: int
    text_tokens: int = 256
    fps: int = 24

    @property
    def latent_t(self) -> int:
        if self.frames < 5 or (self.frames - 5) % 17:
            raise ValueError(f"{self.name}: frames must follow H3's 17k+5 grid")
        return 5 * ((self.frames - 5) // 17) + 2

    @property
    def audio_t(self) -> int:
        return round(self.frames / self.fps * 40)

    @property
    def sequence(self) -> int:
        # Official H3 pack: [text | stereo audio | 2x2-patched video].
        latent_h, latent_w = self.height // 16, self.width // 16
        return (self.text_tokens + 2 * self.audio_t
                + self.latent_t * (latent_h // 2) * (latent_w // 2))


WORKLOADS = {
    "864x480_5s": Workload("864x480_5s", 864, 480, 124),
    "1344x768_5s": Workload("1344x768_5s", 1344, 768, 124),
    "1920x1088_5s": Workload("1920x1088_5s", 1920, 1088, 124),
    # The recently observed 15-second FL2VA job uses 362 frames.
    "864x480_15s": Workload("864x480_15s", 864, 480, 362),
}


def stats(samples: list[float]) -> dict:
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "best_seconds": min(samples),
        "worst_seconds": max(samples),
    }


def evaluate_timed(function, warmups: int, repeats: int):
    for _ in range(warmups):
        mx.eval(function())
    samples = []
    peak_growth = []
    peak_total = []
    output = None
    for _ in range(repeats):
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = function()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        peak = mx.get_peak_memory()
        peak_total.append(peak)
        peak_growth.append(max(0, peak - active_before))
    return output, {
        **stats(samples),
        "peak_incremental_memory_bytes": max(peak_growth),
        "peak_incremental_memory_gib": max(peak_growth) / (1024 ** 3),
        "peak_active_memory_bytes": max(peak_total),
        "peak_active_memory_gib": max(peak_total) / (1024 ** 3),
    }


def error(reference: mx.array, candidate: mx.array) -> dict:
    delta = mx.abs(reference.astype(mx.float32) - candidate.astype(mx.float32))
    mx.eval(delta)
    return {
        "bit_exact": bool(mx.array_equal(reference, candidate).item()),
        "max_abs_error": float(mx.max(delta).item()),
        "mean_abs_error": float(mx.mean(delta).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS,
                        default=["864x480_5s", "1344x768_5s"])
    parser.add_argument("--modes", nargs="+",
                        choices=("baseline", "mlx_fused", "chunked_exact"),
                        default=("baseline", "mlx_fused", "chunked_exact"))
    parser.add_argument("--chunk-sizes", nargs="+", type=int,
                        default=(256, 512, 1024, 2048, 4096))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path,
                        default=Path("h3_attention_backends.json"))
    args = parser.parse_args()
    if args.repeats < 3 or args.warmups < 1:
        parser.error("use at least one warmup and three measured repeats")
    if any(size <= 0 or size % 32 for size in args.chunk_sizes):
        parser.error("chunk sizes must be positive multiples of official MLX BQ=32")

    heads, head_dim = 56, 128
    scale = head_dim ** -0.5
    report = {
        "title": "H3 exact-BF16 attention backend benchmark",
        "hardware": {"python": platform.python_version(), "mlx": mx.__version__,
                     "device": mx.device_info()},
        "semantics": {"dtype": "bfloat16", "heads": heads,
                      "head_dim": head_dim, "mask": None, "causal": False,
                      "scale": scale, "softmax_accumulation": "float32 (MLX API contract)"},
        "mode_definitions": {
            "baseline": "current production: contiguous BHSD + MLX fused SDPA",
            "mlx_fused": "explicit control using the same official fused primitive",
            "chunked_exact": "query slices, each evaluated by official fused SDPA",
        },
        "results": [],
    }

    mx.random.seed(20260822)
    for workload_name in args.workloads:
        workload = WORKLOADS[workload_name]
        sequence = workload.sequence
        print(f"{workload.name}: S={sequence}, shape=[1,{heads},{sequence},{head_dim}]",
              flush=True)
        q = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
        k = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
        v = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
        mx.eval(q, k, v)
        fused = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=None)
        reference, baseline_timing = evaluate_timed(
            fused, args.warmups, args.repeats)
        flops = 4 * heads * sequence * sequence * head_dim
        item = {
            "workload": {**asdict(workload), "latent_t": workload.latent_t,
                         "audio_t": workload.audio_t, "sequence": sequence},
            "q_shape": list(q.shape), "k_shape": list(k.shape),
            "v_shape": list(v.shape), "occurrences_per_h3_step": 50,
            "attention_flops_per_call": flops,
            "modes": {},
        }
        if "baseline" in args.modes:
            item["modes"]["baseline"] = {
                **baseline_timing,
                "estimated_seconds_per_h3_step": baseline_timing["median_seconds"] * 50,
                "observed_tflops": flops / baseline_timing["median_seconds"] / 1e12,
                "speedup_vs_baseline": 1.0,
            }
        if "mlx_fused" in args.modes:
            candidate, timing = evaluate_timed(fused, args.warmups, args.repeats)
            item["modes"]["mlx_fused"] = {
                **timing, **error(reference, candidate),
                "estimated_seconds_per_h3_step": timing["median_seconds"] * 50,
                "observed_tflops": flops / timing["median_seconds"] / 1e12,
                "speedup_vs_baseline": baseline_timing["median_seconds"] / timing["median_seconds"],
                "same_implementation_as_baseline": True,
            }
        if "chunked_exact" in args.modes:
            chunks = {}
            for chunk_size in args.chunk_sizes:
                def chunked(size=chunk_size):
                    return mx.concatenate([
                        mx.fast.scaled_dot_product_attention(
                            q[:, :, start:start + size], k, v,
                            scale=scale, mask=None)
                        for start in range(0, sequence, size)
                    ], axis=2)
                candidate, timing = evaluate_timed(chunked, args.warmups, args.repeats)
                chunks[str(chunk_size)] = {
                    **timing, **error(reference, candidate),
                    "estimated_seconds_per_h3_step": timing["median_seconds"] * 50,
                    "speedup_vs_baseline": baseline_timing["median_seconds"] / timing["median_seconds"],
                    "dispatches_per_call": math.ceil(sequence / chunk_size),
                }
                print(f"  chunk={chunk_size}: {timing['median_seconds']:.4f}s "
                      f"{chunks[str(chunk_size)]['speedup_vs_baseline']:.3f}x "
                      f"maxerr={chunks[str(chunk_size)]['max_abs_error']:.3g}", flush=True)
            item["modes"]["chunked_exact"] = chunks
        print(f"  baseline: {baseline_timing['median_seconds']:.4f}s "
              f"({flops / baseline_timing['median_seconds'] / 1e12:.2f} TFLOP/s)", flush=True)
        report["results"].append(item)
        del q, k, v, reference
        mx.clear_cache()

    args.json_output.write_text(json.dumps(report, indent=2))
    print(f"saved {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
