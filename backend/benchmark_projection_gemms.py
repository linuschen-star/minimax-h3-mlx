#!/usr/bin/env python3
"""Standalone BF16 GEMM A/B for every production H3 projection shape.

This benchmark does not import or execute the H3 model.  It compares the
current checkpoint layout (row-contiguous [N,K], consumed as ``weight.T``)
with a one-time physically transposed [K,N] BF16 weight.  Both paths call the
same MLX 0.32 matmul/addmm primitives and preserve the numerical format.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    m: int
    k: int
    n: int
    calls_per_step: int
    bias: bool = False
    measured_family: str | None = None
    measured_accumulated_seconds: float | None = None


# 832x480, 124 frames, 256 text tokens: packed sequence length 15,100.
# FP32 boundary projections (video/audio input, timestep, final video/audio)
# are intentionally excluded: this experiment is BF16 projection-only.
PRODUCTION_BF16_SHAPES = (
    ProjectionShape("condition_projection", 256, 5120, 5376, 1, True),
    ProjectionShape("refiner_qkv", 256, 5376, 21504, 2),
    ProjectionShape("refiner_attention_out", 256, 7168, 5376, 2),
    ProjectionShape("refiner_mlp_in", 256, 5376, 28672, 2),
    ProjectionShape("refiner_mlp_out", 256, 14336, 5376, 2),
    ProjectionShape("dit_adaln_first_step", 1, 2688, 96768, 50, True),
    ProjectionShape("dit_adaln_steady_step", 2, 2688, 96768, 50, True,
                    "adaln_projection", 0.2910626670),
    ProjectionShape("dit_qkv", 15100, 5376, 21504, 50, False,
                    "attention_qkv_projection", 12.2076964176),
    ProjectionShape("dit_attention_out", 15100, 7168, 5376, 50, False,
                    "attention_output_projection", 4.2285664219),
    ProjectionShape("dit_mlp_in", 15100, 5376, 28672, 50, False,
                    "mlp_input_projection", 16.4620353321),
    ProjectionShape("dit_mlp_out", 15100, 14336, 5376, 50, False,
                    "mlp_output_projection", 8.5008323737),
    ProjectionShape("final_adaln_first_step", 1, 2688, 10752, 1, True),
    ProjectionShape("final_adaln_steady_step", 2, 2688, 10752, 1, True),
)


def stats(samples):
    ordered = sorted(samples)
    return {
        "samples": samples,
        "median_seconds": statistics.median(samples),
        "best_seconds": ordered[0],
        "worst_seconds": ordered[-1],
    }


def run_case(case: ProjectionShape, repeats: int, scheduling_candidates: bool):
    x = mx.random.normal((case.m, case.k)).astype(mx.bfloat16)
    weight = mx.random.normal((case.n, case.k)).astype(mx.bfloat16)
    bias = mx.random.normal((case.n,)).astype(mx.bfloat16) if case.bias else None
    mx.eval(x, weight, bias)

    # Candidate setup is one-time and deliberately outside timed inference.
    weight_kn = mx.contiguous(weight.T)
    mx.eval(weight_kn)

    def nt_view():
        return mx.addmm(bias, x, weight.T) if bias is not None else x @ weight.T

    def nn_pretransposed():
        return mx.addmm(bias, x, weight_kn) if bias is not None else x @ weight_kn

    # Warm both exact dispatch paths once, then alternate to reduce drift bias.
    mx.eval(nt_view(), nn_pretransposed())
    nt_times, nn_times = [], []
    baseline = candidate = None
    for index in range(repeats):
        order = (("nt", nt_view), ("nn", nn_pretransposed))
        if index % 2:
            order = tuple(reversed(order))
        for name, function in order:
            started = time.perf_counter()
            output = function()
            mx.eval(output)
            elapsed = time.perf_counter() - started
            (nt_times if name == "nt" else nn_times).append(elapsed)
            if name == "nt":
                baseline = output
            else:
                candidate = output

    max_abs = float(mx.max(mx.abs(baseline.astype(mx.float32) - candidate.astype(mx.float32))).item())
    exact = bool(mx.array_equal(baseline, candidate).item())
    nt = stats(nt_times)
    nn = stats(nn_times)
    speedup = nt["median_seconds"] / nn["median_seconds"]
    result = {
        **asdict(case),
        "baseline_layout": "NT: x[M,K] @ row_contiguous_weight[N,K].T",
        "candidate_layout": "NN: x[M,K] @ pretransposed_weight[K,N]",
        "baseline": nt,
        "candidate": nn,
        "median_speedup": speedup,
        "candidate_change_percent": (1.0 - nn["median_seconds"] / nt["median_seconds"]) * 100.0,
        "outputs_exact": exact,
        "max_abs_difference": max_abs,
        "estimated_accumulated_baseline_seconds": nt["median_seconds"] * case.calls_per_step,
        "estimated_accumulated_candidate_seconds": nn["median_seconds"] * case.calls_per_step,
    }
    if scheduling_candidates and case.m >= 10_000:
        x_parts = mx.split(x, 2, axis=0)
        w_parts = mx.split(weight, 2, axis=0)

        def split_m2():
            return mx.concatenate([part @ weight.T for part in x_parts], axis=0)

        def split_n2():
            return mx.concatenate([x @ part.T for part in w_parts], axis=1)

        variants = {}
        for name, function in (("split_m2_nt", split_m2), ("split_n2_nt", split_n2)):
            mx.eval(function())
            samples = []
            output = None
            for _ in range(repeats):
                started = time.perf_counter()
                output = function()
                mx.eval(output)
                samples.append(time.perf_counter() - started)
            variant = stats(samples)
            variant["speedup_vs_baseline"] = nt["median_seconds"] / variant["median_seconds"]
            variant["outputs_exact"] = bool(mx.array_equal(baseline, output).item())
            variant["max_abs_difference"] = float(mx.max(mx.abs(
                baseline.astype(mx.float32) - output.astype(mx.float32))).item())
            variants[name] = variant
        result["scheduling_candidates"] = variants
    del x, weight, weight_kn, bias, baseline, candidate
    gc.collect()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", action="append", help="run only a named shape; repeatable")
    parser.add_argument("--scheduling-candidates", action="store_true",
                        help="for large-M shapes, also test two M or N dispatches plus concatenate")
    parser.add_argument("--json-output", type=Path, default=Path("projection_gemm_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    selected = [case for case in PRODUCTION_BF16_SHAPES if not args.only or case.name in args.only]
    unknown = set(args.only or ()) - {case.name for case in PRODUCTION_BF16_SHAPES}
    if unknown:
        parser.error(f"unknown --only shape(s): {sorted(unknown)}")

    mx.random.seed(123)
    report = {
        "title": "H3 production BF16 projection GEMM microbenchmark",
        "hardware": {"python": platform.python_version(), "mlx": mx.__version__, "device": mx.device_info()},
        "policy": {"dtype": "bfloat16", "quantization": False, "sdpa": False, "h3_model_loaded": False},
        "cases": [],
    }
    for case in selected:
        print(f"{case.name}: M={case.m} K={case.k} N={case.n} calls={case.calls_per_step}", flush=True)
        result = run_case(case, args.repeats, args.scheduling_candidates)
        report["cases"].append(result)
        print(f"  NT {result['baseline']['median_seconds']:.6f}s | NN {result['candidate']['median_seconds']:.6f}s | {result['median_speedup']:.3f}x | exact={result['outputs_exact']}", flush=True)

    weighted_baseline = sum(item["estimated_accumulated_baseline_seconds"] for item in report["cases"])
    weighted_candidate = sum(item["estimated_accumulated_candidate_seconds"] for item in report["cases"])
    report["weighted_projection_estimate"] = {
        "baseline_seconds": weighted_baseline,
        "candidate_seconds": weighted_candidate,
        "speedup": weighted_baseline / weighted_candidate if weighted_candidate else None,
        "change_percent": (1.0 - weighted_candidate / weighted_baseline) * 100.0 if weighted_baseline else None,
    }
    args.json_output.write_text(json.dumps(report, indent=2))
    print(f"saved {args.json_output}")


if __name__ == "__main__":
    main()
