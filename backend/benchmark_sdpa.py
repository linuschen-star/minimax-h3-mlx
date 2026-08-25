#!/usr/bin/env python3
"""Standalone exact-BF16 MLX SDPA benchmark for production H3 shapes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx


def summary(samples):
    return {
        "samples": samples,
        "median_seconds": statistics.median(samples),
        "best_seconds": min(samples),
        "worst_seconds": max(samples),
    }


def timed(function, repeats):
    mx.eval(function())
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
    return output, summary(samples)


def layout_metadata(array, layout, inner_strides):
    # MLX 0.32's Python array does not expose strides. These element strides
    # follow directly from the explicit contiguous allocation/view operations
    # used below and are also the values consumed by the C++ dispatcher.
    return {"shape": list(array.shape), "layout": layout,
            "inner_hsd_element_strides": list(inner_strides),
            "dtype": str(array.dtype)}


def make_production_layout(sequence, heads=56, head_dim=128):
    # Q/K after qk-norm + RoPE are compact [S,H,D], then metadata-transposed.
    q_shd = mx.random.normal((sequence, heads, head_dim)).astype(mx.bfloat16)
    k_shd = mx.random.normal((sequence, heads, head_dim)).astype(mx.bfloat16)
    # V is a view into fused [all-Q|all-K|all-V] output, retaining 3*H*D
    # sequence stride exactly as Attention.__call__ does.
    fused = mx.random.normal((sequence, heads * head_dim * 3)).astype(mx.bfloat16)
    v_shd = fused[:, 2 * heads * head_dim:].reshape(sequence, heads, head_dim)
    q = q_shd.transpose(1, 0, 2)[None]
    k = k_shd.transpose(1, 0, 2)[None]
    v = v_shd.transpose(1, 0, 2)[None]
    mx.eval(q, k, v)
    return q, k, v, fused


def compare(reference, output):
    return {
        "outputs_exact": bool(mx.array_equal(reference, output).item()),
        "max_abs_difference": float(mx.max(mx.abs(
            reference.astype(mx.float32) - output.astype(mx.float32))).item()),
    }


def production_case(name, sequence, occurrences, repeats, alternatives=True):
    heads, head_dim = 56, 128
    scale = head_dim ** -0.5
    q, k, v, fused = make_production_layout(sequence, heads, head_dim)
    baseline_fn = lambda: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=None)
    reference, baseline = timed(baseline_fn, repeats)
    flops = 4 * heads * sequence * sequence * head_dim
    result = {
        "name": name,
        "q": layout_metadata(q, "sequence-major compact, metadata-transposed to BHSD",
                             (head_dim, heads * head_dim, 1)),
        "k": layout_metadata(k, "sequence-major compact, metadata-transposed to BHSD",
                             (head_dim, heads * head_dim, 1)),
        "v": layout_metadata(v, "fused-QKV view, metadata-transposed to BHSD",
                             (head_dim, heads * head_dim * 3, 1)),
        "batch": 1, "heads": heads, "query_sequence": sequence,
        "key_sequence": sequence, "head_dim": head_dim,
        "mask": None, "scale": scale, "occurrences_per_step": occurrences,
        "baseline": baseline,
        "attention_flops_per_call": flops,
        "observed_tflops": flops / baseline["median_seconds"] / 1e12,
        "estimated_accumulated_seconds": baseline["median_seconds"] * occurrences,
        "production_profile_accumulated_seconds": 28.2401712523 if sequence == 15100 else None,
    }
    if alternatives:
        candidates = {}

        # Isolate V's fused-QKV gap without changing its logical layout.
        v_compact_shd = mx.contiguous(v[0].transpose(1, 0, 2))
        v_compact = v_compact_shd.transpose(1, 0, 2)[None]
        mx.eval(v_compact)
        output, timing = timed(lambda: mx.fast.scaled_dot_product_attention(
            q, k, v_compact, scale=scale, mask=None), repeats)
        candidates["compact_sequence_major_v_sdpa_only"] = {
            **timing, **compare(reference, output),
            "v": layout_metadata(v_compact, "sequence-major compact BHSD view",
                                 (head_dim, heads * head_dim, 1))}

        # Head-major contiguous is another layout accepted by the same API.
        q_hsd, k_hsd, v_hsd = mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)
        mx.eval(q_hsd, k_hsd, v_hsd)
        output, timing = timed(lambda: mx.fast.scaled_dot_product_attention(
            q_hsd, k_hsd, v_hsd, scale=scale, mask=None), repeats)
        candidates["head_major_sdpa_only"] = {
            **timing, **compare(reference, output),
            "q": layout_metadata(q_hsd, "head-major contiguous BHSD",
                                 (sequence * head_dim, head_dim, 1)),
            "k": layout_metadata(k_hsd, "head-major contiguous BHSD",
                                 (sequence * head_dim, head_dim, 1)),
            "v": layout_metadata(v_hsd, "head-major contiguous BHSD",
                                 (sequence * head_dim, head_dim, 1))}

        # Production would have to pay these copies for every layer.
        output, timing = timed(lambda: mx.fast.scaled_dot_product_attention(
            mx.contiguous(q), mx.contiguous(k), mx.contiguous(v),
            scale=scale, mask=None), repeats)
        candidates["head_major_pack_plus_sdpa"] = {
            **timing, **compare(reference, output)}

        half_heads = heads // 2
        def split_heads():
            return mx.concatenate([
                mx.fast.scaled_dot_product_attention(
                    q[:, start:stop], k[:, start:stop], v[:, start:stop],
                    scale=scale, mask=None)
                for start, stop in ((0, half_heads), (half_heads, heads))
            ], axis=1)
        output, timing = timed(split_heads, repeats)
        candidates["split_heads_2"] = {**timing, **compare(reference, output)}

        half_sequence = sequence // 2
        def split_queries():
            return mx.concatenate([
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:stop], k, v, scale=scale, mask=None)
                for start, stop in ((0, half_sequence), (half_sequence, sequence))
            ], axis=2)
        output, timing = timed(split_queries, repeats)
        candidates["split_queries_2"] = {**timing, **compare(reference, output)}

        for candidate in candidates.values():
            candidate["speedup_vs_baseline"] = (
                baseline["median_seconds"] / candidate["median_seconds"])
        result["candidates"] = candidates

    del q, k, v, fused, reference
    gc.collect()
    mx.clear_cache()
    return result


def diagnostic_case(sequence, heads, head_dim, repeats):
    scale = head_dim ** -0.5
    q = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, heads, sequence, head_dim)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    _, timing = timed(lambda: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=None), repeats)
    flops = 4 * heads * sequence * sequence * head_dim
    result = {
        "sequence": sequence, "heads": heads, "head_dim": head_dim,
        "dtype": "bfloat16", "mask": None, "scale": scale,
        **timing, "attention_flops": flops,
        "observed_tflops": flops / timing["median_seconds"] / 1e12,
    }
    del q, k, v
    gc.collect()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path, default=Path("h3_sdpa_benchmark.json"))
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    mx.random.seed(321)
    report = {
        "title": "H3 exact BF16 fused SDPA standalone benchmark",
        "hardware": {"python": platform.python_version(), "mlx": mx.__version__,
                     "device": mx.device_info()},
        "policy": {"dtype": "bfloat16", "mask": None, "quantization": False,
                   "approximation": False, "h3_model_loaded": False},
        "production_cases": [], "diagnostics": [],
    }
    for name, sequence, occurrences in (("token_refiner", 256, 2), ("dit_full_attention", 15100, 50)):
        print(f"{name}: S={sequence}, H=56, D=128, calls={occurrences}", flush=True)
        case = production_case(name, sequence, occurrences, args.repeats)
        report["production_cases"].append(case)
        print(f"  baseline {case['baseline']['median_seconds']:.6f}s, {case['observed_tflops']:.2f} TFLOP/s", flush=True)
        for cname, candidate in case.get("candidates", {}).items():
            print(f"  {cname}: {candidate['median_seconds']:.6f}s, {candidate['speedup_vs_baseline']:.3f}x, exact={candidate['outputs_exact']}", flush=True)

    if not args.skip_diagnostics:
        for sequence in (1024, 2048, 4096, 8192, 12000, 15100):
            case = diagnostic_case(sequence, 56, 128, args.repeats)
            report["diagnostics"].append(case)
            print(f"diagnostic S={sequence}: {case['median_seconds']:.6f}s, {case['observed_tflops']:.2f} TFLOP/s", flush=True)
        for heads in (14, 28):
            report["diagnostics"].append(diagnostic_case(15100, heads, 128, args.repeats))
        report["diagnostics"].append(diagnostic_case(15100, 56, 64, args.repeats))

    args.json_output.write_text(json.dumps(report, indent=2))
    print(f"saved {args.json_output}")


if __name__ == "__main__":
    main()
