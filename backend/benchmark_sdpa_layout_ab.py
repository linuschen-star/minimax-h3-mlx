#!/usr/bin/env python3
"""Focused interleaved BF16 layout A/B for H3's full production SDPA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from benchmark_sdpa import compare, make_production_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--json-output", type=Path, default=Path("h3_sdpa_layout_ab.json"))
    args = parser.parse_args()
    if args.repeats < 7:
        parser.error("--repeats must be at least 7")

    mx.random.seed(987)
    sequence, heads, head_dim = 15100, 56, 128
    scale = head_dim ** -0.5
    q, k, v, _ = make_production_layout(sequence, heads, head_dim)
    qh, kh, vh = mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)
    mx.eval(qh, kh, vh)

    functions = {
        "production_layout": lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=None),
        "pack_v_plus_sdpa": lambda: mx.fast.scaled_dot_product_attention(
            q, k, mx.contiguous(v), scale=scale, mask=None),
        "pack_qk_plus_sdpa": lambda: mx.fast.scaled_dot_product_attention(
            mx.contiguous(q), mx.contiguous(k), v, scale=scale, mask=None),
        "pack_qkv_plus_sdpa": lambda: mx.fast.scaled_dot_product_attention(
            mx.contiguous(q), mx.contiguous(k), mx.contiguous(v), scale=scale, mask=None),
        "prepacked_qkv_sdpa_only": lambda: mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=scale, mask=None),
        "packing_qkv_only": lambda: (mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)),
    }
    for function in functions.values():
        mx.eval(function())

    names = list(functions)
    samples = {name: [] for name in names}
    last = {}
    # Rotate order every repetition so each variant occupies every thermal and
    # allocator position instead of running all baselines first.
    for repeat in range(args.repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        if (repeat // len(names)) % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            output = functions[name]()
            mx.eval(output)
            samples[name].append(time.perf_counter() - started)
            last[name] = output

    reference = last["production_layout"]
    baseline = statistics.median(samples["production_layout"])
    results = {}
    for name in names:
        median = statistics.median(samples[name])
        item = {
            "samples": samples[name], "median_seconds": median,
            "best_seconds": min(samples[name]), "worst_seconds": max(samples[name]),
            "speedup_vs_production_layout": baseline / median,
        }
        if name not in ("packing_qkv_only", "production_layout"):
            item.update(compare(reference, last[name]))
        results[name] = item

    flops = 4 * heads * sequence * sequence * head_dim
    report = {
        "title": "H3 production exact-BF16 SDPA interleaved layout A/B",
        "shape": [1, heads, sequence, head_dim], "dtype": "bfloat16",
        "mask": None, "scale": scale, "repeats": args.repeats,
        "attention_flops": flops,
        "results": results,
        "production_observed_tflops": flops / baseline / 1e12,
        "prepacked_observed_tflops": flops / results["prepacked_qkv_sdpa_only"]["median_seconds"] / 1e12,
    }
    args.json_output.write_text(json.dumps(report, indent=2))
    for name, item in results.items():
        print(f"{name:28s} {item['median_seconds']:.6f}s  {item['speedup_vs_production_layout']:.3f}x")
    print(f"saved {args.json_output}")


if __name__ == "__main__":
    main()
