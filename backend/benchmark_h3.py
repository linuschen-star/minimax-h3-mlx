#!/usr/bin/env python3
"""Short H3 Apple-Silicon benchmark; never performs a full generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.model import (
    H3Config,
    MiniMaxH3Transformer,
    T2VLayout,
    load_sharded_safetensors,
    time_shift_sigma_float,
)
from minimax_h3_mlx.profiling import BackendMetrics, RegionProfiler
from minimax_h3_mlx.sampling import sigma_schedule


GIB = 1024**3


def seconds(start):
    return time.perf_counter() - start


def memory():
    return {
        "active_gib": mx.get_active_memory() / GIB,
        "cache_gib": mx.get_cache_memory() / GIB,
        "peak_gib": mx.get_peak_memory() / GIB,
    }


def tiny_correctness():
    cfg = H3Config(
        hidden_size=64,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_dim=128,
        text_dim=48,
        freq_dim=32,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=2,
    )
    model = MiniMaxH3Transformer(cfg)
    video = mx.zeros((1, 24, 1, 2, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, 32, 2, 1), dtype=mx.bfloat16)
    context = mx.zeros((1, 4, 48), dtype=mx.bfloat16)
    started = time.perf_counter()
    out_v, out_a = model(video, audio, context, 1.0)
    mx.eval(out_v, out_a)
    if out_v.shape != video.shape or out_a.shape != audio.shape:
        raise RuntimeError(f"tiny output mismatch: {out_v.shape}, {out_a.shape}")
    if not bool(mx.all(mx.isfinite(out_v)).item()) or not bool(mx.all(mx.isfinite(out_a)).item()):
        raise RuntimeError("tiny forward produced non-finite output")
    return {"seconds": seconds(started), "sequence_tokens": 7, "status": "PASS"}


def load_context(path: Path | None, tokens: int, text_dim: int, metrics: BackendMetrics):
    started = time.perf_counter()
    if path:
        data = mx.load(str(path))
        context = data["context"].astype(mx.bfloat16)
        tags = [int(value) for value in data.get("text_token_tags", mx.ones((context.shape[1],))).tolist()]
        # tolist() is a deliberate setup-only CPU materialization; never in the hot loop.
        metrics.cpu_materializations += 1
        metrics.dtype_conversions += 1
    else:
        context = mx.zeros((1, tokens, text_dim), dtype=mx.bfloat16)
        tags = [1] * tokens
    mx.eval(context)
    metrics.explicit_sync += 1
    return context, tags, seconds(started)


def benchmark(args):
    report = {
        "title": "H3 Apple Silicon Benchmark",
        "hardware": {"python": platform.python_version(), "machine": platform.machine(), "mlx": mx.__version__, "device": mx.device_info()},
        "workload": {"width": args.width, "height": args.height, "frames": 124, "audio_latents": 207, "reference_evaluations": args.reference_steps},
    }
    report["test_a_tiny"] = tiny_correctness()

    metrics = BackendMetrics()
    cfg = H3Config.from_json(args.transformer_dir / "config.json")
    cfg = replace(cfg, head_major_sdpa=args.attention_layout == "head-major")
    context, tags, condition_seconds = load_context(args.conditioning, args.context_tokens, cfg.text_dim, metrics)
    report["conditioning"] = {"seconds": condition_seconds, "tokens": context.shape[1], "source": str(args.conditioning) if args.conditioning else "synthetic-zero"}

    mx.reset_peak_memory()
    model_started = time.perf_counter()
    model = MiniMaxH3Transformer(cfg)
    load_sharded_safetensors(model, args.transformer_dir / "model.safetensors.index.json", metrics=metrics)
    report["model_load"] = {"seconds": seconds(model_started), "memory": memory(), "qkv_relayouts": metrics.qkv_relayouts}
    if metrics.qkv_relayouts != cfg.num_layers + cfg.num_refiner_layers:
        raise RuntimeError(f"expected 52 QKV relayouts, got {metrics.qkv_relayouts}")

    latent_h, latent_w = args.height // 16, args.width // 16
    video = mx.random.normal((1, cfg.in_channels, 37, latent_h, latent_w)).astype(mx.bfloat16)
    audio = mx.random.normal((1, cfg.audio_in_channels, 2, 207)).astype(mx.bfloat16)
    layout = T2VLayout(context.shape[1], 37, latent_h, latent_w, 207)
    sigmas_v = sigma_schedule(args.reference_steps + 1, cfg.sigma_shift_video)
    sigmas_a = [time_shift_sigma_float(s, cfg.sigma_shift_video, cfg.sigma_shift_audio) if s else 0.0 for s in sigmas_v]
    mx.eval(video, audio)
    metrics.explicit_sync += 1
    hot_before = metrics.snapshot()
    report["production_sequence_tokens"] = layout.seq_len
    report["steps"] = []

    for index in range(2):
        region_profiler = RegionProfiler() if args.deep_profile and index == 0 else None
        step_peak_before = mx.get_peak_memory()
        transform_started = time.perf_counter()
        velocity_v, velocity_a = model(
            video, audio, context, sigmas_v[index], layout=layout,
            text_token_tags=tags, _profiler=region_profiler,
        )
        if cfg.head_major_sdpa:
            # Exact BF16 Q/K/V packs before each of the 50 long-sequence SDPA
            # calls. The two short token-refiner calls stay sequence-major.
            metrics.layout_conversions += 3 * cfg.num_layers
        mx.eval(velocity_v, velocity_a)
        metrics.explicit_sync += 1
        transform_seconds = seconds(transform_started)
        if transform_seconds > args.abort_step_seconds:
            report["steps"].append({"index": index + 1, "transformer_seconds": transform_seconds, "aborted": True})
            report["abort_reason"] = f"step exceeded {args.abort_step_seconds:.1f}s catastrophic threshold"
            break

        update_started = time.perf_counter()
        video = (video.astype(mx.float32) + velocity_v.astype(mx.float32) * (sigmas_v[index] - sigmas_v[index + 1])).astype(mx.bfloat16)
        audio = (audio.astype(mx.float32) + velocity_a.astype(mx.float32) * (sigmas_a[index] - sigmas_a[index + 1])).astype(mx.bfloat16)
        metrics.dtype_conversions += 6
        mx.eval(video, audio)
        metrics.explicit_sync += 1
        report["steps"].append({
            "index": index + 1,
            "sigma_video": sigmas_v[index],
            "transformer_seconds": transform_seconds,
            "sampler_update_seconds": seconds(update_started),
            "memory": memory(),
            "peak_growth_gib": (mx.get_peak_memory() - step_peak_before) / GIB,
        })
        if region_profiler:
            regions = region_profiler.report()
            measured = sum(item["seconds"] for item in regions.values())
            report["deep_profile"] = {
                "mode": "synchronized major regions on step 1",
                "regions": regions,
                "measured_region_seconds": measured,
                "unattributed_seconds": max(0.0, transform_seconds - measured),
                "explicit_sync": region_profiler.explicit_sync,
                "warning": "Deep mode prevents fusion across region boundaries; compare its total with the normal second step to estimate perturbation.",
            }

    report["backend_transitions_hot_loop"] = metrics.delta(hot_before)
    report["backend_transitions_total"] = metrics.snapshot()
    report["final_memory"] = memory()
    if "deep_profile" in report and len(report["steps"]) == 2:
        deep = report["steps"][0]["transformer_seconds"]
        normal = report["steps"][1]["transformer_seconds"]
        report["deep_profile"]["estimated_overhead_seconds"] = deep - normal
        report["deep_profile"]["estimated_overhead_percent"] = (deep / normal - 1.0) * 100.0
    return report


def print_report(report):
    print("\nH3 Apple Silicon Benchmark")
    print(f"Tiny correctness             {report['test_a_tiny']['status']} {report['test_a_tiny']['seconds']:.3f} s")
    print(f"Conditioning                 {report['conditioning']['seconds']:.3f} s ({report['conditioning']['tokens']} tokens)")
    print(f"Weights load + QKV prepare   {report['model_load']['seconds']:.3f} s")
    print(f"Persistent active memory     {report['model_load']['memory']['active_gib']:.2f} GiB")
    print(f"Production sequence          {report['production_sequence_tokens']} tokens")
    for step in report["steps"]:
        print(f"Step {step['index']} transformer          {step['transformer_seconds']:.3f} s" + (" ABORT" if step.get("aborted") else ""))
        if not step.get("aborted"):
            print(f"Step {step['index']} sampler update       {step['sampler_update_seconds']:.3f} s")
            print(f"Step {step['index']} peak memory          {step['memory']['peak_gib']:.2f} GiB")
    if "deep_profile" in report:
        print("\nDEEP REGION PROFILE — STEP 1")
        regions = report["deep_profile"]["regions"]
        measured = report["deep_profile"]["measured_region_seconds"]
        for name, item in sorted(regions.items(), key=lambda pair: pair[1]["seconds"], reverse=True):
            percentage = item["seconds"] / measured * 100.0 if measured else 0.0
            print(f"{name:32s} {item['seconds']:8.3f} s {percentage:6.1f}%  ({item['calls']} calls)")
        print(f"Estimated profiler overhead   {report['deep_profile']['estimated_overhead_seconds']:.3f} s ({report['deep_profile']['estimated_overhead_percent']:.1f}%)")
    hot = report["backend_transitions_hot_loop"]
    print("\nBACKEND TRANSITIONS — HOT LOOP")
    for key in ("torch_to_mlx", "mlx_to_torch", "mlx_to_numpy", "cpu_materializations", "device_transitions", "explicit_sync", "dtype_conversions", "layout_conversions"):
        print(f"{key:28s} {hot[key]}")
    if "abort_reason" in report:
        print(f"\nABORTED: {report['abort_reason']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transformer-dir", type=Path, required=True)
    p.add_argument("--conditioning", type=Path)
    p.add_argument("--context-tokens", type=int, default=256, help="synthetic text length when --conditioning is omitted")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--reference-steps", type=int, default=8, help="schedule used to select the first two production sigmas; only two evaluations are executed")
    p.add_argument("--abort-step-seconds", type=float, default=600.0)
    p.add_argument("--json-output", type=Path, default=Path("h3_benchmark.json"))
    p.add_argument("--tiny-only", action="store_true")
    p.add_argument("--deep-profile", action="store_true", help="synchronize broad regions in step 1; step 2 remains the normal low-overhead control")
    p.add_argument("--attention-layout", choices=("baseline", "head-major"),
                   default="head-major", help="exact BF16 SDPA input layout; baseline is retained for A/B")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.width % 32 or args.height % 32:
        p.error("production dimensions must be divisible by 32")
    if not args.worker:
        args.json_output.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        result = subprocess.run(command)
        if result.returncode != 0 and not args.json_output.exists():
            failure = {
                "title": "H3 Apple Silicon Benchmark",
                "status": "FATAL_WORKER_EXIT",
                "returncode": result.returncode,
                "hint": "Native Metal/runtime termination occurred; inspect terminal output.",
            }
            args.json_output.write_text(json.dumps(failure, indent=2))
            print(f"Saved fatal-worker log: {args.json_output}", file=sys.stderr)
        raise SystemExit(result.returncode)

    mx.random.seed(0)
    try:
        if args.tiny_only:
            report = {"title": "H3 Apple Silicon Benchmark", "test_a_tiny": tiny_correctness()}
            print(f"Tiny correctness: {report['test_a_tiny']['status']} {report['test_a_tiny']['seconds']:.3f} s")
        else:
            report = benchmark(args)
            print_report(report)
    except Exception as error:
        report = {
            "title": "H3 Apple Silicon Benchmark",
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "memory_at_error": memory(),
        }
        args.json_output.write_text(json.dumps(report, indent=2))
        print(f"BENCHMARK ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"Saved failure log: {args.json_output}", file=sys.stderr)
        raise
    else:
        args.json_output.write_text(json.dumps(report, indent=2))
        print(f"\nSaved machine-readable log: {args.json_output}")


if __name__ == "__main__":
    main()
