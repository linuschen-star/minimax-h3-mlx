# H3 production exact-BF16 SDPA investigation

## Production workloads

Both workloads use batch 1, 56 heads, head dimension 128, BF16 Q/K/V,
`mask=None`, and scale `1/sqrt(128)`.

| Site | Q/K/V logical shape | Calls per H3 step | Standalone median | Estimated contribution |
|---|---|---:|---:|---:|
| Token refiner | `[1,56,256,128]` | 2 | 0.000576 s | 0.00115 s |
| DiT full attention | `[1,56,15100,128]` | 50 | 0.561573 s | 28.079 s |

The standalone DiT estimate matches the synchronized production profile
(28.240 s) within 0.6%.

Production Q/K are compact sequence-major `[S,H,D]` results viewed as BHSD.
Their inner `(H,S,D)` element strides are `(128,7168,1)`. V remains a view of
the fused QKV projection and has strides `(128,21504,1)`. MLX accepts all
three without inserting a copy, but each kernel threadgroup working on one
head traverses interleaved sequence-major storage.

## Scaling measurements

Head-major contiguous diagnostic inputs produced:

| Sequence | Median | Observed attention throughput |
|---:|---:|---:|
| 1,024 | 0.002470 s | 12.17 TFLOP/s |
| 2,048 | 0.009510 s | 12.65 TFLOP/s |
| 4,096 | 0.037925 s | 12.68 TFLOP/s |
| 8,192 | 0.152795 s | 12.59 TFLOP/s |
| 12,000 | 0.333102 s | 12.39 TFLOP/s |
| 15,100 | 0.535925 s | 12.20 TFLOP/s |

At sequence 15,100, changing head count from 14 to 28 to 56 retained
12.19-12.20 TFLOP/s. Head dimension 64 reached 12.46 TFLOP/s versus 12.20 for
dimension 128. Runtime is therefore quadratic in sequence length and linear
in heads/head dimension, with occupancy already saturated by 14 heads.
Dispatch count is not the limiter.

MLX 0.32's non-NAX M4 full-attention kernel uses `BQ=32`, `BK=16` for head
dimension 128. The production call dispatches 472 query tiles per head and
26,432 threadgroups. Each query tile streams the K/V sequence while performing
the exact score and value work. The call contains approximately 6.54 TFLOP.
An idealized tile-level K/V traffic estimate is about 204 GB per call before
cache effects, corresponding to roughly 378 GB/s at the optimized 0.541 s.
The kernel is limited by the exact quadratic attention-matrix work plus its
tiled K/V traffic, not Python, insufficient heads, or launch count.

## Exact BF16 layout alternatives

Initial five-repeat results:

| Candidate | Time | Speedup | Exact |
|---|---:|---:|---:|
| Production layout | 0.561573 s | 1.000x | reference |
| Compact sequence-major V | 0.556482 s | 1.009x | yes |
| Prepacked head-major Q/K/V, SDPA only | 0.537162 s | 1.045x | yes |
| Pack head-major Q/K/V + SDPA | 0.538397 s | 1.043x | yes |
| Split heads into two calls | 0.576002 s | 0.975x | yes |
| Split queries into two calls | 0.576656 s | 0.974x | yes |

A dedicated 15-repeat interleaved A/B confirmed:

- production layout: 0.563198 s;
- V pack + SDPA: 0.548957 s (1.026x);
- Q/K pack + SDPA: 0.554807 s (1.015x);
- Q/K/V pack + SDPA: 0.541515 s (1.040x);
- prepacked SDPA alone: 0.540718 s (1.042x);
- three BF16 packs alone: 0.004375 s;
- every attention output was bit-exact, max absolute difference 0.

## Production A/B and decision

Fixed workload: original BF16 weights, synthetic 256-token conditioning,
832x480, 124 frames, seed 0, same 20-evaluation sigma grid, exactly two model
evaluations.

| Metric | Baseline | Head-major Q/K/V |
|---|---:|---:|
| Step 1 | 70.964 s | 69.887 s |
| Step 2 | 71.087 s | 69.905 s |
| Two-step mean | 71.026 s | 69.896 s |
| Peak memory | 64.96 GiB | 65.27 GiB |

The candidate saves 1.129 seconds per step on average (1.59% total-step
improvement), with a 1.182-second steady-state improvement (1.66%). Peak
memory increases by approximately 0.31 GiB. There are 150 deliberate BF16
layout materializations per step and still no backend, CPU, NumPy, or dtype
transition.

Decision: **KEEP** for sequence lengths at least 1,024. The 256-token refiner
remains on the original layout because packing made that workload slower.

After this layout correction, the exact fused MLX kernel sustains about 12.1
TFLOP/s at the full production shape, close to the 12.2 TFLOP/s contiguous
diagnostic and within about 5% of the best measured sequence-size point. No
existing MLX 0.32 layout or dispatch candidate produced another meaningful
win. A replacement exact-BF16 attention kernel is not integrated.

Artifacts in the parent outputs directory:

- `h3_sdpa_bf16_standalone.json`
- `h3_sdpa_bf16_layout_ab.json`
- `h3_sdpa_ab_baseline.json`
- `h3_sdpa_ab_head_major.json`

## 2026-08-22 official-path ceiling check

This follow-up did not redefine the old sequence-major implementation as the
baseline. The baseline is the current production path: contiguous BHSD BF16
Q/K/V passed to MLX 0.32.0's official
`mx.fast.scaled_dot_product_attention`, with `mask=None`, scale
`1/sqrt(128)`, and FP32 softmax accumulation as required by the API contract.
The explicit `mlx_fused` control is consequently the same implementation and
measured 1.001x, i.e. benchmark noise rather than a new optimization.

Official references inspected before the experiment:

- MLX 0.32.0 `python/src/fast.cpp` API and its FP32-softmax contract;
- MLX 0.32.0 `mlx/backend/metal/scaled_dot_product_attention.cpp` dispatcher;
- Apple's `mlx-examples/video/wan2.1/wan/layers.py`, which uses BHSD inputs and
  the same fused SDPA primitive for video attention.

For the M4 Max non-NAX path at D=128, the official 0.32.0 dispatcher selects
Steel full attention with BQ=32 and BK=16. Query-only `chunked_exact` therefore
keeps the official fused kernel for every chunk; it does not materialize a
score matrix or introduce an alternative softmax.

| Real H3 workload | Sequence | Current fused median | Throughput | 50-call SDPA estimate |
|---|---:|---:|---:|---:|
| 864x480, 124 frames | 15,655 | 0.5834 s | 12.04 TFLOP/s | 29.17 s |
| 1344x768, 124 frames | 37,966 | 3.5631 s | 11.60 TFLOP/s | 178.16 s |
| 864x480, 362 frames | 44,797 | 5.0080 s | 11.49 TFLOP/s | 250.40 s |
| 1920x1088, 124 frames | 76,150 | 15.0240 s | 11.07 TFLOP/s | 751.20 s |

Exact query-chunk A/B results:

| Workload | Query chunk | Median | Relative | Numerical result |
|---|---:|---:|---:|---|
| 864x480/5s | 512 | 0.6128 s | 0.952x | bit-exact |
| 864x480/5s | 1024 | 0.6029 s | 0.968x | bit-exact |
| 864x480/5s | 2048 | 0.5982 s | 0.975x | bit-exact |
| 864x480/5s | 4096 | 0.5991 s | 0.974x | bit-exact |
| 1344x768/5s | 2048 | 3.6778 s | 0.969x | bit-exact |
| 1344x768/5s | 4096 | 3.6277 s | 0.982x | bit-exact |

The stable 11--12 TFLOP/s rate across a 4.9x sequence-length span, runtime
following the exact S-squared work, and failure of all extra query dispatches
to win show that the existing fused Steel kernel is the practical MLX 0.32 API
ceiling for these shapes. The limiter is the exact quadratic score/value work
and tiled K/V traffic. It is not a hidden full score allocation, Python launch
overhead, inadequate occupancy, a mask path, or repeated RoPE construction.

Decision: **no production integration**. `chunked_exact` is retained only in
the standalone benchmark because it is slower at every measured production
shape. A custom exact BF16 Metal kernel would have to beat these standalone
numbers before it could justify any H3 change.

New artifacts:

- `benchmark_attention_backends.py`
- `h3_attention_864x480.json`
- `h3_attention_1344x768.json`
- `h3_attention_large_baselines.json`
- `h3_attention_memory_ab.json`
