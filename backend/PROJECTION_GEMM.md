# Production BF16 projection GEMM isolation

This investigation is intentionally limited to BF16 projection GEMMs. It does
not change or benchmark SDPA, RoPE, sampling, quantization, or the H3 model.

## Production shapes

The standard 832x480 / 124-frame / 256-token workload has a packed sequence
length of 15,100. The dominant 50-layer projections are:

| Projection | MLX multiplication | Calls/step | Accumulated deep-profile time |
|---|---|---:|---:|
| DiT QKV | `[15100,5376] @ [21504,5376].T` | 50 | 12.208 s |
| DiT attention out | `[15100,7168] @ [5376,7168].T` | 50 | 4.229 s |
| DiT MLP in | `[15100,5376] @ [28672,5376].T` | 50 | 16.462 s |
| DiT MLP out | `[15100,14336] @ [5376,14336].T` | 50 | 8.501 s |
| DiT AdaLN steady step | `[2,2688] @ [96768,2688].T + bias` | 50 | 0.291 s |

The standalone manifest also includes the one-shot condition projection, all
four 256-token refiner projection shapes, first-step AdaLN `M=1`, and final
AdaLN `M=1/2`. FP32 boundary projections are excluded by design.

## Exact MLX 0.32 dispatch

Source inspected: official MLX tag `v0.32.0`, particularly
`mlx/backend/metal/matmul.cpp`.

- Checkpoint weights are row-contiguous `[N,K]`; `weight.T` is recognized by
  `check_transpose` and passed as `transpose_b=true`. No contiguous copy occurs.
- A physically transposed `[K,N]` weight is row-contiguous and dispatches with
  `transpose_b=false`. It also requires no runtime copy.
- The M4 Max reports architecture `applegpu_g16s`. MLX classifies suffix `s`
  as the medium-device Steel path, selecting `BM=64, BN=64, BK=16, WM=2,
  WN=2` for both BF16 NN and NT large GEMMs.
- All four dominant shapes exceed the non-NAX split-K threshold and therefore
  use one regular Steel GEMM dispatch. `M=1` AdaLN uses GEMV.
- NAX is unavailable for `g16s` in MLX 0.32; the source requires generation 17
  or newer for architecture suffix `s`.

## Standalone results

Five timed repetitions were alternated between layouts after warmup. All four
large projections produced bit-identical BF16 outputs.

| Projection | Current NT | Pretransposed NN | NN speedup | Current throughput |
|---|---:|---:|---:|---:|
| DiT QKV | 0.24708 s | 0.24618 s | 1.004x | 14.13 TFLOP/s |
| DiT attention out | 0.08300 s | 0.08236 s | 1.008x | 14.02 TFLOP/s |
| DiT MLP in | 0.32824 s | 0.32746 s | 1.002x | 14.18 TFLOP/s |
| DiT MLP out | 0.17290 s | 0.16850 s | 1.026x | 13.46 TFLOP/s |

Across every BF16 projection shape, a fully pretransposed layout estimated
only 0.70% less projection time. Selectively keeping only exact local winners
would save roughly 0.3 seconds from a roughly 70.5-second Transformer step,
before accounting for the one-time physical weight transforms and production
loader complexity.

Two-dispatch scheduling candidates were also tested on every large shape:

- splitting M in half: 0.972-0.980x (slower);
- splitting N in half: 0.969-0.983x (slower).

Both variants were bit-identical, but extra dispatch and concatenation made
them consistently slower. First-step AdaLN NN/GEMV also failed bit identity
(max absolute difference 1.0), so it is not an eligible exact candidate.

## Decision

No candidate achieved a material standalone win. The four dominant shapes all
cluster at 13.46-14.18 TFLOP/s, while changing NT to NN or splitting the same
work does not improve throughput materially. This is evidence that these H3
shapes are close to the current MLX 0.32 regular Steel BF16 GEMM path's observed
limit on this M4 Max.

Per policy, no projection layout change is integrated into H3 and no full
2-step H3 A/B is run. The production numerical path remains unchanged.

Artifacts:

- `benchmark_projection_gemms.py`: reproducible standalone benchmark.
- `h3_projection_gemm_bf16_ab.json`: all BF16 production shapes.
- `h3_projection_gemm_bf16_dispatch_ab.json`: dominant-shape dispatch tests.
