# H3 native MLX performance profile

## Fixed workload

- Apple M4 Max 40-core GPU, 128 GB unified memory
- MLX 0.32.0, original BF16 H3 weights
- 832x480, 124 frames, 37 video latent frames, 207 stereo audio latents
- 256 conditioning tokens, packed sequence length 15,100
- Same seed and 20-evaluation sigma grid; exactly two Transformer evaluations executed

Machine-readable result: `h3_832x480_deep_profile.json` in the parent outputs directory.

## Measured breakdown

Deep step total: 71.871 s. Normal control step: 70.474 s. The synchronized
profiler therefore added 1.397 s (2.0%). Percentages below use accumulated
deep-region time.

| Rank | Region | Time | Share |
|---:|---|---:|---:|
| 1 | Fused full SDPA | 28.240 s | 39.3% |
| 2 | MLP input projection | 16.462 s | 22.9% |
| 3 | Attention QKV projection | 12.208 s | 17.0% |
| 4 | MLP output projection | 8.501 s | 11.8% |
| 5 | Attention output projection | 4.229 s | 5.9% |
| 6 | QK normalization and RoPE | 0.775 s | 1.1% |
| 7 | All remaining measured regions | 1.450 s | 2.0% |

Grouped by execution family, projection GEMMs consume 41.400 s (57.6%),
and SDPA consumes 28.240 s (39.3%). Everything else is about 2.2 s (3.1%).
Sampler update remains below 1 ms.

## Root cause

Inspection of the exact MLX 0.32.0 source (tag `v0.32.0`) confirms that the
H3 attention shape `[1,56,15100,128]` uses the fused Metal full-attention
kernel. Q, K and V have unit head-dimension stride, so the dispatcher does not
insert contiguous copies. The output layout is intentionally produced in the
stride order consumed by the following reshape and output projection.

The projection weights are row-contiguous BF16 arrays. `nn.Linear` passes the
weight transpose as GEMM metadata; it does not materialize a transposed weight
per layer or per step. The production shapes route to the regular large Metal
GEMM rather than a fallback.

The measured 96.9% is therefore required full-sequence attention and BF16 GEMM
work, not Python overhead, backend conversion, repeated weight packing,
explicit synchronization, or accidental layout copies.

## Candidate decisions

### Native affine INT8 weight-only GEMM — REJECT

- Measured bottleneck: projection GEMMs, 57.6% of the step.
- Test shape: QKV `[15100,5376] @ [21504,5376].T`, group size 64.
- BF16 best of three: 0.2409 s.
- INT8 best of three: 0.2755 s.
- Result: 0.874x, approximately 14.4% slower.
- Decision: REJECT. No production code or weight format changed.

### Native affine INT4 weight-only GEMM — REJECT

- Same real QKV shape and group size.
- BF16 best of three: 0.2411 s.
- INT4 best of three: 0.2748 s.
- Result: 0.877x, approximately 14.0% slower.
- Decision: REJECT. It is both slower and numerically riskier.

## Current conclusion

There is no measured, behavior-preserving removal of work large enough to
materially reduce the approximately 71-second BF16 step. Caching positional
tables or prompt refinement would target substantially less than 0.1% of the
step and is intentionally not presented as acceleration. Compiling small
elementwise regions has a theoretical ceiling near the remaining 3% and does
not address the demonstrated bottleneck.

A materially faster next candidate must directly improve the measured SDPA or
large projection kernels. Such work would require an alternative kernel,
attention formulation, or numerical representation and must be evaluated as a
separate A/B with correctness and 6-8-step quality validation; none is silently
enabled in the full-quality production path.
