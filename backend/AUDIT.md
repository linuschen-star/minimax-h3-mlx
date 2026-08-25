# MiniMax H3 Apple Silicon audit

## Sources inspected

1. Official MiniMax H3 modular Diffusers sources in `work/minimax_h3_official/`,
   including encoder, packed-layout, denoise, scheduler and decoder stages.
2. Official transformer, scheduler and visual VAE sources in
   `work/transformer_minimax_h3.py`, `work/scheduling_minimax_h3.py` and
   `work/autoencoder_kl_minimax_h3.py`.
3. Current ComfyUI H3 transformer, video VAE and audio VAE under
   `work/ComfyUI/comfy/ldm/minimax/`.
4. This project's model, sampler, text encoder, video VAE, audio VAE, runner,
   validators and tests.
5. Installed MLX 0.32.0 APIs and local source, especially fast SDPA, RMSNorm,
   lazy evaluation, allocator statistics and native QuantizedLinear.
6. `uetuluk2/minimax-h3-mlx-rebuild`, including its README, conversion logic
   and findings on the original checkpoint and published MLX pack.
7. Original BF16 weight indices and the official converted Diffusers shard.

## Confirmed facts

- Original transformer size: 33,122,992,896 parameters; 535 tensors in 13 shards.
- Production base shape used here: 832x480, 124 RGB frames, 37 video latent
  frames and 207 stereo audio latent frames.
- H3 packs one joint sequence in T2VA order `[text | audio | video]`.
- The denoiser uses shift 12 for video and shift 3 for audio.
- The original-format 52 fused QKV tensors are per-head interleaved. Relayout
  to global Q/K/V was compared with official converted block-0 weights:
  Q, K and V each had max absolute error 0.
- MLX fast SDPA performs the softmax in FP32 for BF16 inputs.
- MLX `QuantizedLinear` calls native `mx.quantized_matmul`; it does not
  dequantize a full weight on every forward.
- The current BF16 hot loop contains only MLX arrays and operations.
- `mx.eval` at the outer denoising-step boundary is required to bound the lazy
  graph. There is no layer-level synchronization.
- On an Apple M4 Max, the 832x480/124-frame BF16 workload loads in 51.79 s,
  executes its first two transformer evaluations in 71.07/71.06 s, and peaks
  at 64.96 GiB with no hot-loop framework or device transitions.

## Behavior confirmed by tests

- Native tiny transformer forward, shifted schedule, patch/audio round trips.
- Qwen layer-50 PyTorch/MLX correlation 0.99999984.
- Visual VAE small reference max absolute error 2.47e-6.
- Full-length audio VAE decode succeeds.
- QKV relayout unit test and official-shard exact comparison succeed.
- Previous output before QKV relayout was invalid noise.
- The attempted full run after adding lazy QKV relayout terminated with a
  Metal command-buffer error before completion; this motivated shard-wise
  materialization and the two-step benchmark policy.

## Current optimization decisions

### Accepted

- **Shard-wise QKV preparation:** removes retention of all lazy source QKV
  tensors, reducing peak unified-memory pressure during load.
- **One RoPE trig table per step:** removes 49 redundant cos/sin calculations
  across the 50 transformer blocks.
- **Persistent MLX denoising zone:** removes all framework transitions from
  the accumulated hot path.
- **Native MLX SDPA:** avoids materializing the quadratic attention matrix and
  provides FP32 softmax.

### Not yet accepted as optimizations

- INT4/INT8: native support exists, but no identical-workload speed/quality
  result has been recorded for this backend.
- Turbo/4-step: excluded from the quality target because it changes sampling
  behavior and can lose action choreography.
- Custom Metal attention: not justified until native SDPA timing establishes a
  bottleneck and a candidate beats it.
- Moving one-time preprocessing between frameworks: low priority until stage
  timing shows material wall-clock impact.

## Assumptions requiring benchmark evidence

- Shard-wise materialization is sufficient to prevent the prior Metal failure.
- The final non-Turbo production step count remains a user quality choice;
  development uses 2 evaluations and short visual checks use 6-8.
- Step 2 approximates steady-state after step-1 compilation for a fixed shape.
