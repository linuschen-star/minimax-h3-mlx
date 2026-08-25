# H3 Apple Silicon execution architecture

## Confirmed execution graph

```text
prompt
  -> CPU tokenizer (once)
  -> native MLX Qwen3-VL language layers 0..49 (once, text-only)
  -> MLX context [1,L,5120]
  -> initialize MLX BF16 video/audio noise (once)
  -> build sigma grids and packed layout on CPU (once)
  -> convert layout constants to MLX (once)
  -> repeat N denoising evaluations entirely in MLX:
       video/audio patch projection
       text projection + two-layer token refiner
       timestep embedding
       MM-RoPE table (once per evaluation, shared by all 50 blocks)
       50 x [AdaLN -> QKV -> RMSNorm/RoPE -> MLX SDPA -> output projection
              -> residual -> AdaLN -> SwiGLU MLP -> residual]
       video/audio output projections
       FP32 Euler update -> BF16 persistent latent
  -> native MLX video VAE decode (once)
  -> native MLX audio VAE decode (once)
  -> CPU writes WAV and streams RGB bytes to ffmpeg (once)
```

## Frequency and optimization priority

| component | frequency | backend | priority |
|---|---:|---|---:|
| tokenizer | once | CPU | low |
| Qwen text encoder | once | MLX | medium |
| packed layout / sigma grid | once | CPU -> MLX | low |
| transformer | steps x 50 layers | MLX | highest |
| SDPA | steps x 50 layers | MLX fast SDPA | highest |
| sampler update | once per step | MLX | medium |
| video/audio VAE | once after sampling | MLX | profiling-dependent |
| ffmpeg mux | once | CPU / VideoToolbox or libx264 | low |

## Backend boundary map

The denoising hot loop has no Torch, NumPy, or CPU tensor boundary.

```text
CPU prompt string
  -> token IDs -> MLX                         one setup boundary
MLX context + latent
  -> N complete transformer/sampler steps    persistent MLX zone
MLX decoded frames/audio
  -> NumPy/CPU output buffers                 one final boundary per output chunk
```

PyTorch/MPS exists only in `prepare_blank_fl2va.py`, an optional one-time legacy
FL2VA keyframe preprocessor. It is not used by normal text-only generation or
the denoising loop.

## Weight loading

The released original transformer contains 52 fused QKV tensors in per-head
`[q,k,v]` row order. The runtime requires global `[all-q;all-k;all-v]` order.
Each shard is loaded, relaid out, materialized, installed, and released before
the next shard. This prevents lazy relayout graphs from retaining all source
QKV tensors and increasing unified-memory peak.

The BF16 path does not quantize or dequantize weights. Quantized execution is
not called an optimization until native `mx.quantized_matmul` is benchmarked
against the identical BF16 workload and its quality tradeoff is measured.

## Synchronization policy

- Loader: one `mx.eval` per weight shard so transformed source tensors can be released.
- Denoising: one evaluation for the transformer result and one for the sampler
  update in benchmark mode so the two stages can be timed separately.
- Normal generation currently evaluates once at the outer step boundary to
  prevent an N-step lazy graph from retaining every activation.
- No layer-level synchronization exists in the production path.

## Confirmed facts vs pending measurement

Confirmed from source and local tests:

- MLX 0.32 fast SDPA performs softmax in FP32.
- The hot loop contains no framework conversion.
- QKV relayout exactly matches official Diffusers `to_q/to_k/to_v` tensors.
- Video VAE, audio VAE, and text layer-50 paths have independent numerical checks.

Pending measurement:

- first and steady-state production step time after memory-safe QKV loading;
- production peak unified memory;
- whether one or two outer-loop evaluations is faster for normal sampling;
- BF16 MLX versus current ComfyUI/MPS wall-clock under an identical workload.

