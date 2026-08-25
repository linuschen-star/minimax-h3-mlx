# REQUIRES_APPLE_METAL_VALIDATION

Only the following require a real Apple GPU:

1. ~~Cold-load, first-step, second-step and peak-memory figures at 832x480, 124 frames.~~ Completed: 51.79 s load, 71.07/71.06 s steps, 64.96 GiB peak.
2. ~~Confirmation that the shard-wise QKV loader removes the previous Metal command-buffer failure.~~ Completed for two production-shape evaluations with stable memory.
3. ~~Short end-to-end decoded pipeline check after QKV relayout.~~ Completed with 2 evaluations: recognizable decoded structure, 832x480/124-frame H.264, and non-silent 32 kHz stereo AAC. A 6-8-step run remains optional when visual quality—not pipeline correctness—needs judgment.
4. Identical-workload MLX versus ComfyUI/MPS wall-clock comparison.
5. Any future INT4/INT8 speed and quality acceptance decision.

Use `benchmark_h3.py` first. Do not run a normal generation until its
two production-shape steps finish below the configured catastrophic threshold
without memory growth or backend transitions.
