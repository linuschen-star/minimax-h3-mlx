# Restoration Guide

Goal:

Reconstruct the known-good MiniMax H3 MLX development environment after a machine reset.

## 1. Clone repository

Clone this repository to the desired development directory.

## 2. Create Python environment

The known-good development environment used:

- Python 3.14.5
- mlx 0.32.0
- mlx-vlm 0.6.15

The backend pyproject.toml defines the preserved Python dependencies.

Do not blindly upgrade MLX or mlx-vlm during baseline restoration.

## 3. Install backend

Install the package under:

backend/

using an isolated Python environment.

## 4. Install ComfyUI integration

The custom-node integration is under:

ComfyUI-H3-MLX/

The exact installation location must match the target ComfyUI installation custom-node layout.

Do not overwrite an existing working installation without first making a backup.

## 5. Restore model files

Model weights are intentionally not stored in Git.

The exact model/checkpoint manifest still needs to be preserved separately before machine reset.

Do not substitute checkpoints based only on similar filenames.

## 6. Preserve production behavior

Known-good behavior includes:

- token_refiner QKV conversion
- main DiT QKV conversion
- Qwen attention_mask=None
- Base BF16
- beta scheduler
- res_multistep
- official ModelSamplingAV behavior
- raw prompt handling
- official image preprocessing
- no quantization
- no Turbo LoRA

## 7. Validation

The only production workflow currently recorded as VERIFIED is T2V.

Known-good validation case:

- 832x480
- 124 frames
- 20 steps
- seed 20260823
- Base BF16
- beta
- res_multistep

Compare against the separately preserved evidence bundle.

I2V and R2V should not be declared verified until independent successful production validation exists.
