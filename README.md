# MiniMax H3 MLX

Native Apple-Silicon / MLX implementation of the full-quality MiniMax H3 inference path, together with a ComfyUI integration layer.

This repository is reconstructed from the known-good development state preserved on 2026-08-25.

## Repository layout

backend/
    Native MiniMax H3 MLX backend

ComfyUI-H3-MLX/
    ComfyUI custom-node integration
    workflow builder
    T2V / I2V / R2V workflow JSON files

docs/
    known-good baseline and restoration metadata

## Current validation status

### T2V

VERIFIED PASS.

Known-good production smoke:

- resolution: 832x480
- frames: 124
- steps: 20
- model: MiniMax H3 Base BF16
- scheduler: beta
- sampler: res_multistep
- seed: 20260823
- quantization: none
- Turbo LoRA: none

The validated output correctly produced the requested outdoor martial-arts combat scene without the previous catastrophic semantic failure.

### I2V

Workflow and implementation are preserved.

NOT claimed production-verified in this repository snapshot.

### R2V

Workflow and implementation are preserved.

NOT claimed production-verified in this repository snapshot.

Presence of a workflow does not imply verified production correctness.

## Critical known-good behavior

The following behavior is part of the known-good baseline and should not be changed without a demonstrated production regression:

- token_refiner QKV row conversion
- main DiT QKV row conversion
- Qwen semantic path uses attention_mask=None
- Base BF16 transformer path
- beta scheduler
- res_multistep
- official ModelSamplingAV behavior
- raw prompt handling
- official Qwen preprocessing behavior
- no quantization
- no Turbo LoRA
- unsupported functionality must fail explicitly rather than silently substitute another behavior

## Dependencies

The preserved backend currently specifies:

- Python >= 3.10
- mlx == 0.32.0
- mlx-vlm == 0.6.15
- numpy >= 2
- scipy >= 1.14
- tokenizers >= 0.22

The known-good development machine used Python 3.14.5.

Exact restoration metadata is documented under docs/.

## Important

Model weights are not included in this repository.

Large diagnostic tensors, videos, virtual environments and checkpoints are intentionally excluded from Git.

See docs/KNOWN_GOOD.md and docs/RESTORE.md.

## Workflow input images

The included I2V and R2V workflows may contain an example relative input
filename such as `DragonballZ.png`.

The image itself is not part of the repository.

After importing those workflows into ComfyUI, choose your own local input
image/reference images before running them.
