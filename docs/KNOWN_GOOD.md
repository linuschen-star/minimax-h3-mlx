# Known-Good Baseline

Preservation date: 2026-08-25

## Trusted source archives

### Core backend

Archive:

minimax_h3_mlx-2026-08-25-clean-source.tar.gz

SHA256:

80906f592b30f4552f84972a2167dde4f2dad3332e8460381a6df3f295f05304

### ComfyUI integration

Archive:

ComfyUI-H3-MLX-2026-08-25-clean-integration.tar.gz

SHA256:

d1410750bac3fa911db048d8f73098cc43d7d7487b4090941155a92b091a5ff1

### Verified T2V evidence

Archive:

h3-mlx-2026-08-25-verified-evidence.tar.gz

SHA256:

8f52c2951c54261be99dd0589a37d51c74c2efebc87dc11b20ae1a909deb35e3

The evidence archive is intentionally NOT committed to this Git repository.

## Verified T2V production smoke

Status: PASS

- 832x480
- 124 frames
- 20 steps
- seed 20260823
- Base BF16
- beta scheduler
- res_multistep
- no quantization
- no Turbo LoRA

## Verified evidence hashes

MLX output video:

563947ba2dad7ec5e6227661c1b28bc96d4fbd7bf4e40eb0a8bf9c06420b43ae

Final latent:

452b322a6b33662ee90686a33cd52769cfc84b396d55933ffb1f2fce294cd574

Diagnostics:

616f55cfa5c48621b17d519fcb073b4c8813138ebd351bec3e14c0b50aa4e27b

MLX dense contact sheet:

562e82808fa379d953a0b14f01fe1789fd970ae3d7ce8c0797286cdd1706bce1

Official reference video:

c5fc7127de04f2a93f0ab6d2bdaa6200903c1f1478c080ea6b66c704c08bce9d

## Critical fixes

### QKV layout conversion

Raw HF QKV organization differs from the production ComfyUI runtime organization.

Verified conversion is required for:

- token_refiner QKV
- main DiT QKV

Do not remove the verified conversion based on raw-HF-only parity tests.

### Qwen causal attention

The H3 MLX multimodal wrapper must pass:

attention_mask=None

This allows the Qwen implementation to construct causal attention.

The previous all-True mask incorrectly produced full/bidirectional attention and caused catastrophic prompt-semantic failure.

After the causal-mask fix, the full production smoke recovered correct prompt semantics.

## Remaining numerical parity differences

Some MLX versus PyTorch / ComfyUI intermediate tensors are not bit-exact.

Those remaining differences did not prevent the verified production T2V generation from succeeding.

Do not resume numerical-parity investigation solely to reduce small intermediate differences unless a real production workflow is broken.
