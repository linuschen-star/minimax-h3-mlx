# Validation Status

## T2V

Status: VERIFIED PASS

Known-good production smoke:

- Resolution: 832x480
- Frames: 124
- Steps: 20
- Seed: 20260823
- Model precision: Base BF16
- Scheduler: beta
- Sampler: res_multistep
- Quantization: none
- Turbo LoRA: none

Observed result:

- Prompt semantics correct
- Two martial artists continuously perform combat actions
- Scene and subject continuity maintained
- No fake-logo / fake-text semantic collapse
- No singing/dancing regression
- No mosaic failure
- No gray/yellow corruption

Verified artifact SHA256 values:

MLX output video:

563947ba2dad7ec5e6227661c1b28bc96d4fbd7bf4e40eb0a8bf9c06420b43ae

Final latent:

452b322a6b33662ee90686a33cd52769cfc84b396d55933ffb1f2fce294cd574

Diagnostics:

616f55cfa5c48621b17d519fcb073b4c8813138ebd351bec3e14c0b50aa4e27b

Dense MLX contact sheet:

562e82808fa379d953a0b14f01fe1789fd970ae3d7ce8c0797286cdd1706bce1

Official comparison video:

c5fc7127de04f2a93f0ab6d2bdaa6200903c1f1478c080ea6b66c704c08bce9d

The large validation artifacts are intentionally not stored in Git.

## I2V

The implementation and workflow are included.

This public snapshot does not claim independent production verification.

When importing the provided workflow, select an appropriate local input
image if the example image filename is unavailable.

## R2V

The implementation and workflow are included.

This public snapshot does not claim independent production verification.

When importing the provided workflow, select the required local reference
images if the example image filename is unavailable.

## Critical known-good behavior

Do not change these without a demonstrated production regression:

- token_refiner QKV row conversion
- main DiT QKV row conversion
- Qwen attention_mask=None causal attention
- Base BF16
- beta scheduler
- res_multistep sampler
- official ModelSamplingAV behavior
- raw prompt handling
- official Qwen preprocessing
- no quantization
- no Turbo LoRA
