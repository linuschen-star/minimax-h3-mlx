# MiniMax H3 MLX — Restore From Scratch

This document describes how to reconstruct the known-good MiniMax H3 MLX baseline after a complete machine reset.

The goal is restoration first, not optimization.

## 1. Canonical known-good version

Git repository:

    https://github.com/linuschen-star/minimax-h3-mlx.git

Immutable known-good tag:

    known-good-2026-08-25

Known-good commit:

    2a5908253a7e4ea9f324ec519f4cd843e57aee5f

For strict restoration:

    git clone https://github.com/linuschen-star/minimax-h3-mlx.git
    cd minimax-h3-mlx
    git checkout known-good-2026-08-25
    git rev-parse HEAD

Expected result:

    2a5908253a7e4ea9f324ec519f4cd843e57aee5f

Do not use an arbitrary future main commit when reproducing this baseline.

## 2. What is preserved

The repository preserves:

- MLX MiniMax H3 backend source
- ComfyUI H3 MLX integration
- T2V workflow
- I2V workflow
- R2V workflow
- architecture and implementation documentation
- model asset inventory
- known-good Python and MLX environment inventory
- software baseline commit information

Large model weights are intentionally not stored in Git.

Internal forensic/debug scripts and diagnostic binaries were intentionally excluded from the public repository.

## 3. Validation status

### T2V

VERIFIED PASS.

Known-good production smoke baseline:

    Base BF16
    832 x 480
    124 frames
    approximately 5.17 seconds
    20 denoising steps
    beta scheduler
    res_multistep
    seed 20260823
    synchronized video and audio

The known catastrophic semantic failure caused by supplying an all-True Qwen attention mask was fixed.

The known-good semantic encoder path allows Qwen / mlx-vlm to construct causal attention.

### I2V

PRESERVED — NOT CLAIMED PRODUCTION VERIFIED.

### R2V

PRESERVED — NOT CLAIMED PRODUCTION VERIFIED.

Do not silently substitute unsupported conditioning behavior.

## 4. Known software baseline

Known-good Python:

    Python 3.14.5

Critical MLX packages:

    mlx==0.32.0
    mlx-vlm==0.6.15

Known ComfyUI baseline:

    version 0.33.0
    commit 0696f61dced6340086cdca64a96200c50f306c66

Known MiniMax-H3 official source baseline:

    commit d21241f0a4b3acbb34c97dae47fa417b7065e438

Complete Python package inventory:

    docs/ENVIRONMENT_LOCK_2026-08-25.txt

Do not blindly upgrade MLX, mlx-vlm, ComfyUI, or related dependencies before reproducing the known-good baseline.

## 5. Reconstruct the Python environment

Create a Python 3.14.5 virtual environment.

Example:

    python3.14 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

Backend project:

    backend/

ComfyUI integration requirements:

    ComfyUI-H3-MLX/requirements.txt

Initial reconstruction:

    python -m pip install -e backend
    python -m pip install -r ComfyUI-H3-MLX/requirements.txt

Verify critical versions:

    python -c 'import platform, importlib.metadata as m; print(platform.python_version()); print(m.version("mlx")); print(m.version("mlx-vlm"))'

Expected:

    3.14.5
    0.32.0
    0.6.15

If dependency resolution differs materially, compare against:

    docs/ENVIRONMENT_LOCK_2026-08-25.txt

before debugging model behavior.

## 6. Restore the model bundle

Model weights are external to Git.

Known runtime bundle roots:

    FL2VA/
    Ref2VA/
    vae/

Major runtime components:

    FL2VA/transformer
    FL2VA/text_encoder
    FL2VA/audio_vae
    Ref2VA/transformer
    vae

Exact known filenames and byte sizes:

    docs/MODEL_ASSET_INVENTORY_2026-08-25.txt

Clean inventory contains:

    63 runtime assets
    1 extra asset not asserted required for restoration

The extra file is:

    transformer/diffusion_pytorch_model-00001-of-00014.safetensors

That file was present in the working bundle but is not asserted to be required for restoration.

Hugging Face .cache files are not model assets.

Restore or re-download the model bundle and configure the MLX / ComfyUI integration to use it.

Machine-specific absolute paths are intentionally not part of the public restore procedure.

## 7. Install the recorded ComfyUI baseline

For strict reproduction, begin with:

    0696f61dced6340086cdca64a96200c50f306c66

After cloning/installing ComfyUI:

    cd <COMFYUI_ROOT>
    git checkout 0696f61dced6340086cdca64a96200c50f306c66

Do not begin restoration against an arbitrary newer ComfyUI revision.

## 8. Install ComfyUI-H3-MLX

Preserved integration:

    ComfyUI-H3-MLX/

Install/copy it to:

    <COMFYUI_ROOT>/custom_nodes/ComfyUI-H3-MLX

The backend package must be importable by the Python interpreter used to launch ComfyUI.

Do not rewrite production integration during initial restoration.

## 9. Preserved workflows

T2V:

    ComfyUI-H3-MLX/workflows/h3_mlx_t2v.json

I2V:

    ComfyUI-H3-MLX/workflows/h3_mlx_i2v.json

R2V:

    ComfyUI-H3-MLX/workflows/h3_mlx_r2v.json

Restore and test T2V first.

I2V/R2V workflow examples may refer to:

    DragonballZ.png

The example image itself is not included.

Use an appropriate local reference image when testing those workflows.

## 10. First smoke test after restoration

Do not optimize first.

Reproduce T2V first using:

    832 x 480
    124 frames
    20 steps
    Base BF16
    beta scheduler
    res_multistep
    seed 20260823

Successful restoration means semantically coherent video, coherent temporal behavior, correct conditioning, and synchronized audio.

The earlier catastrophic pre-fix failure included:

- gray/yellow output
- fake logo/text-like artifacts
- incorrect semantic content
- output unrelated to the intended martial-arts scene

The post-fix known-good smoke produced coherent outdoor martial-arts action.

Exact pixel identity with official ComfyUI is not required.

## 11. Critical implementation facts

These are deliberate known-good compatibility behaviors.

Do not casually "clean them up."

### Main DiT QKV

Raw Hugging Face conceptual layout:

    [head, qkv, head_dim, in]

Production layout:

    [qkv, head, head_dim, in]

Known 56-head / 128-head-dim conversion:

    raw.reshape(56, 3, 128, 5376).transpose(1, 0, 2, 3).reshape(21504, 5376)

The proven QKV mapping is frozen unless an actual production workflow demonstrates failure.

### Token-refiner QKV

Token-refiner QKV also requires the proven row conversion.

Do not remove it during restoration.

### Qwen causal attention

Do not supply an all-True semantic attention mask.

The known-good path uses:

    attention_mask=None

This allows Qwen / mlx-vlm to construct causal attention.

The previous all-True wrapper mask caused catastrophic semantic failure.

### Sampling

Known-good production behavior includes:

    official beta scheduler
    res_multistep
    ModelSamplingAV-compatible timestep/carry behavior
    audio scale 12 / 3 = 4

### Qwen image preprocessing

Known preprocessing:

    mean = 0.5
    std = 0.5
    bilinear interpolation
    align_corners = False

### Velocity sign

The known-good implementation matches official negative video/audio velocity output behavior where required.

## 12. Do not do these before T2V passes

Do not:

- introduce SageAttention
- quantize the model
- add Turbo LoRA
- rewrite prompts
- change QKV mappings
- replace Qwen causal-mask behavior
- optimize SDPA
- rewrite production conditioning
- chase tiny numerical parity differences
- silently substitute unsupported I2V/R2V behavior
- upgrade dependencies simply because newer versions exist

Restore first.

Optimize later.

## 13. Recommended recovery order

    1. Install macOS development prerequisites
    2. Install Python 3.14.5
    3. Clone this Git repository
    4. Checkout known-good-2026-08-25
    5. Reconstruct Python / MLX environment
    6. Restore or download MiniMax H3 model bundle
    7. Verify filenames against MODEL_ASSET_INVENTORY_2026-08-25.txt
    8. Install recorded ComfyUI baseline
    9. Install ComfyUI-H3-MLX
    10. Make backend importable to ComfyUI Python
    11. Load h3_mlx_t2v.json
    12. Run known-good T2V smoke test
    13. Confirm semantic and temporal correctness
    14. Only then consider newer dependencies, I2V/R2V verification, or optimization

## 14. Canonical preservation references

Known-good tag:

    known-good-2026-08-25

Known-good commit:

    2a5908253a7e4ea9f324ec519f4cd843e57aee5f

Model inventory:

    docs/MODEL_ASSET_INVENTORY_2026-08-25.txt

Environment lock:

    docs/ENVIRONMENT_LOCK_2026-08-25.txt

Workflow validation statement:

    ComfyUI-H3-MLX/VALIDATION.md

## 15. Final principle

The 2026-08-25 baseline is a preservation target, not an invitation to refactor.

If a future restored machine reproduces the verified T2V production behavior, restoration is successful.

Only after that baseline works should changes be introduced one at a time.
