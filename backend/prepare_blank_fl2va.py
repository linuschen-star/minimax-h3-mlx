#!/usr/bin/env python3
"""Prepare the fixed blank-keyframe conditioning required by legacy FL2VA.

This is a one-shot reference preprocessing utility.  The denoising and decode
hot path remains native MLX; the official PyTorch Qwen vision tower and VAE
encoder are used here because their outputs are fixed for the blank anchor.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)


PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


def load_partial_qwen(model_dir: Path, processor_dir: Path):
    config = Qwen3VLConfig.from_pretrained(processor_dir, local_files_only=True)
    # hidden_states[50] needs outputs of layers 0..49.  Keep layer 50 so that
    # Transformers exposes that intermediate state before its final norm.
    config.text_config.num_hidden_layers = 51
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted_shards = {
        shard for key, shard in index.items()
        if key.startswith("model.visual.")
        or key == "model.language_model.embed_tokens.weight"
        or key == "model.language_model.norm.weight"
        or (key.startswith("model.language_model.layers.") and int(key.split(".")[3]) <= 50)
    }
    for shard in sorted(wanted_shards):
        state = load_file(model_dir / shard, device="cpu")
        state = {key: value for key, value in state.items() if key in index and (
            key.startswith("model.visual.")
            or key == "model.language_model.embed_tokens.weight"
            or key == "model.language_model.norm.weight"
            or (key.startswith("model.language_model.layers.") and int(key.split(".")[3]) <= 50)
        )}
        model.load_state_dict(state, strict=False, assign=True)
        del state
    missing_meta = [name for name, value in model.model.named_parameters() if value.is_meta]
    if missing_meta:
        raise RuntimeError(f"partial Qwen load left required meta parameters: {missing_meta[:8]}")
    # Non-persistent RoPE buffers are derived from config and therefore absent
    # from safetensors.  A meta construction leaves them unmaterialized.
    vision_rope = model.model.visual.rotary_pos_emb
    fresh_vision_rope = Qwen3VLVisionRotaryEmbedding(vision_rope.dim, vision_rope.theta)
    vision_rope._buffers["inv_freq"] = fresh_vision_rope.inv_freq
    fresh_text_rope = Qwen3VLTextRotaryEmbedding(config.text_config)
    model.model.language_model.rotary_emb._buffers["inv_freq"] = fresh_text_rope.inv_freq
    model.model.language_model.rotary_emb._buffers["original_inv_freq"] = fresh_text_rope.original_inv_freq
    return model


def encode_prompt(model, processor, tokenizer, image, prompt, device):
    vision = processor.image_processor(images=[image], return_tensors="pt")
    grid = vision["image_grid_thw"]
    num_image_tokens = int(grid[0].prod()) // processor.image_processor.merge_size**2
    label = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    vision_ids = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = label + vision_ids + prompt_ids
    tags = [1] * len(label) + [0] * len(vision_ids) + [1] * len(prompt_ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    # Transformers 4.x exposes this through processor(...,
    # return_mm_token_type_ids=True); constructing an already-tokenized H3
    # presentation means the equivalent rule is image-pad rows => 1.
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    mm_types = torch.tensor([[1 if token == image_pad_id else 0 for token in ids]], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_types,
            pixel_values=vision["pixel_values"].to(device, model.dtype),
            image_grid_thw=grid.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
    return outputs.hidden_states[50].float().cpu().numpy(), np.asarray(tags, dtype=np.int32)


def load_vae(source: Path, model_dir: Path):
    name = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_local"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = json.loads((model_dir / "config.json").read_text())
    vae = module.AutoencoderKLMiniMaxH3.from_config(config)
    index = json.loads((model_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted(set(index.values())):
        state.update(load_file(model_dir / shard, device="cpu"))
    vae.load_state_dict(state, strict=True)
    return vae, config


def encode_keyframe(vae, config, image, device):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)[None, :, None].to(device)
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    with torch.inference_mode():
        posterior = vae.encode((pixels.float() / 255 - mean) / std, return_dict=False)[0]
        latent = posterior.sample(generator=torch.Generator().manual_seed(42)).half().float().cpu()
    lmean = torch.tensor(config["latents_mean"]).view(1, -1, 1, 1, 1)
    lstd = torch.tensor(config["latents_std"]).view(1, -1, 1, 1, 1)
    return ((latent - lmean) / lstd).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-encoder-dir", type=Path, required=True)
    p.add_argument("--processor-dir", type=Path, required=True)
    p.add_argument("--vae-dir", type=Path, required=True)
    p.add_argument("--vae-source", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("mps")
    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.processor_dir, local_files_only=True)
    qwen = load_partial_qwen(args.text_encoder_dir, args.processor_dir)
    qwen.model.to(device).eval()
    context, tags = encode_prompt(qwen, processor, tokenizer, image, args.prompt, device)
    print(f"encoded vision-aware presentation: {context.shape}", flush=True)
    del qwen
    gc.collect()
    torch.mps.empty_cache()
    vae, config = load_vae(args.vae_source, args.vae_dir)
    vae.enable_tiling()
    vae = vae.to(device).eval()
    condition = encode_keyframe(vae, config, image, device)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(condition.shape, dtype=np.float32)
    save_file({
        "context": context,
        "text_token_tags": tags,
        "condition_video": condition.astype(np.float32),
        "condition_noise": noise,
    }, str(args.output))
    print(f"saved {args.output}; condition={condition.shape}")


if __name__ == "__main__":
    main()
