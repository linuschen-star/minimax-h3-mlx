"""Native MLX Qwen3-VL layer-50 conditioning for H3 keyframes."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_vl.config import ModelConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
from tokenizers import Tokenizer


TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."


def preprocess_qwen3vl_images(
    images: Sequence,
    *,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple[mx.array, mx.array]:
    """Port ComfyUI Qwen3VL preprocessing with explicit mean/std 0.5."""
    all_patches = []
    grids = []
    factor = patch_size * merge_size
    for image in images:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width, channels = pixels.shape
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            resize_factor = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / resize_factor / factor) * factor)
            w_bar = max(factor, math.floor(width / resize_factor / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            resize_factor = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * resize_factor / factor) * factor
            w_bar = math.ceil(width * resize_factor / factor) * factor
        source = mx.array(pixels)[None]
        resized = nn.Upsample(
            scale_factor=(h_bar / height, w_bar / width),
            mode="linear",
            align_corners=False,
        )(source)[0]
        if resized.shape[:2] != (h_bar, w_bar):
            raise RuntimeError(
                f"MLX Qwen3-VL resize produced {resized.shape[:2]}, expected {(h_bar, w_bar)}"
            )
        normalized = (resized - 0.5) / 0.5
        grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
        repeated = mx.repeat(normalized[None], temporal_patch_size, axis=0)
        patches = repeated.reshape(
            1, temporal_patch_size, grid_h // merge_size, merge_size, patch_size,
            grid_w // merge_size, merge_size, patch_size, channels,
        )
        patches = patches.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7)
        all_patches.append(patches.reshape(grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size))
        grids.append((1, grid_h, grid_w))
    if not all_patches:
        raise ValueError("Qwen3-VL image preprocessing requires at least one image")
    return mx.concatenate(all_patches), mx.array(grids, dtype=mx.int32)


def _wanted_key(key: str, output_layer: int) -> bool:
    if key.startswith(VISION_PREFIX):
        return True
    if key == TEXT_PREFIX + "embed_tokens.weight":
        return True
    if key.startswith(TEXT_PREFIX + "layers."):
        return int(key.split(".")[3]) < output_layer
    return False


def load_h3_qwen3vl(model: Model, index_path: str | Path, output_layer: int = 50) -> None:
    index_path = Path(index_path)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    wanted = {key: shard for key, shard in weight_map.items() if _wanted_key(key, output_layer)}
    for shard in sorted(set(wanted.values())):
        raw = mx.load(str(index_path.parent / shard))
        selected = {key: raw[key] for key, name in wanted.items() if name == shard}
        selected = model.sanitize(selected)
        selected = model.vision_tower.sanitize(selected)
        model.load_weights(list(selected.items()), strict=False)
        del raw, selected
        gc.collect()


def encode_keyframe_prompt(
    text_encoder_dir: str | Path,
    images: Sequence,
    prompt: str,
    output_layer: int = 50,
) -> tuple[mx.array, tuple[int, ...]]:
    """Encode the official ``<Picture i>`` presentation without Torch tensors."""
    text_encoder_dir = Path(text_encoder_dir)
    config_data = json.loads((text_encoder_dir / "config.json").read_text())
    config_data["text_config"]["num_hidden_layers"] = output_layer
    model = Model(ModelConfig.from_dict(config_data))
    load_h3_qwen3vl(model, text_encoder_dir / "model.safetensors.index.json", output_layer)

    pixels, grids = preprocess_qwen3vl_images(images)
    tokenizer = Tokenizer.from_file(str(text_encoder_dir / "tokenizer.json"))
    image_pad = tokenizer.token_to_id("<|image_pad|>")
    vision_start = tokenizer.token_to_id("<|vision_start|>")
    vision_end = tokenizer.token_to_id("<|vision_end|>")
    merge_area = 4
    ids: list[int] = []
    tags: list[int] = []
    for index, grid in enumerate(grids):
        label = tokenizer.encode(f"<Picture {index + 1}>: ", add_special_tokens=False).ids
        count = int(grid.prod()) // merge_area
        vision_ids = [vision_start] + [image_pad] * count + [vision_end]
        ids.extend(label + vision_ids)
        tags.extend([1] * len(label) + [0] * len(vision_ids))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    ids.extend(prompt_ids)
    tags.extend([1] * len(prompt_ids))

    input_ids = mx.array([ids], dtype=mx.int32)
    image_grid = grids
    hidden = model.hidden_state_at_layer(
        input_ids, output_layer, pixel_values=pixels, mask=None,
        image_grid_thw=image_grid,
    ).astype(mx.bfloat16)
    mx.eval(hidden)
    del model, pixels
    gc.collect()
    mx.clear_cache()
    return hidden, tuple(tags)
