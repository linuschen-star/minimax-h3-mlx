#!/usr/bin/env python3
"""Validate official H3 Qwen3-VL hidden_states[50] on Apple Silicon."""

import argparse
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.text_encoder import H3TextEncoder, TextConfig, load_text_encoder, tokenize_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_encoder_dir", type=Path)
    parser.add_argument("--prompt", default="A red fox walks through snow while wind moves the pine trees.")
    args = parser.parse_args()

    cfg = TextConfig.from_json(args.text_encoder_dir / "config.json")
    model = H3TextEncoder(cfg, output_layer=50)
    started = time.perf_counter()
    load_text_encoder(model, args.text_encoder_dir / "model.safetensors.index.json")
    ids = tokenize_prompt(args.text_encoder_dir / "tokenizer.json", args.prompt)
    output = model(ids)
    mx.eval(output)
    assert output.shape == (1, ids.shape[1], 5120)
    assert mx.all(mx.isfinite(output)).item()
    print(
        f"PASS official MLX Qwen3-VL hidden_states[50]: {time.perf_counter()-started:.3f}s, "
        f"tokens={ids.shape[1]}, shape={output.shape}"
    )


if __name__ == "__main__":
    main()
