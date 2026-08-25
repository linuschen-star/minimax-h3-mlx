"""Native MLX text-only Qwen3-VL conditioner used by MiniMax H3."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class TextConfig:
    def __init__(self, root):
        cfg = root["text_config"] if "text_config" in root else root
        for key, value in cfg.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text()))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.empty((dim,))
        self.eps = eps

    def __call__(self, x):
        y = x.astype(mx.float32) * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)
        return y.astype(x.dtype) * self.weight


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, mask):
        b, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, length, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, length, self.kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        groups = self.heads // self.kv_heads
        k, v = mx.repeat(k, groups, axis=1), mx.repeat(v, groups, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, length, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class H3TextEncoder(nn.Module):
    """Qwen3-VL language stack truncated at H3's exact hidden_states[50]."""

    def __init__(self, cfg: TextConfig, output_layer=50):
        super().__init__()
        if output_layer > cfg.num_hidden_layers:
            raise ValueError("output layer exceeds official Qwen3-VL depth")
        self.cfg, self.output_layer = cfg, output_layer
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(output_layer)]
        self.rope_theta = cfg.rope_theta

    def __call__(self, input_ids):
        x = self.embed_tokens(input_ids)
        length = input_ids.shape[-1]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, self.cfg.head_dim, 2, dtype=mx.float32) / self.cfg.head_dim)
        )
        freqs = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None]
        emb = mx.concatenate((freqs, freqs), axis=-1)[None, None]
        cos, sin = mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)
        mask = mx.where(
            mx.triu(mx.ones((length, length), dtype=mx.bool_), k=1),
            mx.array(-1e9, dtype=x.dtype), mx.array(0.0, dtype=x.dtype),
        )
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        # Deliberately no final RMSNorm: official H3 reads hidden_states[50].
        return x


def load_text_encoder(encoder: H3TextEncoder, index_path: str | Path) -> None:
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    prefix = "model.language_model."
    wanted = {}
    for key, shard in index["weight_map"].items():
        if key == prefix + "embed_tokens.weight":
            wanted[key] = shard
            continue
        if key.startswith(prefix + "layers."):
            layer = int(key.split(".")[3])
            if layer < encoder.output_layer:
                wanted[key] = shard

    expected = set(dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(encoder.parameters())))
    mapped = {key[len(prefix):] for key in wanted}
    if mapped != expected:
        raise ValueError(f"text encoder design mismatch: missing={sorted(expected-mapped)}, extra={sorted(mapped-expected)}")

    for shard in sorted(set(wanted.values())):
        raw = mx.load(str(index_path.parent / shard))
        weights = [(key[len(prefix):], raw[key]) for key, name in wanted.items() if name == shard]
        encoder.load_weights(weights, strict=False)


def tokenize_prompt(tokenizer_json: str | Path, prompt: str) -> mx.array:
    # Official H3: verbatim prompt, no chat template, no special tokens.
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(tokenizer_json)).encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError("prompt tokenized to an empty sequence")
    return mx.array([ids], dtype=mx.int32)
