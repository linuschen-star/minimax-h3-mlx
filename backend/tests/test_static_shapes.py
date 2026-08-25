import ast
from pathlib import Path

import mlx.core as mx
import pytest

import minimax_h3_mlx.multimodal_encoder as multimodal_encoder
from minimax_h3_mlx.model import Attention, FL2VLayout, H3Config, MiniMaxH3Transformer, Ref2VImageLayout, T2VLayout, reorder_per_head_qkv
from minimax_h3_mlx.profiling import RegionProfiler
from minimax_h3_mlx.sampling import flow_add_noise, resume_schedule, sigma_schedule
from benchmark_projection_gemms import PRODUCTION_BF16_SHAPES
from minimax_h3_mlx.runtime import ExactH3Runtime, H3GenerationSpec, H3LatentState, resolve_ffmpeg
from minimax_h3_mlx.video_vae import H3VideoEncoder, H3VideoVAE


def test_model_source_parses():
    source = Path(__file__).parents[1] / "minimax_h3_mlx" / "model.py"
    ast.parse(source.read_text())


def test_no_torch_hot_path():
    root = Path(__file__).parents[1]
    source = "\n".join(p.read_text() for p in (root / "minimax_h3_mlx").glob("*.py"))
    assert "import torch" not in source
    assert "quantized_" not in source.lower()


def test_qwen_conditioning_uses_native_causal_mask(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"text_config": {}}')
    (tmp_path / "tokenizer.json").touch()
    (tmp_path / "model.safetensors.index.json").touch()
    captured = {}

    class FakeConfig:
        @staticmethod
        def from_dict(value):
            return value

    class FakeModel:
        def __init__(self, _config): pass

        def hidden_state_at_layer(self, input_ids, output_layer, **kwargs):
            captured.update(kwargs)
            return mx.zeros((1, input_ids.shape[1], 8), dtype=mx.bfloat16)

    class Encoded:
        def __init__(self, ids): self.ids = ids

    class FakeTokenizer:
        @staticmethod
        def from_file(_path): return FakeTokenizer()

        def token_to_id(self, token):
            return {"<|image_pad|>": 3, "<|vision_start|>": 1,
                    "<|vision_end|>": 2}[token]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return Encoded([4])

    monkeypatch.setattr(multimodal_encoder, "ModelConfig", FakeConfig)
    monkeypatch.setattr(multimodal_encoder, "Model", FakeModel)
    monkeypatch.setattr(multimodal_encoder, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(multimodal_encoder, "load_h3_qwen3vl", lambda *_a, **_k: None)
    monkeypatch.setattr(
        multimodal_encoder,
        "preprocess_qwen3vl_images",
        lambda _images: (mx.zeros((4, 8)), mx.array([[1, 2, 2]], dtype=mx.int32)),
    )

    context, tags = multimodal_encoder.encode_keyframe_prompt(
        tmp_path, [object()], "prompt", output_layer=50,
    )
    assert context.shape == (1, 5, 8)
    assert len(tags) == 5
    assert "mask" in captured and captured["mask"] is None


def test_480p_geometry_is_exact():
    assert 832 % 32 == 0 and 480 % 32 == 0
    assert (37 - 2) // 5 * 17 + 5 == 124
    assert round(124 / 24 * 40) == 207


def test_production_layout_token_count():
    layout = T2VLayout(256, 37, 30, 52, 207)
    assert layout.seq_len == 256 + 2 * 207 + 37 * 15 * 26


def test_first_and_last_keyframes_have_official_endpoint_positions():
    layout = FL2VLayout(4, 7, 4, 6, 5, ("first", "last"))
    rows_per_frame = 2 * 3
    assert layout.condition == (4, 4 + 2 * rows_per_frame, "cond")
    first_times = layout.position_ids[4:4 + rows_per_frame, 0].tolist()
    last_times = layout.position_ids[4 + rows_per_frame:4 + 2 * rows_per_frame, 0].tolist()
    assert first_times == [4.0] * rows_per_frame
    expected_last = 4.0 + sum([1.0, 4.0, 4.0, 4.0, 4.0, 1.0, 4.0]) * 5 / 3 - 5 / 3
    assert last_times == [pytest.approx(expected_last)] * rows_per_frame


def test_twenty_evaluations_have_twenty_one_grid_points():
    sigmas = sigma_schedule(21, 12.0)
    assert len(sigmas) == 21
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0


def test_resume_step_count_is_independent_from_strength():
    grid, index, evaluations = resume_schedule(8, 0.5, 12.0)
    assert index == 0 and evaluations == 8 and len(grid) == 9
    assert grid[0] == pytest.approx(0.5)
    assert grid[-1] == 0.0
    assert all(a > b for a, b in zip(grid, grid[1:]))


def test_zero_strength_is_an_exact_noop_schedule():
    grid, index, evaluations = resume_schedule(8, 0.0, 12.0)
    assert grid == [0.0] and index == 8 and evaluations == 0


def test_flow_noise_matches_official_const_interpolation():
    clean = mx.full((1, 2), 2.0, dtype=mx.bfloat16)
    noise = mx.full((1, 2), -2.0, dtype=mx.bfloat16)
    assert flow_add_noise(clean, noise, 0.0).tolist() == clean.tolist()
    assert flow_add_noise(clean, noise, 1.0).tolist() == noise.tolist()
    assert flow_add_noise(clean, noise, 0.25).tolist() == [[1.0, 1.0]]


def test_spatial_resize_preserves_bcthw_time_audio_and_context():
    spec = H3GenerationSpec("prompt", 832, 480, 124, 20, 7)
    video = mx.zeros((1, 24, 37, 30, 52), dtype=mx.bfloat16)
    audio = mx.zeros((1, 32, 2, 207), dtype=mx.bfloat16)
    context = mx.zeros((1, 3, 8), dtype=mx.bfloat16)
    state = H3LatentState(video, audio, context, (1, 1, 1), spec)
    resized = ExactH3Runtime.resize_latent(state, 1.5)
    assert resized.video.shape == (1, 24, 37, 46, 78)
    assert resized.audio is audio and resized.context is context
    assert resized.spec.frames == 124
    assert (resized.spec.width, resized.spec.height) == (1248, 736)


def test_explicit_1080p_resize_does_not_lose_a_column_to_float_rounding():
    spec = H3GenerationSpec("prompt", 832, 480, 124, 20, 7)
    state = H3LatentState(
        mx.zeros((1, 24, 37, 30, 52), dtype=mx.bfloat16),
        mx.zeros((1, 32, 2, 207), dtype=mx.bfloat16),
        mx.zeros((1, 3, 8), dtype=mx.bfloat16), (1, 1, 1), spec,
    )
    resized = ExactH3Runtime.resize_latent(
        state, target_width=1920, target_height=1088,
    )
    assert resized.video.shape == (1, 24, 37, 68, 120)
    assert (resized.spec.width, resized.spec.height) == (1920, 1088)


def test_qkv_relayout_is_global_q_then_k_then_v():
    # two heads, one row per q/k/v, one input column
    source = mx.array([[0], [1], [2], [3], [4], [5]])
    actual = reorder_per_head_qkv(source, heads=2, head_dim=1)
    assert actual.tolist() == [[0], [3], [1], [4], [2], [5]]


def test_deep_profiler_preserves_tiny_forward():
    cfg = H3Config(hidden_size=64, num_layers=2, num_refiner_layers=1,
                   num_attention_heads=4, attention_head_dim=16, ffn_dim=128,
                   text_dim=48, freq_dim=32, time_embed_hidden_dim=64,
                   time_embed_dim=32, rope_freq_dim=2)
    model = MiniMaxH3Transformer(cfg)
    video = mx.zeros((1, 24, 1, 2, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, 32, 2, 1), dtype=mx.bfloat16)
    context = mx.zeros((1, 4, 48), dtype=mx.bfloat16)
    expected = model(video, audio, context, 1.0)
    profiler = RegionProfiler()
    actual = model(video, audio, context, 1.0, _profiler=profiler)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(actual[0], expected[0]).item())
    assert bool(mx.array_equal(actual[1], expected[1]).item())
    assert profiler.calls["attention_sdpa"] == 2
    assert profiler.calls["mlp_input_projection"] == 2


def test_tiny_first_and_last_frame_forward():
    cfg = H3Config(hidden_size=64, num_layers=1, num_refiner_layers=1,
                   num_attention_heads=4, attention_head_dim=16, ffn_dim=128,
                   text_dim=48, freq_dim=32, time_embed_hidden_dim=64,
                   time_embed_dim=32, rope_freq_dim=2)
    model = MiniMaxH3Transformer(cfg)
    video = mx.zeros((1, 24, 2, 2, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, 32, 2, 2), dtype=mx.bfloat16)
    context = mx.zeros((1, 4, 48), dtype=mx.bfloat16)
    conditions = mx.zeros((1, 24, 2, 2, 2), dtype=mx.bfloat16)
    layout = FL2VLayout(4, 2, 2, 2, 2, ("first", "last"))
    output = model(video, audio, context, 1.0, layout=layout,
                   condition_video=conditions, condition_noise=conditions)
    mx.eval(output)
    assert output[0].shape == video.shape and output[1].shape == audio.shape


def test_tiny_image_reference_forward_keeps_reference_grid_independent():
    cfg = H3Config(hidden_size=64, num_layers=1, num_refiner_layers=1,
                   num_attention_heads=4, attention_head_dim=16, ffn_dim=128,
                   text_dim=48, freq_dim=32, time_embed_hidden_dim=64,
                   time_embed_dim=32, rope_freq_dim=2)
    model = MiniMaxH3Transformer(cfg)
    video = mx.zeros((1, 24, 2, 2, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, 32, 2, 2), dtype=mx.bfloat16)
    context = mx.zeros((1, 4, 48), dtype=mx.bfloat16)
    reference = mx.zeros((1, 24, 1, 4, 6), dtype=mx.bfloat16)
    layout = Ref2VImageLayout(4, 2, 2, 2, 2, ((1, 4, 6),))
    output = model(video, audio, context, 1.0, layout=layout,
                   condition_video=(reference,), condition_noise=(reference,))
    mx.eval(output)
    assert layout.condition == (4, 10, "cond")
    assert output[0].shape == video.shape and output[1].shape == audio.shape


def test_projection_manifest_covers_profiled_dit_gemms():
    cases = {case.name: case for case in PRODUCTION_BF16_SHAPES}
    assert (cases["dit_qkv"].m, cases["dit_qkv"].k, cases["dit_qkv"].n) == (15100, 5376, 21504)
    assert (cases["dit_attention_out"].m, cases["dit_attention_out"].k, cases["dit_attention_out"].n) == (15100, 7168, 5376)
    assert (cases["dit_mlp_in"].m, cases["dit_mlp_in"].k, cases["dit_mlp_in"].n) == (15100, 5376, 28672)
    assert (cases["dit_mlp_out"].m, cases["dit_mlp_out"].k, cases["dit_mlp_out"].n) == (15100, 14336, 5376)
    assert all(cases[name].calls_per_step == 50 for name in
               ("dit_qkv", "dit_attention_out", "dit_mlp_in", "dit_mlp_out"))
    measured = sum(cases[name].measured_accumulated_seconds for name in
                   ("dit_qkv", "dit_attention_out", "dit_mlp_in", "dit_mlp_out"))
    assert abs(measured - 41.3991305453) < 1e-9


def test_head_major_sdpa_pack_is_exact_bf16():
    mx.random.seed(19)
    attention = Attention(hidden=128, heads=1, head_dim=128, eps=1e-5,
                          head_major_sdpa=False)
    x = mx.random.normal((1024, 128)).astype(mx.bfloat16)
    baseline = attention(x)
    attention.head_major_sdpa = True
    candidate = attention(x)
    mx.eval(baseline, candidate)
    assert bool(mx.array_equal(baseline, candidate).item())


def test_runtime_production_geometry_and_early_validation():
    assert H3GenerationSpec("prompt").validate() == (37, 207)
    with pytest.raises(ValueError, match="17.n . 5"):
        H3GenerationSpec("prompt", frames=125).validate()


def test_native_i2v_encoder_topology_and_kernel_layout():
    encoder = H3VideoEncoder()
    assert len(encoder.down_blocks) == 6
    assert [len(block.resnets) for block in encoder.down_blocks] == [2] * 6
    assert [len(block.downsamplers) for block in encoder.down_blocks] == [1, 1, 1, 1, 0, 0]
    assert encoder.conv_in.conv.weight.shape == (128, 3, 3, 3, 3)
    assert encoder.conv_out.conv.weight.shape == (48, 3, 3, 3, 1024)


def test_video_vae_temporal_chunk_uses_official_17_plus_5_split():
    decoded = mx.zeros((1, 3, 28, 1, 1))
    body, overlap = H3VideoVAE._split_temporal_chunk(decoded)
    assert body.shape[2] == 17
    assert overlap.shape[2] == 5
    # Seven bodies plus the final overlap produce exactly 124 frames.
    assert 7 * body.shape[2] + overlap.shape[2] == 124


def test_ffmpeg_resolves_homebrew_when_gui_path_is_minimal(monkeypatch):
    monkeypatch.delenv("H3_MLX_FFMPEG", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    path = resolve_ffmpeg()
    assert path.is_absolute()
    assert path.name == "ffmpeg"
