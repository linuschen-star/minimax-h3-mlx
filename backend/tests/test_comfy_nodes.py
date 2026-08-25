from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest


PACKAGE = Path(__file__).parents[2] / "ComfyUI-H3-MLX"


@pytest.fixture
def loaded_nodes(tmp_path, monkeypatch):
    models = tmp_path / "models"
    output = tmp_path / "output"
    temp = tmp_path / "temp"
    models.mkdir(); output.mkdir(); temp.mkdir()
    registered = {}
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models)
    folder_paths.add_model_folder_path = lambda name, path, is_default=False: registered.setdefault(name, [path])
    folder_paths.get_folder_paths = lambda name: registered[name]
    folder_paths.get_temp_directory = lambda: str(temp)
    folder_paths.get_output_directory = lambda: str(output)

    def save_path(prefix, output_dir, *_):
        subfolder = str(Path(prefix).parent) if Path(prefix).parent != Path(".") else ""
        directory = Path(output_dir) / subfolder
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory), Path(prefix).name, 1, subfolder, prefix
    folder_paths.get_save_image_path = save_path

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.throw_exception_if_processing_interrupted = lambda: None
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total): self.total, self.current = total, 0
        def update_absolute(self, value, total=None): self.current = value

    utils.ProgressBar = ProgressBar
    comfy.model_management = model_management

    comfy_api = types.ModuleType("comfy_api")
    comfy_api.__path__ = []
    comfy_api_latest = types.ModuleType("comfy_api.latest")
    class VideoFromFile:
        def __init__(self, file): self.file = file
        def get_stream_source(self): return self.file
    comfy_api_latest.VideoFromFile = VideoFromFile

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest)
    package_name = "comfyui_h3_mlx_test"
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name, PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{package_name}.nodes"], models, output, temp


def make_bundle(root: Path):
    for directory in ("FL2VA/transformer", "FL2VA/text_encoder", "FL2VA/audio_vae", "vae"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for filename in (
        "FL2VA/transformer/config.json", "FL2VA/transformer/model.safetensors.index.json",
        "FL2VA/text_encoder/config.json", "FL2VA/text_encoder/model.safetensors.index.json",
        "FL2VA/text_encoder/tokenizer.json", "FL2VA/audio_vae/config.json",
        "FL2VA/audio_vae/model.safetensors", "vae/diffusion_pytorch_model.safetensors.index.json",
    ):
        (root / filename).touch()


def test_nodes_register_and_use_custom_opaque_types(loaded_nodes):
    nodes, *_ = loaded_nodes
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "H3MLXModelLoader", "H3MLXResolution", "H3MLXDuration",
        "H3MLXTextConditioning", "H3MLXGenerationSettings",
        "H3MLXGenerate", "H3MLXOutput", "H3MLXVAEDecode",
        "H3MLXKeyframeSample",
        "H3MLXRefModelLoader", "H3MLXReferenceSample",
    }
    assert nodes.H3MLXModelLoader.RETURN_TYPES == ("H3_MLX_MODEL",)
    assert nodes.H3MLXGenerate.RETURN_TYPES == ("VIDEO",)
    assert nodes.H3MLXVAEDecode.RETURN_TYPES == ("VIDEO",)
    source = (PACKAGE / "nodes.py").read_text()
    assert "import torch" not in source
    # IMAGE is a Comfy Torch input and must cross once into the private MLX
    # runtime; native MLX tensors never travel in the opposite direction.
    assert "import torch" not in source


def test_model_bundle_discovery_and_safe_resolution(loaded_nodes):
    nodes, models, *_ = loaded_nodes
    make_bundle(models / "minimax_h3" / "MiniMax-H3")
    assert list(nodes.discover_model_bundles()) == ["MiniMax-H3"]
    paths = nodes.resolve_model_bundle("MiniMax-H3")
    assert paths.transformer.name == "transformer"
    with pytest.raises(FileNotFoundError):
        nodes.resolve_model_bundle("../../escape")


@pytest.mark.parametrize("width,height,frames,error", [
    (833, 480, 124, "divisible by 32"),
    (832, 481, 124, "divisible by 32"),
    (832, 480, 125, "17*n + 5"),
])
def test_invalid_shapes_fail_before_runtime(loaded_nodes, width, height, frames, error):
    nodes, *_ = loaded_nodes
    with pytest.raises(ValueError) as exc:
        nodes.H3MLXGenerationSettings().build(width, height, frames, 2, 0)
    assert error in str(exc.value)


def test_opaque_handle_reuses_runtime_and_forwards_parameters(loaded_nodes):
    nodes, _, _, temp = loaded_nodes
    calls = []

    class Runtime:
        backend = "exact_bf16_full_attention"
        paths = types.SimpleNamespace(variant="fl2va")

        def sample_keyframes(self, spec, images, anchors, **_kwargs):
            calls.append((spec, images, anchors))
            return types.SimpleNamespace(spec=spec)

        def decode(self, state, mp4_path, diagnostics_path, *_args, **_kwargs):
            Path(mp4_path).write_bytes(b"mp4")
            Path(diagnostics_path).write_text("{}")
            return nodes.H3GenerationResult(
                Path(mp4_path), Path(diagnostics_path), state.spec.width, state.spec.height,
                state.spec.frames, 24, 32000, 2, state.spec.frames / 24,
                (1.0,) * state.spec.steps,
            )

        def close(self): pass

    handle = nodes.H3MLXModel(Runtime(), "bundle")
    generator = nodes.H3MLXGenerate()
    first = generator.generate(handle, nodes.H3MLXConditioning("one"), nodes.H3MLXSettings(832, 480, 124, 2, 7))[0]
    second = generator.generate(handle, nodes.H3MLXConditioning("two"), nodes.H3MLXSettings(832, 480, 124, 2, 8))[0]
    assert len(calls) == 2
    assert calls[0][0].seed == 7 and calls[1][0].prompt == "two"
    assert calls[0][2] == ("first", "last") and len(calls[0][1]) == 2
    assert first.metadata["backend"] == "exact_bf16_full_attention"
    assert first.metadata["audio_channels"] == 2
    assert first.mp4_path.parent == temp
    del first, second


def test_output_copies_only_files_and_reports_audio_metadata(loaded_nodes):
    nodes, _, output, temp = loaded_nodes
    mp4, diagnostics = temp / "source.mp4", temp / "source.json"
    mp4.write_bytes(b"video+audio")
    diagnostics.write_text('{"audio_channels":2}')
    native = nodes.H3GenerationResult(
        mp4, diagnostics, 832, 480, 124, 24, 32000, 2,
        124 / 24, (70.0, 70.0),
    )
    result = nodes.H3MLXVideo(native, temporary=False)
    response = nodes.H3MLXOutput().save(result, "h3/test")
    saved_mp4, saved_json, timings = response["result"]
    assert Path(saved_mp4).read_bytes() == b"video+audio"
    assert Path(saved_json).is_file()
    assert Path(saved_mp4).is_relative_to(output)
    assert response["ui"]["gifs"][0]["format"] == "video/h264-mp4"
    assert response["ui"]["text"] == [timings]
    assert "Step 1: 70.000 s" in timings
    assert "Average per step: 70.000 s" in timings


def test_current_frontend_workflow_schema_and_connections():
    workflow = json.loads((PACKAGE / "workflows" / "h3_mlx_t2v.json").read_text())
    assert workflow["version"] == 0.4
    assert workflow["revision"] == 0
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == 9 and len(workflow["groups"]) == 5
    assert nodes[3]["type"] == "H3MLXTextConditioning"
    assert nodes[4]["type"] == "H3MLXResolution"
    assert nodes[5]["type"] == "H3MLXDuration"
    assert nodes[6]["type"] == "H3MLXGenerationSettings"
    assert nodes[6]["widgets_values"][:7] == [
        "bfloat16", "none", "none", "res_multistep", "beta",
        "mlx_fused_full_sdpa", 20,
    ]
    assert nodes[7]["type"] == "H3MLXGenerate"
    assert nodes[8]["type"] == "H3MLXOutput"
    assert workflow["links"][-3:] == [
        [7, 3, 0, 7, 1, "H3_MLX_CONDITIONING"],
        [8, 6, 0, 7, 2, "H3_MLX_SETTINGS"],
        [9, 7, 0, 8, 0, "VIDEO"],
    ]


def test_i2v_workflow_leaves_last_frame_optional():
    workflow = json.loads((PACKAGE / "workflows" / "h3_mlx_i2v.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    sampler_inputs = {item["name"]: item for item in nodes[9]["inputs"]}
    assert sampler_inputs["first_frame"]["link"] == 7
    assert sampler_inputs["last_frame"]["link"] is None
    assert nodes[5]["outputs"][0]["links"] == []


def test_r2v_workflow_uses_ref_loader_and_optional_references():
    workflow = json.loads((PACKAGE / "workflows" / "h3_mlx_r2v.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[2]["type"] == "H3MLXRefModelLoader"
    assert nodes[13]["type"] == "H3MLXTextConditioning"
    assert "subject_definitions:" in nodes[13]["widgets_values"][0]
    assert not nodes[13]["widgets_values"][0].startswith("integrated_multimodal_description:")
    assert nodes[17]["type"] == "H3MLXReferenceSample"
    inputs = {item["name"]: item for item in nodes[17]["inputs"]}
    assert inputs["reference_image_1"]["link"] == 4
    assert inputs["reference_image_2"]["link"] is None
    assert inputs["reference_image_3"]["link"] is None
    assert inputs["reference_image_9"]["link"] is None
    assert all(nodes[node_id]["type"] == "LoadImage" for node_id in range(4, 13))
    assert nodes[19]["inputs"][0]["type"] == "VIDEO"


def test_ref2va_rejects_unsupported_reference_modalities(loaded_nodes):
    nodes, *_ = loaded_nodes
    runtime = types.SimpleNamespace(paths=types.SimpleNamespace(variant="ref2va"))
    model = nodes.H3MLXModel(runtime, "bundle")
    with pytest.raises(ValueError, match="supports image references only"):
        nodes.H3MLXReferenceSample().sample(
            model,
            nodes.H3MLXConditioning("raw prompt"),
            nodes.H3MLXSettings(832, 480, 124, 20, 0),
            reference_video=object(),
        )


def test_duration_snaps_to_h3_grid(loaded_nodes):
    nodes, *_ = loaded_nodes
    frames, seconds = nodes.H3MLXDuration().convert(5.0)
    assert frames == 124
    assert seconds == pytest.approx(124 / 24)


def test_raw_prompt_is_preserved_exactly(loaded_nodes):
    nodes, *_ = loaded_nodes
    raw = "  [Shot 1] Rooftop chase\n\nwind; low score  "
    conditioning, = nodes.H3MLXTextConditioning().build(raw)
    assert conditioning.prompt == raw


def test_t2v_compatibility_frame_is_the_verified_official_asset(loaded_nodes):
    nodes, *_ = loaded_nodes
    assert nodes.COMPATIBILITY_FRAME.is_file()
    assert hashlib.sha256(nodes.COMPATIBILITY_FRAME.read_bytes()).hexdigest() == (
        "d19cc49c86929f50691073662fdd21bd9c98243cdc0b633fd4836ddb53d7dbe5"
    )
    images = nodes.H3MLXKeyframeSample._compatibility_keyframes(832, 480)
    assert [image.size for image in images] == [(832, 480), (832, 480)]


def test_custom_resolution_is_not_silently_resized(loaded_nodes):
    nodes, *_ = loaded_nodes
    assert nodes.H3MLXResolution().select(
        "Custom width / height below", 1920, 1088
    ) == (1920, 1088)


def test_primary_resolution_presets_are_h3_aligned(loaded_nodes):
    nodes, *_ = loaded_nodes
    expected = {
        "0.7 MP landscape — 1152×640": (1152, 640),
        "720p landscape (H3-aligned) — 1280×736": (1280, 736),
        "Official 768p landscape — 1344×768": (1344, 768),
        "1080p landscape (H3-aligned) — 1920×1088": (1920, 1088),
    }
    for label, size in expected.items():
        assert nodes.H3MLXResolution.PRESETS[label] == size
        assert size[0] % 32 == 0 and size[1] % 32 == 0


def test_reference_resize_matches_official_down_only_modes(loaded_nodes):
    nodes, *_ = loaded_nodes

    class FakeImage:
        def __init__(self, array): self.array = array
        def __getitem__(self, _index): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.array

    np = pytest.importorskip("numpy")
    image = FakeImage(np.zeros((900, 1600, 3), dtype=np.float32))
    match = nodes.H3MLXReferenceSample._reference(image, 1152, 640, "match")
    maximum = nodes.H3MLXReferenceSample._reference(image, 1152, 640, "max")
    assert match.size == (1152, 640)
    assert maximum.size == (1600, 896)

    small = FakeImage(np.zeros((360, 640, 3), dtype=np.float32))
    assert nodes.H3MLXReferenceSample._reference(small, 1152, 640, "match").size == (640, 352)
    assert nodes.H3MLXReferenceSample._reference(small, 1152, 640, "max").size == (640, 352)

    with pytest.raises(ValueError, match="Unknown reference_image_size"):
        nodes.H3MLXReferenceSample._reference(image, 1152, 640, "invalid")


def test_second_pass_resolution_presets_are_h3_aligned(loaded_nodes):
    nodes, *_ = loaded_nodes
    for width, height in nodes.H3MLXLatentResize.TARGET_PRESETS.values():
        assert width % 32 == 0 and height % 32 == 0
    assert nodes.H3MLXLatentResize.TARGET_PRESETS["4K landscape — 3840×2176"] == (3840, 2176)
