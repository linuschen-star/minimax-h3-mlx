"""Thin ComfyUI nodes for the Phase-1 native MLX H3 runtime."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import folder_paths
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.latest import VideoFromFile

from minimax_h3_mlx.runtime import (
    ExactH3Runtime,
    H3GenerationResult,
    H3GenerationSpec,
    H3LatentState,
    H3ModelPaths,
    resolve_ffmpeg,
)


MODEL_FOLDER = "minimax_h3"
MODEL_ROOT = Path(folder_paths.models_dir) / MODEL_FOLDER
COMPATIBILITY_FRAME = Path(__file__).parent / "assets" / "GrayBackGround.png"
folder_paths.add_model_folder_path(MODEL_FOLDER, str(MODEL_ROOT), is_default=True)


def _valid_bundle(path: Path) -> bool:
    try:
        H3ModelPaths.from_bundle(path)
    except FileNotFoundError:
        return False
    return True


def discover_model_bundles() -> dict[str, Path]:
    candidates: list[Path] = []
    for base_name in folder_paths.get_folder_paths(MODEL_FOLDER):
        base = Path(base_name).resolve()
        if _valid_bundle(base):
            candidates.append(base)
        if base.is_dir():
            candidates.extend(path.resolve() for path in sorted(base.iterdir())
                              if path.is_dir() and _valid_bundle(path))
    result: dict[str, Path] = {}
    for path in candidates:
        name = path.name
        if name in result and result[name] != path:
            suffix = 2
            while f"{name} [{suffix}]" in result:
                suffix += 1
            name = f"{name} [{suffix}]"
        result[name] = path
    return result


def resolve_model_bundle(name: str) -> H3ModelPaths:
    bundles = discover_model_bundles()
    if name not in bundles:
        raise FileNotFoundError(
            f"MiniMax H3 model bundle '{name}' was not found. Place it under "
            f"{MODEL_ROOT} or configure an extra '{MODEL_FOLDER}' model path."
        )
    return H3ModelPaths.from_bundle(bundles[name])


def resolve_ref_model_bundle(name: str) -> H3ModelPaths:
    bundles = discover_model_bundles()
    if name not in bundles:
        raise FileNotFoundError(f"MiniMax H3 model bundle '{name}' was not found")
    return H3ModelPaths.from_bundle(bundles[name], variant="ref2va")


class H3MLXModel:
    """Opaque ComfyUI value; native MLX arrays stay inside ExactH3Runtime."""

    __slots__ = ("_runtime", "bundle_name")

    def __init__(self, runtime: ExactH3Runtime, bundle_name: str):
        self._runtime = runtime
        self.bundle_name = bundle_name

    @property
    def backend(self):
        return self._runtime.backend

    def generate(self, *args, **kwargs):
        return self._runtime.generate(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self._runtime.sample(*args, **kwargs)

    def sample_keyframes(self, *args, **kwargs):
        return self._runtime.sample_keyframes(*args, **kwargs)

    def sample_references(self, *args, **kwargs):
        return self._runtime.sample_references(*args, **kwargs)

    def resume(self, *args, **kwargs):
        return self._runtime.resume(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._runtime.decode(*args, **kwargs)

    def resize_latent(self, *args, **kwargs):
        return self._runtime.resize_latent(*args, **kwargs)

    def close(self):
        self._runtime.close()


class H3MLXVideo(VideoFromFile):
    """Standard file-backed ComfyUI VIDEO carrying private H3 diagnostics."""

    __slots__ = ("_result", "_temporary")

    def __init__(self, result: H3GenerationResult, temporary: bool = True):
        super().__init__(str(result.mp4_path))
        self._result = result
        self._temporary = temporary

    @property
    def mp4_path(self):
        return self._result.mp4_path

    @property
    def diagnostics_path(self):
        return self._result.diagnostics_path

    @property
    def metadata(self):
        return {
            "width": self._result.width, "height": self._result.height,
            "frames": self._result.frames, "fps": self._result.fps,
            "audio_sample_rate": self._result.audio_sample_rate,
            "audio_channels": self._result.audio_channels,
            "duration_seconds": self._result.duration_seconds,
            "step_seconds": self._result.step_seconds,
            "backend": self._result.backend,
        }

    def __del__(self):
        if self._temporary:
            self.mp4_path.unlink(missing_ok=True)
            self.diagnostics_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class H3MLXConditioning:
    """Small opaque recipe. Native tensors are created only inside the runtime."""

    prompt: str


@dataclass(frozen=True)
class H3MLXSettings:
    width: int
    height: int
    frames: int
    steps: int
    seed: int
    dtype: str = "bfloat16"
    quantization: str = "none"
    lora: str = "none"
    sampler: str = "res_multistep"
    scheduler: str = "beta"
    attention_backend: str = "mlx_fused_full_sdpa"


class H3MLXLatent:
    """Opaque Comfy value; native BF16 video/audio/context arrays stay private."""

    __slots__ = ("_state",)

    def __init__(self, state: H3LatentState):
        self._state = state

    @property
    def metadata(self):
        return {
            "layout": self._state.layout, "shape": self._state.shape,
            "dtype": str(self._state.video.dtype), "width": self._state.spec.width,
            "height": self._state.spec.height, "frames": self._state.spec.frames,
            "scheduler_index": self._state.scheduler_index,
            "start_sigma": self._state.start_sigma, "pass": self._state.pass_name,
        }


class H3MLXResolution:
    PRESETS = {
        "16:9 landscape — 832×480": (832, 480),
        "9:16 portrait — 480×832": (480, 832),
        "0.7 MP landscape — 1152×640": (1152, 640),
        "0.7 MP portrait — 640×1152": (640, 1152),
        "720p landscape (H3-aligned) — 1280×736": (1280, 736),
        "720p portrait (H3-aligned) — 736×1280": (736, 1280),
        "Official 768p landscape — 1344×768": (1344, 768),
        "Official 768p portrait — 768×1344": (768, 1344),
        "1080p landscape (H3-aligned) — 1920×1088": (1920, 1088),
        "1080p portrait (H3-aligned) — 1088×1920": (1088, 1920),
        "1:1 square — 640×640": (640, 640),
        "4:3 landscape — 768×576": (768, 576),
        "3:4 portrait — 576×768": (576, 768),
        "Custom width / height below": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (list(cls.PRESETS),),
            "custom_width": ("INT", {"default": 832, "min": 32, "max": 2048, "step": 32}),
            "custom_height": ("INT", {"default": 480, "min": 32, "max": 2048, "step": 32}),
        }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def select(self, preset, custom_width, custom_height):
        selected = self.PRESETS[preset]
        return selected if selected is not None else (custom_width, custom_height)


class H3MLXDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"duration_seconds": (
            "FLOAT", {"default": 5.0, "min": 0.25, "max": 30.0, "step": 0.25}
        )}}

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("valid_frames", "actual_seconds")
    FUNCTION = "convert"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def convert(self, duration_seconds):
        # H3 accepts F=17*n+5. Choose the nearest valid duration at fixed 24 fps.
        n = max(0, round((duration_seconds * 24 - 5) / 17))
        frames = 17 * n + 5
        return (frames, frames / 24.0)


class H3MLXTextConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {
            "multiline": True,
            "dynamicPrompts": True,
            "default": "Describe the scene, motion, camera work, dialogue, and sound.",
        })}}

    RETURN_TYPES = ("H3_MLX_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def build(self, prompt):
        if not prompt.strip():
            raise ValueError("MiniMax H3 prompt must not be empty.")
        return (H3MLXConditioning(prompt),)


class H3MLXGenerationSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"forceInput": True}),
            "height": ("INT", {"forceInput": True}),
            "frames": ("INT", {"forceInput": True}),
            "dtype": (["bfloat16"],),
            "quantization": (["none"],),
            "lora": (["none"],),
            "sampler": (["res_multistep"],),
            "scheduler": (["beta"],),
            "attention_backend": (["mlx_fused_full_sdpa"],),
            "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        }}

    RETURN_TYPES = ("H3_MLX_SETTINGS",)
    RETURN_NAMES = ("generation_settings",)
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def build(self, width, height, frames, steps, seed, dtype="bfloat16",
              quantization="none", lora="none", sampler="res_multistep",
              scheduler="beta", attention_backend="mlx_fused_full_sdpa"):
        supported = ("bfloat16", "none", "none", "res_multistep", "beta", "mlx_fused_full_sdpa")
        selected = (dtype, quantization, lora, sampler, scheduler, attention_backend)
        if selected != supported:
            raise ValueError(
                "Unsupported H3 MLX generation configuration. This parity pass supports only "
                "Base BF16, no quantization, no LoRA, res_multistep + beta, and MLX fused full SDPA."
            )
        # Validate dimensions/grid here without fabricating a prompt.
        H3GenerationSpec("validation", width, height, frames, steps, seed).validate()
        return (H3MLXSettings(width, height, frames, steps, seed, *selected),)


class H3MLXModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        bundles = list(discover_model_bundles())
        if not bundles:
            bundles = ["<no valid H3 model bundle found>"]
        return {"required": {"model_bundle": (bundles,)}}

    RETURN_TYPES = ("H3_MLX_MODEL",)
    RETURN_NAMES = ("h3_mlx_model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def load(self, model_bundle):
        resolve_ffmpeg()
        paths = resolve_model_bundle(model_bundle)
        progress_bar = ProgressBar(13)

        def progress(_stage, current, total):
            progress_bar.update_absolute(current, total)
            model_management.throw_exception_if_processing_interrupted()

        runtime = ExactH3Runtime(paths, progress=progress)
        return (H3MLXModel(runtime, model_bundle),)


class H3MLXRefModelLoader(H3MLXModelLoader):
    """Load only the official Ref2VA transformer partition."""

    def load(self, model_bundle):
        resolve_ffmpeg()
        paths = resolve_ref_model_bundle(model_bundle)
        progress_bar = ProgressBar(13)
        def progress(_stage, current, total):
            progress_bar.update_absolute(current, total)
            model_management.throw_exception_if_processing_interrupted()
        return (H3MLXModel(ExactH3Runtime(paths, progress=progress), model_bundle),)


class H3MLXSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3_mlx_model": ("H3_MLX_MODEL",),
            "conditioning": ("H3_MLX_CONDITIONING",),
            "generation_settings": ("H3_MLX_SETTINGS",),
        }}

    RETURN_TYPES = ("H3_MLX_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/MLX Latent"

    def sample(self, h3_mlx_model, conditioning, generation_settings):
        if not isinstance(h3_mlx_model, H3MLXModel):
            raise TypeError("H3 MLX Sample requires H3 MLX Model Loader.")
        if not isinstance(conditioning, H3MLXConditioning):
            raise TypeError("H3 MLX Sample requires H3 MLX Text Conditioning.")
        if not isinstance(generation_settings, H3MLXSettings):
            raise TypeError("H3 MLX Sample requires H3 MLX Sampling Settings.")
        s = generation_settings
        spec = H3GenerationSpec(conditioning.prompt, s.width, s.height, s.frames, s.steps, s.seed)
        bar = ProgressBar(spec.steps + 1)

        def progress(stage, current, total):
            value = current if stage == "conditioning" else 1 + current
            bar.update_absolute(value, spec.steps + 1)

        state = h3_mlx_model.sample(
            spec, progress=progress,
            cancel=model_management.throw_exception_if_processing_interrupted,
        )
        return (H3MLXLatent(state),)


class H3MLXLatentResize:
    TARGET_PRESETS = {
        "480p landscape — 832×480": (832, 480),
        "720p landscape — 1280×736": (1280, 736),
        "1080p landscape — 1920×1088": (1920, 1088),
        "2K landscape — 2048×1088": (2048, 1088),
        "4K landscape — 3840×2176": (3840, 2176),
        "720p portrait — 736×1280": (736, 1280),
        "1080p portrait — 1088×1920": (1088, 1920),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("H3_MLX_LATENT",),
            "mode": (["scale factor", *cls.TARGET_PRESETS, "explicit pixel size"],),
            "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 4.0, "step": 0.05}),
            "target_width": ("INT", {"default": 1296, "min": 32, "max": 4096, "step": 32}),
            "target_height": ("INT", {"default": 736, "min": 32, "max": 4096, "step": 32}),
        }}

    RETURN_TYPES = ("H3_MLX_LATENT", "STRING")
    RETURN_NAMES = ("latent", "shape_info")
    FUNCTION = "resize"
    CATEGORY = "MiniMax H3/MLX Latent"

    def resize(self, latent, mode, scale_factor, target_width, target_height):
        if not isinstance(latent, H3MLXLatent):
            raise TypeError("H3 MLX Latent Resize requires H3_MLX_LATENT.")
        target = self.TARGET_PRESETS.get(mode)
        if mode == "explicit pixel size":
            target = (target_width, target_height)
        state = ExactH3Runtime.resize_latent(
            latent._state, scale_factor,
            target[0] if target else None, target[1] if target else None,
        )
        info = f"{latent._state.shape} -> {state.shape} {state.layout}; frames={state.spec.frames}"
        return (H3MLXLatent(state), info)


class H3MLXResume:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3_mlx_model": ("H3_MLX_MODEL",),
            "latent": ("H3_MLX_LATENT",),
            "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "second_pass_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        }}

    RETURN_TYPES = ("H3_MLX_LATENT", "STRING")
    RETURN_NAMES = ("latent", "schedule_info")
    FUNCTION = "resume"
    CATEGORY = "MiniMax H3/MLX Latent"

    def resume(self, h3_mlx_model, latent, denoise_strength, second_pass_steps, seed):
        if not isinstance(h3_mlx_model, H3MLXModel) or not isinstance(latent, H3MLXLatent):
            raise TypeError("H3 MLX Resume requires the native model and latent objects.")
        evaluations = second_pass_steps if denoise_strength > 0.0 else 0
        bar = ProgressBar(max(1, evaluations))
        state = h3_mlx_model.resume(
            latent._state, denoise_strength, second_pass_steps, seed,
            progress=lambda _stage, current, total: bar.update_absolute(current, total),
            cancel=model_management.throw_exception_if_processing_interrupted,
        )
        info = (f"strength={denoise_strength:.3f}; start_sigma={state.start_sigma:.9f}; "
                f"steps={len(state.step_seconds)}; shape={state.shape}")
        return (H3MLXLatent(state), info)


class H3MLXVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3_mlx_model": ("H3_MLX_MODEL",),
            "latent": ("H3_MLX_LATENT",),
        }}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/MLX Latent"

    def decode(self, h3_mlx_model, latent):
        if not isinstance(h3_mlx_model, H3MLXModel) or not isinstance(latent, H3MLXLatent):
            raise TypeError("H3 MLX VAE Decode requires the native model and latent objects.")
        temp_dir = Path(folder_paths.get_temp_directory()); temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="h3_mlx_", suffix=".mp4", dir=temp_dir, delete=False) as file:
            mp4_path = Path(file.name)
        mp4_path.unlink(missing_ok=True)
        diagnostics_path = mp4_path.with_suffix(".json")
        try:
            result = h3_mlx_model.decode(
                latent._state, mp4_path, diagnostics_path, temp_dir,
                cancel=model_management.throw_exception_if_processing_interrupted,
            )
        except BaseException:
            mp4_path.unlink(missing_ok=True); diagnostics_path.unlink(missing_ok=True); raise
        return (H3MLXVideo(result),)


class H3MLXKeyframeSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_mlx_model": ("H3_MLX_MODEL",),
                "conditioning": ("H3_MLX_CONDITIONING",),
                "generation_settings": ("H3_MLX_SETTINGS",),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("H3_MLX_LATENT", "STRING")
    RETURN_NAMES = ("latent", "mode")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    @staticmethod
    def _resize_keyframe(keyframe, width, height, *, follower=False):
        if keyframe.size == (width, height):
            return keyframe
        if not follower:
            # Official FL2VA: the first supplied geometry anchor is stretched
            # directly onto an explicitly requested canvas (no cover crop).
            return keyframe.resize((width, height), Image.Resampling.LANCZOS)
        # Official follower arithmetic (round, then centred cover crop).  This
        # intentionally differs by a pixel from PIL ImageOps.fit in some sizes.
        scale = max(width / keyframe.size[0], height / keyframe.size[1])
        resized_size = (max(width, round(keyframe.size[0] * scale)),
                        max(height, round(keyframe.size[1] * scale)))
        left = max(0, (resized_size[0] - width) // 2)
        top = max(0, (resized_size[1] - height) // 2)
        resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
        return resized.crop((left, top, left + width, top + height))

    @classmethod
    def _keyframe(cls, image, width, height, *, follower=False):
        array = np.clip(image[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        return cls._resize_keyframe(
            Image.fromarray(array, "RGB"), width, height, follower=follower,
        )

    @classmethod
    def _compatibility_keyframes(cls, width, height):
        if not COMPATIBILITY_FRAME.is_file():
            raise FileNotFoundError(
                f"Missing official H3 T2V compatibility frame: {COMPATIBILITY_FRAME}"
            )
        with Image.open(COMPATIBILITY_FRAME) as image:
            source = image.convert("RGB")
            return [
                cls._resize_keyframe(source, width, height),
                cls._resize_keyframe(source, width, height, follower=True),
            ]

    def sample(self, h3_mlx_model, conditioning, generation_settings,
               first_frame=None, last_frame=None):
        if not isinstance(h3_mlx_model, H3MLXModel):
            raise TypeError("H3 MLX Keyframe Sample requires the native model loader output")
        if not isinstance(conditioning, H3MLXConditioning) or not isinstance(generation_settings, H3MLXSettings):
            raise TypeError("H3 MLX Keyframe Sample requires H3 conditioning and settings")
        settings = generation_settings
        spec = H3GenerationSpec(conditioning.prompt, settings.width, settings.height,
                                settings.frames, settings.steps, settings.seed)
        images, anchors = [], []
        if first_frame is not None:
            images.append(self._keyframe(first_frame, settings.width, settings.height)); anchors.append("first")
        if last_frame is not None:
            images.append(self._keyframe(last_frame, settings.width, settings.height,
                                         follower=bool(images))); anchors.append("last")
        if not images:
            images = self._compatibility_keyframes(settings.width, settings.height)
            anchors = ["first", "last"]
            mode = "t2va:official-compatibility-frames"
        else:
            mode = "fl2va:" + "+".join(anchors)
        print(f"[H3 MLX] FL2VA request: anchors={tuple(anchors)}, steps={settings.steps}, "
              f"target={(settings.width, settings.height, settings.frames)}", flush=True)
        bar = ProgressBar(settings.steps + 1)
        state = h3_mlx_model.sample_keyframes(
            spec, images, tuple(anchors),
            progress=lambda _stage, current, total: bar.update_absolute(current, total),
            cancel=model_management.throw_exception_if_processing_interrupted,
        )
        return H3MLXLatent(state), mode


class H3MLXReferenceSample:
    """Official image-reference Ref2VA path; references retain their own grids."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"reference_image_{index}": ("IMAGE",) for index in range(1, 10)}
        return {"required": {
            "h3_mlx_model": ("H3_MLX_MODEL",),
            "conditioning": ("H3_MLX_CONDITIONING",),
            "generation_settings": ("H3_MLX_SETTINGS",),
            "reference_image_size": (["match", "max"], {"default": "match"}),
        }, "optional": optional}

    RETURN_TYPES = ("H3_MLX_LATENT", "STRING")
    RETURN_NAMES = ("latent", "mode")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    @staticmethod
    def _reference(image, output_width, output_height, reference_image_size):
        array = np.clip(image[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, "RGB")
        width, height = pil.size
        if width > 4 * height or height > 4 * width:
            raise ValueError(f"Ref2VA images must be within 1:4 and 4:1, got {width}x{height}")
        if reference_image_size == "match":
            scale = min(1.0, np.sqrt((output_width * output_height) / (width * height)))
        elif reference_image_size == "max":
            scale = min(1.0, 2048 / min(width, height))
        else:
            raise ValueError(f"Unknown reference_image_size: {reference_image_size!r}")
        target_width = max(32, round(width * scale / 32) * 32)
        target_height = max(32, round(height * scale / 32) * 32)
        return pil if pil.size == (target_width, target_height) else pil.resize(
            (target_width, target_height), Image.Resampling.LANCZOS)

    def sample(self, h3_mlx_model, conditioning, generation_settings,
               reference_image_size="match", **kwargs):
        if not isinstance(h3_mlx_model, H3MLXModel):
            raise TypeError("H3 MLX Reference Sample requires the Ref2VA loader")
        if h3_mlx_model._runtime.paths.variant != "ref2va":
            raise TypeError("Connect H3 MLX Ref2VA Model Loader, not the FL2VA loader")
        if not isinstance(conditioning, H3MLXConditioning) or not isinstance(generation_settings, H3MLXSettings):
            raise TypeError("H3 MLX Reference Sample requires H3 conditioning and settings")
        supported_inputs = {f"reference_image_{index}" for index in range(1, 10)}
        unsupported = sorted(set(kwargs) - supported_inputs)
        if unsupported:
            raise ValueError(
                "Native MLX Ref2VA supports image references only; unsupported reference "
                f"input(s): {', '.join(unsupported)}"
            )
        settings = generation_settings
        images = [self._reference(kwargs[name], settings.width, settings.height, reference_image_size)
                  for name in sorted(kwargs) if kwargs[name] is not None]
        if not images:
            raise ValueError("Ref2VA requires at least one connected reference image")
        spec = H3GenerationSpec(conditioning.prompt, settings.width, settings.height,
                                settings.frames, settings.steps, settings.seed)
        bar = ProgressBar(settings.steps + 1)
        state = h3_mlx_model.sample_references(
            spec, images,
            progress=lambda _stage, current, total: bar.update_absolute(current, total),
            cancel=model_management.throw_exception_if_processing_interrupted,
        )
        return H3MLXLatent(state), f"ref2va:image×{len(images)}:{reference_image_size}"


class H3MLXGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_mlx_model": ("H3_MLX_MODEL",),
                "conditioning": ("H3_MLX_CONDITIONING",),
                "generation_settings": ("H3_MLX_SETTINGS",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3/MLX Phase 1"

    def generate(self, h3_mlx_model, conditioning, generation_settings):
        if not isinstance(h3_mlx_model, H3MLXModel):
            raise TypeError("H3 MLX Generate requires the opaque output of H3 MLX Model Loader.")
        if not isinstance(conditioning, H3MLXConditioning):
            raise TypeError("H3 MLX Generate requires H3 MLX Text Conditioning.")
        if not isinstance(generation_settings, H3MLXSettings):
            raise TypeError("H3 MLX Generate requires H3 MLX Generation Settings.")
        latent, _mode = H3MLXKeyframeSample().sample(
            h3_mlx_model, conditioning, generation_settings,
        )
        return H3MLXVAEDecode().decode(h3_mlx_model, latent)


class H3MLXOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "h3_mlx/H3"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mp4_path", "diagnostics_path", "step_timings")
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/MLX Phase 1"
    OUTPUT_NODE = True

    def save(self, video, filename_prefix):
        if not isinstance(video, H3MLXVideo):
            raise TypeError("H3 MLX Output requires the VIDEO from H3 MLX Sampler + AV Decode.")
        metadata = video.metadata
        output_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(),
            metadata["width"], metadata["height"],
        )
        mp4_name = f"{filename}_{counter:05}.mp4"
        diagnostics_name = f"{filename}_{counter:05}.json"
        mp4_path = Path(output_dir) / mp4_name
        diagnostics_path = Path(output_dir) / diagnostics_name
        shutil.copy2(video.mp4_path, mp4_path)
        shutil.copy2(video.diagnostics_path, diagnostics_path)
        preview = {
            "filename": mp4_name, "subfolder": subfolder, "type": "output",
            "format": "video/h264-mp4", "frame_rate": metadata["fps"],
            "fullpath": str(mp4_path),
        }
        step_seconds = metadata["step_seconds"]
        if step_seconds:
            lines = [f"Step {index}: {seconds:.3f} s"
                     for index, seconds in enumerate(step_seconds, 1)]
            total = sum(step_seconds)
            lines.extend((f"Sampling total: {total:.3f} s",
                          f"Average per step: {total / len(step_seconds):.3f} s"))
            timing_summary = "\n".join(lines)
        else:
            timing_summary = "No sampling step timings were recorded."
        print(f"[H3 MLX]\n{timing_summary}")
        return {
            "ui": {"gifs": [preview], "text": [timing_summary]},
            "result": (str(mp4_path), str(diagnostics_path), timing_summary),
        }


NODE_CLASS_MAPPINGS = {
    "H3MLXModelLoader": H3MLXModelLoader,
    "H3MLXRefModelLoader": H3MLXRefModelLoader,
    "H3MLXVAEDecode": H3MLXVAEDecode,
    "H3MLXKeyframeSample": H3MLXKeyframeSample,
    "H3MLXReferenceSample": H3MLXReferenceSample,
    "H3MLXResolution": H3MLXResolution,
    "H3MLXDuration": H3MLXDuration,
    "H3MLXTextConditioning": H3MLXTextConditioning,
    "H3MLXGenerationSettings": H3MLXGenerationSettings,
    "H3MLXGenerate": H3MLXGenerate,
    "H3MLXOutput": H3MLXOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MLXModelLoader": "H3 MLX Model Loader (Exact BF16)",
    "H3MLXRefModelLoader": "H3 MLX Ref2VA Model Loader (Exact BF16)",
    "H3MLXVAEDecode": "H3 MLX VAE Decode (Video + Audio)",
    "H3MLXKeyframeSample": "H3 MLX T2V / First / Last Frame Sample",
    "H3MLXReferenceSample": "H3 MLX Ref2VA Image Reference Sample",
    "H3MLXResolution": "H3 MLX Resolution / Aspect Ratio",
    "H3MLXDuration": "H3 MLX Duration → Valid Frames",
    "H3MLXTextConditioning": "H3 MLX Text Conditioning",
    "H3MLXGenerationSettings": "H3 MLX Sampling Settings",
    "H3MLXGenerate": "H3 MLX Sampler + AV Decode (Exact BF16)",
    "H3MLXOutput": "H3 MLX Video + Audio Output",
}
