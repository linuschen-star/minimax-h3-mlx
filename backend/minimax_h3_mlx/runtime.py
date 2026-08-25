"""Stable Phase-1 exact H3 runtime API for CLI and thin UI adapters."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio_vae import H3AudioVAE, load_audio_vae_decoder
from .model import H3Config, MiniMaxH3Transformer, load_sharded_safetensors
from .model import FL2VLayout, Ref2VImageLayout
from .multimodal_encoder import encode_keyframe_prompt
from .sampling import res_multistep_sample
from .text_encoder import H3TextEncoder, TextConfig, load_text_encoder, tokenize_prompt
from .video_vae import H3VideoVAE, load_video_vae_decoder, load_video_vae_encoder


ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], None]


def resolve_ffmpeg() -> Path:
    """Find ffmpeg even when a GUI-launched ComfyUI has a minimal PATH."""
    candidates = [
        os.environ.get("H3_MLX_FFMPEG"),
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "ffmpeg is required to encode the H3 audiovisual MP4. Install it with "
        "Homebrew, or set H3_MLX_FFMPEG to its absolute executable path."
    )


@dataclass(frozen=True)
class H3ModelPaths:
    root: Path
    transformer: Path
    text_encoder: Path
    video_vae: Path
    audio_vae: Path
    variant: str = "fl2va"

    @classmethod
    def from_bundle(cls, root: str | Path, variant: str = "fl2va") -> "H3ModelPaths":
        root = Path(root).resolve()
        if variant not in ("fl2va", "ref2va"):
            raise ValueError(f"unsupported H3 model variant: {variant}")
        fl2va = root / "FL2VA" if (root / "FL2VA").is_dir() else root
        transformer = (root / "Ref2VA" / "transformer" if variant == "ref2va"
                       else fl2va / "transformer")
        paths = cls(root, transformer, fl2va / "text_encoder",
                    root / "vae", fl2va / "audio_vae", variant)
        required = {
            paths.transformer / "config.json": "transformer config",
            paths.transformer / "model.safetensors.index.json": "transformer weights",
            paths.text_encoder / "config.json": "text encoder config",
            paths.text_encoder / "model.safetensors.index.json": "text encoder weights",
            paths.text_encoder / "tokenizer.json": "tokenizer",
            paths.video_vae / "diffusion_pytorch_model.safetensors.index.json": "video VAE weights",
            paths.audio_vae / "config.json": "audio VAE config",
            paths.audio_vae / "model.safetensors": "audio VAE weights",
        }
        missing = [f"{label}: {path}" for path, label in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete MiniMax H3 bundle:\n" + "\n".join(missing))
        return paths


@dataclass(frozen=True)
class H3GenerationSpec:
    prompt: str
    width: int = 832
    height: int = 480
    frames: int = 124
    steps: int = 2
    seed: int = 0
    fps: int = 24

    def validate(self) -> tuple[int, int]:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if self.width % 32 or self.height % 32:
            raise ValueError("H3 width and height must both be divisible by 32.")
        if self.width < 32 or self.height < 32:
            raise ValueError("H3 width and height must be at least 32 pixels.")
        if self.frames < 5 or (self.frames - 5) % 17:
            raise ValueError("H3 frame count must equal 17*n + 5 (for example 124).")
        if self.steps < 1:
            raise ValueError("Sampling steps must be at least 1.")
        if self.fps != 24:
            raise ValueError("Phase 1 H3 output is fixed at 24 fps.")
        latent_t = 5 * ((self.frames - 5) // 17) + 2
        audio_t = round(self.frames / self.fps * 40)
        return latent_t, audio_t


@dataclass(frozen=True)
class H3GenerationResult:
    mp4_path: Path
    diagnostics_path: Path
    width: int
    height: int
    frames: int
    fps: int
    audio_sample_rate: int
    audio_channels: int
    duration_seconds: float
    step_seconds: tuple[float, ...]
    backend: str = "exact_bf16_full_attention"


@dataclass(frozen=True)
class H3LatentState:
    """Opaque native H3 state at sigma zero; arrays never cross into Torch."""

    video: mx.array
    audio: mx.array
    context: mx.array
    text_token_tags: tuple[int, ...]
    spec: H3GenerationSpec
    step_seconds: tuple[float, ...] = ()
    condition_video: mx.array | None = None
    condition_noise: mx.array | None = None
    scheduler_index: int = 0
    start_sigma: float = 1.0
    pass_name: str = "first"

    @property
    def layout(self):
        return "BCTHW"

    @property
    def shape(self):
        return tuple(self.video.shape)


class ExactH3Runtime:
    """Own the persistent MLX Transformer; all native arrays remain private."""

    backend = "exact_bf16_full_attention"

    def __init__(self, paths: H3ModelPaths, progress: ProgressCallback | None = None):
        self.paths = paths
        self.cfg = H3Config.from_json(paths.transformer / "config.json")
        if not self.cfg.head_major_sdpa:
            raise RuntimeError("Phase 1 runtime requires head-major long-sequence SDPA.")
        self.model = MiniMaxH3Transformer(self.cfg)
        load_sharded_safetensors(
            self.model, paths.transformer / "model.safetensors.index.json",
            progress_callback=(lambda current, total: progress("loading_transformer", current, total))
            if progress else None,
        )
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.model = None
        self.closed = True
        gc.collect()
        mx.clear_cache()

    def _check_open(self):
        if self.closed or self.model is None:
            raise RuntimeError("This H3 MLX runtime has been released; reload it in ComfyUI.")

    def _unload_transformer_for_conditioning(self):
        """Prevent the 33B transformer and 26.5B vision conditioner overlapping."""
        self._check_open()
        self.model = None
        gc.collect()
        mx.clear_cache()

    def _reload_transformer_after_conditioning(self, progress=None):
        started = time.perf_counter()
        print("[H3 MLX] Reloading transformer after releasing multimodal conditioner", flush=True)
        self.model = MiniMaxH3Transformer(self.cfg)
        load_sharded_safetensors(
            self.model, self.paths.transformer / "model.safetensors.index.json",
            progress_callback=(lambda current, total: progress("loading_transformer", current, total))
            if progress else None,
        )
        print(f"[H3 MLX] Transformer reload completed in {time.perf_counter() - started:.3f} s", flush=True)

    def _encode_prompt(self, prompt: str):
        config = TextConfig.from_json(self.paths.text_encoder / "config.json")
        encoder = H3TextEncoder(config, output_layer=50)
        load_text_encoder(encoder, self.paths.text_encoder / "model.safetensors.index.json")
        input_ids = tokenize_prompt(self.paths.text_encoder / "tokenizer.json", prompt)
        context = encoder(input_ids).astype(mx.bfloat16)
        mx.eval(context)
        token_count = context.shape[1]
        del encoder, input_ids
        gc.collect()
        mx.clear_cache()
        return context, [1] * token_count

    def sample(
        self,
        spec: H3GenerationSpec,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3LatentState:
        """Run the original full schedule and return the native pre-VAE state."""
        self._check_open()
        latent_t, audio_t = spec.validate()
        notify = progress or (lambda *_: None)
        interrupt = cancel or (lambda: None)
        interrupt(); notify("conditioning", 0, 1)
        context, tags = self._encode_prompt(spec.prompt)
        notify("conditioning", 1, 1); interrupt()
        mx.random.seed(spec.seed)
        video = mx.random.normal((1, self.cfg.in_channels, latent_t,
                                  spec.height // 16, spec.width // 16)).astype(mx.bfloat16)
        audio = mx.random.normal((1, self.cfg.audio_in_channels, 2, audio_t)).astype(mx.bfloat16)
        mx.eval(video, audio, context)
        timings: list[float] = []
        last_step = time.perf_counter()

        def sampled(step, _sigma, *_):
            nonlocal last_step
            now = time.perf_counter(); elapsed = now - last_step; last_step = now
            timings.append(elapsed)
            print(f"[H3 MLX] First pass step {step + 1}/{spec.steps}: {elapsed:.3f} s", flush=True)
            notify("sampling", step + 1, spec.steps); interrupt()

        video, audio = res_multistep_sample(
            self.model, video, audio, context, steps=spec.steps,
            text_token_tags=tags, callback=sampled,
        )
        print(f"[H3 MLX] First-pass latent shape: {tuple(video.shape)} BCTHW", flush=True)
        return H3LatentState(video, audio, context, tuple(tags), spec, tuple(timings))

    def sample_keyframes(
        self,
        spec: H3GenerationSpec,
        images,
        anchors: tuple[str, ...],
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3LatentState:
        """Run the official FL2VA presentation with one or two endpoint images."""
        self._check_open()
        latent_t, audio_t = spec.validate()
        if not images or len(images) != len(anchors) or len(images) > 2:
            raise ValueError("FL2VA requires one or two images matching their endpoint anchors")
        notify = progress or (lambda *_: None)
        interrupt = cancel or (lambda: None)
        mode = "+".join(anchors)
        started = time.perf_counter()
        print(f"[H3 MLX] FL2VA ({mode}): multimodal conditioning started", flush=True)
        interrupt(); notify("conditioning", 0, 1)
        # The persistent 33B denoiser plus the 26.5B Qwen3-VL vision path exceeds
        # the practical M4 Max working set and pages transformer weights out.
        # Drop it before conditioning, then reload only after the conditioner is
        # fully evaluated and released. This makes model-load time explicit and
        # keeps every denoising step on the normal resident BF16 path.
        self._unload_transformer_for_conditioning()
        context, tags = encode_keyframe_prompt(self.paths.text_encoder, images, spec.prompt)
        print(f"[H3 MLX] FL2VA ({mode}): Qwen3-VL layer-50 conditioning "
              f"completed in {time.perf_counter() - started:.3f} s; tokens={context.shape[1]}", flush=True)

        vae_started = time.perf_counter()
        print(f"[H3 MLX] FL2VA ({mode}): Visual VAE keyframe encode started", flush=True)
        vae = H3VideoVAE()
        load_video_vae_encoder(vae, self.paths.video_vae / "diffusion_pytorch_model.safetensors.index.json")
        condition_parts = []
        for index, image in enumerate(images):
            pixels = mx.array(np.asarray(image, dtype=np.float32) / 255.0)[None]
            if pixels.shape[1:3] != (spec.height, spec.width):
                raise ValueError("keyframes must already match the target H3 canvas")
            condition_parts.append(vae.encode_image(pixels, seed=42 + index))
        condition_video = mx.concatenate(condition_parts, axis=2)
        del vae, condition_parts
        gc.collect(); mx.clear_cache()
        print(f"[H3 MLX] FL2VA ({mode}): Visual VAE keyframe encode completed in "
              f"{time.perf_counter() - vae_started:.3f} s; shape={tuple(condition_video.shape)}", flush=True)
        self._reload_transformer_after_conditioning(progress)
        notify("conditioning", 1, 1); interrupt()

        mx.random.seed(spec.seed)
        # Official FL2VA condition augmentation is FP32; keep the clean VAE
        # condition and its noise in that dtype until model patchification.
        condition_noise = mx.random.normal(condition_video.shape).astype(mx.float32)
        video = mx.random.normal((1, self.cfg.in_channels, latent_t,
                                  spec.height // 16, spec.width // 16)).astype(mx.bfloat16)
        audio = mx.random.normal((1, self.cfg.audio_in_channels, 2, audio_t)).astype(mx.bfloat16)
        mx.eval(condition_video, condition_noise, video, audio, context)
        layout = FL2VLayout(context.shape[1], latent_t, spec.height // 16,
                            spec.width // 16, audio_t, anchors)
        timings = []
        last_step = time.perf_counter()
        print(f"[H3 MLX] FL2VA ({mode}): sampling started; steps={spec.steps}, "
              f"target={(spec.width, spec.height, spec.frames)}", flush=True)

        def sampled(step, _sigma, *_):
            nonlocal last_step
            now = time.perf_counter(); elapsed = now - last_step; last_step = now
            timings.append(elapsed)
            print(f"[H3 MLX] FL2VA step {step + 1}/{spec.steps}: {elapsed:.3f} s", flush=True)
            notify("sampling", step + 1, spec.steps); interrupt()

        video, audio = res_multistep_sample(
            self.model, video, audio, context, steps=spec.steps,
            text_token_tags=tags, condition_video=condition_video,
            condition_noise=condition_noise, callback=sampled, layout=layout,
        )
        total = sum(timings)
        print(f"[H3 MLX] FL2VA ({mode}): sampling completed in {total:.3f} s; "
              f"average={total / len(timings):.3f} s/step", flush=True)
        return H3LatentState(
            video, audio, context, tags, spec, tuple(timings),
            condition_video, condition_noise, 0, 1.0, "first",
        )

    def sample_references(
        self,
        spec: H3GenerationSpec,
        images,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3LatentState:
        """Run official image-reference Ref2VA with independent reference grids."""
        self._check_open()
        if self.paths.variant != "ref2va":
            raise RuntimeError("Ref2VA sampling requires the Ref2VA model loader")
        latent_t, audio_t = spec.validate()
        if not 1 <= len(images) <= 9:
            raise ValueError("Ref2VA image mode requires between 1 and 9 reference images")
        notify = progress or (lambda *_: None)
        interrupt = cancel or (lambda: None)
        started = time.perf_counter()
        print(f"[H3 MLX] Ref2VA: conditioning {len(images)} image reference(s)", flush=True)
        interrupt(); notify("conditioning", 0, 1)
        self._unload_transformer_for_conditioning()
        context, tags = encode_keyframe_prompt(self.paths.text_encoder, images, spec.prompt)
        print(f"[H3 MLX] Ref2VA: Qwen3-VL layer-50 completed in "
              f"{time.perf_counter() - started:.3f} s; tokens={context.shape[1]}", flush=True)

        vae_started = time.perf_counter()
        vae = H3VideoVAE()
        load_video_vae_encoder(vae, self.paths.video_vae / "diffusion_pytorch_model.safetensors.index.json")
        conditions = []
        for index, image in enumerate(images):
            pixels = mx.array(np.asarray(image, dtype=np.float32) / 255.0)[None]
            conditions.append(vae.encode_image(pixels, seed=42))
        del vae
        gc.collect(); mx.clear_cache()
        print(f"[H3 MLX] Ref2VA: Visual VAE encoded references in "
              f"{time.perf_counter() - vae_started:.3f} s; "
              f"shapes={[tuple(value.shape) for value in conditions]}", flush=True)
        self._reload_transformer_after_conditioning(progress)
        notify("conditioning", 1, 1); interrupt()

        mx.random.seed(spec.seed)
        noises = tuple(mx.random.normal(value.shape).astype(mx.float32) for value in conditions)
        conditions = tuple(conditions)
        video = mx.random.normal((1, self.cfg.in_channels, latent_t,
                                  spec.height // 16, spec.width // 16)).astype(mx.bfloat16)
        audio = mx.random.normal((1, self.cfg.audio_in_channels, 2, audio_t)).astype(mx.bfloat16)
        layout = Ref2VImageLayout(
            context.shape[1], latent_t, spec.height // 16, spec.width // 16,
            audio_t, tuple(tuple(value.shape[2:5]) for value in conditions),
        )
        mx.eval(video, audio, context, *conditions, *noises)
        timings = []
        last_step = time.perf_counter()
        print(f"[H3 MLX] Ref2VA: sampling started; steps={spec.steps}, "
              f"target={(spec.width, spec.height, spec.frames)}", flush=True)

        def sampled(step, _sigma, *_):
            nonlocal last_step
            now = time.perf_counter(); elapsed = now - last_step; last_step = now
            timings.append(elapsed)
            print(f"[H3 MLX] Ref2VA step {step + 1}/{spec.steps}: {elapsed:.3f} s", flush=True)
            notify("sampling", step + 1, spec.steps); interrupt()

        video, audio = res_multistep_sample(
            self.model, video, audio, context, steps=spec.steps,
            text_token_tags=tags, condition_video=conditions,
            condition_noise=noises, callback=sampled, layout=layout,
        )
        total = sum(timings)
        print(f"[H3 MLX] Ref2VA: sampling completed in {total:.3f} s; "
              f"average={total / len(timings):.3f} s/step", flush=True)
        return H3LatentState(video, audio, context, tags, spec, tuple(timings),
                            conditions, noises, 0, 1.0, "first")

    @staticmethod
    def resize_latent(
        state: H3LatentState,
        scale_factor: float = 1.5,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> H3LatentState:
        """Resize only H/W of BCTHW video latent with MLX linear interpolation."""
        if not isinstance(state, H3LatentState):
            raise TypeError("resize_latent requires an H3LatentState")
        old_h, old_w = state.video.shape[-2:]
        if target_width is None and target_height is None:
            if scale_factor <= 0:
                raise ValueError("scale_factor must be positive")
            new_w, new_h = round(old_w * scale_factor), round(old_h * scale_factor)
        elif target_width is not None and target_height is not None:
            if target_width % 16 or target_height % 16:
                raise ValueError("target pixel width and height must be divisible by 16")
            new_w, new_h = target_width // 16, target_height // 16
        else:
            raise ValueError("set both target_width and target_height, or neither")
        # H3 patchifies latent H/W by 2, so both target latent axes must be even.
        new_h = max(2, math.floor(new_h / 2 + 0.5) * 2)
        new_w = max(2, math.floor(new_w / 2 + 0.5) * 2)
        b, c, t, _, _ = state.video.shape
        frames = state.video.transpose(0, 2, 3, 4, 1).reshape(b * t, old_h, old_w, c)
        # MLX derives output size with integer truncation from scale_factor.
        # Ratios such as 120/52 can round just below the exact quotient and
        # incorrectly produce 119 columns. Move each ratio by one representable
        # float toward +inf, then crop defensively to the requested exact grid.
        scale_h = math.nextafter(new_h / old_h, math.inf)
        scale_w = math.nextafter(new_w / old_w, math.inf)
        resized = nn.Upsample((scale_h, scale_w), mode="linear")(frames)
        if resized.shape[1] < new_h or resized.shape[2] < new_w:
            raise RuntimeError(
                f"MLX Upsample returned {resized.shape[1:3]}, smaller than requested {(new_h, new_w)}"
            )
        resized = resized[:, :new_h, :new_w]
        resized = resized.reshape(b, t, new_h, new_w, c).transpose(0, 4, 1, 2, 3).astype(state.video.dtype)
        mx.eval(resized)
        spec = H3GenerationSpec(state.spec.prompt, new_w * 16, new_h * 16,
                                state.spec.frames, state.spec.steps, state.spec.seed, state.spec.fps)
        print(f"[H3 MLX] Latent resize: {tuple(state.video.shape)} -> {tuple(resized.shape)} BCTHW", flush=True)
        return H3LatentState(
            resized, state.audio, state.context, state.text_token_tags, spec,
            state.step_seconds, state.condition_video, state.condition_noise,
            state.scheduler_index, state.start_sigma, state.pass_name,
        )

    def resume(
        self,
        state: H3LatentState,
        denoise_strength: float,
        steps: int | None = None,
        seed: int | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3LatentState:
        """Second-pass parity has not been established and is intentionally unsupported."""
        raise RuntimeError(
            "H3 MLX second-pass sampling is unsupported until its official ComfyUI "
            "sampling semantics have passed numerical parity."
        )

    def generate(
        self,
        spec: H3GenerationSpec,
        mp4_path: str | Path,
        diagnostics_path: str | Path,
        temp_directory: str | Path,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3GenerationResult:
        self._check_open()
        latent_t, audio_t = spec.validate()
        mp4_path, diagnostics_path = Path(mp4_path), Path(diagnostics_path)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = resolve_ffmpeg()

        notify = progress or (lambda *_: None)
        interrupt = cancel or (lambda: None)
        interrupt()
        notify("conditioning", 0, 1)
        context, tags = self._encode_prompt(spec.prompt)
        notify("conditioning", 1, 1)
        interrupt()

        mx.random.seed(spec.seed)
        video = mx.random.normal((1, self.cfg.in_channels, latent_t,
                                  spec.height // 16, spec.width // 16)).astype(mx.bfloat16)
        audio = mx.random.normal((1, self.cfg.audio_in_channels, 2, audio_t)).astype(mx.bfloat16)
        mx.eval(video, audio, context)
        step_seconds: list[float] = []
        last_step = time.perf_counter()

        def sampled(step, _sigma, *_):
            nonlocal last_step
            now = time.perf_counter()
            elapsed = now - last_step
            step_seconds.append(elapsed)
            last_step = now
            print(f"[H3 MLX] Sampling step {step + 1}/{spec.steps}: {elapsed:.3f} s", flush=True)
            notify("sampling", step + 1, spec.steps)
            interrupt()

        video, audio = res_multistep_sample(
            self.model, video, audio, context, steps=spec.steps,
            text_token_tags=tags, callback=sampled,
        )
        del context, tags
        gc.collect()
        mx.clear_cache()

        notify("decoding_audio", 0, 1)
        audio_vae = H3AudioVAE.from_config(self.paths.audio_vae / "config.json")
        load_audio_vae_decoder(audio_vae, self.paths.audio_vae / "model.safetensors")
        waveform = audio_vae.decode(audio).astype(mx.float32)
        mx.eval(waveform)
        pcm = np.clip(np.asarray(waveform[0]).T, -1.0, 1.0)
        pcm = (pcm * 32767.0).round().astype("<i2")
        del audio_vae, audio, waveform
        gc.collect()
        mx.clear_cache()
        notify("decoding_audio", 1, 1)
        interrupt()

        temp_directory = Path(temp_directory)
        temp_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_directory, delete=False) as temp_wav:
            wav_path = Path(temp_wav.name)
        try:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(32000)
                wav.writeframes(pcm.tobytes())
            del pcm

            notify("decoding_video", 0, 1)
            video_vae = H3VideoVAE()
            load_video_vae_decoder(video_vae, self.paths.video_vae / "diffusion_pytorch_model.safetensors.index.json")
            frames = video_vae.decode(video)
            mx.eval(frames)
            del video_vae, video
            gc.collect()
            mx.clear_cache()
            notify("decoding_video", 1, 1)
            interrupt()

            notify("encoding_mp4", 0, 1)
            command = [
                str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{spec.width}x{spec.height}",
                "-r", str(spec.fps), "-i", "-", "-i", str(wav_path),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(mp4_path),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for index in range(frames.shape[2]):
                    interrupt()
                    frame = (np.asarray(frames[0, :, index]).transpose(1, 2, 0) * 255.0).round().astype("uint8")
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("ffmpeg failed while encoding the H3 MP4.")
            except BaseException:
                process.kill()
                raise
            finally:
                del frames
                gc.collect()
                mx.clear_cache()
            notify("encoding_mp4", 1, 1)
        finally:
            wav_path.unlink(missing_ok=True)

        diagnostics = {
            "backend": self.backend,
            "dtype": "bfloat16", "quantization": "none", "lora": "none",
            "sampler": "res_multistep", "scheduler": "beta",
            "attention_backend": "mlx_fused_full_sdpa",
            "mlx": mx.__version__,
            "prompt": spec.prompt,
            "prompt_sha256": hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest(),
            "transformer_checkpoint": str(self.paths.transformer),
            "width": spec.width, "height": spec.height, "frames": spec.frames,
            "fps": spec.fps, "steps": spec.steps, "seed": spec.seed,
            "step_seconds": step_seconds,
            "hot_loop": {"torch_to_mlx": 0, "mlx_to_numpy": 0,
                         "cpu_round_trips": 0, "device_round_trips": 0,
                         "head_major_sdpa": True,
                         "layout_materializations_per_step": 3 * self.cfg.num_layers},
            "output": {"mp4": str(mp4_path), "audio_sample_rate": 32000,
                       "audio_channels": 2},
        }
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
        return H3GenerationResult(
            mp4_path, diagnostics_path, spec.width, spec.height, spec.frames,
            spec.fps, 32000, 2, spec.frames / spec.fps,
            tuple(step_seconds), self.backend,
        )

    def decode(
        self,
        state: H3LatentState,
        mp4_path: str | Path,
        diagnostics_path: str | Path,
        temp_directory: str | Path,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> H3GenerationResult:
        """Decode an exposed first- or second-pass state and mux standard VIDEO."""
        self._check_open()
        if not isinstance(state, H3LatentState):
            raise TypeError("decode requires an H3LatentState")
        spec = state.spec
        spec.validate()
        if state.video.shape != (1, self.cfg.in_channels, spec.validate()[0],
                                  spec.height // 16, spec.width // 16):
            raise ValueError("latent video shape and generation metadata disagree")
        mp4_path, diagnostics_path = Path(mp4_path), Path(diagnostics_path)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        notify = progress or (lambda *_: None)
        interrupt = cancel or (lambda: None)
        ffmpeg = resolve_ffmpeg()

        notify("decoding_audio", 0, 1); interrupt()
        audio_vae = H3AudioVAE.from_config(self.paths.audio_vae / "config.json")
        load_audio_vae_decoder(audio_vae, self.paths.audio_vae / "model.safetensors")
        waveform = audio_vae.decode(state.audio).astype(mx.float32)
        mx.eval(waveform)
        pcm = np.clip(np.asarray(waveform[0]).T, -1.0, 1.0)
        pcm = (pcm * 32767.0).round().astype("<i2")
        del audio_vae, waveform
        gc.collect(); mx.clear_cache()
        notify("decoding_audio", 1, 1); interrupt()

        temp_directory = Path(temp_directory)
        temp_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_directory, delete=False) as temp_wav:
            wav_path = Path(temp_wav.name)
        try:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(32000)
                wav.writeframes(pcm.tobytes())
            del pcm
            notify("decoding_video", 0, 1)
            video_vae = H3VideoVAE()
            load_video_vae_decoder(video_vae, self.paths.video_vae / "diffusion_pytorch_model.safetensors.index.json")
            frames = video_vae.decode(state.video)
            mx.eval(frames)
            del video_vae
            gc.collect(); mx.clear_cache()
            notify("decoding_video", 1, 1); interrupt()

            notify("encoding_mp4", 0, 1)
            command = [
                str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{spec.width}x{spec.height}",
                "-r", str(spec.fps), "-i", "-", "-i", str(wav_path),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(mp4_path),
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for index in range(frames.shape[2]):
                    interrupt()
                    frame = (np.asarray(frames[0, :, index]).transpose(1, 2, 0) * 255.0).round().astype("uint8")
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("ffmpeg failed while encoding the H3 MP4.")
            except BaseException:
                process.kill(); raise
            finally:
                del frames
                gc.collect(); mx.clear_cache()
            notify("encoding_mp4", 1, 1)
        finally:
            wav_path.unlink(missing_ok=True)

        diagnostics = {
            "backend": self.backend, "mlx": mx.__version__, "prompt": spec.prompt,
            "dtype": "bfloat16", "quantization": "none", "lora": "none",
            "sampler": "res_multistep", "scheduler": "beta",
            "attention_backend": "mlx_fused_full_sdpa",
            "prompt_sha256": hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest(),
            "transformer_checkpoint": str(self.paths.transformer),
            "width": spec.width, "height": spec.height, "frames": spec.frames,
            "fps": spec.fps, "steps": spec.steps, "seed": spec.seed,
            "pass": state.pass_name, "latent_layout": state.layout,
            "latent_shape": list(state.shape), "latent_dtype": str(state.video.dtype),
            "scheduler_index": state.scheduler_index, "start_sigma": state.start_sigma,
            "step_seconds": list(state.step_seconds),
            "output": {"mp4": str(mp4_path), "audio_sample_rate": 32000, "audio_channels": 2},
        }
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
        return H3GenerationResult(
            mp4_path, diagnostics_path, spec.width, spec.height, spec.frames,
            spec.fps, 32000, 2, spec.frames / spec.fps,
            state.step_seconds, self.backend,
        )
