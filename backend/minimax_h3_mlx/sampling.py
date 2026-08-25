"""Official ComfyUI H3 flow schedule and RES multistep sampling semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx
import numpy as np
from scipy.stats import beta as beta_distribution

from .model import FL2VLayout, MiniMaxH3Transformer, T2VLayout, time_shift_sigma_float


VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def shifted_sigma(base_sigma: float, shift: float) -> float:
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def beta_sigma_schedule(
    steps: int,
    shift: float = VIDEO_SHIFT,
    *,
    alpha: float = 0.6,
    beta: float = 0.6,
    timesteps: int = 1000,
) -> list[float]:
    """Port of ``comfy.samplers.beta_scheduler`` for discrete-flow sampling."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    base_grid = np.arange(1, timesteps + 1, dtype=np.float32) / np.float32(timesteps)
    shift32 = np.float32(shift)
    model_sigmas = shift32 * base_grid / (
        np.float32(1.0) + (shift32 - np.float32(1.0)) * base_grid
    )
    ts = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False)
    ts = np.rint(beta_distribution.ppf(ts, alpha, beta) * (timesteps - 1))
    sigmas: list[float] = []
    last_t = -1
    for timestep in ts:
        index = int(timestep)
        if index != last_t:
            sigmas.append(float(model_sigmas[index]))
        last_t = index
    sigmas.append(0.0)
    return sigmas


def sigma_schedule(num_steps: int, shift: float = VIDEO_SHIFT) -> list[float]:
    """Compatibility wrapper: ``num_steps`` includes the terminal zero point."""
    if num_steps < 2:
        raise ValueError("num_steps must include at least one evaluation and terminal zero")
    return beta_sigma_schedule(num_steps - 1, shift)


def _audio_model_input_and_carried_velocity(
    carried_audio: mx.array,
    own_velocity: mx.array,
    sigma_video: float,
    shift_video: float,
    shift_audio: float,
) -> tuple[mx.array, mx.array, float]:
    """Port MiniMaxH3.forward's ModelSamplingAV carry conversion."""
    sigma_audio = time_shift_sigma_float(sigma_video, shift_video, shift_audio)
    carry = sigma_audio / sigma_video
    audio_scale = shift_video / shift_audio
    model_audio = carried_audio * carry
    carried_velocity = (
        (1.0 - audio_scale) * model_audio
        + (1.0 + (audio_scale - 1.0) * sigma_audio) * own_velocity
    )
    return model_audio, carried_velocity, sigma_audio


def res_multistep_sample(
    model: MiniMaxH3Transformer,
    video: mx.array,
    audio: mx.array,
    context: mx.array,
    *,
    steps: int,
    text_token_tags: Sequence[int] | None = None,
    condition_video: mx.array | Sequence[mx.array] | None = None,
    condition_noise: mx.array | Sequence[mx.array] | None = None,
    callback: Callable[[int, float, mx.array, mx.array], None] | None = None,
    sigma_video: Sequence[float] | None = None,
    layout: T2VLayout | FL2VLayout | None = None,
) -> tuple[mx.array, mx.array]:
    """Exact non-ancestral ``res_multistep`` with H3 ``ModelSamplingAV`` carry.

    ``audio`` enters as the common-schedule carried variable. For a normal run
    at sigma=1 this is ordinary unit noise. The returned audio is converted
    through official ``process_latent_out`` and is ready for the audio VAE.
    """
    cfg = model.cfg
    if layout is None:
        layout_cls = FL2VLayout if condition_video is not None else T2VLayout
        layout = layout_cls(context.shape[1], video.shape[2], video.shape[3], video.shape[4], audio.shape[-1])
    sigmas = list(sigma_video) if sigma_video is not None else beta_sigma_schedule(steps, cfg.sigma_shift_video)
    if len(sigmas) < 2:
        raise ValueError("sigma_video must contain at least two schedule points")
    if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("sigma_video must be strictly decreasing")

    audio_scale = cfg.sigma_shift_video / cfg.sigma_shift_audio
    old_sigma_down: float | None = None
    old_denoised_video: mx.array | None = None
    old_denoised_audio: mx.array | None = None

    for step, (sigma, sigma_down) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        model_audio, _, _ = _audio_model_input_and_carried_velocity(
            audio, mx.zeros_like(audio), sigma,
            cfg.sigma_shift_video, cfg.sigma_shift_audio,
        )
        velocity_video, velocity_audio_own = model(
            video,
            model_audio,
            context,
            sigma,
            layout=layout,
            text_token_tags=text_token_tags,
            condition_video=condition_video,
            condition_noise=condition_noise,
        )
        _, velocity_audio, _ = _audio_model_input_and_carried_velocity(
            audio, velocity_audio_own, sigma,
            cfg.sigma_shift_video, cfg.sigma_shift_audio,
        )
        denoised_video = video - velocity_video * sigma
        denoised_audio = audio - velocity_audio * sigma

        if sigma_down == 0.0 or old_denoised_video is None:
            dt = sigma_down - sigma
            video = video + ((video - denoised_video) / sigma) * dt
            audio = audio + ((audio - denoised_audio) / sigma) * dt
        else:
            t = -np.log(sigma)
            t_old = -np.log(old_sigma_down)
            t_next = -np.log(sigma_down)
            t_prev = -np.log(sigmas[step - 1])
            h = float(t_next - t)
            c2 = float((t_prev - t_old) / h)
            phi1 = np.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            b1 = float(np.nan_to_num(phi1 - phi2 / c2, nan=0.0))
            b2 = float(np.nan_to_num(phi2 / c2, nan=0.0))
            decay = float(np.exp(-h))
            video = decay * video + h * (b1 * denoised_video + b2 * old_denoised_video)
            audio = decay * audio + h * (b1 * denoised_audio + b2 * old_denoised_audio)

        old_denoised_video = denoised_video
        old_denoised_audio = denoised_audio
        old_sigma_down = sigma_down
        mx.eval(video, audio)
        if callback is not None:
            callback(step, sigma, video, audio / audio_scale)

    return video, audio / audio_scale


def resume_schedule(total_steps: int, denoise_strength: float, shift: float = VIDEO_SHIFT):
    """Legacy second-pass schedule; second-pass parity is outside this pass."""
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1")
    if not 0.0 <= denoise_strength <= 1.0:
        raise ValueError("denoise_strength must be between 0 and 1")
    if denoise_strength == 0.0:
        return [0.0], total_steps, 0
    base_start = denoise_strength / (shift + denoise_strength * (1.0 - shift))
    grid = [shifted_sigma(base_start * i / total_steps, shift) for i in range(total_steps, -1, -1)]
    return grid, 0, total_steps


def flow_add_noise(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
    """Official Comfy H3/CONST flow interpolation at a schedule sigma."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise shapes must match")
    if not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be between 0 and 1")
    return sigma * noise + (1.0 - sigma) * clean
