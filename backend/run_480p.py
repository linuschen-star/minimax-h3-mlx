#!/usr/bin/env python3
"""Minimal native-MLX H3 864x480 / 124-frame (~5 s) latent pipeline.

Conditioning is supplied as an MLX-readable safetensors file so this runner can
validate the H3 hot path before the Qwen3-VL port is complete.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.audio_vae import H3AudioVAE, load_audio_vae_decoder
from minimax_h3_mlx.text_encoder import H3TextEncoder, TextConfig, load_text_encoder, tokenize_prompt
from minimax_h3_mlx.model import H3Config, MiniMaxH3Transformer, load_sharded_safetensors
from minimax_h3_mlx.sampling import euler_sample
from minimax_h3_mlx.video_vae import H3VideoVAE, load_video_vae_decoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transformer-dir", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--conditioning", type=Path,
                   help="safetensors with context [1,L,5120 or 5376], optional text_token_tags [L]")
    source.add_argument("--prompt", type=str, help="Encode a verbatim text prompt with native MLX Qwen3-VL")
    p.add_argument("--text-encoder-dir", type=Path,
                   help="Required with --prompt; official FL2VA/text_encoder directory")
    p.add_argument("--steps", type=int, required=True,
                   help="Number of transformer evaluations (use 2 while developing, 6-8 for short quality checks)")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("h3_480p_latents.safetensors"))
    p.add_argument("--vae-dir", type=Path, help="Optional diffusers vae directory; decodes RGB frames")
    p.add_argument("--audio-vae-dir", type=Path,
                   help="Optional official audio_vae directory; decodes a 32 kHz stereo WAV")
    p.add_argument("--mp4-output", type=Path,
                   help="Mux decoded RGB and stereo audio with ffmpeg; requires both VAE options")
    args = p.parse_args()
    if args.width % 32 or args.height % 32:
        p.error("--width and --height must be divisible by 32")
    if args.mp4_output and (not args.vae_dir or not args.audio_vae_dir):
        p.error("--mp4-output requires --vae-dir and --audio-vae-dir")

    cfg = H3Config.from_json(args.transformer_dir / "config.json")
    if args.prompt is not None:
        if args.text_encoder_dir is None:
            p.error("--text-encoder-dir is required with --prompt")
        text_cfg = TextConfig.from_json(args.text_encoder_dir / "config.json")
        text_encoder = H3TextEncoder(text_cfg, output_layer=50)
        load_text_encoder(text_encoder, args.text_encoder_dir / "model.safetensors.index.json")
        input_ids = tokenize_prompt(args.text_encoder_dir / "tokenizer.json", args.prompt)
        context = text_encoder(input_ids).astype(mx.bfloat16)
        mx.eval(context)
        tags = [1] * context.shape[1]  # Official H3 modality tag: video=0, text=1, audio=2.
        print(f"encoded {context.shape[1]} prompt tokens with Qwen3-VL hidden_states[50]")
        del text_encoder
        gc.collect()
        mx.clear_cache()
    else:
        cond = mx.load(str(args.conditioning))
        context = cond["context"].astype(mx.bfloat16)
        condition_video = cond.get("condition_video")
        condition_noise = cond.get("condition_noise")
        tags = None
        if "text_token_tags" in cond:
            tags = [int(v) for v in cond["text_token_tags"].tolist()]
    if args.prompt is not None:
        condition_video = condition_noise = None

    # Load the 33B denoiser only after the one-shot text encoder has been freed.
    model = MiniMaxH3Transformer(cfg)
    load_sharded_safetensors(model, args.transformer_dir / "model.safetensors.index.json")

    # 124 frames is the official ~5 s 17k+5 grid.
    mx.random.seed(args.seed)
    video = mx.random.normal((1, 24, 37, args.height // 16, args.width // 16)).astype(mx.bfloat16)
    audio = mx.random.normal((1, 32, 2, 207)).astype(mx.bfloat16)
    # The shard loader already materializes each weight once. Evaluating the
    # entire 33B parameter tree here forces a redundant full-device pass before
    # the first forward; only the newly-created sampler state needs evaluation.
    mx.eval(video, audio, context)

    started = time.perf_counter()
    def progress(step, sigma, *_):
        print(f"step {step + 1:>3}/{args.steps} sigma={sigma:.6f}", flush=True)
    video, audio = euler_sample(
        model, video, audio, context, num_steps=args.steps + 1,
        text_token_tags=tags, condition_video=condition_video,
        condition_noise=condition_noise, callback=progress,
    )
    print(f"sampling: {time.perf_counter() - started:.1f}s")
    mx.save_safetensors(str(args.output), {"video": video, "audio": audio})
    print(f"saved {args.output}")
    del model
    gc.collect()
    mx.clear_cache()

    if args.vae_dir:
        vae = H3VideoVAE()
        load_video_vae_decoder(vae, args.vae_dir / "diffusion_pytorch_model.safetensors.index.json")
        frames = vae.decode(video)
        mx.eval(frames)
        frame_path = args.output.with_name(args.output.stem + "_frames.safetensors")
        mx.save_safetensors(str(frame_path), {"frames": frames})
        print(f"saved decoded [B,C,T,H,W] RGB frames to {frame_path}")
        del vae
        gc.collect()
        mx.clear_cache()

    if args.audio_vae_dir:
        audio_vae = H3AudioVAE.from_config(args.audio_vae_dir / "config.json")
        load_audio_vae_decoder(audio_vae, args.audio_vae_dir / "model.safetensors")
        waveform = audio_vae.decode(audio).astype(mx.float32)
        mx.eval(waveform)
        pcm = np.clip(np.asarray(waveform[0]).T, -1.0, 1.0)
        pcm = (pcm * 32767.0).round().astype("<i2")
        wav_path = args.output.with_suffix(".wav")
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(audio_vae.sample_rate)
            wav.writeframes(pcm.tobytes())
        print(f"saved {wav_path} ({waveform.shape[-1] / audio_vae.sample_rate:.3f}s stereo)")

    if args.mp4_output:
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", "24", "-i", "-",
            "-i", str(wav_path), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.mp4_output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for index in range(frames.shape[2]):
                frame = (np.asarray(frames[0, :, index]).transpose(1, 2, 0) * 255.0).round().astype("uint8")
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg mux failed")
        except Exception:
            process.kill()
            raise
        print(f"saved muxed H.264/AAC video to {args.mp4_output}")


if __name__ == "__main__":
    main()
