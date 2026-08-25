# MiniMax H3 native MLX runtime

MiniMax H3 Base 的 Apple Silicon 原生 MLX inference runtime。這是官方 MiniMax H3／ComfyUI 行為的實作層 port，不是重新設計。

## Production contract

本版固定支援：

- Base BF16
- 50 層 FL2VA／Ref2VA transformer
- token-refiner 與 main DiT 的 production-checkpoint QKV row conversion
- `beta` scheduler
- `res_multistep` sampler
- 官方 `ModelSamplingAV` common schedule、audio timestep conversion 與 output scale
- Qwen layer-50 conditioning，`attention_mask=None`
- Qwen image mean/std `0.5` 與 bilinear `align_corners=False`
- 原始 prompt，無 rewrite、template 或自動 mode 推斷
- MLX fused full SDPA
- Visual VAE、Audio VAE 與同步 audiovisual MP4

不支援的 dtype、量化、LoRA、sampler、scheduler 或 attention backend 必須在 UI 邊界明確失敗。Production 不包含 INT8/FP8、Turbo、SageAttention、稀疏／近似 attention 或 silent fallback。

## 支援模式

- T2V：透過官方 ComfyUI 相容 first＋last gray frames 進入 FL2VA presentation；使用者只輸入文字。
- I2V／L2V／FL2V：支援 optional first frame、optional last frame，至少一個真實 keyframe 時使用對應 endpoint anchor。
- R2V：Ref2VA image mode，支援 1–9 張獨立比例圖片。

目前不支援 reference video、reference audio、任意 guide、二次採樣或 2K regeneration。Runtime 不會把這些模式替換成較簡單路徑。

## 已凍結的三項關鍵修正

1. `token_refiner.blocks.*.attn.qkv_proj.weight` 使用 per-head `[q,k,v]` 到 `[all-q,all-k,all-v]` row conversion。
2. `blocks.*.attn.qkv_proj.weight` 使用同一個、已對 production ComfyUI single-file checkpoint 驗證 bit-exact 的 conversion。
3. Qwen semantic path 傳入 `attention_mask=None`，由 Qwen 建立 causal attention mask。

不要因小幅 tensor parity 差異修改這些映射、vision packing、RoPE、SDPA 或 dtype policy；只有 production workflow 實際失敗時才重新開啟調查。

## 480p correctness baseline

```text
832×480 / 124 frames / 24 fps / 20 steps
Base BF16 / no LoRA / no quantization
beta / res_multistep / MLX fused full SDPA
seed 20260823
official first+last compatibility frames
```

該測試正確生成兩名武術家在明亮戶外場地持續格鬥，影音完整；沒有 mosaic、fake logo、唱歌／跳舞或災難性語意偏離。詳見相鄰 ComfyUI integration 的 `VALIDATION.md`。

## 安裝與測試

```bash
cd /absolute/path/to/minimax_h3_mlx
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q tests
```

模型 bundle 應包含 `FL2VA/transformer`、`FL2VA/text_encoder`、`FL2VA/audio_vae`、`Ref2VA/transformer` 與 `vae`。完整 ComfyUI 使用方式與 workflow 路徑見 `ComfyUI-H3-MLX/README.md`。
