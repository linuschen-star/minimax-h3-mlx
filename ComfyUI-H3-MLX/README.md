# ComfyUI-H3-MLX

MiniMax H3 Base 的原生 MLX ComfyUI production integration。大型 MLX tensor 始終保留在 opaque runtime／latent 物件內，只有最終影音輸出會轉成可編碼資料。

## 固定的 production 設定

- Base BF16 checkpoint
- 無量化、無 Turbo LoRA、無 silent fallback
- `beta` scheduler
- `res_multistep` sampler
- 官方 `ModelSamplingAV` common schedule 與 audio scale
- MLX fused full SDPA
- 使用者 prompt 原文傳入，不改寫、不拆欄、不套 chat template
- token-refiner 與 main DiT QKV 都使用已對 production ComfyUI checkpoint 驗證的 per-head row conversion
- Qwen 使用 `attention_mask=None`，由模型建立 causal mask
- Qwen 圖片 normalization 為 mean/std `0.5`，bilinear resize 使用 `align_corners=False`

其他 dtype、量化、LoRA、sampler、scheduler 或 attention backend 會直接報錯，不會改走其他路徑。

## 安裝

將本目錄安裝為：

```text
ComfyUI/custom_nodes/ComfyUI-H3-MLX
```

並在 ComfyUI 的 Python 環境安裝 runtime：

```bash
python -m pip install -e /absolute/path/to/minimax_h3_mlx
```

模型 bundle：

```text
ComfyUI/models/minimax_h3/MiniMax-H3/
├── FL2VA/
│   ├── transformer/
│   ├── text_encoder/
│   └── audio_vae/
├── Ref2VA/
│   └── transformer/
└── vae/
```

模型目錄可使用 symlink 或 ComfyUI 的額外 `minimax_h3` 搜尋路徑。

## Production workflows

### T2V

`workflows/h3_mlx_t2v.json`

純文字工作流。使用者不需要提供圖片；節點內部使用目前官方 ComfyUI H3 路徑需要的 first＋last compatibility frames。內建資產是已驗證的 `GrayBackGround.png`，SHA-256：

```text
d19cc49c86929f50691073662fdd21bd9c98243cdc0b633fd4836ddb53d7dbe5
```

輸出為 H.264 MP4，包含模型同步產生的 32 kHz stereo AAC 音訊。

### I2V / FL2V

`workflows/h3_mlx_i2v.json`

- 只連 first frame：I2V
- 只連 last frame：L2V
- first 與 last 都連接：FL2V
- last frame 不是必填

圖片經官方 Qwen3-VL Picture-token presentation 與 native MLX Visual VAE encoder。Prompt 保持原文。

### R2V / Ref2VA

`workflows/h3_mlx_r2v.json`

目前 production implementation 支援 1–9 張 reference images，依 `<Picture 1>` 至 `<Picture 9>` 的輸入順序 conditioning。`match` 只在必要時縮到輸出畫布面積；`max` 只在必要時限制至 2048px 短邊，兩者都不放大來源。

目前不支援 native MLX reference-video、reference-audio 或任意 guide conditioning。這些組合不會被替換成 I2V、T2V 或 image-only Ref2VA；API 傳入未支援 reference 類型時會明確報錯。這不影響最終生成影片內由 H3 同步產生的音訊。

## 已驗證基準

T2V correctness baseline：

```text
832×480
124 frames / 24 fps
20 steps
seed 20260823
Base BF16
beta + res_multistep
official compatibility first/last frames
```

結果為可辨識且連續的戶外武術格鬥；沒有 mosaic、灰黃假標誌、唱歌／跳舞或災難性語意錯誤。產物與其 diagnostics 記錄在 `VALIDATION.md`。

## 限制

- Apple Silicon；目前 production baseline 為 M4 Max 128 GB。
- 尺寸必須能被 32 整除。
- 幀數必須符合 `17*n + 5`；固定 24 fps。
- 目前只交付 T2V、I2V/FL2V 與 image-only R2V。
- 二次採樣、2K regeneration、SageAttention、量化、Turbo 與任意 LoRA 不屬於本版 production workflow。
- 高解析度與長片會因 full attention 顯著增加時間與 unified memory；480p/124-frame 是已驗證基準。

## 測試

```bash
cd /absolute/path/to/minimax_h3_mlx
.venv/bin/python -m pytest -q tests
```
