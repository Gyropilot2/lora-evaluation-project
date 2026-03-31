# ComfyUI Input Type Field Catalog

## Purpose

This document records **what each ComfyUI input type actually contains at runtime** - its Python class, keys/attributes, tensor shapes, dtypes, and value properties - discovered empirically via the lab probe's `_raw_inputs` section.

This is **not** a description of what the probe outputs. That is `.docs/lab/LAB_FIELD_STATUS.md`.
This is a description of the **raw inputs** that arrive at the probe node, before any extraction.

**Why this matters:** trusting the runtime object is always better than asking the user to supply a string ("I think this is 4D float32"). The lab probe reads the truth directly.

## How to read this document

- **Always present**: field exists in every observed run under this model architecture.
- **Conditional**: field may or may not be present depending on ComfyUI setup, model type, or user choices.
- **Architecture notes**: differences observed between Flux, SDXL, SD1.5.

## Last updated from

- Probe version: **v0.9.6**
- Source dump: `2026-03-02T02-01-04_3105e5a8.json`
- Model: Flux Schnell Q8 GGUF + Core Physics LoRA
- CLIP Vision: SigLIP (image_size=512) + ViT-L (image_size=224)
- Image: 512x512; seed=0; lora_strength=1.0; 7 faces detected

---

## MODEL  (`ModelPatcher` or `GGUFModelPatcher`)

**Python type:** `comfy.model_patcher.ModelPatcher` (standard) or `comfy_gguf.GGUFModelPatcher` (GGUF models)

**Patcher class note:** GGUF quantized models (`.gguf` files) use `GGUFModelPatcher`, not `ModelPatcher`. Confirmed from v0.9.4 run with `flux1-schnell-q8_0.gguf`.

**Top-level attributes observed:**

| Attribute | Type | Always present | Notes |
|-----------|------|---------------|-------|
| `model` | `comfy.model_base.*` (e.g. `Flux`, `SDXL`) | Yes | Inner diffusion model |
| `patches` | `dict[str, list]` | Yes | Empty dict for baseline; populated for LoRA |
| `model_options` | `dict` | Yes | `{"transformer_options": {}}` - usually sparse |
| `model_options["transformer_options"]` | `dict` | Yes | Often empty; populated by model-modifying nodes (ModelSamplingFlux, FreeU, etc.) |
| `load_device` | `torch.device` | Yes | GPU/CPU the model is on |
| `offload_device` | `torch.device` | Yes | Where model is offloaded when idle |

**`model.model_config` attributes (inner diffusion config):**

| Attribute | Type | Observed value (Flux Schnell) | Notes |
|-----------|------|------|-------|
| `unet_config` | `dict` | `{"image_model": "flux", "in_channels": 16, ...}` | Full config dict; architecture-specific |
| `latent_format` | object | `comfy.latent_formats.Flux` | Determines VAE encode/decode space; class name distinguishes architectures |
| `model_type` | `ModelType` enum | `ModelType.FLOW` | Flow-matching vs diffusion |
| `adm_channels` | `int` | `0` (Flux), `2816` (SDXL) | ADM/pooled text conditioning channels |
| `manual_cast_dtype` | `dtype \| None` | `None` | Override for mixed-precision |
| `supported_inference_dtypes` | `list` | `[torch.bfloat16, torch.float16, ...]` | Conditional |

**`model.diffusion_model`:** the actual neural network (`nn.Module`). Has `named_parameters()`. Used for base model hash (structural + value anchor).

**Base model name:** accessible from the pipeline - confirmed `"flux1-schnell-q8_0.gguf"`. Available via `folder_paths` lookup on the file hash or from the patcher's load path.

**LoRA patches structure (confirmed v0.9.4):**

`patches[layer_key]` is a list; each item is a tuple of 5 elements:
```
(apply_fn,         # index 0: callable - how to apply the patch
 LoRAAdapter,      # index 1: the adapter object with lora_up/lora_down/alpha
 strength_float,   # index 2: float - lora_strength at apply time
 None,             # index 3
 None)             # index 4
```

`LoRAAdapter` fields: `lora_up_shape`, `lora_down_shape`, `alpha`, `has_mid`, `has_dora_scale`.

**Flux Schnell + Core Physics LoRA:** 304 patches total. `double_blocks`: 190, `single_blocks`: 114. `img_attn`: 38, `txt_attn`: 38. Rank 384, alpha 128.0.

**Probe functions that use MODEL:**
- `_extract_model_pipeline()` - full pipeline dissection
- `_compute_base_model_content_hash()` - structural + value anchor hash

---

## LATENT  (`dict`)

**Python type:** `dict`

**Keys observed (Flux Schnell, 512x512):**

| Key | Type | Shape | Always present | Notes |
|-----|------|-------|---------------|-------|
| `samples` | `torch.Tensor` | `[1, 16, 64, 64]` (Flux) / `[1, 4, 64, 64]` (SD) | Yes | The latent tensor; `[B, C, H/8, W/8]` |
| `noise_mask` | `torch.Tensor` | `[1, 1, H/8, W/8]` | Conditional | Present only when inpainting mask is connected |
| `batch_index` | `list[int]` | - | Conditional | Present when batch processing specific frames |

**`samples` tensor properties (Flux Schnell 512x512, confirmed v0.9.4):**
- Shape: `[1, 16, 64, 64]` - batch=1, 16 latent channels (Flux), 64x64 latent spatial
- Dtype: `torch.float32`
- Device: `cpu` (returned from KSampler)
- Value range: not normalized to [0,1] (raw latent space); observed min~-10, max~7.8

**SD1.5/SDXL difference:** `samples` shape is `[1, 4, H/8, W/8]` (4 channels vs Flux's 16).

**Latent channel count** (`samples.shape[1]`) distinguishes architectures:
- 4 channels: SD1.5, SDXL
- 16 channels: Flux, SD3 (same channel count - further distinction via base_model.hash)

**No separate "type" field:** The `latent_format` class name on the model (e.g. "Flux", "SD15") identifies which empty latent node to use, but this is derived from MODEL not LATENT. The LATENT dict itself carries no architecture label.

**`noise_mask` and `batch_index`:** confirmed absent in standard generation run (v0.9.4). Conditional for inpainting/batch workflows.

**Probe functions that use LATENT:**
- `_extract_latent()` - shape, dtype, stats, content hash (from `samples` only)
- `_inspect_latent_raw()` [v0.9.4+] - all keys + full per-key metadata

---

## CONDITIONING  (`list[tuple[Tensor, dict]]`)

**Python type:** `list` of `(context_tensor, metadata_dict)` tuples

**Structure:**

```
positive = [
    (context_tensor,  # torch.Tensor - text embedding from CLIP/T5
     metadata_dict),  # dict - extra data per conditioning pair
    ...  # additional pairs for area conditioning, etc.
]
```

**`context_tensor` properties (Flux Schnell, confirmed v0.9.4):**
- Shape: `[1, 256, 4096]` - batch=1, 256 tokens, 4096-dim T5 XXL embedding
- Dtype: `torch.float32`
- Device: `cpu`
- SD1.5: `[1, 77, 768]`; SDXL: `[1, 77, 2048]` (dual CLIP)

**`metadata_dict` keys - Flux (confirmed v0.9.4):**

| Key | Type | Value (Flux Schnell) | Always present | Notes |
|-----|------|------|---------------|-------|
| `pooled_output` | `torch.Tensor` | shape `[1, 768]` f32 | Flux/SDXL only | Pooled CLIP embedding; absent in SD1.5 |
| `guidance` | `float` | `3.5` | Flux only | Guidance distillation value |

**Flux meta keys confirmed absent:** `cross_attn_kwargs`, `area`, `mask`, `set_area_to_bounds`, `strength` - these are SD1.x/SDXL-specific (regional conditioning, ControlNet masking, etc.). The `meta_all_keys` for this run was exactly `["pooled_output", "guidance"]`.

**Negative conditioning structure:** Same tensor structure as positive even when "empty". Confirmed: negative CONDITIONING has a real `[1, 256, 4096]` context tensor even when the user used an empty or zero-text prompt. `negative_text` is null in the dump because it came from an empty string node input, but the tensor exists.

**Architecture notes:**
- Flux: `guidance` key is populated (guidance distillation). SD1.5: absent.
- SDXL: `pooled_output` present from both CLIP-G and CLIP-L encoders.
- Area/regional conditioning: adds more pairs to the list with `area`, `mask`, `strength` keys - not used in this run.

**Probe functions that use CONDITIONING:**
- `_extract_conditioning()` - pair count, first tensor stats, pooled_output, guidance, full tensor hash
- `_inspect_conditioning_raw()` [v0.9.4+] - all meta keys + full type/value inspection per key

---

## VAE

**Python type:** `comfy.sd.VAE`

**Top-level attributes (confirmed v0.9.4):**

| Attribute | Type | Observed | Notes |
|-----------|------|----------|-------|
| `first_stage_model` | `nn.Module` | Yes | The actual VAE network (encoder + decoder) |
| `first_stage_model` class | `AutoencodingEngine` | Yes | Confirmed v0.9.4 for Flux VAE |
| `device` | `torch.device` | `cpu` | Compute device |
| `working_dtypes` | `list` | `[torch.bfloat16, torch.float32]` | Allowed dtypes |
| `upscale_ratio` | `int` | `8` | 8x spatial upscale from latent to pixel |
| `downscale_ratio` | `int` | `8` | 8x spatial downscale from pixel to latent |
| `latent_channels` | `int` | `16` (Flux) | 4 for SD1.5/SDXL |
| `output_device` | `torch.device` | `cpu` | Output tensor device |

**`first_stage_model` sub-modules:** `encoder_class` = `Encoder`; `decoder_class` = `Decoder`.

**VAE name:** only "VAE" (the Python class name) is available without extra hooks into ComfyUI's model manager. The hash fully identifies the VAE; the name is cosmetic. Decision: hash is enough, no filename extraction.

**What we currently capture:** weight hash via `_hash_model_weights(vae)` - hashes `first_stage_model.named_parameters()` -> full float32 content hash.

**Probe functions that use VAE:**
- `_extract_vae_model()` - weight hash only
- `vae.decode(latent["samples"])` - decodes to IMAGE in `probe()`
- `_inspect_vae_raw()` [v0.9.4+] - class hierarchy, first_stage_model config, common attrs

---

## CLIP_VISION  (raw input object, before encode)

**Python type:** `comfy.clip_vision.ClipVisionModel` (wrapper around a HuggingFace model)

**Top-level attributes:**

| Attribute | Type | Observed | Notes |
|-----------|------|----------|-------|
| `model` | `nn.Module` | Yes | Inner HF vision model |
| `patcher` | `ModelPatcher` | Yes | ComfyUI patcher wrapper |
| `load_device` | `torch.device` | `cpu` | ModelManagement attr |

**`model` inner class:** `CLIPVisionModelProjection` (confirmed for both SigLIP and ViT-L models in this run).

**Architecture variants (confirmed v0.9.4):**

| Model | `image_size` | `visual_projection_class` | `image_embeds` shape | `last_hidden_state` shape | `is_aliased` | CLS token |
|-------|-------------|--------------------------|---------------------|--------------------------|--------------|-----------|
| SigLIP (9f9c...) | 512 | `"function"` (no real projection) | `[1, 1024, 1152]` | same (aliased) | True | No - 1024 pure spatial patches |
| ViT-L (96d2...) | 224 | `Linear` | `[1, 768]` (projected) | `[1, 257, 1024]` | False | Yes - index 0 is CLS, patches 1..256 |

**Aliasing note (SigLIP without projection):** `image_embeds` and `last_hidden_state` are the same tensor in memory. Do NOT treat them as independent signals. Use only one (prefer `image_embeds` mean-pooled -> `[1, 1152]`).

**`encode_image(image)` return object:**
- `image_embeds`: projected embedding (or spatial features for SigLIP without proj)
- `last_hidden_state`: spatial token features `[1, num_patches+CLS, hidden_dim]` (or same as image_embeds for SigLIP)
- `penultimate_hidden_states`: second-to-last layer features (same shape as last_hidden_state)

**Patch pool coverage note:** Face mask covers only ~1.2% of the image (small faces). For SigLIP: 13/1024 patches covered. For ViT-L: 1/256 patches. Very sparse - patch pool results for face mask are low-confidence at 512x512 with small faces. A minimum-patch guard should be applied for low-coverage regions.

**Probe functions that use CLIP_VISION:**
- `_extract_clip_vision_slot()` - hashes model, encodes, extracts stats + mean-pooled embedding
- `_inspect_clip_vision_raw()` [v0.9.4+] - raw object structure before encode

---

## MASK  (`torch.Tensor`)

**Python type:** `torch.Tensor`

**Properties (confirmed from `_raw_inputs` in v0.9.4 run):**

| Property | Value | Notes |
|----------|-------|-------|
| Shape | `[1, H, W]` = `[1, 512, 512]` | `[B, H, W]` - no channel dim |
| Dtype | `torch.float32` | Standard from ComfyUI segmentation nodes |
| Device | `cpu` | Confirmed |
| Value range | `[0.0, 1.0]` | White = subject, black = background |

**Binary vs soft - confirmed v0.9.4:**

| Mask | Binary? | Max observed | Notes |
|------|---------|-------------|-------|
| `face` | YES (binary) | 1.0 | Face segmentation - hard boundary |
| `main_subject` | **NO (soft)** | 0.996 | MainSubject mask has soft gradient edges from segmentation |
| `skin` | YES (binary) | 1.0 | Skin region - hard boundary |
| `clothing` | YES (binary) | 1.0 | Clothing region - hard boundary |
| `hair` | YES (binary) | 1.0 | Hair region - hard boundary |

**Coverage values (from v0.9.4 dump, 512x512 image, small faces in crowd scene):**

| Mask | Coverage (fraction) | Pixel count | CLIP SigLIP patches covered | CLIP ViT-L patches covered |
|------|---------------------|-------------|-----------------------------|-----------------------------|
| face | ~1.2% | 3,099 | 13 / 1024 | 1 / 256 |
| main_subject | ~28.4% | ~74,424 | 310 / 1024 | ~72 / 256 |
| skin | ~5.2% | 13,548 | 54 / 1024 | ~13 / 256 |
| clothing | ~23.4% | 61,320 | 256 / 1024 | ~62 / 256 |
| hair | ~1.7% | 4,424 | 21 / 1024 | ~5 / 256 |

**Shape notes:**
- ComfyUI segmentation nodes produce `[1, H, W]` - no channel dim.
- Some older nodes may produce `[H, W]` or `[B, H, W, 1]`.
- Probe's `_resize_mask_to_image()` handles size mismatch.

**Probe functions that use MASK:**
- `_masked_patch_pool()` - weight CLIP spatial tokens by mask coverage
- `_resize_mask_to_image()` - handles size mismatch before writing ValueRef
- `_inspect_mask_raw()` - shape, ndim, dtype, device, coverage, binary check

---

## IMAGE  (`torch.Tensor`)

**Python type:** `torch.Tensor`

**Properties (always-decoded output - produced internally by `vae.decode()`, confirmed v0.9.4):**

| Property | Value | Notes |
|----------|-------|-------|
| Shape | `[1, 512, 512, 3]` | `[B, H, W, C]` - channels LAST (ComfyUI convention) |
| Dtype | `torch.float32` | Always float32 from VAE decode |
| Device | `cpu` | VAE decode returns CPU tensor |
| Value range | `[0.0, 1.0]` | Clamped by VAE decode |
| Channel order | RGB | Not BGR |
| Memory | 3.146 MB | `1 * 512 * 512 * 3 * 4 bytes` |

**InsightFace conversion:** `[B,H,W,C] float32 [0,1]` -> `[H,W,C] uint8 [0,255] BGR` for InsightFace input.

**Probe functions that use IMAGE:**
- `_extract_image()` - shape, dtype, pixel stats, content hash
- `_compute_luminance_map()` - BT.709 luminance from RGB channels
- `_extract_clip_vision_slot()` - feeds decoded image to CLIP encode
- `_extract_insightface()` - converts float32 RGB -> uint8 BGR for InsightFace
- `_inspect_image_raw()` [v0.9.4+] - shape, ndim, dtype, device, channel means, memory

---

## InsightFace `Face` object

**Python type:** `insightface.app.common.Face` (or equivalent named tuple / SimpleNamespace)

**Returned by:** `FaceAnalysis.app.get(img_bgr_uint8)` - returns a `list[Face]`

**Model pack used in this run:** AntelopeV2 or buffalo_l (standard open pack)

**All attributes - confirmed v0.9.4 (via `_inspect_insightface_faces_raw()`):**

| Attribute | Shape / Type | Present in this run | Currently stored by probe | Notes |
|-----------|-------------|---------------------|--------------------------|-------|
| `bbox` | `ndarray [4]` float32 | YES | Yes | `[x0, y0, x1, y1]` pixel coords |
| `kps` | `ndarray [5, 2]` float32 | YES | Yes (v0.9.6) | 5 facial keypoints: left_eye, right_eye, nose, left_mouth, right_mouth |
| `det_score` | scalar float32 | YES | Yes | Detection confidence `[0,1]` |
| `landmark_3d_68` | `ndarray [68, 3]` float32 | YES | **No (not currently stored)** | 68-point 3D landmark - no settled metric mapped |
| `landmark_2d_106` | `ndarray [106, 2]` float32 | YES | **No (not currently stored)** | 106-point 2D landmark - no settled metric mapped |
| `pose` | `ndarray [3]` float32 | YES | Yes (v0.9.5) | Head pose `[pitch, yaw, roll]` in degrees - `[-1.65, ..., 17.6]` range observed |
| `age` | `int` | YES | Yes | Estimated age (e.g. 33) |
| `gender` | scalar int64 | YES | Yes | `0=female 1=male` |
| `embedding` | `ndarray [512]` float32 | YES | Yes (hash+norm+.npy) | ArcFace identity embedding |
| `normed_embedding` | `ndarray [512]` float32 | YES | Yes (v0.9.5, .npy) | L2-normalized embedding (norm ~ unit sphere); preferred for cosine similarity |
| `sex` | `str` | YES | Yes (v0.9.5) | Alternative gender: `'M'` or `'F'`; observed `"F"` |
| `face_token` | varies | **NO** (not present) | No | Pack-dependent - present in some premium/commercial packs; absent with standard open packs |

**face_token note:** `getattr(face, "face_token", None)` returned `None` for this model pack. This does NOT mean it is permanently unavailable - a different InsightFace model pack may expose it. Probe correctly detects it if present.

**extra_attrs:** no extra attributes detected beyond the known list in this run.

**Multi-face behavior (7 faces detected in this run):**
- `app.get()` returns all faces; probe stores ALL of them sorted by `det_score` descending
- Best face `det_score` = 0.90; lowest detected = 0.54
- Faces are paired across runs by index position or by closest bbox match

**Detection behavior:**
- Input: `[H, W, C] uint8 BGR` - probe converts from `[B, H, W, C] float32 RGB`
- Returns list sorted by detection score (descending in some packs) or unsorted
- Probe stores all detected faces in sorted order

**Probe functions that use InsightFace:**
- `_get_insightface_app()` - model loading + caching
- `_extract_insightface()` - face detection + embedding extraction
- `_inspect_insightface_faces_raw()` [v0.9.4+] - dumps ALL attributes of best face + all face bboxes

---

## Aux Images (depth / pose / normal / edge)

**Python type:** `torch.Tensor` - same format as decoded IMAGE

**Properties:** identical to decoded IMAGE - `[B, H, W, 3]` float32 RGB `[0,1]`.

**Source:** user-provided from external preprocessor nodes (DepthAnythingV2, DW-Pose, DSINE, LineArt, etc.).

**What's captured:** content hash + pixel stats. No semantic interpretation; that happens at comparison time.

**Assignment:** always Sample domain. Aux images are measurement tools: preprocessor outputs that describe the image structurally. They answer "did the structural skeleton change between baseline and LoRA?" - e.g. "LineArt moved a lot, image probably changed." They are stored because they are usable measurements, not because of how they were wired.

---

## Revision history

| Date | Probe version | Update |
|------|--------------|--------|
| 2026-03-01 | v0.9.3 | Initial document - seeded from existing dump + code analysis |
| 2026-03-01 | v0.9.4 | Added `_raw_inputs` net-fishing; `[confirm]` entries pending first v0.9.4 run |
| 2026-03-01 | v0.9.4 | Resolved all `[confirm]` entries from first v0.9.4 run (dump `2026-03-01T19-42-24_060bc3c2.json`) |
