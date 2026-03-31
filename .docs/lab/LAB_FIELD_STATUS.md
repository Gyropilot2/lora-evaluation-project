# Lab Field Status Reference

This document is the authoritative reference for every field written by `lab/probe.py`
into a JSON dump file. It records what each field represents, how the probe
obtains it, and the current reliability or limitation of that field inside lab
output.

Production-canon storage law now lives in `../06_EVIDENCE_REFERENCE.md` and
`../08_ASSET_AND_VALUEREF_POLICY.md`. This file stays focused on probe output,
probe field meaning, and lab-facing extraction notes.

**Keep this document updated whenever probe.py changes.**
Probe version this document was written against: **v0.9.6**

---

## Status legend

| Tag | Meaning |
|-----|---------|
| `RELIABLE` | Value is correctly extracted and can be trusted for analysis |
| `PARTIAL` | Value is present but incomplete or limited in scope |
| `USER-PROVIDED` | Not auto-detected; relies on user input; must be validated by the user |
| `REDUNDANT` | Present in the dump but identical to another field under certain conditions |
| `ARCHITECTURAL NOTE` | Not a defect; explains expected behaviour that looks surprising |
| `DEFERRED` | Concept is visible in probe output but not part of settled stored or reviewed use |
| `CONDITIONAL` | Present only when the relevant optional input was wired |
| `BROKEN` | Currently broken in the probe; known bug; data not reliably available |

---

## 3-object Evidence format

Probe output is a JSON file with three top-level records. Identity hashes are computed in probe and match the production Evidence schema.

### Method record

Represents the generative harness - everything except LoRA identity, seed, and lora_strength.

**`method_hash`** = BLAKE3 of: `(base_model_hash, vae_hash, conditioning_pos_hash, conditioning_neg_hash, steps, denoise, sampler, scheduler, cfg, latent_width, latent_height, latent_shape, model_extras_hash)`

**Note on `latent_type` / `latent_channels`:** Both obsolete names for what is now `latent_shape` - the full tensor shape [B, C, H_latent, W_latent]. Stored as an array; channel count is derivable as shape[1].

| Field | Source type | Part of method_hash | Status | Notes |
|-------|------------|---------------------|--------|-------|
| `base_model.hash` | IC - in-memory weight hash | YES | `RELIABLE` | Structural + value anchor; no file path needed |
| `base_model.name` | IC - from ModelPatcher/folder_paths | No | `RELIABLE` | "flux1-schnell-q8_0.gguf" confirmed accessible |
| `base_model.arch` | IC - from model config | No (extras) | `RELIABLE` | class, config_class, unet_config, model_type; Bouncer moves to extras |
| `vae_model.hash` | IC - in-memory weight hash | YES | `RELIABLE` | Full float32 hash of first_stage_model params |
| `vae_model.name` | IC - class name only | No | `ARCHITECTURAL NOTE` | "VAE" - no filename available; hash is identity |
| `conditioning.positive_hash` | IC - BLAKE3 of tensors | YES | `RELIABLE` | Hashes what the diffusion model actually received |
| `conditioning.positive_text` | UI - STRING NodeInput | No | `USER-PROVIDED` | Metadata label only; user must match what they actually used |
| `conditioning.positive_guidance` | NI - from conditioning meta | YES | `RELIABLE` | Float from `pair[1]["guidance"]` |
| `conditioning.negative_hash` | IC - BLAKE3 of tensors | YES | `RELIABLE` | Structure confirmed: same as positive even when empty |
| `conditioning.negative_text` | UI - STRING NodeInput | No | `USER-PROVIDED` | Metadata label only |
| `conditioning.negative_guidance` | NI - from conditioning meta | YES | `RELIABLE` | Same structure as positive |
| `settings.steps` | UI | YES | `RELIABLE` | |
| `settings.denoise` | UI | YES | `RELIABLE` | |
| `settings.sampler` | UI (COMBO) | YES | `RELIABLE` | |
| `settings.scheduler` | UI (COMBO) | YES | `RELIABLE` | |
| `settings.cfg` | UI | YES | `RELIABLE` | |
| `latent.width` | IC - samples.shape[3] x 8 | YES | `RELIABLE` | |
| `latent.height` | IC - samples.shape[2] x 8 | YES | `RELIABLE` | |
| `latent.shape` | IC - full samples.shape [B,C,H,W] | YES | `RELIABLE` | [1,16,H,W] for Flux/SD3; [1,4,H,W] for SD/SDXL |
| `model_extras` | IC - non-LoRA model modifications | YES | `RELIABLE` | Null for standard configs (ModelSamplingFlux modifies BaseModel, not model_options). Hash present when custom CFG wrappers, FreeU, etc. are attached. |

**Source types:** `IC` = InternalCompute (derived from NodeInputs by probe), `UI` = User-provided scalar/string NodeInput, `NI` = raw NodeInput value.

### Eval record

Represents one LoRA identity under a Method (or baseline when lora_hash is null).

**`eval_hash`** = BLAKE3(method_hash, lora_hash)

| Field | Source type | Status | Notes |
|-------|------------|--------|-------|
| `lora.hash` | IC - BLAKE3 of weight tensors in sorted key order | `RELIABLE` | True LoRA identity; stable to file re-encoding |
| `lora.file_hash` | IC - BLAKE3 of .safetensors file bytes | `RELIABLE` | File-level identity; different from lora.hash if file is re-saved |
| `lora.name` | IC - resolved from folder_paths COMBO enumeration | `RELIABLE` | Filename from system-enumerated COMBO, not free-text user input. Used for display and path resolution. |

### Sample record

Represents one atomic output: one (seed, lora_strength) pair under one Eval.

**`sample_hash`** = BLAKE3(eval_hash, seed, lora_strength)

**Vital fields** (must be present - missing any = hard refusal by Bouncer):
`sample_hash`, `eval_hash`, `seed`, `latent_hash`, `image_hash`

| Field | Source type | Status | Notes |
|-------|------------|--------|-------|
| `seed` | UI | `RELIABLE` | |
| `lora_strength` | UI | `RELIABLE` | |
| `latent_hash` | IC - BLAKE3 of LATENT samples tensor | `RELIABLE` | Vital integrity field; NOT_ZERO_DELTA if changes on re-ingest |
| `image_hash` | IC - BLAKE3 of decoded image tensor | `RELIABLE` | Vital integrity field |
| `ingest_status` | Bouncer | `RELIABLE` | OK / WARN / ERROR |
| `is_dirty` | Bouncer | `RELIABLE` | Review assembly excludes dirty samples by default |

---

## Sample domain - What to store and why

Column "Evaluation question" shows what comparison each field enables between baseline and LoRA sample records.

### `face_analysis` domain

**Structure: array of all detected faces, sorted by `det_score` descending.** No artificial cap - store all detected faces so faces can be paired across runs without pre-selection. Each element in `faces[]` contains the fields below.

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| `embedding` [512] | IC / InsightFace | YES | ValueRef .npy per face | Face identity drift: cosine similarity vs baseline | `RELIABLE` |
| `normed_embedding` [512] | IC / InsightFace | YES | ValueRef .npy per face | Better for cosine: pre-normalized; preferred over raw embedding | `RELIABLE` |
| `pose` [3] pitch/yaw/roll | IC / InsightFace | YES | JSON scalars per face | Head orientation: did LoRA shift the subject's pose? | `RELIABLE` |
| `det_score` | IC / InsightFace | YES | JSON scalar per face | Reliability gate: low score -> treat measurement with caution | `RELIABLE` |
| `bbox` [4] | IC / InsightFace | YES | JSON per face | Face position/size: was the subject repositioned? | `RELIABLE` |
| `face_count` | IC / InsightFace | YES | JSON int (top-level) | Total detected faces: did LoRA add/remove faces? | `RELIABLE` |
| `age` int | IC / InsightFace | YES | JSON per face | Age proxy: style drift marker (not a reliable measurement) | `USER-PROVIDED` |
| `gender` / `sex` | IC / InsightFace | YES | JSON per face | Should never change; if it does -> detection failure signal | `USER-PROVIDED` |
| `kps` [5,2] | IC / InsightFace | YES | JSON per face (added v0.9.6) | Keypoint positions: did face geometry shift between runs? | `RELIABLE` |
| `landmark_3d_68` / `landmark_2d_106` | IC / InsightFace | NO | - | Rich geometry; large data; no settled metric or stored use | - |

### `clip_vision` domain

**Record-shape note:** Store global (unmasked) embeddings only. Masked
regional analysis is done at comparison time: the stored global embedding plus
stored mask ValueRef are combined later. Pre-filtered masked embeddings are not
stored in the probe output.

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| `global_embedding_valueref` (mean-pooled) | IC | YES | ValueRef .npy | Global semantic drift: cosine similarity baseline vs LoRA | `RELIABLE` |
| `last_hidden_state_valueref` (spatial features) | IC | YES | ValueRef .npy | Masked pooling at comparison time (regional analysis without re-encoding) | `RELIABLE` |
| Pre-masked embeddings (face/skin/clothing/hair) | IC | **NO - removed** | - | Masks applied to stored features at comparison time | - |
| Patch pool coverage stats per mask | IC | YES | JSON | Sanity: how many patches covered by region | `RELIABLE` |
| Global stats (mean, norm, mean_pooled_norm) | IC | YES | JSON | Quick scalar comparisons; not a substitute for cosine | `RELIABLE` |

**Notes on spatial features per model:**
- **SigLIP** (`last_hidden_state` [1, 1024, 1152]): same tensor as `image_embeds` (aliased); storing `last_hidden_state_valueref` gives 1024 spatial patches to pool against any mask.
- **ViT-L** (`last_hidden_state` [1, 257, 1024]): CLS token at index 0; skip it at comparison time (patches are tokens 1..256). Enables regional analysis that projected `image_embeds` [1, 768] cannot support.

### `masks` domain

**Record-shape note:** store raw mask tensors only. Derived statistics
such as `pixel_stats` or `edge_density` are not pre-baked into the probe dump.
Image plus mask can be combined later when that analysis is actually needed.

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| Mask ValueRef per slot | NI (mask tensor) | YES | ValueRef .npy | Mask applicable to any stored signal (image, lum, CLIP) at comparison time | `RELIABLE` |
| `pixel_stats` per mask | IC - (image + mask) | **NO** | - | Re-derivable at comparison time: image+mask -> stats | removed v0.9.6 |
| `edge_density` per mask | IC - (image + mask + Sobel) | **NO** | - | Re-derivable at comparison time: same reason | removed v0.9.6 |

### `luminance` domain

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| stats (mean/std/min/max) | IC - BT.709 | YES | JSON | Global brightness/contrast shift (e.g. exposure LoRA effect) | `RELIABLE` |
| Luminance map | IC | YES | ValueRef .npy | Spatial brightness distribution; spatial analysis | `RELIABLE` |

### `image` domain

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| pixel_stats | IC | YES | JSON | Global pixel stats (rough sanity check) | `RELIABLE` |
| Image ValueRef | IC | YES | ValueRef .npy | Raw image for visual inspection | `RELIABLE` |

### `aux` domain

Aux images are stored in Sample (not Method) because they are **measurement tools**: preprocessor
outputs that describe the image structurally. They answer "did the structural skeleton change
between baseline and LoRA?" - e.g. "LineArt moved a lot, image probably changed." They are not
stored because they "drove ControlNet" - they are stored because they are usable measurements.

| Field | Type | Store | How | Evaluation question | Status |
|-------|------|-------|-----|-----------------|--------|
| Per-aux pixel_stats | NI | YES | JSON | Quick sanity: did the depth/pose map change significantly? | `RELIABLE` |
| Aux ValueRefs | NI | YES | ValueRef .npy | Structural comparison: same ValueRef hash = same structure | `RELIABLE` (fixed v0.9.5) |

---

## What stays lab-local here

This file still records probe-scope omissions and deferrals, but the broad
production-facing "what we do not store" rationale now belongs in the numbered
canon docs.

Lab-local omissions and deferrals:

- `landmark_2d_106` and `landmark_3d_68` remain outside the stored lab payload
  because the review layer does not have a settled use for them yet
- `face_token` remains pack-dependent and absent in the open pack currently in
  use
- multi-LoRA decomposition remains outside probe output because the combined patch dict does
  not support a reliable split

For canon storage law, use:
- `../06_EVIDENCE_REFERENCE.md`
- `../08_ASSET_AND_VALUEREF_POLICY.md`

---

## Probe-facing settled notes

These are the settled notes that still matter specifically to probe output and
lab interpretation.

| Note | Ruling |
|------|--------|
| `latent_type` / `latent_channels` | Replaced by `latent.shape`; probe output should use the full tensor shape |
| `positive_text` / `negative_text` | Metadata labels only; not part of hash material |
| `face_token` availability | Pack-dependent; absent in the open pack used for this reference |
| VAE name | Class name is acceptable in probe output; hash remains the identity |
| `normed_embedding`, `pose`, `kps` | Belong in the face records and remain useful probe output |
| `face_analysis` structure | Array of detected faces sorted by `det_score` descending; no artificial cap |
| Multi-LoRA stacking | Not part of probe output because there is no reliable decomposition from the combined patch dict |

---

## ValueRef format

Probe output uses the same ValueRef family as the production record model.
The canonical contract now lives in `../08_ASSET_AND_VALUEREF_POLICY.md`.

Lab-specific note:
- when a write fails, the probe still leaves an explicit status wrapper in the
  slot rather than pretending the asset exists

---

## Meta fields

### `_dump_meta.run_id`
- **What**: UUID4 generated at probe execution time. Lab tracking ID only - not a production `run_id`.
- **Status**: `RELIABLE`
- **Evaluation use**: Links the dump file to the run. Keyed in `build_paired_catalog`.

### `_dump_meta.written_at`
- **What**: ISO 8601 timestamp of dump write time.
- **Status**: `RELIABLE`

### `probe_version`
- **What**: Version string of `lab/probe.py` that produced this dump.
- **Status**: `RELIABLE`
- **Use**: Identify dumps that predate a fix. Discard `lora_patches.content_hash` from < v0.5.0.

### `is_baseline`
- **What**: `true` when no LoRA patches were found on the ModelPatcher.
- **How**: `not bool(model.patches)` - auto-detected.
- **Status**: `RELIABLE`
- **Caveats**: LoRA at strength=0 may still leave patches; `is_baseline` would be `false`.
- **Evaluation use**: Baseline records -> `lora_hash = Invalid(NULL)` in production Evidence.

### `lora_name`
- **What**: LoRA filename from the COMBO dropdown (system-enumerated, not free-text user input). Sentinel `"(none)"` stored as `null`.
- **Status**: `IC` - resolved from `folder_paths` COMBO enumeration
- **Note**: Filename is structurally lost after `ModelPatcher.load_lora()`. COMBO is the only runtime source of the filename. The `content_hash` is the true identity.
- **Name resolution priority**: content_hash > lora_metadata.inferred_display_name > lora_name (COMBO).

---

## model_pipeline fields

### `model_pipeline.patcher_class`
- **Status**: `RELIABLE`
- **Note**: `GGUFModelPatcher` = model loaded via GGUF quantization extension.

### `model_pipeline.base_model.class` / `config_class`
- **Status**: `RELIABLE`
- **Evaluation use**: `config_class` (e.g. `"FluxSchnell"`) is the most useful architecture identifier.
  Used in `_settings_fingerprint` for run pairing and baseline lookup.

### `model_pipeline.base_model.unet_config`
- **Status**: `RELIABLE` (non-JSON values stringified)

### `model_pipeline.base_model.model_type`
- **Status**: `RELIABLE` (e.g. `"ModelType.FLOW"`)

### `model_pipeline.base_model.state_dict_key_count`
- **Status**: `RELIABLE` - but slow on large models; wrapped in `_try`.

### `model_pipeline.lora_patches.count`
- **Status**: `RELIABLE`

### `model_pipeline.lora_patches.content_hash`
- **What**: BLAKE3 hash of all LoRA weight tensors in sorted key order. True LoRA identity.
- **Status**: `RELIABLE`
- **Evaluation use**: This is `lora_hash` in the production Evidence schema.

### `model_pipeline.lora_patches.keys_sample` / `keys_tail_sample`
- **Status**: `RELIABLE`
- **Note**: Keys are string-sorted. `single_blocks.9` sorting last doesn't mean coverage stops
  at block 9 - "9" > "37" in lexicographic order.

### `model_pipeline.lora_patches.layer_components` / `block_types` / `attn_types`
- **Status**: `RELIABLE` - fragment lists are hardcoded; may miss architecture-specific names.

### `model_pipeline.lora_patches.patch_structs_sample`
- **Status**: `RELIABLE` (shows LoRAAdapter internals).

### `model_pipeline.model_options`
- **Status**: `RELIABLE` - often `{"transformer_options": {}}` in standard workflows.

---

## latent fields

### `latent.shape`
- **What**: Tensor shape of the raw latent from KSampler (e.g. `[1, 16, 128, 128]` for Flux 1024x1024).
- **Status**: `RELIABLE`

### `latent.dtype`
- **Status**: `RELIABLE`

### `latent.stats`
- **What**: min/max/mean/std of latent tensor values.
- **Status**: `RELIABLE`
- **Evaluation use**: Latent statistics are a fast proxy for generation health.
  A latent with mean near 0 and std near 1 is typical; extreme values indicate unusual generation.

### `latent.content_hash`
- **What**: BLAKE3 hash of the raw latent tensor bytes (float32 converted).
- **Status**: `RELIABLE`
- **Evaluation use**: Stable identity for the undecoded sample.

### `latent.asset_note`
- **What**: Informational string noting that the lab dump records hash + stats only, not a latent asset file.
- **Status**: `DEFERRED` - Lab captures hash + stats only.

---

## image fields

### `image.shape`
- **What**: Tensor shape of the decoded image (e.g. `[1, 1024, 1024, 3]` = BxHxWxC).
- **Status**: `RELIABLE`

### `image.dtype`
- **Status**: `RELIABLE` (always `torch.float32` from ComfyUI VAE decode)

### `image.pixel_stats`
- **What**: min/max/mean of pixel values [0, 1].
- **Status**: `RELIABLE`
- **Evaluation use**: Fast signal that image changed. Absolute value has limited meaning;
  delta between baseline and LoRA run is the useful measurement.

### `image.content_hash`
- **What**: BLAKE3 hash of raw pixel tensor bytes.
- **Status**: `RELIABLE`
- **Evaluation use**: This is `sample_id` in the production Evidence schema.

### `image.asset_note`
- **Status**: `DEFERRED` - Lab captures hash + stats only.

---

## vae fields

### `vae.value.class`
- **What**: Python class name of the VAE wrapper (e.g. `"VAE"`).
- **Status**: `RELIABLE`

### `vae.value.inner_class`
- **What**: Python class name of the inner model (e.g. `"AutoencoderKL"`).
- **Status**: `RELIABLE`

### `vae.value.model_hash`
- **What**: BLAKE3 of all VAE parameter tensors (float32, sorted by name).
- **How**: Unwraps via `first_stage_model` -> `model` -> self. The real autoencoder for
  ComfyUI VAE objects lives at `first_stage_model`, NOT `.model`.
- **Status**: `RELIABLE`
- **Evaluation use**: Run-hash material. Different VAE -> different run.
  Production Evidence carries the same role through `vae_hash`.

### `vae.value.param_count`
- **Status**: `RELIABLE`

---

## conditioning fields

### `conditioning.positive.hash`
- **What**: BLAKE3 hash of the positive conditioning tensors (context + pooled_output).
  This is what the diffusion model actually received. Replaces `prompt.hash`.
- **Status**: `RELIABLE`
- **Evaluation use**: Primary identity for "what prompt was used". Production
  Evidence carries the same role through `conditioning_hash_positive`, and the
  same signal is used in `_settings_fingerprint` for run pairing.
- **Key insight**: Two different prompt strings that produce the same conditioning
  (same text encoder, same text) are treated as the same experiment. This is correct.

### `conditioning.positive.first_tensor_shape`
- **What**: Shape of the first context tensor (e.g. `[1, 256, 4096]` for Flux T5-XXL output).
- **Status**: `RELIABLE`

### `conditioning.positive.norm` / `mean`
- **Status**: `RELIABLE`
- **Evaluation use**: Conditioning norm as a sanity check. Very low norm could indicate empty/failed encoding.

### `conditioning.positive.pooled_shape` / `pooled_norm`
- **What**: Shape and L2 norm of the pooled_output tensor (e.g. CLIP-L output for Flux).
- **Status**: `RELIABLE` (when pooled_output is present in the conditioning dict)

### `conditioning.negative`
- **What**: Same structure as positive. `{"status": "not_provided"}` if negative not wired.
- **Status**: `RELIABLE`

---

## clip_vision fields

`clip_vision` is a dict keyed by `model_hash[:16]` (first 16 hex chars of the model's BLAKE3
parameter hash). An absent key means that model was not connected. An empty dict means no
ClipVision models were connected.

Each entry has three sub-keys: `model` (identity), `output` (global embedding stats), and
`patch_pools` (per-mask patch pool stats).

**Why hash keys, not slot positions:** Hash-keyed slots ensure that SigLIP in any slot always
produces the same `clip_vision.{siglip_hash}.*` key path, so `field_catalog --paired` can
match measurements across baseline and LoRA runs regardless of which slot the model was wired into.

Suggested slot assignments (conventions, not enforced):
- Semantic similarity - SigLIP / SigLIP2
- Structural/spatial - DINOv2
- Provisional - EVA02 or other

### `clip_vision.{model_hash_16}.model.value.class`
- **Status**: `RELIABLE`

### `clip_vision.{model_hash_16}.model.value.model_hash`
- **What**: Full BLAKE3 of all CLIP_VISION model parameters (float32, sorted by name).
  The dict key is the first 16 chars of this value.
- **Status**: `RELIABLE` - slow for large models (~1-10s). Acceptable for Lab.
- **Evaluation use**: NOT run_signature material (see S-22 in design brief - CV is a
  measurement instrument, not a generative input). Records which model produced the
  measurement.

### `clip_vision.{model_hash_16}.model.value.param_count`
- **Status**: `RELIABLE`

### `clip_vision.{model_hash_16}.output.is_aliased`
- **What**: `true` when `image_embeds` and `last_hidden_state` are the same tensor in memory.
- **Status**: `ARCHITECTURAL NOTE` - expected for SigLIP/SigLIP2 (no projection_dim).
- **Evaluation use**: When `true`, do NOT count `image_embeds` and `last_hidden_state` as
  independent signals. Use only `image_embeds` mean_pooled stats.

### `clip_vision.{model_hash_16}.output.image_embeds`
- **What**: Primary image embedding. Shape varies:
  - SigLIP: `[B, seq_len, dim]` e.g. `[1, 1024, 1152]` - full patch token sequence
  - Standard CLIP / DINOv2: `[B, dim]` - CLS token projection
- **Status**: `RELIABLE`
- **Additional fields (3D shapes only)**:
  - `mean_pooled_shape`: shape after mean-pooling over sequence dim (e.g. `[1, 1152]`)
  - `mean_pooled_norm`: L2 norm of pooled vector - cosine similarity signal
- **Evaluation use**: Mean-pooled `[B, dim]` vector is the primary cosine similarity input.

### `clip_vision.{model_hash_16}.output.last_hidden_state`
- **Status**: `REDUNDANT` when `is_aliased=true`. `RELIABLE` and independent otherwise.

### `clip_vision.{model_hash_16}.output.penultimate_hidden_states`
- **What**: Output from second-to-last transformer encoder layer (pre-final-layernorm).
- **Status**: `RELIABLE` - genuinely different from `image_embeds`.
  Wider value range (unnormalized activations). Potential secondary signal.

### `clip_vision.{model_hash_16}.patch_pools`
- **What**: Per-mask patch pool stats dict. Keys are active mask names (`face`, `main_subject`, etc.).
  Present only when at least one mask input is wired.
- **Status**: `CONDITIONAL`

### `clip_vision.{model_hash_16}.patch_pools.<mask_name>.patch_pool`
- **What**: Mask-weighted pooling of spatial patch features from the GLOBAL encode's
  `last_hidden_state` (CLS token at index 0 is skipped; patches 1..N are spatial).
  Mask is downsampled to the `sqrt(N) x sqrt(N)` patch grid; each patch weighted by
  its mask coverage. Produces: `covered_patches`, `total_patches`, `coverage_ratio`,
  `pooled_norm`, `pooled_mean`.
- **Status**: `RELIABLE` when `last_hidden_state` has a square patch grid (DINOv2, standard ViT).
  Returns `{"status": "error"}` for non-square grids.
- **Evaluation use**: Character-specific spatial feature vector without FeatUp. Cosine
  similarity of `pooled_norm` across runs is a structural consistency signal.
- **Note**: Uses ORIGINAL global encode features, not the masked encode. The mask only
  determines WHICH patches to weight, not the feature values themselves.

---

## face_analysis fields

`face_analysis` is an array of all detected faces sorted by `det_score` descending. `face_count`
is a top-level int. Each element in `faces[]` contains:

### `face_analysis.status`
- `"ok"` - InsightFace ran successfully
- `"not_provided"` - `insightface_model_path` was empty
- `"not_available"` - `insightface` package not installed
- `"error"` - runtime failure
- **Status**: `CONDITIONAL`

### `face_analysis.face_count`
- **What**: Total number of faces detected (array length).
- **Status**: `RELIABLE` (when status="ok")

### Per-face fields (`face_analysis.faces[N].*`)

### `faces[N].embedding` / `normed_embedding`
- **What**: 512-dim ArcFace identity embedding. `embedding` is the raw output;
  `normed_embedding` is L2-normalized (preferred for cosine similarity).
- **How**: Both stored as ValueRef `.npy`. Hash from `embedding` is the face identity key.
- **Status**: `RELIABLE`
- **Evaluation use**: Cosine similarity of stored vectors (`compare.face_cosine_delta.v1`).
  Very low norm may indicate a degraded face region.

### `faces[N].det_score`
- **What**: InsightFace detection confidence `[0, 1]`.
- **Status**: `RELIABLE` - low score -> treat measurement with caution.

### `faces[N].bbox`
- **What**: `[x0, y0, x1, y1]` bounding box in pixel coordinates.
- **Status**: `RELIABLE`

### `faces[N].kps`
- **What**: `[[x,y],...]` for 5 facial keypoints (left_eye, right_eye, nose, left_mouth, right_mouth).
- **Status**: `RELIABLE`

### `faces[N].pose`
- **What**: `[pitch, yaw, roll]` head orientation in degrees.
- **Status**: `RELIABLE`

### `faces[N].age` / `gender` / `sex`
- **What**: Estimated age (int), gender (int: 1=male, 0=female), sex string ("M"/"F").
- **Status**: `USER-PROVIDED` - model output, unverified. Useful as a metadata drift signal,
  not a reliable measurement primitive.

---

## aux fields

Present only when aux image inputs (`aux_depth`, `aux_pose`, `aux_canvas`) are wired.
Each sub-key mirrors the `image` field structure: `shape`, `dtype`, `pixel_stats`, `content_hash`.

### `aux.depth` / `aux.pose` / `aux.canvas`
- **What**: Image metadata captured from auxiliary comparison images produced by external nodes.
  `depth` - depth map (DepthAnything, MiDaS, etc.)
  `pose` - pose skeleton visualization (DWPose, OpenPose, etc.)
  `canvas` - edge/line art (Canny, etc.)
- **Status**: `CONDITIONAL` - present only when wired.
- **Evaluation use**: Structural consistency signals.
  - `aux.depth.content_hash` delta: same hash = same structure.
  - `aux.depth.pixel_stats.mean` delta: indicates depth composition shift.
  - Similar analysis for pose and canvas.
  - These are rough sanity checks, not a settled structural comparison package.

### `aux.<name>.content_hash`
- **What**: BLAKE3 hash of the aux image pixel bytes. Stable structural fingerprint.
- **Status**: `RELIABLE` (when wired)

### `aux.<name>.pixel_stats`
- **What**: min/max/mean of the aux image pixels.
- **Status**: `RELIABLE` (when wired)

---

## lora_metadata fields

### `lora_metadata.status`
- `"ok"` - file found and header read successfully
- `"error"` - file not found or parse failure
- `"not_provided"` - lora_name was `"(none)"` or not given

### `lora_metadata.file_path`
- **What**: Absolute path of the .safetensors file as resolved by `folder_paths`.
- **Status**: `RELIABLE` - but path is machine-dependent (not portable).

### `lora_metadata.raw_metadata`
- **What**: The full `__metadata__` dict from the safetensors header. All values stringified.
- **Status**: `PARTIAL` - content is whatever the LoRA creator wrote. Unverified.
  Useful as informational metadata; NOT a measurement primitive.

### `lora_metadata.inferred_display_name`
- **What**: Display name extracted from `raw_metadata` using this priority:
  `ss_output_name` -> `modelspec.title` -> `name` -> `LoRA_name`.
- **Status**: `USER-PROVIDED` (creator-written, extracted by the probe).
- **Evaluation use**: Best available human-readable name when COMBO filename is unhelpful.
  Supersedes COMBO in name resolution priority.

### `lora_metadata.metadata_key_count`
- **Status**: `RELIABLE` - tells you how much information the creator embedded.

---

## prompt fields (v0.6.0: label only)

### `prompt.hint`
- **What**: Optional STRING input from the user - a human-readable label for the run.
  E.g. "street scene, Flux Schnell, step 4".
- **Status**: `USER-PROVIDED` - metadata only. NOT used for hashing or fingerprinting.
- **Note**: The raw prompt text is gone by the time conditioning reaches the probe.
  If you want to record what the prompt was, use this field.

---

## settings fields

All settings are values passed into the Lab Probe node inputs. User is responsible
for matching these to the actual KSampler settings.

### `settings.seed` / `steps` / `cfg` / `denoise`
- **Status**: `RELIABLE`

### `settings.sampler_name`
- **What**: Sampler name from `KSampler.SAMPLERS` COMBO (v0.6.0+). Was a STRING in v0.5.
- **Status**: `RELIABLE` from v0.6.0. `USER-PROVIDED` (typo risk) in v0.5 and earlier.

### `settings.scheduler`
- **What**: Scheduler name from `KSampler.SCHEDULERS` COMBO (v0.6.0+). Was a STRING in v0.5.
- **Status**: `RELIABLE` from v0.6.0. `USER-PROVIDED` (typo risk) in v0.5 and earlier.

---

## LoRA name resolution - multi-source strategy

The LoRA filename is structurally lost after `ModelPatcher.load_lora()`. From v0.6.0,
the probe uses three complementary sources:

| Source | Field | Reliability | Notes |
|--------|-------|-------------|-------|
| Weight tensor hash | `lora_patches.content_hash` | **GOLD** - true identity | Reliable from v0.5 |
| Safetensors metadata | `lora_metadata.inferred_display_name` | High | Creator-written, unverified |
| COMBO selection | `lora_name` | Medium | User-provided; needed for file path resolution |

This keeps identity anchored first to content hash and only then to display
metadata or the UI-selected name.
