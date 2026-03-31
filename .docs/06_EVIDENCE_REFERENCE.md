# 06 - Evidence Reference

This document describes the raw measurement domains that the Extractor produces
and stores in the DB. Everything here is Evidence - facts only, no interpretation,
no cross-sample comparison.

For how Evidence is compared and what derived metrics mean, see
`07_SYNTHESIS_PROVISIONAL.md`.
For detailed ValueRef and asset-storage law, see
`08_ASSET_AND_VALUEREF_POLICY.md`.

---

## 1 - What Evidence Is

Evidence is the record produced for one sample: one seed, one LoRA (or baseline),
one strength, under one Method. It contains the generative inputs that identify
the sample and the measurement domains that describe what came out.

Measurement domains are populated by measurement instruments (CLIP models,
InsightFace, segmentation preprocessors, depth/normal/edge preprocessors, pose
preprocessors). These instruments observe the output after generation. They do
not affect what was generated and can be added or enriched post-hoc without
changing sample identity.

Evidence contains no opinions. It does not score, rank, or compare. It records
what happened on one run.

---

## 2 - Measurement Domains

### `image`

The generated image stored as a float32 `.npy` asset.

- Format: float32, exact roundtrip required (re-ingest hashes against this file)
- Inline: `pixel_stats` (mean, std, min, max per channel)
- Purpose: source of truth for all downstream pixel-level and semantic measurements

### `luminance`

BT.709 luminance map of the generated image.

- Format: 16-bit grayscale PNG (genuine >8-bit precision confirmed)
- Inline: `stats` (mean, std, min, max of the luminance field)
- Decode contract: read via Pillow mode `I`, divide by 65535.0 to restore float32 [0,1]
- Purpose: brightness and contrast structure; separates tone from color

Can answer:
- Did overall brightness change?
- Did contrast spread change?
- Did a specific region (face, clothing, background) get brighter or darker?

### `clip_vision`

CLIP/SigLIP embeddings keyed by `model_hash[:16]`.

Per key:
- Global embedding ValueRef - full-image semantic vector
- `patch_pools` - per-mask pooled spatial tokens (32x32 grid for SigLIP, 16x16 for ViT-L/14)

Models currently stored: SigLIP and ViT-L/14. Both are stored together; comparison
procedures may use either or both.

Note: keys use the prefix `clip_`, but the stored measurements include
both SigLIP and ViT-L/14 embeddings regardless of key naming.

Can answer:
- Did the full-image semantic meaning change?
- Did a specific region (face, hair, clothing, background) change semantically?

### `masks`

Segmentation masks keyed by mask name. Stored as binary PNG (lossless, always binary
regardless of upstream softness).

Default mask slots:
- `face` - face region
- `main_subject` - `MainSubject_Mask`: white = subject foreground, black = scene background.
  True scene background is derived as `1 - main_subject`.
- `skin` - skin region
- `clothing` - clothing region
- `hair` - hair region

Masks enable region-constrained measurements across all other domains. A luminance
measurement constrained to the clothing mask answers "did clothing brightness change"
rather than "did the whole image brightness change."

**Soft component mask caution:** `skin`, `clothing`, `face`, and `hair` are sparse
soft probability maps, not reliable full-subject silhouettes. Their union does not
cleanly reconstruct a full body region and tends to approximate `main_subject`
anyway. Use `main_subject` as the primary subject-region prior. Component masks are
useful for targeted region questions; do not composite them into a fake full-body
region without threshold calibration.

### `face_analysis`

Array of all detected faces sorted by detection score descending. Per face:

- ArcFace embedding ValueRef (normed and raw)
- Keypoints (`kps`)
- Head pose: pitch, yaw, roll
- Bounding box (`bbox`)
- Detection score (`det_score`, 0-1)
- Age estimate
- Gender probability (`face_gender_f`)

Can answer:
- Was a face detected? How confidently?
- What is the face identity (embedding)?
- What is the head orientation?
- Did face count change?

Detection failure is itself a measurement outcome - face loss may be LoRA-induced,
not a neutral missing value.

### `aux`

Preprocessor outputs keyed by label. Each entry: ValueRef + `pixel_stats`.

Default slots:
- `depth` - depth estimation map (8-bit PNG, confirmed 8-bit-equivalent precision)
- `normal` - surface normal map, 8-bit RGB PNG
- `edge` - edge/structure map, 8-bit PNG

Can answer:
- Did spatial structure / scene geometry change?
- Did the camera or viewpoint shift?
- Did surface material or form change (normal)?
- Did structural outlines change (edge)?

Depth and normal are currently the best evidence for camera/viewpoint geometry
drift as distinct from semantic scene-meaning drift. Edge is useful for
shape-level change detection in constrained regions.

### `pose_evidence`

Structured keypoint evidence from pose preprocessors. Fixed source keys:
`openpose_body`, `dw_body`.

Per source:
- Raw grouped `people[]` payload from the preprocessor
- Normalized per-person grouped joints with per-joint `x`, `y`, `confidence`, missingness
- Per-person scene support facts: `main_subject_overlap`, `densepose_overlap`,
  `recognized_joint_count`, `core_joint_count`, `missing_joints`, scene flags
- Per-joint support against `MainSubject_Mask` and DensePose-derived human region

Storage direction:
- Prefer upstream grouped keypoints over raster pose parsing when available
- Body-only joints are the primary evidence surface (hand/finger and face dots
  are secondary; head orientation is better sourced from InsightFace)
- Preserve per-person grouping; do not flatten to a joint soup

Can answer:
- What is the body pose of the main subject?
- How complete is the detected skeleton?
- Which preprocessor produced cleaner subject coverage?
- Is the detected person likely the main subject (via MainSubject_Mask overlap)?

---

## 3 - Measurement Determinism

Evidence extraction is deterministic and lossless. The same generated output
will always produce the same Evidence record. There is no noise or rounding
introduced by the measurement process itself.

This was confirmed empirically: feeding a baseline output as a fake LoRA sample
produced byte-identical measurements to the original baseline record - zero delta,
no drift of any kind. Measurement deltas observed in Synthesis are LoRA-induced,
not instrument noise.

---

## 4 - What Evidence Cannot Answer

Evidence is per-sample. It cannot by itself answer:

- "Did the LoRA change anything?" - that requires a Pair (baseline + LoRA sample
  under the same method and seed). Pairs and deltas live in Synthesis.
- "Is this LoRA good or bad?" - Evidence records facts; judgment is the operator's.
- "Is this LoRA's effect consistent?" - that requires multiple seeds. Consistency
  analysis lives in Synthesis.
- "What does this LoRA do when triggered?" - Evidence under a prompt that does not
  invoke the LoRA's trigger words measures unprompted behavior only. Trigger-gated
  behavior requires a different Method containing the trigger.

---

## 5 - ValueRef Use In Evidence

Non-scalar measurement outputs may be stored as ValueRefs: portable JSON objects
that point at content-addressed assets.

Evidence-side notes:

- the Evidence record stores the ValueRef object, not the live tensor or image object
- if an asset write fails, the slot carries an explicit Invalid wrapper instead
  of pretending the asset exists
- `path` is a machine-local resolution hint; content identity comes from
  `content_hash`

Canonical ValueRef structure, hashing policy, storage formats, and lifecycle law
live in `08_ASSET_AND_VALUEREF_POLICY.md`.

---

## 6 - What Evidence Does Not Store

Core principle: store raw signals; do not pre-bake derived values that can be
recomputed losslessly from what is already stored.

| Criterion | Examples |
|-----------|---------|
| Re-derivable from stored raw assets | per-mask `pixel_stats` and `edge_density`; masked CLIP region readings are computed at comparison time from stored image, mask, and spatial features |
| Process metadata, not measurement | InsightFace model name, CLIP architecture class names - model hashes already carry identity |
| Consumer-layer or report-layer output | rankings, recipe selections, and other rolled-up judgments belong in Synthesis rather than Evidence |
