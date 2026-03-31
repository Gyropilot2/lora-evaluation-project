"""
contracts/metrics_registry.py — canonical metric registry.

Every metric key used in Evidence measurements must be registered here.
Registration is the single source of truth for metric identity, semantics, and
metric-owned metadata.

This registry is not a frontend convenience layer and it is not a recipe file.
Its job is to answer "what is this metric?" for both raw metrics and compound
metrics. If a downstream consumer needs stable facts about a metric and cannot
infer them honestly, those facts belong here if they are truly about the metric
itself.

Each entry defines:
  key            — unique snake_case identifier used in Evidence measurements dict
  unit           — measurement unit (ratio, cosine_distance, l2_distance, degrees,
                   count, score, embedding_norm, embedding_activation, bool, scalar)
  value_min      — optional intrinsic lower bound for the metric value
  value_max      — optional intrinsic upper bound for the metric value
  value_type     — float | int | bool | str
  polarity       — higher_is_better | lower_is_better | neutral
  tier           — metric taxonomy tier:
                     raw_evidence   : direct measurement, no baseline comparison
                     delta          : direct comparison vs. baseline (same-seed pairing)
                     derived        : computed from structured evidence (not direct pixel diff)
                     composite_exp  : provisional weighted combination, not yet validated
  reliability    — validation status:
                     confirmed      : validated by calibration + multi-method evidence
                     provisional    : plausible signal, not yet fully validated
  reliability_metric_key — optional companion metric key for this metric's reliability slot
  dropped_metric_key     — optional companion metric key for this metric's dropped/failure slot
  component_metric_keys  — optional ordered list of literal ingredient metrics that
                           actually participate in building this metric
  peer_metric_keys       — optional ordered list of corroborating / second-opinion
                           metrics that help interpret this metric but do not build it
  selection_metric_keys  — optional ordered list of source-pick / gate / status
                           metrics explaining which upstream path was chosen
  inspection_graph_metric_keys — optional ordered list of numeric companion metrics
                           worth graphing in review inspection surfaces
  metric_def_hash — BLAKE3 hash of this metric's definition (computed at import time)
  deprecated     — True if this metric key should no longer be produced
  description    — explanation of what is measured

Rules:
  - Never reuse a key for a different semantic meaning. Add _v2 suffix instead.
  - If metric semantics change, bump the definition and its metric_def_hash changes automatically.
  - Deprecated metrics emit a WARN diagnostic when produced; they are never silently dropped.
  - All keys in Evidence measurements must exist here; the review assembly layer enforces this.
  - Registry fields describe metric anatomy, not recipe promotion. A metric may declare
    hero-style anatomy here without being selected by a consumer recipe.
  - Compound metrics are still metrics. Their anatomy may live here when it is part of
    the metric definition rather than a consumer's presentation choice.
  - Downstream consumers should consume this metadata, not recreate it from ad hoc maps.
  - Legacy authoring may still carry older range-ceiling hints in the raw definitions during
    migration, but the public registry contract exported from this module is `value_min` /
    `value_max` / `value_type`, not `range_hint` / `ceiling` / `sample_ceiling`.

Usage:
    from contracts.metrics_registry import get_metric, all_metrics, METRICS_REGISTRY

    entry = get_metric("img_px_diff")
    # → {"key": "clip_delta_cos", "unit": "cosine_distance", ..., "metric_def_hash": "..."}
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Metric definitions (source of truth — edit here, hash updates automatically)
# ---------------------------------------------------------------------------

# Format: list of dicts. Each dict must have all keys except metric_def_hash
# (that is computed from the other fields at module load time).
_METRIC_DEFINITIONS: list[dict] = [

    # ── Legacy entries (deprecated — superseded by current metric naming) ──────


    # ── Luminance (raw_evidence) ───────────────────────────────────────────────

    {
        "key": "lum_mean",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average brightness across all pixels. 0 = pure black, 1 = pure white.",
    },
    {
        "key": "lum_std",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Standard deviation of pixel brightness. Higher = more contrast between dark and light areas.",
    },
    {
        "key": "lum_min",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Darkest pixel luminance value in the image.",
    },
    {
        "key": "lum_max",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Brightest pixel luminance value in the image.",
    },

    # ── Raw image stats (raw_evidence) ────────────────────────────────────────

    {
        "key": "px_mean",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average raw RGB pixel value across the whole image.",
    },
    {
        "key": "px_std",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Standard deviation of raw RGB pixel values. Higher = more varied / contrasty.",
    },

    # ── Face detection (raw_evidence) ─────────────────────────────────────────

    {
        "key": "face_count",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Number of distinct faces found by the face analysis model.",
    },
    {
        "key": "det_score",
        "unit": "score",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Face detector certainty score. Higher = more confident a face is present. "
            "Low values do not indicate LoRA failure — detector confidence reflects pose/angle, "
            "not LoRA quality."
        ),
    },

    # ── Head pose angles (raw_evidence) ───────────────────────────────────────

    {
        "key": "pitch",
        "unit": "degrees",
        "value_min": -90.0,
        "value_max": 90.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Vertical head angle in degrees. Negative = looking down, positive = looking up.",
    },
    {
        "key": "yaw",
        "unit": "degrees",
        "value_min": -90.0,
        "value_max": 90.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Horizontal head rotation in degrees. Negative = turned left, positive = turned right.",
    },
    {
        "key": "roll",
        "unit": "degrees",
        "value_min": -90.0,
        "value_max": 90.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Head tilt angle in degrees. Positive = tilted right (clockwise), negative = tilted left.",
    },

    # ── CLIP ViT-H/14 face-region spatial pools (raw_evidence) ───────────────

    {
        "key": "clip_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Fraction of CLIP's 1024 spatial patches that overlap the face mask region. "
            "Higher = face takes up more of the image from CLIP's perspective."
        ),
    },
    {
        "key": "clip_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "L2 magnitude of the CLIP embedding averaged over face-region patches. "
            "Reflects how strongly CLIP encodes the face content."
        ),
    },
    {
        "key": "clip_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 1024.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of CLIP spatial patches (out of 1024 total) that overlap the face mask.",
    },

    # ── CLIP ViT-H/14 background, skin, clothing, hair (raw_evidence) ─────────

    {
        "key": "clip_bg_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Fraction of CLIP's 1024 patches that overlap the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "clip_bg_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "L2 magnitude of CLIP embedding pooled over the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "clip_bg_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 1024.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Count of CLIP patches overlapping the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "clip_skin_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of CLIP's 1024 patches that overlap the skin mask region.",
    },
    {
        "key": "clip_skin_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of CLIP embedding pooled over skin-mask patches.",
    },
    {
        "key": "clip_skin_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 1024.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of CLIP patches overlapping the skin mask region.",
    },
    {
        "key": "clip_cloth_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of CLIP's 1024 patches that overlap the clothing mask region.",
    },
    {
        "key": "clip_cloth_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of CLIP embedding pooled over clothing-mask patches.",
    },
    {
        "key": "clip_cloth_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 1024.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of CLIP patches overlapping the clothing mask region.",
    },
    {
        "key": "clip_hair_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of CLIP's 1024 patches that overlap the hair mask region.",
    },
    {
        "key": "clip_hair_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of CLIP embedding pooled over hair-mask patches.",
    },
    {
        "key": "clip_hair_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 1024.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of CLIP patches overlapping the hair mask region.",
    },

    # ── Depth map stats (raw_evidence) ────────────────────────────────────────

    {
        "key": "depth_mean",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average depth map value. Higher = depth estimator perceives scene as more spatially deep.",
    },
    {
        "key": "depth_std",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Standard deviation of the depth map. Higher = more variation in perceived depth across the scene.",
    },

    # ── Face extras (raw_evidence) ────────────────────────────────────────────

    {
        "key": "face_age",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Estimated age of the detected face in years.",
    },
    {
        "key": "face_emb_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of the raw face embedding vector before normalisation.",
    },
    {
        "key": "face_bbox_area",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Face bounding box area as a fraction of total image pixels. Higher = face is larger in frame.",
    },
    {
        "key": "face_gender_f",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "1.0 if detected gender is female, 0.0 if male. Averaged across samples: fraction detected as female.",
    },

    # ── Normal and edge map stats (raw_evidence) ──────────────────────────────

    {
        "key": "normal_mean",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average pixel value of the surface-normal map output.",
    },
    {
        "key": "normal_std",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Standard deviation of the normal map. Higher = more varied surface orientations in the scene.",
    },
    {
        "key": "edge_mean",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average pixel value of the edge-detection map. Higher = more / stronger edges.",
    },
    {
        "key": "edge_std",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Standard deviation of the edge map.",
    },

    # ── SigLIP pooled_mean per mask region (raw_evidence) ─────────────────────

    {
        "key": "siglip_face_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation value of SigLIP spatial tokens in the face region.",
    },
    {
        "key": "siglip_bg_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Average activation of SigLIP tokens in the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "siglip_skin_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of SigLIP tokens in the skin region.",
    },
    {
        "key": "siglip_cloth_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of SigLIP tokens in the clothing region.",
    },
    {
        "key": "siglip_hair_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of SigLIP tokens in the hair region.",
    },

    # ── CLIP ViT-L/14 per-mask spatial pools (raw_evidence) ───────────────────

    {
        "key": "vitl_face_cov",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of CLIP ViT-L/14's 256 spatial patches overlapping the face mask.",
    },
    {
        "key": "vitl_face_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of ViT-L/14 spatial embedding pooled over face-region patches.",
    },
    {
        "key": "vitl_face_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 256.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of ViT-L/14 patches overlapping the face mask.",
    },
    {
        "key": "vitl_face_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of ViT-L/14 spatial tokens in the face region.",
    },
    {
        "key": "vitl_bg_cov",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Fraction of ViT-L/14's 256 patches overlapping the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "vitl_bg_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "L2 magnitude of ViT-L/14 embedding pooled over the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "vitl_bg_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 256.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Count of ViT-L/14 patches overlapping the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "vitl_bg_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Average activation of ViT-L/14 tokens in the true scene background, "
            "derived as the inverse of `masks.main_subject`."
        ),
    },
    {
        "key": "vitl_skin_cov",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of ViT-L/14's 256 patches overlapping the skin mask.",
    },
    {
        "key": "vitl_skin_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of ViT-L/14 embedding pooled over skin-region patches.",
    },
    {
        "key": "vitl_skin_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 256.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of ViT-L/14 patches overlapping the skin mask.",
    },
    {
        "key": "vitl_skin_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of ViT-L/14 tokens in the skin region.",
    },
    {
        "key": "vitl_cloth_cov",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of ViT-L/14's 256 patches overlapping the clothing mask.",
    },
    {
        "key": "vitl_cloth_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of ViT-L/14 embedding pooled over clothing-region patches.",
    },
    {
        "key": "vitl_cloth_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 256.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of ViT-L/14 patches overlapping the clothing mask.",
    },
    {
        "key": "vitl_cloth_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of ViT-L/14 tokens in the clothing region.",
    },
    {
        "key": "vitl_hair_cov",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Fraction of ViT-L/14's 256 patches overlapping the hair mask.",
    },
    {
        "key": "vitl_hair_norm",
        "unit": "embedding_norm",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "L2 magnitude of ViT-L/14 embedding pooled over hair-region patches.",
    },
    {
        "key": "vitl_hair_patches",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 256.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Count of ViT-L/14 patches overlapping the hair mask.",
    },
    {
        "key": "vitl_hair_mean",
        "unit": "embedding_activation",
        "value_min": None,
        "value_max": None,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Average activation of ViT-L/14 tokens in the hair region.",
    },

    # ── Pose evidence — raw keypoint counts and coverage (raw_evidence) ────────

    {
        "key": "pose_openpose_person_count",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Number of people detected in the OpenPose grouped keypoint output.",
    },
    {
        "key": "pose_openpose_joint_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy scalar: fraction of OpenPose joints inside the MainSubject mask.",
    },
    {
        "key": "pose_openpose_primary_recognized",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy scalar: number of non-missing joints on the primary (largest area) detected person per OpenPose.",
    },
    {
        "key": "pose_dw_person_count",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Number of people detected in the DW-Pose grouped keypoint output.",
    },
    {
        "key": "pose_dw_joint_coverage",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy scalar: fraction of DW-Pose joints inside the MainSubject mask.",
    },
    {
        "key": "pose_dw_primary_recognized",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "raw_evidence",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy scalar: number of non-missing joints on the primary person per DW-Pose.",
    },

    # ── Pixel-level drift — same-seed pairing (delta) ─────────────────────────

    {
        "key": "img_px_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Mean absolute RGB pixel difference between this LoRA sample and its same-seed baseline. "
            "Total visible change in the rendered image."
        ),
    },
    {
        "key": "depth_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute pixel diff between LoRA and baseline depth maps. High = spatial structure changed.",
    },
    {
        "key": "normal_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute pixel diff between LoRA and baseline surface-normal maps. High = surface orientations changed.",
    },
    {
        "key": "edge_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute pixel diff between LoRA and baseline edge-detection maps. Very sensitive to structural changes.",
    },
    {
        "key": "lum_face_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute luminance-map difference inside the baseline face mask.",
    },
    {
        "key": "lum_cloth_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute luminance-map difference inside the baseline clothing mask.",
    },
    {
        "key": "lum_bg_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute luminance-map difference inside the baseline background mask.",
    },
    {
        "key": "lum_character_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Mean absolute luminance difference inside the character mask. "
            "Captures total subject brightness shift, excluding background."
        ),
    },
    {
        "key": "cloth_edge_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Mean absolute edge-map difference inside the baseline clothing mask.",
    },
    {
        "key": "bg_depth_diff",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "reliability_metric_key": None,
        "dropped_metric_key": None,
        "component_metric_keys": [],
        "peer_metric_keys": ["composition_package_exp", "depth_diff", "normal_diff", "lum_bg_diff"],
        "inspection_graph_metric_keys": ["composition_package_exp", "depth_diff", "normal_diff", "lum_bg_diff"],
        "deprecated": False,
        "description": "Mean absolute depth-map difference inside the baseline background mask.",
    },
    {
        "key": "face_bbox_area_delta",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Absolute change in primary-face bounding-box area fraction vs baseline. Higher = face got larger/smaller.",
    },
    {
        "key": "face_center_shift",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Euclidean movement of the primary-face bbox center vs baseline, normalised by image diagonal. "
            "Higher = face moved within the frame."
        ),
    },

    # ── Pose evidence paired deltas (delta) ───────────────────────────────────

    {
        "key": "pose_openpose_person_count_delta",
        "unit": "count",
        "value_min": None,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Change in OpenPose person count vs baseline (LoRA minus baseline).",
    },
    {
        "key": "pose_openpose_joint_coverage_delta",
        "unit": "ratio",
        "value_min": -1.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy delta for the old OpenPose joint-coverage scalar.",
    },
    {
        "key": "pose_openpose_primary_recognized_delta",
        "unit": "count",
        "value_min": None,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy delta for the old OpenPose primary-joints scalar.",
    },
    {
        "key": "pose_dw_person_count_delta",
        "unit": "count",
        "value_min": None,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "Change in DW-Pose person count vs baseline.",
    },
    {
        "key": "pose_dw_joint_coverage_delta",
        "unit": "ratio",
        "value_min": -1.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy delta for the old DW-Pose joint-coverage scalar.",
    },
    {
        "key": "pose_dw_primary_recognized_delta",
        "unit": "count",
        "value_min": None,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": True,
        "description": "Deprecated legacy delta for the old DW-Pose primary-joints scalar.",
    },

    # ── Cosine distances — global embeddings (delta) ──────────────────────────

    {
        "key": "face_cos_dist",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Cosine distance (0–2) between L2-normalised face embeddings of LoRA and same-seed baseline. "
            "Directly measures identity drift. Note: unreliable for pose-changing LoRAs "
            "(angled faces inflate the score even when identity is preserved)."
        ),
    },
    {
        "key": "clip_global_cos_dist",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "component_metric_keys": [],
        "peer_metric_keys": ["vitl_global_cos_dist"],
        "inspection_graph_metric_keys": ["vitl_global_cos_dist"],
        "deprecated": False,
        "description": (
            "Cosine distance between global SigLIP image embeddings of LoRA and same-seed baseline. "
            "Overall semantic drift across the whole image."
        ),
    },
    {
        "key": "vitl_global_cos_dist",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Cosine distance between global CLIP ViT-L/14 CLS embeddings of LoRA and same-seed baseline. "
            "Second-opinion semantic drift from a different CLIP architecture."
        ),
    },

    # ── Cosine distances — per mask region (delta) ────────────────────────────

    {
        "key": "siglip_cos_face",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over baseline face-mask patches.",
    },
    {
        "key": "siglip_cos_bg",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "reliability_metric_key": None,
        "dropped_metric_key": None,
        "component_metric_keys": [],
        "peer_metric_keys": ["background_package_exp", "vitl_cos_bg", "lum_bg_diff"],
        "inspection_graph_metric_keys": ["background_package_exp", "vitl_cos_bg", "lum_bg_diff"],
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over background-mask patches.",
    },
    {
        "key": "siglip_cos_skin",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over skin-mask patches.",
    },
    {
        "key": "siglip_cos_cloth",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "component_metric_keys": [],
        "peer_metric_keys": ["vitl_cos_cloth", "lum_cloth_diff"],
        "inspection_graph_metric_keys": ["vitl_cos_cloth", "lum_cloth_diff"],
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over clothing-mask patches.",
    },
    {
        "key": "siglip_cos_hair",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over hair-mask patches.",
    },
    {
        "key": "siglip_cos_character",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "SigLIP spatial cosine distance pooled over the character mask. Overall subject semantic drift.",
    },
    {
        "key": "vitl_cos_face",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "ViT-L/14 spatial cosine distance pooled over face-mask patches. "
            "Coarser patches (32px) — treat with care on small faces."
        ),
    },
    {
        "key": "vitl_cos_bg",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "ViT-L/14 spatial cosine distance pooled over background-mask patches.",
    },
    {
        "key": "vitl_cos_skin",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "ViT-L/14 spatial cosine distance pooled over skin-mask patches.",
    },
    {
        "key": "vitl_cos_cloth",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "ViT-L/14 spatial cosine distance pooled over clothing-mask patches.",
    },
    {
        "key": "vitl_cos_hair",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "ViT-L/14 spatial cosine distance pooled over hair-mask patches.",
    },
    {
        "key": "vitl_cos_character",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "delta",
        "reliability": "provisional",
        "deprecated": False,
        "description": "ViT-L/14 spatial cosine distance pooled over the character mask. Second opinion on subject semantic drift.",
    },

    # ── Derived metrics (derived) ──────────────────────────────────────────────

    {
        "key": "head_rot_drift",
        "unit": "degrees",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "3D orientation change magnitude: sqrt(Δpitch² + Δyaw² + Δroll²). "
            "Single number capturing total head pose shift from baseline. "
            "Note: method quality matters — anarchy prompt inflates this regardless of LoRA."
        ),
    },
    {
        "key": "cross_seed_face_dist",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "provisional",
        "reliability_metric_key": None,
        "dropped_metric_key": None,
        "component_metric_keys": [],
        "deprecated": True,
        "description": (
            "Deprecated retired group-level metric: mean pairwise cosine distance between "
            "face embeddings across seeds at one eval+strength. Removed from live production "
            "because cross-seed consistency belongs to recipe/aggregation work, not per-sample review payloads."
        ),
    },
    {
        "key": "pose_selected_source",
        "unit": "label",
        "value_min": None,
        "value_max": None,
        "value_type": "str",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "Winning pose detector source used for the headline pose metrics: `openpose` or `dw`. "
            "None when neither source can produce a comparable pose reading."
        ),
    },
    {
        "key": "pose_openpose_baseline_has_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when OpenPose baseline evidence has a person0 entry (`people[0]`).",
    },
    {
        "key": "pose_openpose_lora_has_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when OpenPose LoRA evidence has a person0 entry (`people[0]`).",
    },
    {
        "key": "pose_openpose_comparable_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when both baseline and LoRA OpenPose evidence have a person0 to compare.",
    },
    {
        "key": "pose_openpose_baseline_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of non-missing OpenPose baseline joints on person0.",
    },
    {
        "key": "pose_openpose_lora_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of non-missing OpenPose LoRA joints on person0.",
    },
    {
        "key": "pose_openpose_comparable_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of baseline-visible person0 joints that OpenPose also retained on the LoRA side.",
    },
    {
        "key": "pose_openpose_baseline_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of angle-computable OpenPose baseline person0 triplets across the 8 tracked limb triplets.",
    },
    {
        "key": "pose_openpose_lora_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of angle-computable OpenPose LoRA person0 triplets across the 8 tracked limb triplets.",
    },
    {
        "key": "pose_openpose_comparable_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of tracked limb triplets that remain angle-computable on both baseline and LoRA OpenPose person0.",
    },
    {
        "key": "pose_openpose_angle_drift",
        "unit": "degrees",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Mean absolute OpenPose person0 angle drift across comparable limb triplets. Lower = more faithful pose.",
    },
    {
        "key": "pose_openpose_reliability",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "higher_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "OpenPose person0 joint-retention ratio against baseline: comparable_joints / baseline_joints.",
    },
    {
        "key": "pose_openpose_dropped",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when OpenPose cannot produce a comparable person0 angle-drift score for this pair.",
    },
    {
        "key": "pose_dw_baseline_has_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when DW-Pose baseline evidence has a person0 entry (`people[0]`).",
    },
    {
        "key": "pose_dw_lora_has_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when DW-Pose LoRA evidence has a person0 entry (`people[0]`).",
    },
    {
        "key": "pose_dw_comparable_person0",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when both baseline and LoRA DW-Pose evidence have a person0 to compare.",
    },
    {
        "key": "pose_dw_baseline_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of non-missing DW-Pose baseline joints on person0.",
    },
    {
        "key": "pose_dw_lora_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of non-missing DW-Pose LoRA joints on person0.",
    },
    {
        "key": "pose_dw_comparable_joints",
        "unit": "count",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of baseline-visible person0 joints that DW-Pose also retained on the LoRA side.",
    },
    {
        "key": "pose_dw_baseline_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of angle-computable DW-Pose baseline person0 triplets across the 8 tracked limb triplets.",
    },
    {
        "key": "pose_dw_lora_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of angle-computable DW-Pose LoRA person0 triplets across the 8 tracked limb triplets.",
    },
    {
        "key": "pose_dw_comparable_triplets",
        "unit": "count",
        "value_min": 0.0,
        "value_max": 8.0,
        "value_type": "int",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Count of tracked limb triplets that remain angle-computable on both baseline and LoRA DW-Pose person0.",
    },
    {
        "key": "pose_dw_angle_drift",
        "unit": "degrees",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "Mean absolute DW-Pose person0 angle drift across comparable limb triplets. Lower = more faithful pose.",
    },
    {
        "key": "pose_dw_reliability",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "higher_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "DW-Pose person0 joint-retention ratio against baseline: comparable_joints / baseline_joints.",
    },
    {
        "key": "pose_dw_dropped",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": "True when DW-Pose cannot produce a comparable person0 angle-drift score for this pair.",
    },
    {
        "key": "pose_angle_drift",
        "unit": "degrees",
        "value_min": 0.0,
        "value_max": None,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "reliability_metric_key": "pose_reliability",
        "dropped_metric_key": "pose_dropped",
        "component_metric_keys": [],
        "inspection_graph_metric_keys": [
            "pose_openpose_angle_drift",
            "pose_dw_angle_drift",
            "pose_openpose_reliability",
            "pose_dw_reliability",
        ],
        "selection_metric_keys": [
            "pose_selected_source",
            "pose_openpose_angle_drift",
            "pose_openpose_reliability",
            "pose_openpose_dropped",
            "pose_dw_angle_drift",
            "pose_dw_reliability",
            "pose_dw_dropped",
        ],
        "deprecated": False,
        "description": (
            "Selected pose angle drift from the better of the two same-source person0 comparisons "
            "(OpenPose vs. OpenPose, DW-Pose vs. DW-Pose). Winner is chosen by higher "
            "`pose_reliability`, with lower drift as the tie-break. Lower = more faithful pose."
        ),
    },
    {
        "key": "pose_detection_lost",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": True,
        "description": (
            "Deprecated legacy flag: baseline had some pose-detected person while the LoRA side "
            "had none across both sources. Retired because it conflates genuine pose failure with "
            "subject collapse and is superseded by the explicit per-source person0 facts."
        ),
    },

    # ── Provisional composite metrics (composite_exp) ─────────────────────────

    {
        "key": "pose_reliability",
        "unit": "ratio",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "higher_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "Selected-source joint-retention reliability against baseline person0: "
            "`comparable_joints / baseline_joints` for the source chosen by `pose_selected_source`. "
            "Higher = the LoRA retained more of the baseline body structure."
        ),
    },
    {
        "key": "pose_dropped",
        "unit": "boolean",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "True when neither pose source can supply a selected pose angle-drift score. "
            "False when `pose_selected_source` can provide headline pose metrics."
        ),
    },
    {
        "key": "face_detection_lost",
        "unit": "boolean",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "True when baseline had >=1 face detected and LoRA has 0. False when LoRA still "
            "has >=1 face. None when baseline had no face to lose."
        ),
    },
    {
        "key": "identity_dropped",
        "unit": "boolean",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "lower_is_better",
        "tier": "derived",
        "reliability": "provisional",
        "deprecated": True,
        "description": (
            "Deprecated alias of `face_detection_lost`. Retired from live production so the "
            "identity dropped slot points directly at the real face-loss signal instead of a duplicate metric."
        ),
    },
    {
        "key": "identity_gate_status",
        "unit": "label",
        "value_min": None,
        "value_max": None,
        "value_type": "str",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "Identity-package face-anchor gate status. `open` means the live package scores were "
            "allowed to surface normally; `gated_no_lora_face` means the LoRA side had no detected face, "
            "so the live identity package scores were intentionally nulled."
        ),
    },
    {
        "key": "identity_region_drift_pre_gate_exp",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Visibility/debug companion for `identity_region_drift_exp`: the raw region-only package "
            "value before the current face-anchor gate is applied. Not a judgment-bearing live score "
            "when `identity_gate_status != open`."
        ),
    },
    {
        "key": "identity_region_drift_usable",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "True when `identity_region_drift_exp` has a live usable score after gate application. "
            "False when the score was gated off or no package value could be formed."
        ),
    },
    {
        "key": "identity_region_drift_exp",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "component_metric_keys": [
            "identity_region_drift_pre_gate_exp",
            "identity_region_drift_usable",
            "identity_gate_status",
            "siglip_cos_face",
            "vitl_cos_face",
            "siglip_cos_hair",
            "vitl_cos_hair",
        ],
        "inspection_graph_metric_keys": [
            "identity_region_drift_pre_gate_exp",
            "siglip_cos_face",
            "vitl_cos_face",
            "siglip_cos_hair",
            "vitl_cos_hair",
        ],
        "deprecated": False,
        "description": (
            "Provisional identity metric: mean of face-region and hair-region semantic drift "
            "across SigLIP and ViT-L/14. Intended to capture character drift while being "
            "less sensitive to background/clothing noise. Live score remains gated to None when "
            "the LoRA side has no detected face; inspect the pre-gate/usability companions for visibility."
        ),
    },
    {
        "key": "identity_region_plus_arcface_pre_gate_exp",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Visibility/debug companion for `identity_region_plus_arcface_exp`: the raw ArcFace-inclusive "
            "package value before the current face-anchor gate is applied. Not a judgment-bearing live score "
            "when `identity_gate_status != open`."
        ),
    },
    {
        "key": "identity_region_plus_arcface_usable",
        "unit": "bool",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "bool",
        "polarity": "neutral",
        "tier": "derived",
        "reliability": "confirmed",
        "deprecated": False,
        "description": (
            "True when `identity_region_plus_arcface_exp` has a live usable score after gate application. "
            "False when the score was gated off or no package value could be formed."
        ),
    },
    {
        "key": "identity_region_plus_arcface_exp",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "reliability_metric_key": "det_score",
        "dropped_metric_key": "face_detection_lost",
        "component_metric_keys": [
            "face_cos_dist",
            "siglip_cos_face",
            "vitl_cos_face",
            "siglip_cos_hair",
            "vitl_cos_hair",
        ],
        "peer_metric_keys": ["identity_region_drift_exp"],
        "inspection_graph_metric_keys": [
            "face_cos_dist",
            "siglip_cos_face",
            "vitl_cos_face",
            "siglip_cos_hair",
            "vitl_cos_hair",
            "det_score",
            "identity_region_drift_exp",
        ],
        "selection_metric_keys": [
            "identity_region_plus_arcface_pre_gate_exp",
            "identity_region_plus_arcface_usable",
            "identity_gate_status",
        ],
        "deprecated": False,
        "description": (
            "Provisional identity metric: region-based identity package plus face_cos_dist. "
            "Keeps ArcFace in the bundle while still leaning on face/hair regional semantic drift. "
            "Live score remains gated to None when the LoRA side has no detected face; inspect the "
            "pre-gate/usability companions for visibility."
        ),
    },
    {
        "key": "background_package_exp",
        "unit": "cosine_distance",
        "value_min": 0.0,
        "value_max": 2.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Provisional background/scenery package: mean of background-region semantic drift "
            "across SigLIP and ViT-L/14."
        ),
    },
    {
        "key": "composition_package_exp",
        "unit": "scalar",
        "value_min": 0.0,
        "value_max": 1.0,
        "value_type": "float",
        "polarity": "lower_is_better",
        "tier": "composite_exp",
        "reliability": "provisional",
        "deprecated": False,
        "description": (
            "Provisional composition/viewpoint package: mean of structural background drift "
            "from depth and normal surfaces."
        ),
    },
]

# ---------------------------------------------------------------------------
# Registry construction (do not edit below this line)
# ---------------------------------------------------------------------------


_RANGE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def _infer_value_type(entry: dict) -> str:
    unit = entry.get("unit")
    if unit in {"bool", "boolean"}:
        return "bool"
    if unit == "count":
        return "int"
    return "float"


def _parse_range_bounds(range_hint: object) -> tuple[float | None, float | None]:
    if not isinstance(range_hint, str):
        return None, None
    normalized = range_hint.strip()
    if normalized in {"True/False", "True/False/None"}:
        return 0.0, 1.0
    if normalized.endswith("+"):
        try:
            return float(normalized[:-1]), None
        except ValueError:
            return None, None
    match = _RANGE_PATTERN.match(normalized)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _normalize_metric_definition(raw: dict) -> dict:
    entry = dict(raw)
    explicit_value_min = entry.get("value_min")
    explicit_value_max = entry.get("value_max")
    explicit_value_type = entry.get("value_type")
    legacy_range_hint = entry.pop("range_hint", None)
    entry.pop("ceiling", None)
    entry.pop("sample_ceiling", None)
    if explicit_value_min is None and explicit_value_max is None:
        value_min, value_max = _parse_range_bounds(legacy_range_hint)
        entry["value_min"] = value_min
        entry["value_max"] = value_max
    else:
        entry["value_min"] = explicit_value_min
        entry["value_max"] = explicit_value_max
    entry["value_type"] = explicit_value_type or _infer_value_type(entry)
    return entry


def _compute_def_hash(defn: dict) -> str:
    """Compute a stable BLAKE3 hash for a metric definition.

    Excludes the 'metric_def_hash' key itself from the hash input so that
    the hash represents only the semantic content of the definition.
    """
    from core.hashing import hash_bytes
    from core.json_codec import canonical_json

    hashable = {k: v for k, v in defn.items() if k != "metric_def_hash"}
    return hash_bytes(canonical_json(hashable).encode("utf-8"))


def _build_registry(definitions: list[dict]) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for raw in definitions:
        entry = _normalize_metric_definition(raw)
        entry["metric_def_hash"] = _compute_def_hash(entry)
        key = entry["key"]
        if key in registry:
            raise ValueError(f"Duplicate metric key in registry: {key!r}")
        registry[key] = entry
    return registry


#: The canonical metric registry, keyed by metric key string.
METRICS_REGISTRY: dict[str, dict] = _build_registry(_METRIC_DEFINITIONS)


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_metric(key: str) -> dict | None:
    """Return the registry entry for the given metric key, or None if not registered."""
    return METRICS_REGISTRY.get(key)


def all_metrics() -> dict[str, dict]:
    """Return a copy of the full metrics registry keyed by metric key."""
    return dict(METRICS_REGISTRY)


def is_deprecated(key: str) -> bool:
    """Return True if the given metric key is registered and marked deprecated."""
    entry = METRICS_REGISTRY.get(key)
    return bool(entry and entry.get("deprecated"))


def metrics_by_tier(tier: str) -> dict[str, dict]:
    """Return all non-deprecated registry entries for the given tier."""
    return {
        k: v for k, v in METRICS_REGISTRY.items()
        if v.get("tier") == tier and not v.get("deprecated")
    }


def metrics_by_reliability(reliability: str) -> dict[str, dict]:
    """Return all non-deprecated registry entries for the given reliability level."""
    return {
        k: v for k, v in METRICS_REGISTRY.items()
        if v.get("reliability") == reliability and not v.get("deprecated")
    }
