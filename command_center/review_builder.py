"""Shared heavy review-data assembly for the review JSON export and the operator app."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from core.diagnostics import emit

try:
    from PIL import Image
except ImportError:
    import subprocess

    emit(
        "WARN",
        "REVIEW.PILLOW_AUTO_INSTALL",
        _WHERE,
        "Pillow not found; installing into active Python environment",
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image

import core.paths as paths
from contracts.procedures_registry import (
    face_detection_lost as procedure_face_detection_lost,
    gate_identity_score,
    pose_pair_metrics as procedure_pose_pair_metrics,
)

_WHERE = "command_center.review_builder"


def method_label(method: dict[str, Any]) -> str:
    """Display method label derived from method inputs."""
    model_name = (method.get("base_model") or {}).get("name") or "unknown-model"
    settings = method.get("settings") or {}
    sampler = settings.get("sampler", "?")
    scheduler = settings.get("scheduler", "?")
    steps = settings.get("steps", "?")
    latent = method.get("latent") or {}
    width, height = latent.get("width", "?"), latent.get("height", "?")
    prompt_hash = ((method.get("conditioning") or {}).get("positive_hash") or "")[:6] or "------"
    return f"{model_name} \u00b7 {sampler}/{scheduler} \u00b7 {steps}s \u00b7 {width}\u00d7{height} \u00b7 prompt:{prompt_hash}"


def method_prompt_text(method: dict[str, Any]) -> str | None:
    """Return the best available declared prompt text from a Method record."""
    conditioning = method.get("conditioning") or {}
    for key in ("positive_text", "prompt_hint", "prompt", "positive_prompt_hint"):
        value = conditioning.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def method_prompt_hint(method: dict[str, Any]) -> str | None:
    """Compatibility alias for older prompt-hint callers."""
    return method_prompt_text(method)

def load_npy_ref(ref, project_root: Path, cache: dict[str, np.ndarray | None] | None = None):
    """Load a ValueRef asset into a float32 numpy array.

    Handles both legacy all-`.npy` assets and the post-4.5 migrated PNG families.
    Returns None on any failure.
    """
    if not ref or not isinstance(ref, dict):
        return None
    path = ref.get("path")
    if not path:
        return None
    if cache is not None and path in cache:
        return cache[path]

    fmt = ref.get("format", "npy")
    asset_type = ref.get("asset_type", "")
    abs_path = str(project_root / path)
    try:
        if fmt == "npy":
            arr = np.load(abs_path).astype(np.float32)
        elif fmt == "png":
            img = Image.open(abs_path)
            if asset_type == "luminance":
                arr = np.array(img.convert("I"), dtype=np.float32) / 65535.0
            else:
                arr = np.array(img, dtype=np.float32) / 255.0
        else:
            arr = None
    except Exception:
        arr = None
    if cache is not None:
        cache[path] = arr
    return arr


def cosine_dist(a, b) -> float | None:
    """Cosine distance = 1 - cosine_similarity. Inputs are flattened + L2-normalised.
    Returns None if either input is None or degenerate (zero-norm)."""
    if a is None or b is None:
        return None
    try:
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return None
        sim = float(np.dot(a / na, b / nb))
        return float(1.0 - max(-1.0, min(1.0, sim)))  # clamp for numerical safety
    except Exception:
        return None


def px_abs_diff(arr_a, arr_b) -> float | None:
    """Mean absolute element-wise difference between two numeric arrays.
    Both cast to float32. Returns None if either input is None."""
    if arr_a is None or arr_b is None:
        return None
    try:
        return float(np.mean(np.abs(
            arr_a.astype(np.float32) - arr_b.astype(np.float32)
        )))
    except Exception:
        return None


def _squeeze_leading_units(arr):
    """Drop leading singleton dims so [1,H,W,C] -> [H,W,C], [1,H,W] -> [H,W]."""
    out = arr
    try:
        while hasattr(out, "ndim") and out.ndim > 2 and out.shape[0] == 1:
            out = out[0]
    except Exception:
        return arr
    return out



def masked_abs_diff(arr_a, arr_b, mask_arr, threshold: float = 0.5) -> float | None:
    """Mean absolute difference inside a binary-ish mask."""
    if arr_a is None or arr_b is None or mask_arr is None:
        return None
    try:
        a = _squeeze_leading_units(arr_a).astype(np.float32)
        b = _squeeze_leading_units(arr_b).astype(np.float32)
        m = _squeeze_leading_units(mask_arr).astype(np.float32)
        if m.ndim != 2 or a.shape[:2] != m.shape[:2] or b.shape[:2] != m.shape[:2]:
            return None
        mask = m > threshold
        if not np.any(mask):
            return None
        diff = np.abs(a - b)
        selected = diff[mask]
        return float(np.mean(selected)) if selected.size else None
    except Exception:
        return None


def bbox_area_and_center(face_analysis: dict | None, image_shape) -> tuple[float | None, tuple[float, float] | None]:
    """Return (area_fraction, center_xy) for the primary face bbox, or (None, None)."""
    fa = face_analysis or {}
    bbox = fa.get("bbox")
    if not bbox or len(bbox) != 4:
        return None, None
    try:
        h, w = (image_shape or [512, 512, 3])[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = abs((x2 - x1) * (y2 - y1)) / float(h * w)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return float(area), center
    except Exception:
        return None, None


def mean_of_present(*vals) -> float | None:
    """Average numeric values, ignoring None. Returns None if nothing is present."""
    xs = [float(v) for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def masked_clip_pool(lhs, mask_arr, n_grid: int, has_cls: bool = False):
    """Pool CLIP spatial tokens whose patches overlap a binary mask.

    lhs:      last_hidden_state [1, N, D] or [1, N+1, D] when has_cls=True
    mask_arr: float mask [1, H, W] or [H, W]; non-zero = region of interest
    n_grid:   patches per side (32 for SigLIP/1024-patch; 16 for ViT-L/14/256-patch)
    has_cls:  True when token[0] is CLS — spatial patches start at index 1

    Returns mean-pooled [D] float32 array, or None if no patches overlap / any input is None.
    """
    if lhs is None or mask_arr is None:
        return None
    try:
        mask_2d = np.asarray(mask_arr).squeeze()     # [H, W]
        if mask_2d.ndim != 2:
            return None

        img_h, img_w = mask_2d.shape[:2]
        ph = img_h // n_grid                         # patch height in pixels
        pw = img_w // n_grid                         # patch width in pixels
        if ph <= 0 or pw <= 0:
            return None

        tokens = lhs[0]                              # [N, D] or [N+1, D]

        # Preserve the old "mean mask coverage per patch > 0.1" rule, but do it
        # in one reshape/mean pass instead of 32x32 or 16x16 Python loops.
        if ph * n_grid == img_h and pw * n_grid == img_w:
            patch_means = (
                mask_2d.reshape(n_grid, ph, n_grid, pw)
                .mean(axis=(1, 3))
                .reshape(-1)
            )
            indices = np.flatnonzero(patch_means > 0.1)
        else:
            indices = []
            for i in range(n_grid):
                for j in range(n_grid):
                    region = mask_2d[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                    if float(region.mean()) > 0.1:
                        indices.append(i * n_grid + j)
            indices = np.asarray(indices, dtype=np.int64)

        if indices.size == 0:
            return None
        if has_cls:
            indices = indices + 1                    # shift past CLS token at position 0

        selected = tokens[indices]                  # [K, D]
        return selected.mean(axis=0)                # [D]
    except Exception:
        return None


def npy_to_b64_jpeg(npy_path: str, quality: int = 88) -> str:
    """Load float32 [H, W, 3] npy image, return JPEG as base64 string."""
    arr = np.load(npy_path)
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_to_hw_float(mask_arr):
    """Normalize a stored mask asset to float32 [H, W], or None on shape mismatch."""
    if mask_arr is None:
        return None
    try:
        mask_2d = _squeeze_leading_units(mask_arr).astype(np.float32)
        return mask_2d if mask_2d.ndim == 2 else None
    except Exception:
        return None


def _true_background_mask(mask_arr):
    """Return the true scene background mask as the inverse of `main_subject`."""
    subject_mask = _mask_to_hw_float(mask_arr)
    if subject_mask is None:
        return None
    return np.clip(1.0 - subject_mask, 0.0, 1.0)


def _resize_mask_bilinear(mask_arr, out_h: int, out_w: int):
    """Resize a 2D mask with the same half-pixel bilinear convention as PyTorch."""
    try:
        mask = _mask_to_hw_float(mask_arr)
        if mask is None:
            return None
        in_h, in_w = mask.shape
        if in_h == out_h and in_w == out_w:
            return mask

        ys = ((np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h)) - 0.5
        xs = ((np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w)) - 0.5

        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, in_h - 1)
        x1 = np.clip(x0 + 1, 0, in_w - 1)
        y0 = np.clip(y0, 0, in_h - 1)
        x0 = np.clip(x0, 0, in_w - 1)

        wy = (ys - y0).reshape(out_h, 1)
        wx = (xs - x0).reshape(1, out_w)

        top_left = mask[y0[:, None], x0[None, :]]
        top_right = mask[y0[:, None], x1[None, :]]
        bottom_left = mask[y1[:, None], x0[None, :]]
        bottom_right = mask[y1[:, None], x1[None, :]]

        top = top_left * (1.0 - wx) + top_right * wx
        bottom = bottom_left * (1.0 - wx) + bottom_right * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)
    except Exception:
        return None


def _masked_patch_pool_stats(last_hidden_state, mask_arr):
    """Pure-numpy patch-pool stats matching the stored extractor math."""
    if last_hidden_state is None or mask_arr is None:
        return {}
    try:
        lhs = np.asarray(last_hidden_state, dtype=np.float32)
        if lhs.ndim != 3 or lhs.shape[0] != 1:
            return {}

        total_tokens = lhs.shape[1]
        n_no_cls = total_tokens - 1
        gs_no_cls = int(n_no_cls ** 0.5)
        has_cls = gs_no_cls * gs_no_cls == n_no_cls

        if has_cls:
            patch_features = lhs[:, 1:, :]
        else:
            gs_all = int(total_tokens ** 0.5)
            if gs_all * gs_all != total_tokens:
                return {
                    "status": "error",
                    "detail": (
                        f"non-square patch grid: tried {n_no_cls} (CLS skip) "
                        f"and {total_tokens} (no CLS skip)"
                    ),
                }
            patch_features = lhs

        n_patches = patch_features.shape[1]
        grid_size = int(n_patches ** 0.5)
        resized_mask = _resize_mask_bilinear(mask_arr, grid_size, grid_size)
        if resized_mask is None:
            return {}

        patch_weights = resized_mask.reshape(-1).astype(np.float32)
        total_weight = float(patch_weights.sum())
        if total_weight < 1e-6:
            return {"status": "empty_mask", "covered_patches": 0, "total_patches": n_patches}

        features = patch_features[0]
        pooled = (features * patch_weights[:, None]).sum(axis=0) / total_weight
        return {
            "covered_patches": int((patch_weights > 0.1).sum()),
            "total_patches": n_patches,
            "coverage_ratio": round(total_weight / n_patches, 4),
            "pooled_norm": float(np.linalg.norm(pooled)),
            "pooled_mean": float(pooled.mean()),
            "has_cls_token": has_cls,
        }
    except Exception:
        return {}


def _derived_true_background_patch_pool(sample: dict, clip_key: str | None, load_ref=None) -> dict[str, Any]:
    """Recompute the raw background patch pool from the inverse `main_subject` mask.

    Stored patch pools are authored for the literal stored mask keys. Since
    `main_subject` is the subject/foreground region, raw `_bg` metrics must be
    derived from its inverse rather than read from the stored `main_subject`
    pool under a misleading `_bg` name.
    """
    if not clip_key or load_ref is None:
        return {}
    try:
        clip_slot = ((sample.get("clip_vision") or {}).get(clip_key) or {})
        lhs = load_ref(clip_slot.get("last_hidden_state"))
        masks = sample.get("masks") or {}
        subject_mask = load_ref((masks.get("main_subject") or {}).get("output"))
        true_bg_mask = _true_background_mask(subject_mask)
        if lhs is None or true_bg_mask is None:
            return {}
        return _masked_patch_pool_stats(lhs, true_bg_mask)
    except Exception:
        return {}


def extract_metrics(sample: dict, *, load_ref=None) -> dict:
    """Pull flat numeric metrics from a Treasurer sample record."""
    m = {}

    lum = (sample.get("luminance") or {}).get("stats") or {}
    m["lum_mean"] = lum.get("mean")
    m["lum_std"]  = lum.get("std")
    m["lum_min"]  = lum.get("min")
    m["lum_max"]  = lum.get("max")

    img_px = (sample.get("image") or {}).get("pixel_stats") or {}
    m["px_mean"] = img_px.get("mean")
    m["px_std"]  = img_px.get("std")

    fa = sample.get("face_analysis") or {}
    m["face_count"] = fa.get("face_count")
    m["det_score"]  = fa.get("det_score")

    pose = fa.get("pose") or {}
    if isinstance(pose, dict):
        m["pitch"] = pose.get("pitch")
        m["yaw"]   = pose.get("yaw")
        m["roll"]  = pose.get("roll")
    else:
        m["pitch"] = m["yaw"] = m["roll"] = None

    # ── Identify CLIP model keys by embedding dimension ───────────────────────
    # SigLIP:     global_embedding_shape[-1] == 1152, 1024 spatial patches (32×32)
    # ViT-L/14:   global_embedding_shape[-1] == 768,  256  spatial patches (16×16) + CLS
    siglip_key = vitl_key = None
    for k, v in (sample.get("clip_vision") or {}).items():
        sh = (v or {}).get("global_embedding_shape") or []
        if sh and sh[-1] == 1152:
            siglip_key = k
        elif sh and sh[-1] == 768:
            vitl_key = k

    # SigLIP patch pools (keep backward-compat field names; add pooled_mean)
    if siglip_key:
        cv_s = sample["clip_vision"][siglip_key] or {}
        pp = cv_s.get("patch_pools", {}).get("face", {}).get("patch_pool", {})
        m["clip_coverage"]    = pp.get("coverage_ratio")
        m["clip_norm"]        = pp.get("pooled_norm")
        m["clip_patches"]     = pp.get("covered_patches")
        m["siglip_face_mean"] = pp.get("pooled_mean")
        bg_pp = _derived_true_background_patch_pool(sample, siglip_key, load_ref=load_ref)
        for mask_name, short in [("main_subject","bg"),("skin","skin"),("clothing","cloth"),("hair","hair")]:
            if short == "bg":
                pp2 = bg_pp
            else:
                pp2 = cv_s.get("patch_pools", {}).get(mask_name, {}).get("patch_pool", {})
            m[f"clip_{short}_coverage"]  = pp2.get("coverage_ratio")
            m[f"clip_{short}_norm"]      = pp2.get("pooled_norm")
            m[f"clip_{short}_patches"]   = pp2.get("covered_patches")
            m[f"siglip_{short}_mean"]    = pp2.get("pooled_mean")
    else:
        m["clip_coverage"] = m["clip_norm"] = m["clip_patches"] = m["siglip_face_mean"] = None
        for short in ["bg", "skin", "cloth", "hair"]:
            m[f"clip_{short}_coverage"] = m[f"clip_{short}_norm"] = m[f"clip_{short}_patches"] = None
            m[f"siglip_{short}_mean"] = None

    # CLIP ViT-L/14 patch pools (new)
    if vitl_key:
        cv_v = sample["clip_vision"][vitl_key] or {}
        bg_pp = _derived_true_background_patch_pool(sample, vitl_key, load_ref=load_ref)
        for mask_name, short in [("face","face"),("main_subject","bg"),("skin","skin"),("clothing","cloth"),("hair","hair")]:
            if short == "bg":
                pp = bg_pp
            else:
                pp = cv_v.get("patch_pools", {}).get(mask_name, {}).get("patch_pool", {})
            m[f"vitl_{short}_cov"]     = pp.get("coverage_ratio")
            m[f"vitl_{short}_norm"]    = pp.get("pooled_norm")
            m[f"vitl_{short}_patches"] = pp.get("covered_patches")
            m[f"vitl_{short}_mean"]    = pp.get("pooled_mean")
    else:
        for short in ["face","bg","skin","cloth","hair"]:
            m[f"vitl_{short}_cov"] = m[f"vitl_{short}_norm"] = m[f"vitl_{short}_patches"] = m[f"vitl_{short}_mean"] = None

    aux_depth = ((sample.get("aux") or {}).get("depth") or {}).get("pixel_stats") or {}
    m["depth_mean"] = aux_depth.get("mean")
    m["depth_std"]  = aux_depth.get("std")

    # Face extras
    m["face_age"]      = fa.get("age")
    m["face_emb_norm"] = fa.get("embedding_norm")
    img_shape  = (sample.get("image") or {}).get("shape") or [512, 512, 3]
    img_pixels = img_shape[0] * img_shape[1]
    bbox = fa.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        m["face_bbox_area"] = abs((x2 - x1) * (y2 - y1)) / img_pixels
    else:
        m["face_bbox_area"] = None
    gender = fa.get("gender")
    m["face_gender_f"] = 1.0 if gender == "F" else (0.0 if gender == "M" else None)

    # Aux — normal and edge only. Legacy raster pose maps are retired; pose
    # review metrics now come from structured pose_evidence instead.
    for aux_name in ["normal", "edge"]:
        ps2 = ((sample.get("aux") or {}).get(aux_name) or {}).get("pixel_stats") or {}
        m[f"{aux_name}_mean"] = ps2.get("mean")
        m[f"{aux_name}_std"]  = ps2.get("std")

    # Pose evidence scalars (optional domain — skipped when absent)
    m.update(_extract_pose_evidence_metrics(sample.get("pose_evidence") or {}))

    return m


def _extract_pose_evidence_metrics(pe: dict) -> dict:
    """Extract flat scalars from the pose_evidence domain.

    Called from extract_metrics(); safe to call with an empty dict.
    Does not raise; missing data produces None values.
    """
    out: dict = {}
    for prefix, source_key in [("pose_openpose", "openpose_body"), ("pose_dw", "dw_body")]:
        src = pe.get(source_key)
        if not isinstance(src, dict):
            continue
        out[f"{prefix}_person_count"] = src.get("raw_people_count")
    return out

def resolved_lora_label(lora_info: dict, fallback_hash: str | None = None) -> str:
    """Return the best available LoRA label for the gallery tree.

    Keep the heuristic intentionally narrow: only replace obviously trash names.
    """
    raw_name = (lora_info or {}).get("name")
    meta = (lora_info or {}).get("raw_metadata") or {}

    def _clean(v):
        return v.strip() if isinstance(v, str) and v.strip() else None

    name = _clean(raw_name)
    trash_names = {"lora", "unknown", "unknown-model", "untitled"}
    if name and name.lower() not in trash_names:
        return name

    for key in ("ss_output_name", "modelspec.title", "name", "LoRA_name"):
        candidate = _clean(meta.get(key))
        if candidate and candidate.lower() not in trash_names:
            return candidate

    return name or (fallback_hash[:12] if fallback_hash else "Unnamed LoRA")


def build_data(
    treasurer,
    method_prefix: str | None = None,
    *,
    include_images: bool = True,
    jpeg_quality: int = 88,
) -> dict:
    """Query DB via Treasurer and assemble the full gallery data structure."""
    project_root = Path(paths.get_path("project_root"))
    asset_cache: dict[str, np.ndarray | None] = {}

    def load_cached(ref, _project_root: Path | None = None):
        return load_npy_ref(ref, project_root, asset_cache)

    # Enumerate methods via evals (Treasurer has no list_methods)
    all_evals = treasurer.query_evals()
    method_hashes = sorted(set(
        e["method_hash"] for e in all_evals
        if not method_prefix or e["method_hash"].startswith(method_prefix)
    ))

    total_images = 0
    out = {"methods": []}

    for mhash in method_hashes:
        method = treasurer.get_method(mhash)
        if method is None:
            continue
        mlabel = method_label(method)
        mprompt = method_prompt_text(method)

        evals_for_method = [e for e in all_evals if e["method_hash"] == mhash]

        baselines_by_seed = {}   # seed → sample_obj
        lora_groups       = {}   # (lhash, lname) → {strength → [sample_obj, ...]}

        for eval_rec in evals_for_method:
            lora_info = eval_rec.get("lora") or {}
            samples   = treasurer.query_samples(filters={"eval_hash": eval_rec["eval_hash"]})

            for s in samples:
                if s.get("is_dirty"):
                    continue

                img_rel = (s.get("image") or {}).get("output", {}).get("path")
                if not img_rel:
                    continue

                img_b64 = npy_to_b64_jpeg(str(project_root / img_rel), quality=jpeg_quality) if include_images else None
                metrics = extract_metrics(s, load_ref=load_cached)
                total_images += 1

                sample_obj = {
                    "id":       s["sample_hash"],
                    "seed":     s["seed"],
                    "strength": s.get("lora_strength"),
                    "label":    f"seed {s['seed']}",
                    "img":      img_b64,
                    "image_path": img_rel,
                    "metrics":  metrics,
                    "_raw":     s,   # temp; used for cosine dist; stripped before JSON
                }

                if not lora_info:
                    # Baseline eval
                    baselines_by_seed[s["seed"]] = sample_obj
                else:
                    lhash = lora_info.get("hash", "unknown")
                    lname = resolved_lora_label(lora_info, lhash)
                    key   = (lhash, lname)
                    if key not in lora_groups:
                        lora_groups[key] = {}
                    strength = s.get("lora_strength")
                    if strength not in lora_groups[key]:
                        lora_groups[key][strength] = []
                    lora_groups[key][strength].append(sample_obj)

        # ── Pairing pass: attach baseline refs + compute all paired metrics ──
        # All metrics that require two samples (LoRA vs. baseline) are computed
        # here rather than in extract_metrics(), which only sees one sample.
        _PAIRED_NONE = [
            "face_cos_dist", "clip_global_cos_dist", "vitl_global_cos_dist",
            "img_px_diff", "depth_diff", "pose_selected_source", "pose_angle_drift", "pose_reliability", "pose_dropped", "normal_diff", "edge_diff",
            "lum_face_diff", "lum_cloth_diff", "lum_bg_diff", "cloth_edge_diff", "bg_depth_diff",
            "face_bbox_area_delta", "face_center_shift",
            "identity_region_drift_exp", "identity_region_plus_arcface_exp",
            "identity_region_drift_pre_gate_exp", "identity_region_drift_usable",
            "identity_region_plus_arcface_pre_gate_exp", "identity_region_plus_arcface_usable",
            "identity_gate_status",
            "head_rot_drift",
            "siglip_cos_face","siglip_cos_bg","siglip_cos_skin","siglip_cos_cloth","siglip_cos_hair",
            "vitl_cos_face",  "vitl_cos_bg", "vitl_cos_skin", "vitl_cos_cloth",  "vitl_cos_hair",
            "lum_character_diff", "siglip_cos_character", "vitl_cos_character",
            "pose_openpose_baseline_has_person0", "pose_openpose_lora_has_person0",
            "pose_openpose_comparable_person0", "pose_openpose_baseline_joints",
            "pose_openpose_lora_joints", "pose_openpose_comparable_joints",
            "pose_openpose_baseline_triplets", "pose_openpose_lora_triplets",
            "pose_openpose_comparable_triplets", "pose_openpose_angle_drift",
            "pose_openpose_reliability", "pose_openpose_dropped",
            "pose_dw_baseline_has_person0", "pose_dw_lora_has_person0",
            "pose_dw_comparable_person0", "pose_dw_baseline_joints",
            "pose_dw_lora_joints", "pose_dw_comparable_joints",
            "pose_dw_baseline_triplets", "pose_dw_lora_triplets",
            "pose_dw_comparable_triplets", "pose_dw_angle_drift",
            "pose_dw_reliability", "pose_dw_dropped",
            # Pose side-evidence deltas
            "pose_openpose_person_count_delta", "pose_dw_person_count_delta",
            "face_detection_lost",
        ]

        for (lhash, lname), strengths_dict in lora_groups.items():
            for strength, samples_list in strengths_dict.items():
                for sobj in samples_list:
                    bl = baselines_by_seed.get(sobj["seed"])
                    if not bl:
                        for k in _PAIRED_NONE:
                            sobj["metrics"][k] = None
                        continue

                    if include_images:
                        sobj["bl_img"] = bl["img"]
                    sobj["bl_metrics"] = bl["metrics"]

                    s_raw  = sobj.get("_raw") or {}
                    bl_raw = bl.get("_raw")   or {}

                    # ── Face embedding cosine distance (identity drift) ──────
                    fa_lora = s_raw.get("face_analysis")  or {}
                    fa_bl   = bl_raw.get("face_analysis") or {}
                    lora_face_emb = load_cached(fa_lora.get("normed_embedding"))
                    bl_face_emb = load_cached(fa_bl.get("normed_embedding"))
                    sobj["metrics"]["face_cos_dist"] = cosine_dist(
                        lora_face_emb,
                        bl_face_emb,
                    )
                    bl_face_count = bl["metrics"].get("face_count")
                    lora_face_count = sobj["metrics"].get("face_count")
                    sobj["metrics"]["face_detection_lost"] = procedure_face_detection_lost(bl_face_count, lora_face_count)

                    # ── Head rotation drift (combined 3D pose change magnitude) ─────────
                    p_l = sobj["metrics"].get("pitch"); y_l = sobj["metrics"].get("yaw"); r_l = sobj["metrics"].get("roll")
                    p_b = bl["metrics"].get("pitch");   y_b = bl["metrics"].get("yaw");   r_b = bl["metrics"].get("roll")
                    if all(v is not None for v in [p_l, y_l, r_l, p_b, y_b, r_b]):
                        sobj["metrics"]["head_rot_drift"] = float(np.sqrt(
                            (p_l - p_b)**2 + (y_l - y_b)**2 + (r_l - r_b)**2
                        ))
                    else:
                        sobj["metrics"]["head_rot_drift"] = None

                    # ── Pose angle drift (structured keypoints, replaces image-map version) ──
                    pe_bl   = bl_raw.get("pose_evidence")   or {}
                    pe_lora = s_raw.get("pose_evidence")    or {}
                    sobj["metrics"].update(procedure_pose_pair_metrics(pe_bl, pe_lora))

                    # ── Identify CLIP model keys (same discriminator as extract_metrics) ─
                    siglip_key = vitl_key = None
                    for k, v in (s_raw.get("clip_vision") or {}).items():
                        sh = (v or {}).get("global_embedding_shape") or []
                        if sh and sh[-1] == 1152: siglip_key = k
                        elif sh and sh[-1] == 768: vitl_key  = k

                    # ── SigLIP global cosine distance ───────────────────────
                    if siglip_key:
                        cv_s_lora = s_raw["clip_vision"].get(siglip_key)  or {}
                        cv_s_bl   = (bl_raw.get("clip_vision") or {}).get(siglip_key) or {}
                        sobj["metrics"]["clip_global_cos_dist"] = cosine_dist(
                            load_cached(cv_s_lora.get("global_embedding")),
                            load_cached(cv_s_bl.get("global_embedding")),
                        )
                    else:
                        sobj["metrics"]["clip_global_cos_dist"] = None

                    # ── ViT-L/14 global cosine distance ─────────────────────
                    if vitl_key:
                        cv_v_lora = s_raw["clip_vision"].get(vitl_key)  or {}
                        cv_v_bl   = (bl_raw.get("clip_vision") or {}).get(vitl_key) or {}
                        sobj["metrics"]["vitl_global_cos_dist"] = cosine_dist(
                            load_cached(cv_v_lora.get("global_embedding")),
                            load_cached(cv_v_bl.get("global_embedding")),
                        )
                    else:
                        sobj["metrics"]["vitl_global_cos_dist"] = None

                    # ── Pixel-level diffs (image + preprocessors) ───────────
                    sobj["metrics"]["img_px_diff"] = px_abs_diff(
                        load_cached((s_raw.get("image")  or {}).get("output")),
                        load_cached((bl_raw.get("image") or {}).get("output")),
                    )
                    for aux_name in ["depth", "normal", "edge"]:
                        sobj["metrics"][f"{aux_name}_diff"] = px_abs_diff(
                            load_cached(((s_raw.get("aux")  or {}).get(aux_name) or {}).get("output")),
                            load_cached(((bl_raw.get("aux") or {}).get(aux_name) or {}).get("output")),
                        )

                    # ── Per-mask CLIP cosine distances (from last_hidden_state) ─
                    # Load LHS tensors once per sample, reuse across all 5 masks.
                    # Baseline mask is used as the reference patch-selection region.
                    lum_lora = load_cached((s_raw.get("luminance") or {}).get("output"))
                    lum_bl   = load_cached((bl_raw.get("luminance") or {}).get("output"))
                    depth_lora = load_cached(((s_raw.get("aux")  or {}).get("depth") or {}).get("output"))
                    depth_bl   = load_cached(((bl_raw.get("aux") or {}).get("depth") or {}).get("output"))
                    edge_lora  = load_cached(((s_raw.get("aux")  or {}).get("edge") or {}).get("output"))
                    edge_bl    = load_cached(((bl_raw.get("aux") or {}).get("edge") or {}).get("output"))

                    masks_bl = bl_raw.get("masks") or {}
                    face_mask  = load_cached((masks_bl.get("face")         or {}).get("output"))
                    bg_mask    = load_cached((masks_bl.get("main_subject") or {}).get("output"))
                    cloth_mask = load_cached((masks_bl.get("clothing")     or {}).get("output"))

                    # masks.main_subject is the foreground/character region (white = subject).
                    # Its inverse is the true scene background used for bg_depth_diff, lum_bg_diff.
                    character_mask = _mask_to_hw_float(bg_mask)
                    true_bg_mask = _true_background_mask(bg_mask)

                    sobj["metrics"]["lum_face_diff"]      = masked_abs_diff(lum_lora, lum_bl, face_mask)
                    sobj["metrics"]["lum_cloth_diff"]     = masked_abs_diff(lum_lora, lum_bl, cloth_mask)
                    sobj["metrics"]["lum_bg_diff"]        = masked_abs_diff(lum_lora, lum_bl, true_bg_mask)
                    sobj["metrics"]["cloth_edge_diff"]    = masked_abs_diff(edge_lora, edge_bl, cloth_mask)
                    sobj["metrics"]["bg_depth_diff"]      = masked_abs_diff(depth_lora, depth_bl, true_bg_mask)
                    sobj["metrics"]["lum_character_diff"] = masked_abs_diff(lum_lora, lum_bl, character_mask)

                    img_shape_l = (s_raw.get("image") or {}).get("shape") or [512, 512, 3]
                    img_shape_b = (bl_raw.get("image") or {}).get("shape") or [512, 512, 3]
                    area_l, center_l = bbox_area_and_center(fa_lora, img_shape_l)
                    area_b, center_b = bbox_area_and_center(fa_bl,   img_shape_b)
                    sobj["metrics"]["face_bbox_area_delta"] = (
                        abs(area_l - area_b) if area_l is not None and area_b is not None else None
                    )
                    if center_l is not None and center_b is not None:
                        h, w = img_shape_b[:2]
                        diag = float(np.sqrt(h * h + w * w))
                        sobj["metrics"]["face_center_shift"] = (
                            float(np.sqrt((center_l[0] - center_b[0]) ** 2 + (center_l[1] - center_b[1]) ** 2) / diag)
                            if diag > 0 else None
                        )
                    else:
                        sobj["metrics"]["face_center_shift"] = None

                    lhs_sig_lora = lhs_sig_bl = lhs_vitl_lora = lhs_vitl_bl = None
                    if siglip_key:
                        cv_s_lora = s_raw["clip_vision"].get(siglip_key)  or {}
                        cv_s_bl   = (bl_raw.get("clip_vision") or {}).get(siglip_key) or {}
                        lhs_sig_lora = load_cached(cv_s_lora.get("last_hidden_state"))
                        lhs_sig_bl   = load_cached(cv_s_bl.get("last_hidden_state"))
                    if vitl_key:
                        cv_v_lora = s_raw["clip_vision"].get(vitl_key)  or {}
                        cv_v_bl   = (bl_raw.get("clip_vision") or {}).get(vitl_key) or {}
                        lhs_vitl_lora = load_cached(cv_v_lora.get("last_hidden_state"))
                        lhs_vitl_bl   = load_cached(cv_v_bl.get("last_hidden_state"))

                    masks_bl = bl_raw.get("masks") or {}
                    for mask_name, short in [
                        ("face","face"),("main_subject","bg"),
                        ("skin","skin"),("clothing","cloth"),("hair","hair"),
                    ]:
                        if mask_name == "main_subject":
                            mask_arr = true_bg_mask  # use actual background (inverse of main_subject)
                        else:
                            mask_arr = load_cached((masks_bl.get(mask_name) or {}).get("output"))
                        # SigLIP: 32×32 grid, 16px patches, no CLS token
                        sobj["metrics"][f"siglip_cos_{short}"] = cosine_dist(
                            masked_clip_pool(lhs_sig_lora,  mask_arr, n_grid=32, has_cls=False),
                            masked_clip_pool(lhs_sig_bl,    mask_arr, n_grid=32, has_cls=False),
                        )
                        # ViT-L/14: 16×16 grid, 32px patches, CLS at token index 0
                        sobj["metrics"][f"vitl_cos_{short}"] = cosine_dist(
                            masked_clip_pool(lhs_vitl_lora, mask_arr, n_grid=16, has_cls=True),
                            masked_clip_pool(lhs_vitl_bl,   mask_arr, n_grid=16, has_cls=True),
                        )

                    # Character mask — derived union; not stored, computed here
                    sobj["metrics"]["siglip_cos_character"] = cosine_dist(
                        masked_clip_pool(lhs_sig_lora,  character_mask, n_grid=32, has_cls=False),
                        masked_clip_pool(lhs_sig_bl,    character_mask, n_grid=32, has_cls=False),
                    )
                    sobj["metrics"]["vitl_cos_character"] = cosine_dist(
                        masked_clip_pool(lhs_vitl_lora, character_mask, n_grid=16, has_cls=True),
                        masked_clip_pool(lhs_vitl_bl,   character_mask, n_grid=16, has_cls=True),
                    )

                    region_pre_gate = mean_of_present(
                        sobj["metrics"].get("siglip_cos_face"),
                        sobj["metrics"].get("vitl_cos_face"),
                        sobj["metrics"].get("siglip_cos_hair"),
                        sobj["metrics"].get("vitl_cos_hair"),
                    )
                    plus_arcface_pre_gate = mean_of_present(
                        sobj["metrics"].get("face_cos_dist"),
                        sobj["metrics"].get("siglip_cos_face"),
                        sobj["metrics"].get("vitl_cos_face"),
                        sobj["metrics"].get("siglip_cos_hair"),
                        sobj["metrics"].get("vitl_cos_hair"),
                    )
                    region_gate = gate_identity_score(region_pre_gate, lora_face_count)
                    plus_arcface_gate = gate_identity_score(plus_arcface_pre_gate, lora_face_count)
                    sobj["metrics"]["identity_gate_status"] = plus_arcface_gate.gate_status
                    sobj["metrics"]["identity_region_drift_pre_gate_exp"] = region_gate.pre_gate_score
                    sobj["metrics"]["identity_region_drift_usable"] = region_gate.usable
                    sobj["metrics"]["identity_region_drift_exp"] = region_gate.score
                    sobj["metrics"]["identity_region_plus_arcface_pre_gate_exp"] = plus_arcface_gate.pre_gate_score
                    sobj["metrics"]["identity_region_plus_arcface_usable"] = plus_arcface_gate.usable
                    sobj["metrics"]["identity_region_plus_arcface_exp"] = plus_arcface_gate.score
                    sobj["metrics"]["background_package_exp"] = mean_of_present(
                        sobj["metrics"].get("siglip_cos_bg"),
                        sobj["metrics"].get("vitl_cos_bg"),
                    )
                    sobj["metrics"]["composition_package_exp"] = mean_of_present(
                        sobj["metrics"].get("bg_depth_diff"),
                        sobj["metrics"].get("normal_diff"),
                    )

                    # ── Pose evidence deltas (LoRA vs baseline, per source) ──────────
                    for _pe_pfx in ("pose_openpose", "pose_dw"):
                        for _pe_sfx in ("_person_count",):
                            _key = f"{_pe_pfx}{_pe_sfx}"
                            _lv  = sobj["metrics"].get(_key)
                            _bv  = bl["metrics"].get(_key)
                            _dk  = f"{_key}_delta"
                            if _lv is not None and _bv is not None:
                                sobj["metrics"][_dk] = round(float(_lv) - float(_bv), 4)
                            else:
                                sobj["metrics"][_dk] = None

        # Strip _raw from all sample_objs — not needed past this point
        # and must not enter the JSON payload (large + potentially non-serialisable).
        for bl_sobj in baselines_by_seed.values():
            bl_sobj.pop("_raw", None)
        for strengths_dict in lora_groups.values():
            for samples_list in strengths_dict.values():
                for sobj in samples_list:
                    sobj.pop("_raw", None)

        # ── Build eval tree ───────────────────────────────────────────────────
        evals = []

        if baselines_by_seed:
            evals.append({
                "id":          next(
                    (e.get("eval_hash") for e in evals_for_method if not (e.get("lora") or {})),
                    "baseline",
                ),
                "label":       "Baseline",
                "is_baseline": True,
                "samples":     sorted(baselines_by_seed.values(), key=lambda x: x["seed"]),
            })

        for (lhash, lname), strengths_dict in sorted(lora_groups.items(), key=lambda x: x[0][1].lower()):
            eval_id = next(
                (
                    e.get("eval_hash")
                    for e in evals_for_method
                    if (e.get("lora") or {}).get("hash", "unknown") == lhash
                ),
                lhash,
            )
            strengths = []
            for strength in sorted(strengths_dict.keys()):
                slist = sorted(strengths_dict[strength], key=lambda x: x["seed"])
                strengths.append({
                    "value":   strength,
                    "label":   f"strength {strength}",
                    "samples": slist,
                })
            evals.append({
                "id":          eval_id,
                "label":       lname,
                "lora_hash":   lhash,
                "is_baseline": False,
                "strengths":   strengths,
            })

        out["methods"].append({
            "id":    mhash,
            "label": mlabel,
            "prompt_text": mprompt,
            "prompt_hint": mprompt,
            "evals": evals,
        })
    return out
