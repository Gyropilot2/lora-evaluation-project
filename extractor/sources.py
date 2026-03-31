"""
extractor/sources.py — safe accessor layer: ComfyUI inputs → ExtractionResult.

Converts ComfyUI live objects to a plain-Python ExtractionResult dataclass.
Every field is extracted in its own try/except; failures emit diagnostics and
set the field to None.  No live tensor or ComfyUI object escapes this module.

ComfyUI dependency: YES — only import from within a ComfyUI execution context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import diagnostics
import extractor.primitives as _p

_WHERE = "extractor.sources"


# ---------------------------------------------------------------------------
# ExtractionResult — plain-Python snapshot of one inference pass
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    # ---- Method hash components ----
    base_model_hash: str | None
    base_model_name: str | None
    vae_model_hash:  str | None
    pos_cond_hash:   str | None
    pos_guidance:    float | None
    neg_cond_hash:   str | None
    neg_guidance:    float | None
    steps:           int
    cfg:             float
    sampler:         str
    scheduler:       str
    denoise:         float
    latent_shape:    list | None          # [B, C, H_latent, W_latent]
    model_extras:    str | None

    # ---- Vital integrity fields ----
    latent_hash:     str | None
    image_hash:      str | None

    # ---- Eval inputs ----
    is_baseline:     bool
    lora_hash:       str | None           # in-memory patch hash (canonical for eval_hash)
    lora_file_hash:  str | None           # raw file bytes BLAKE3 (provenance)
    lora_name:       str | None
    lora_strength:   float | None

    # ---- Sample scalar ----
    seed: int
    positive_prompt_hint: str | None
    negative_prompt_hint: str | None

    # ---- Asset payloads — target-format bytes staged by nodes.py, not here ----
    image_npy_bytes:   bytes | None
    image_shape:       list | None
    lum_npy_bytes:     bytes | None
    lum_stats:         dict | None
    image_pixel_stats: dict | None

    # ---- CLIP Vision: list of per-model dicts ----
    # Each dict keys: model_hash, global_embedding_bytes, global_embedding_shape,
    #                 last_hidden_state_bytes, last_hidden_state_shape,
    #                 is_aliased, stats, patch_pools
    clip_vision_slots: list

    # ---- Masks keyed by name ("face","background","skin","clothing","hair") ----
    mask_bytes:  dict
    mask_shapes: dict
    mask_hashes: dict

    # ---- InsightFace: raw result dict (faces list may contain embedding_array ndarrays) ----
    face_analysis_raw: dict

    # ---- Aux images keyed by name ("depth","normal","edge") ----
    # Note: "pose" aux image (legacy raster PNG) removed. Use pose_evidence instead.
    aux_image_bytes:   dict
    aux_image_shapes:  dict
    aux_image_hashes:  dict
    aux_pixel_stats:   dict

    # ---- LoRA safetensors metadata (collected for future 4th table) ----
    lora_metadata: dict | None


# ---------------------------------------------------------------------------
# Hash-only computation (used by SampleGuard — no pixel extraction)
# ---------------------------------------------------------------------------


def compute_hashes_only(
    model: Any,
    vae: Any,
    positive: Any,
    negative: Any,
    latent: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    lora_name: str | None = None,
    lora_strength: float | None = None,
) -> dict:
    """Compute method_hash, eval_hash, and sample_hash from ComfyUI inputs.

    Runs only the hash-computation phase of extract_all() — no pixel
    extraction, no asset staging, no DB writes.  Used by SampleGuard to
    check existence before KSampler runs.

    Hash logic is identical to build_candidates() in extractor/extract.py:
    - None values are passed through (not coerced to ""), so hashes match.
    - Each component is extracted in its own try/except; failures leave the
      field as None, which canonical_json serialises as null — same as the
      Extractor when a hash cannot be computed.

    Never raises.  Returns:
        {
            "method_hash": str,
            "eval_hash":   str,
            "sample_hash": str,
            "lora_hash":   str | None,  # None for baseline
        }
    """
    from core import hashing as _hashing

    # ---- Base model ----
    base_model_hash: str | None = None
    try:
        base_model_hash = _p.hash_base_model(model)
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE, str(exc))

    # ---- VAE ----
    vae_model_hash: str | None = None
    try:
        vae_model_hash = _p.hash_vae_model(vae)
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE, str(exc))

    # ---- Conditioning ----
    pos_cond_hash: str | None  = None
    pos_guidance: float | None = None
    try:
        pos_cond_hash, pos_guidance = _p.hash_conditioning(positive)
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.COND_HASH_FAIL", _WHERE, str(exc))

    neg_cond_hash: str | None  = None
    neg_guidance: float | None = None
    try:
        neg_cond_hash, neg_guidance = _p.hash_conditioning(negative)
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.COND_HASH_FAIL", _WHERE, str(exc))

    # ---- Latent shape ----
    latent_shape: list | None = None
    try:
        _, latent_shape = _p.hash_latent(latent)
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.LATENT_SHAPE_FAIL", _WHERE, str(exc))

    latent_w: int = 0
    latent_h: int = 0
    if latent_shape and len(latent_shape) >= 4:
        latent_h = int(latent_shape[2]) * 8
        latent_w = int(latent_shape[3]) * 8

    # ---- Model extras ----
    model_extras: str | None = None
    try:
        model_extras = _p.extract_model_extras(model)
    except Exception:
        pass

    # ---- LoRA ----
    patch_hash: str | None = None
    try:
        patch_hash = _p.hash_lora_patches(model)
    except Exception:
        pass

    is_baseline = patch_hash is None
    lora_hash: str | None = None

    if not is_baseline:
        content_hash_from_file: str | None = None
        try:
            lora_path = _p.resolve_lora_path(lora_name) if lora_name else None
            if lora_path:
                content_hash_from_file, _ = _p.hash_lora_file(lora_path)
            elif lora_name:
                diagnostics.emit("WARN", "EXTRACTOR.LORA_RESOLVE_FAIL", _WHERE,
                                 f"could not resolve lora path for: {lora_name!r}")
        except Exception as exc:
            diagnostics.emit("WARN", "EXTRACTOR.LORA_RESOLVE_FAIL", _WHERE, str(exc))
        lora_hash = patch_hash or content_hash_from_file

    _lora_strength: float | None = None if is_baseline else lora_strength

    # ---- Compute the three hashes (identical logic to extract.py/build_candidates) ----
    _method_hash = _hashing.method_hash({
        "base_model_hash":                base_model_hash,
        "model_extras":                   model_extras,
        "positive_conditioning_hash":     pos_cond_hash,
        "positive_conditioning_guidance": pos_guidance,
        "negative_conditioning_hash":     neg_cond_hash,
        "negative_conditioning_guidance": neg_guidance,
        "steps":         int(steps),
        "denoise":       float(denoise),
        "sampler":       str(sampler_name),
        "scheduler":     str(scheduler),
        "cfg":           float(cfg),
        "latent_width":  latent_w,
        "latent_height": latent_h,
        "latent_shape":  latent_shape,
        "vae_model_hash": vae_model_hash,
    })
    _eval_hash   = _hashing.eval_hash(_method_hash, lora_hash)
    _sample_hash = _hashing.sample_hash(_eval_hash, seed, _lora_strength)

    return {
        "method_hash": _method_hash,
        "eval_hash":   _eval_hash,
        "sample_hash": _sample_hash,
        "lora_hash":   lora_hash,
    }


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_all(
    model: Any,
    vae: Any,
    positive: Any,
    negative: Any,
    latent: Any,
    image: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    lora_name: str | None = None,
    lora_strength: float | None = None,
    positive_prompt_hint: str | None = None,
    negative_prompt_hint: str | None = None,
    clip_vision_1: Any = None,
    clip_vision_2: Any = None,
    mask_face: Any = None,
    mask_main_subject: Any = None,
    mask_skin: Any = None,
    mask_clothing: Any = None,
    mask_hair: Any = None,
    aux_depth: Any = None,
    aux_normal: Any = None,
    aux_edge: Any = None,
) -> ExtractionResult:
    """Convert ComfyUI live objects to a plain-Python ExtractionResult.

    Each field is extracted in its own try/except.  Failures emit diagnostics
    and set the field to None.  Never raises.
    """

    # ---- Base model ----
    base_model_hash: str | None = None
    base_model_name: str | None = None
    try:
        base_model_hash = _p.hash_base_model(model)
        if base_model_hash is None:
            diagnostics.emit("WARN", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE,
                             "base model hash returned None")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE, str(exc))

    try:
        filename = getattr(model, "filename", None)
        if filename:
            base_model_name = Path(str(filename)).name
        else:
            base_model_name = type(model.model).__name__
    except Exception:
        pass

    # ---- VAE ----
    vae_model_hash: str | None = None
    try:
        vae_model_hash = _p.hash_vae_model(vae)
        if vae_model_hash is None:
            diagnostics.emit("WARN", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE,
                             "VAE model hash returned None")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE, str(exc))

    # ---- Conditioning ----
    pos_cond_hash: str | None = None
    pos_guidance: float | None = None
    try:
        pos_cond_hash, pos_guidance = _p.hash_conditioning(positive)
        if pos_cond_hash is None:
            diagnostics.emit("WARN", "EXTRACTOR.COND_HASH_FAIL", _WHERE,
                             "positive conditioning hash returned None")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.COND_HASH_FAIL", _WHERE, str(exc))

    neg_cond_hash: str | None = None
    neg_guidance: float | None = None
    try:
        neg_cond_hash, neg_guidance = _p.hash_conditioning(negative)
        if neg_cond_hash is None:
            diagnostics.emit("WARN", "EXTRACTOR.COND_HASH_FAIL", _WHERE,
                             "negative conditioning hash returned None")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.COND_HASH_FAIL", _WHERE, str(exc))

    # ---- Latent (vital) ----
    latent_hash: str | None = None
    latent_shape: list | None = None
    try:
        latent_hash, latent_shape = _p.hash_latent(latent)
        if latent_hash is None:
            diagnostics.emit("ERROR", "EXTRACTOR.LATENT_HASH_FAIL", _WHERE,
                             "latent hash returned None — vital field missing")
        if latent_shape is None:
            diagnostics.emit("WARN", "EXTRACTOR.LATENT_SHAPE_FAIL", _WHERE,
                             "latent shape unavailable")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.LATENT_HASH_FAIL", _WHERE, str(exc))

    # ---- Image (vital) ----
    image_hash: str | None = None
    try:
        image_hash = _p.hash_image(image)
        if image_hash is None:
            diagnostics.emit("ERROR", "EXTRACTOR.IMAGE_HASH_FAIL", _WHERE,
                             "image hash returned None — vital field missing")
    except Exception as exc:
        diagnostics.emit("ERROR", "EXTRACTOR.IMAGE_HASH_FAIL", _WHERE, str(exc))

    # ---- Model extras ----
    model_extras: str | None = None
    try:
        model_extras = _p.extract_model_extras(model)
    except Exception:
        pass  # model_extras failure is non-fatal; None is the expected return for standard configs

    # ---- LoRA ----
    # Baseline/non-baseline is inferred from MODEL patches, not from COMBO selection.
    patch_hash: str | None = None
    try:
        patch_hash = _p.hash_lora_patches(model)
    except Exception:
        pass

    is_baseline = patch_hash is None
    _lora_name: str | None       = None if is_baseline else lora_name
    _lora_strength: float | None = None if is_baseline else lora_strength

    lora_hash: str | None       = None
    lora_file_hash: str | None  = None
    lora_metadata: dict | None  = None

    if not is_baseline:
        # File-based hashes (content_hash = fallback for eval_hash; file_hash = provenance)
        # COMBO is used only to resolve file path/metadata; identity comes from MODEL patch hash.
        content_hash_from_file: str | None = None
        try:
            lora_path = _p.resolve_lora_path(_lora_name) if _lora_name else None
            if lora_path:
                content_hash_from_file, file_hash = _p.hash_lora_file(lora_path)
                lora_file_hash = file_hash
                try:
                    lora_metadata = _p.read_lora_safetensors_metadata(lora_path)
                except Exception:
                    pass
            elif _lora_name:
                diagnostics.emit("WARN", "EXTRACTOR.LORA_RESOLVE_FAIL", _WHERE,
                                 f"could not resolve lora path for: {_lora_name!r}")
        except Exception as exc:
            diagnostics.emit("WARN", "EXTRACTOR.LORA_RESOLVE_FAIL", _WHERE, str(exc))

        # Resolve display LoRA name.
        # Preference order: metadata inferred_display_name → COMBO basename stem.
        # COMBO is kept as the path-resolution source; the name field is display-only.
        if lora_metadata and lora_metadata.get("status") == "ok":
            meta_name = lora_metadata.get("inferred_display_name")
            if meta_name:
                _lora_name = meta_name
            elif _lora_name:
                _lora_name = Path(_lora_name).stem
        elif _lora_name:
            _lora_name = Path(_lora_name).stem

        # Canonical lora_hash: patch hash preferred; fall back to file content hash
        lora_hash = patch_hash or content_hash_from_file
        if lora_hash is None:
            diagnostics.emit("WARN", "EXTRACTOR.MODEL_HASH_FAIL", _WHERE,
                             "lora_hash is None for non-baseline MODEL — eval_hash will be degraded")

    # ---- Main image asset bytes (.npy stays canonical for re-ingest integrity) ----
    image_npy_bytes: bytes | None = None
    image_shape: list | None = None
    try:
        image_npy_bytes, _, image_shape = _p.image_to_npy_bytes(image)
    except RuntimeError as exc:
        diagnostics.emit("WARN", "EXTRACTOR.IMAGE_HASH_FAIL", _WHERE, str(exc))

    # ---- Luminance asset bytes (16-bit grayscale PNG) ----
    lum_npy_bytes: bytes | None = None
    lum_stats: dict | None = None
    try:
        lum_npy_bytes, lum_stats = _p.compute_luminance(image)
    except RuntimeError as exc:
        diagnostics.emit("WARN", "EXTRACTOR.IMAGE_HASH_FAIL", _WHERE, str(exc))

    # ---- Pixel stats ----
    image_pixel_stats: dict = _p.compute_pixel_stats(image)

    # ---- Masks (keyed by name; used by CLIP patch pools + asset stage) ----
    raw_masks: dict[str, Any] = {
        "face":         mask_face,
        "main_subject": mask_main_subject,
        "skin":         mask_skin,
        "clothing":     mask_clothing,
        "hair":         mask_hair,
    }
    mask_bytes:  dict = {}
    mask_shapes: dict = {}
    mask_hashes: dict = {}
    for mask_name, mask_t in raw_masks.items():
        if mask_t is None:
            continue
        b, s, h = _mask_to_bytes_hash(mask_t)
        mask_bytes[mask_name]  = b
        mask_shapes[mask_name] = s
        mask_hashes[mask_name] = h

    # ---- CLIP Vision slots ----
    clip_vision_slots: list = []
    for cv_model in (clip_vision_1, clip_vision_2):
        if cv_model is None:
            continue
        slot = _extract_clip_slot(cv_model, image, raw_masks)
        clip_vision_slots.append(slot)

    # ---- InsightFace — always attempted; skips gracefully if package or models absent ----
    face_analysis_raw: dict = {"status": "not_available"}
    try:
        insightface_root = _p.resolve_insightface_root()
        face_analysis_raw = _p.run_insightface(image, insightface_root)
        if face_analysis_raw.get("status") == "error":
            diagnostics.emit("WARN", "EXTRACTOR.INSIGHTFACE_FAIL", _WHERE,
                             face_analysis_raw.get("detail", ""))
    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.INSIGHTFACE_FAIL", _WHERE, str(exc))
        face_analysis_raw = {"status": "error", "detail": str(exc)}

    # ---- Aux images (depth/normal/edge only — pose raster aux removed; use pose_evidence) ----
    raw_aux: dict[str, Any] = {
        "depth":  aux_depth,
        "normal": aux_normal,
        "edge":   aux_edge,
    }
    aux_image_bytes:  dict = {}
    aux_image_shapes: dict = {}
    aux_image_hashes: dict = {}
    aux_pixel_stats:  dict = {}
    for aux_name, aux_t in raw_aux.items():
        if aux_t is None:
            continue
        b, s, h = _aux_image_to_bytes_hash(aux_t)
        aux_image_bytes[aux_name]  = b
        aux_image_shapes[aux_name] = s
        aux_image_hashes[aux_name] = h
        aux_pixel_stats[aux_name]  = _p.compute_pixel_stats(aux_t)

    return ExtractionResult(
        base_model_hash   = base_model_hash,
        base_model_name   = base_model_name,
        vae_model_hash    = vae_model_hash,
        pos_cond_hash     = pos_cond_hash,
        pos_guidance      = pos_guidance,
        neg_cond_hash     = neg_cond_hash,
        neg_guidance      = neg_guidance,
        steps             = steps,
        cfg               = cfg,
        sampler           = sampler_name,
        scheduler         = scheduler,
        denoise           = denoise,
        latent_shape      = latent_shape,
        model_extras      = model_extras,
        latent_hash       = latent_hash,
        image_hash        = image_hash,
        is_baseline       = is_baseline,
        lora_hash         = lora_hash,
        lora_file_hash    = lora_file_hash,
        lora_name         = _lora_name,
        lora_strength     = _lora_strength,
        seed              = seed,
        positive_prompt_hint = (positive_prompt_hint or None),
        negative_prompt_hint = (negative_prompt_hint or None),
        image_npy_bytes   = image_npy_bytes,
        image_shape       = image_shape,
        lum_npy_bytes     = lum_npy_bytes,
        lum_stats         = lum_stats,
        image_pixel_stats = image_pixel_stats,
        clip_vision_slots = clip_vision_slots,
        mask_bytes        = mask_bytes,
        mask_shapes       = mask_shapes,
        mask_hashes       = mask_hashes,
        face_analysis_raw = face_analysis_raw,
        aux_image_bytes   = aux_image_bytes,
        aux_image_shapes  = aux_image_shapes,
        aux_image_hashes  = aux_image_hashes,
        aux_pixel_stats   = aux_pixel_stats,
        lora_metadata     = lora_metadata,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mask_to_bytes_hash(mask_tensor: Any) -> tuple[bytes | None, list | None, str | None]:
    """Convert a mask tensor to binary PNG bytes + shape + BLAKE3 hash."""
    try:
        import blake3 as _blake3

        arr = mask_tensor[0].cpu().float().contiguous().numpy()
        raw = _p.ndarray_to_png8_bytes(arr, "binary")
        return raw, list(arr.shape), _blake3.blake3(raw).hexdigest()
    except Exception:
        return None, None, None


def _aux_image_to_bytes_hash(
    image_tensor: Any,
) -> tuple[bytes | None, list | None, str | None]:
    """Convert an aux IMAGE tensor to 8-bit PNG bytes + BLAKE3 hash."""
    try:
        import blake3 as _blake3

        arr = image_tensor[0].cpu().float().contiguous().numpy()
        mode = "L" if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1) else "RGB"
        raw = _p.ndarray_to_png8_bytes(arr, mode)
        shape = list(arr.shape)
        return raw, shape, _blake3.blake3(raw).hexdigest()
    except Exception:
        return None, None, None


def _extract_clip_slot(cv_model: Any, image: Any, masks_dict: dict) -> dict:
    """Extract a CLIP Vision slot dict for one CLIP_VISION model.

    Returns a plain-Python dict.  Never raises.
    """
    slot: dict = {
        "model_hash":               None,
        "global_embedding_bytes":   None,
        "global_embedding_shape":   None,
        "last_hidden_state_bytes":  None,
        "last_hidden_state_shape":  None,
        "is_aliased":               False,
        "stats":                    {},
        "patch_pools":              {},
    }

    try:
        slot["model_hash"] = _p.hash_clip_model(cv_model)
    except Exception:
        pass

    try:
        clip_output = _p.encode_clip_vision(cv_model, image)
        arrays = _p.extract_clip_vision_arrays(clip_output)

        if arrays["global_embedding_array"] is not None:
            slot["global_embedding_bytes"] = _p.embedding_to_fp16npy_bytes(
                arrays["global_embedding_array"]
            )
            slot["global_embedding_shape"] = arrays["global_embedding_shape"]

        if arrays["last_hidden_state_array"] is not None:
            slot["last_hidden_state_bytes"] = _p.embedding_to_fp16npy_bytes(
                arrays["last_hidden_state_array"]
            )
            slot["last_hidden_state_shape"] = arrays["last_hidden_state_shape"]

        slot["is_aliased"] = arrays["is_aliased"]
        slot["stats"]      = arrays["stats"]

        # Patch pools — use live last_hidden_state tensor + live mask tensors.
        # Must be computed here, before tensors are released.
        lhs_tensor = getattr(clip_output, "last_hidden_state", None)
        if lhs_tensor is not None:
            active_masks = {k: v for k, v in masks_dict.items() if v is not None}
            if active_masks:
                slot["patch_pools"] = _p.compute_patch_pools(lhs_tensor, active_masks)

    except Exception as exc:
        diagnostics.emit("WARN", "EXTRACTOR.CLIP_ENCODE_FAIL", _WHERE, str(exc))

    return slot
