"""
extractor/extract.py — candidate assembly and Bouncer dispatch.

Receives a plain-Python ExtractionResult (from extractor.sources) and committed
ValueRefs (from comfyui.nodes), assembles three Evidence candidate dicts
(Method, Eval, Sample), and routes them through bouncer.gate.

Constraints:
  - No databank import (forbidden pair — extractor ↔ databank)
  - No ComfyUI import
  - treasurer accepted as Any (no Treasurer import)
"""
from __future__ import annotations

from typing import Any

from contracts.validation_errors import is_invalid, make_null
from core import hashing as _hashing

_EXTRACTOR_VERSION = "2.1.0"
_WHERE = "extractor.extract"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    result: Any,          # extractor.sources.ExtractionResult — no import to avoid circular
    valued_refs: dict,    # committed ValueRef dicts from comfyui.nodes
    treasurer: Any,       # SQLiteBackend instance — no databank import
    timestamp: str,
    extra_domains: dict | None = None,
    workflow_ref: dict | None = None,
) -> dict:
    """Assemble candidates and route through Bouncer → DataBank.

    Returns the gate.process() result dict.
    """
    from bouncer import gate

    method_c, eval_c, sample_c = build_candidates(
        result,
        valued_refs,
        timestamp,
        extra_domains,
        workflow_ref=workflow_ref,
    )
    return gate.process(method_c, eval_c, sample_c, treasurer)


def run_replay(
    replay_bundle: dict[str, Any],
    measurement_delta: dict[str, Any],
    treasurer: Any,
) -> dict:
    """Route a replay/backfill enrichment through the same Bouncer flow."""
    from bouncer import gate

    method_c, eval_c, sample_c = build_replay_candidates(replay_bundle, measurement_delta)
    return gate.process(method_c, eval_c, sample_c, treasurer)


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------


def build_candidates(
    result: Any,
    valued_refs: dict,
    timestamp: str,
    extra_domains: dict | None = None,
    workflow_ref: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Assemble (method_candidate, eval_candidate, sample_candidate).

    Hash computation follows plan exactly:
      method_hash = BLAKE3 of canonical method inputs
      eval_hash   = BLAKE3(method_hash, lora_hash)
      sample_hash = BLAKE3(eval_hash, seed, lora_strength)
    """

    # ---- Derive pixel dimensions from latent shape [B, C, H_latent, W_latent] ----
    # Multiply spatial dims by 8 (standard latent downscale factor).
    # Falls back to 0 if shape unavailable (EXTRACTOR.LATENT_SHAPE_FAIL already emitted).
    latent_w: int = 0
    latent_h: int = 0
    if result.latent_shape and len(result.latent_shape) >= 4:
        latent_h = int(result.latent_shape[2]) * 8
        latent_w = int(result.latent_shape[3]) * 8

    # ---- Compute canonical hashes ----
    _method_hash: str = _hashing.method_hash({
        "base_model_hash":                result.base_model_hash,
        "model_extras":                   result.model_extras,
        "positive_conditioning_hash":     result.pos_cond_hash,
        "positive_conditioning_guidance": result.pos_guidance,
        "negative_conditioning_hash":     result.neg_cond_hash,
        "negative_conditioning_guidance": result.neg_guidance,
        "steps":        result.steps,
        "denoise":      result.denoise,
        "sampler":      result.sampler,
        "scheduler":    result.scheduler,
        "cfg":          result.cfg,
        "latent_width": latent_w,
        "latent_height": latent_h,
        "latent_shape": result.latent_shape,
        "vae_model_hash": result.vae_model_hash,
    })
    _eval_hash: str   = _hashing.eval_hash(_method_hash, result.lora_hash)
    _sample_hash: str = _hashing.sample_hash(_eval_hash, result.seed, result.lora_strength)

    # ---- Method candidate ----
    method_candidate = _build_method_candidate(result, _method_hash, workflow_ref=workflow_ref)

    # ---- Eval candidate ----
    eval_candidate = _build_eval_candidate(result, _method_hash, _eval_hash)

    # ---- Sample candidate ----
    sample_candidate = _build_sample_candidate(
        result, valued_refs, _eval_hash, _sample_hash, extra_domains
    )

    return method_candidate, eval_candidate, sample_candidate


def build_replay_candidates(
    replay_bundle: dict[str, Any],
    measurement_delta: dict[str, Any],
) -> tuple[dict, dict, dict]:
    """Assemble candidates for replay/backfill from existing DB records.

    Stored records may already contain Invalid wrappers and sidecar keys that are
    valid for persisted Evidence but not valid as fresh Bouncer input. This path
    sanitizes them back into candidate-shaped records, then overlays only the
    newly supplied measurement delta.
    """
    method_candidate = _sanitize_replay_record(replay_bundle.get("method_record") or {})
    eval_candidate = _sanitize_replay_record(replay_bundle.get("eval_record") or {})
    sample_candidate = _sanitize_replay_record(replay_bundle.get("sample_record") or {})

    sample_candidate.pop("_extras", None)
    sample_candidate.pop("_errors", None)
    sample_candidate.update(measurement_delta or {})

    return method_candidate, eval_candidate, sample_candidate


# ---------------------------------------------------------------------------
# Candidate builders
# ---------------------------------------------------------------------------


def _build_method_candidate(
    result: Any,
    method_hash: str,
    workflow_ref: dict | None = None,
) -> dict:
    """Assemble the method Evidence candidate dict."""

    # Conditioning — required hashes fall back to "" on failure.
    # Optional guidance values included only when available.
    conditioning: dict = {
        "positive_hash": result.pos_cond_hash or "",
        "negative_hash": result.neg_cond_hash or "",
    }
    positive_hint = _normalize_prompt_hint(result.positive_prompt_hint)
    negative_hint = _normalize_prompt_hint(result.negative_prompt_hint)
    if positive_hint is not None:
        conditioning["positive_text"] = positive_hint
    if negative_hint is not None:
        conditioning["negative_text"] = negative_hint
    if result.pos_guidance is not None:
        conditioning["positive_guidance"] = float(result.pos_guidance)
    if result.neg_guidance is not None:
        conditioning["negative_guidance"] = float(result.neg_guidance)

    # Base model — name is optional; hash required (falls back to "" on failure).
    base_model: dict = {"hash": result.base_model_hash or ""}
    if result.base_model_name:
        base_model["name"] = result.base_model_name

    candidate: dict = {
        "method_hash":       method_hash,
        "extractor_version": _EXTRACTOR_VERSION,
        "base_model":        base_model,
        "vae_model":         {"hash": result.vae_model_hash or ""},
        "conditioning":      conditioning,
        "settings": {
            "steps":     int(result.steps),
            "denoise":   float(result.denoise),
            "sampler":   str(result.sampler),
            "scheduler": str(result.scheduler),
            "cfg":       float(result.cfg),
        },
        "latent": {
            "width":  latent_w_from(result),
            "height": latent_h_from(result),
            "shape":  result.latent_shape,
        },
    }

    # Optional top-level field
    if result.model_extras is not None:
        candidate["model_extras"] = result.model_extras
    if workflow_ref is not None:
        candidate["workflow_ref"] = workflow_ref

    return candidate


def _build_eval_candidate(result: Any, method_hash: str, eval_hash: str) -> dict:
    """Assemble the eval Evidence candidate dict."""

    # LoRA object — all sub-fields optional; populated only when available.
    lora_obj: dict = {}
    if result.lora_hash is not None:
        lora_obj["hash"]      = result.lora_hash
    if result.lora_file_hash is not None:
        lora_obj["file_hash"] = result.lora_file_hash
    if result.lora_name is not None:
        lora_obj["name"]      = result.lora_name

    # LoRA metadata for the dedicated LoRAs catalog table (when available).
    metadata = result.lora_metadata if isinstance(result.lora_metadata, dict) else None
    if metadata and metadata.get("status") == "ok":
        if metadata.get("rank") is not None:
            lora_obj["rank"] = metadata.get("rank")
        if metadata.get("network_alpha") is not None:
            lora_obj["network_alpha"] = metadata.get("network_alpha")
        if metadata.get("target_blocks") is not None:
            lora_obj["target_blocks"] = metadata.get("target_blocks")
        if metadata.get("affects_text_encoder") is not None:
            lora_obj["affects_text_encoder"] = metadata.get("affects_text_encoder")
        if metadata.get("raw_metadata") is not None:
            lora_obj["raw_metadata"] = metadata.get("raw_metadata")

    return {
        "eval_hash":  eval_hash,
        "method_hash": method_hash,
        "lora":        lora_obj,
    }


def _build_sample_candidate(
    result: Any,
    valued_refs: dict,
    eval_hash: str,
    sample_hash: str,
    extra_domains: dict | None = None,
) -> dict:
    """Assemble the sample Evidence candidate dict."""

    # Image measurement domain (required)
    image_domain: dict = {
        "output": valued_refs.get("image", make_null("EXTRACTOR.ASSET_COMMIT_FAIL")),
    }
    if result.image_shape is not None:
        image_domain["shape"] = result.image_shape
    if result.image_pixel_stats is not None:
        image_domain["pixel_stats"] = result.image_pixel_stats

    # Luminance measurement domain (required)
    lum_domain: dict = {
        "output": valued_refs.get("luminance", make_null("EXTRACTOR.ASSET_COMMIT_FAIL")),
    }
    if result.lum_stats is not None:
        # Only the four schema-defined sub-fields; extras (shape, dtype, coefficients)
        # would become WARN unknowns in Bouncer. Emit only what the schema expects.
        try:
            lum_domain["stats"] = {
                "mean": float(result.lum_stats["mean"]),
                "std":  float(result.lum_stats["std"]),
                "min":  float(result.lum_stats["min"]),
                "max":  float(result.lum_stats["max"]),
            }
        except (KeyError, TypeError, ValueError):
            pass  # lum_stats malformed — omit stats sub-object

    candidate: dict = {
        "sample_hash":       sample_hash,
        "eval_hash":         eval_hash,
        "seed":              int(result.seed),
        "latent_hash":       result.latent_hash,   # None → Bouncer hard refusal
        "image_hash":        result.image_hash,    # None → Bouncer hard refusal
        "extractor_version": _EXTRACTOR_VERSION,
        "image":             image_domain,
        "luminance":         lum_domain,
    }

    if result.lora_strength is not None:
        candidate["lora_strength"] = float(result.lora_strength)

    # Optional measurement domains — omit entirely if empty
    cv = _build_clip_vision_domain(result, valued_refs)
    if cv:
        candidate["clip_vision"] = cv

    masks = _build_masks_domain(result, valued_refs)
    if masks:
        candidate["masks"] = masks

    face = _build_face_domain(result, valued_refs)
    if face:
        candidate["face_analysis"] = face

    aux = _build_aux_domain(result, valued_refs)
    if aux:
        candidate["aux"] = aux

    # Optional injected domains (e.g. pose_evidence from live Extractor node)
    for _domain_key, _domain_val in (extra_domains or {}).items():
        if _domain_val:
            candidate[_domain_key] = _domain_val

    return candidate


# ---------------------------------------------------------------------------
# Measurement domain builders (private)
# ---------------------------------------------------------------------------


def _build_clip_vision_domain(result: Any, valued_refs: dict) -> dict:
    """Build clip_vision domain dict.

    Dynamic-key object (no fixed schema enforcement in Bouncer).
    Keys: "slot_0", "slot_1", ... — one per CLIP model provided.
    """
    domain: dict = {}
    for idx, slot in enumerate(result.clip_vision_slots):
        model_hash = slot.get("model_hash")
        # Key by model_hash[:16] so the same model is always compared with itself
        # regardless of wiring order (slot-index trap S-21). Falls back to slot_{idx}
        # only when model_hash is unavailable (should not occur in practice).
        key = model_hash[:16] if isinstance(model_hash, str) and len(model_hash) >= 16 else f"slot_{idx}"
        slot_dict: dict = {
            "model_hash":              model_hash,
            "global_embedding_shape":  slot.get("global_embedding_shape"),
            "last_hidden_state_shape": slot.get("last_hidden_state_shape"),
            "is_aliased":              slot.get("is_aliased", False),
            "stats":                   slot.get("stats", {}),
            "patch_pools":             slot.get("patch_pools", {}),
        }

        # Asset ValueRefs
        ge_key  = f"clip_slot_{idx}_global"
        lhs_key = f"clip_slot_{idx}_lhs"
        slot_dict["global_embedding"]   = valued_refs.get(ge_key,  make_null("EXTRACTOR.ASSET_COMMIT_FAIL"))
        slot_dict["last_hidden_state"]  = valued_refs.get(lhs_key, make_null("EXTRACTOR.ASSET_COMMIT_FAIL"))

        domain[key] = slot_dict

    return domain


def _build_masks_domain(result: Any, valued_refs: dict) -> dict:
    """Build masks domain dict.

    Dynamic-key object (no fixed schema enforcement in Bouncer).
    """
    domain: dict = {}
    for mask_name in ("face", "main_subject", "skin", "clothing", "hair"):
        if mask_name not in result.mask_bytes:
            continue
        ref_key = f"mask_{mask_name}"
        domain[mask_name] = {
            "output": valued_refs.get(ref_key, make_null("EXTRACTOR.ASSET_COMMIT_FAIL")),
            "shape":  result.mask_shapes.get(mask_name),
            "hash":   result.mask_hashes.get(mask_name),
        }
    return domain


def _build_face_domain(result: Any, valued_refs: dict) -> dict:
    """Build face_analysis domain dict.

    Schema has fixed nested fields (face_analysis validated by Bouncer).
    Uses the best (highest det_score) face from face_analysis_raw.
    """
    raw = result.face_analysis_raw
    if not isinstance(raw, dict):
        return {}
    status = raw.get("status")
    if status != "ok":
        return {}

    faces = raw.get("faces", [])
    face_count = raw.get("face_count", len(faces))

    domain: dict = {"face_count": int(face_count)}

    if not faces:
        return domain

    # Best face = first (sorted by det_score desc in primitives.run_insightface)
    best = faces[0]

    # Embedding ValueRef (from nodes.py asset commit)
    emb_ref = valued_refs.get("face_0_embedding")
    if emb_ref is not None:
        domain["embedding"] = emb_ref

    # Normed embedding ValueRef — used by compare.face.primary_identity_cos
    normed_emb_ref = valued_refs.get("face_0_normed_embedding")
    if normed_emb_ref is not None:
        domain["normed_embedding"] = normed_emb_ref

    if "embedding_hash" in best:
        domain["embedding_hash"] = str(best["embedding_hash"])
    if "embedding_norm" in best:
        domain["embedding_norm"] = float(best["embedding_norm"])
    if "det_score" in best:
        domain["det_score"] = float(best["det_score"])
    if "bbox" in best:
        domain["bbox"] = [float(x) for x in best["bbox"]]
    if "age" in best:
        domain["age"] = float(best["age"])

    # gender: schema expects string — prefer sex attribute if available
    if "sex" in best:
        domain["gender"] = str(best["sex"])
    elif "gender" in best:
        g = best["gender"]
        domain["gender"] = "M" if g == 1 else ("F" if g == 0 else str(g))

    # Pose: stored inline as {pitch, yaw, roll} floats.
    # InsightFace returns pose as a [pitch, yaw, roll] list.
    # Used by compare.face.primary_pose_delta.v1
    if "pose" in best:
        pose_data = best["pose"]
        if isinstance(pose_data, list) and len(pose_data) >= 3:
            domain["pose"] = {
                "pitch": float(pose_data[0]),
                "yaw":   float(pose_data[1]),
                "roll":  float(pose_data[2]),
            }

    return domain


def _build_aux_domain(result: Any, valued_refs: dict) -> dict:
    """Build aux domain dict.

    Dynamic-key object (no fixed schema enforcement in Bouncer).
    """
    domain: dict = {}
    for aux_name in ("depth", "normal", "edge"):  # "pose" raster aux removed; use pose_evidence instead
        if aux_name not in result.aux_image_bytes:
            continue
        ref_key = f"aux_{aux_name}"
        domain[aux_name] = {
            "output":      valued_refs.get(ref_key, make_null("EXTRACTOR.ASSET_COMMIT_FAIL")),
            "shape":       result.aux_image_shapes.get(aux_name),
            "hash":        result.aux_image_hashes.get(aux_name),
            "pixel_stats": result.aux_pixel_stats.get(aux_name),
        }
    return domain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def latent_w_from(result: Any) -> int:
    """Derive pixel width from latent shape (W_latent × 8)."""
    if result.latent_shape and len(result.latent_shape) >= 4:
        return int(result.latent_shape[3]) * 8
    return 0


def latent_h_from(result: Any) -> int:
    """Derive pixel height from latent shape (H_latent × 8)."""
    if result.latent_shape and len(result.latent_shape) >= 4:
        return int(result.latent_shape[2]) * 8
    return 0


def _normalize_prompt_hint(value: Any) -> str | None:
    """Normalize optional prompt hint text for method.conditioning.*_text.

    Accepts only non-empty strings that are not boolean-like literals.
    This prevents accidental "true"/"false" leakage from miswired nodes.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.casefold() in {"true", "false"}:
        return None
    return cleaned


def _sanitize_replay_record(value: Any) -> Any:
    """Strip persisted-only wrappers back into candidate-safe replay input."""
    if is_invalid(value):
        return None
    if isinstance(value, list):
        return [_sanitize_replay_record(item) for item in value]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            sanitized = _sanitize_replay_record(item)
            if sanitized is None and is_invalid(item):
                continue
            clean[key] = sanitized
        return clean
    return value
