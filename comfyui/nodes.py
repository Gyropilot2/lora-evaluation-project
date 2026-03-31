"""
comfyui/nodes.py — ComfyUI node registration glue.

Registers:
  - LabProbe (Step 1.5)  — on-site Lab field discovery node
  - Extractor (Step 2.1) — production Evidence extraction node

ComfyUI dependency: YES — only importable when ComfyUI is installed.
"""

from __future__ import annotations

import json
from typing import Any

from contracts.validation_errors import make_null
from comfyui.replay_enrichment import ReplayEnricher, SampleNeedsMeasurementGuard, SampleReplayLoad
from core import diagnostics
from databank.treasurer import Treasurer
from lab.probe import LabProbe

_WHERE = "comfyui.nodes"
_NO_LORA_BASELINE = "(baseline)"


# ---------------------------------------------------------------------------
# Extractor node
# ---------------------------------------------------------------------------


class Extractor:
    """Production ComfyUI node for Evidence extraction.

    Receives live workflow outputs, converts them to portable primitives,
    assembles three Evidence candidates (Method, Eval, Sample), and routes
    them through Bouncer into the DataBank.

    Returns LATENT and IMAGE as passthroughs so it can be wired inline
    without disrupting downstream nodes.

    Wiring:
        KSampler.LATENT        → latent_image
        VAEDecode.IMAGE        → image
        VAELoader / CheckpointLoader VAE → vae
    """

    CATEGORY     = "lora_eval"
    FUNCTION     = "execute"
    OUTPUT_NODE  = True          # forces execution even without downstream consumers
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")

    # ---- DB singleton (lazy, class-level) ----
    _treasurer: Any = None

    @classmethod
    def _get_treasurer(cls) -> Any:
        if cls._treasurer is None:
            from databank.treasurer import open_treasurer
            cls._treasurer = open_treasurer()
        return cls._treasurer

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        import comfy.samplers
        import folder_paths

        lora_files    = folder_paths.get_filename_list("loras")
        lora_choices  = [_NO_LORA_BASELINE] + lora_files

        return {
            "required": {
                "model":        ("MODEL",),
                "vae":          ("VAE",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "image":        ("IMAGE",),
                "seed":         ("INT",   {"default": 0,
                                           "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 10000}),
                "cfg":          ("FLOAT", {"default": 7.0,
                                           "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 1.0,
                                           "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lora_name":     (lora_choices,),
                "lora_strength": ("FLOAT", {"default": 1.0,
                                            "min": 0.0, "max": 2.0, "step": 0.1,
                                            "tooltip": "LoRA conditioning strength (from LoraLoader node). 0.0 = baseline (no LoRA applied); valid LoRA range 0.1–2.0."}),
                "positive_prompt_hint": ("STRING", {"default": "",
                                            "tooltip": "Optional positive prompt hint for display. Not used for hashing."}),
                "negative_prompt_hint": ("STRING", {"default": "",
                                            "tooltip": "Optional negative prompt hint for display. Not used for hashing."}),
                "workflow_ref_json": ("STRING", {"default": "",
                                            "tooltip": "Optional JSON-serialized workflow_ref passed in by batch_run. Not used for hashing."}),
                "clip_vision_1": ("CLIP_VISION",
                                  {"tooltip": "Semantic similarity model. Recommended: SigLIP2-SO400M."}),
                "clip_vision_2": ("CLIP_VISION",
                                  {"tooltip": "Structural / spatial model. Recommended: DINOv2 (ViT-H, IP-Adapter)."}),
                "mask_face":         ("MASK", {"tooltip": "Face region."}),
                "mask_main_subject": ("MASK", {"tooltip": "Main subject / foreground region (white = subject, black = background). Used for mask-based measurements and pose joint overlap."}),
                "mask_skin":         ("MASK", {"tooltip": "Exposed skin."}),
                "mask_clothing":     ("MASK", {"tooltip": "Clothing and accessories."}),
                "mask_hair":         ("MASK", {"tooltip": "Hair region."}),
                "aux_depth":  ("IMAGE", {"tooltip": "Depth map (e.g. DepthAnythingV2)."}),
                "aux_normal": ("IMAGE", {"tooltip": "Surface normals (e.g. DSINE)."}),
                "aux_edge":   ("IMAGE", {"tooltip": "Edge / line art (e.g. LineArt-Edge)."}),
                "pose_keypoint_openpose": (
                    "POSE_KEYPOINT",
                    {"tooltip": "Grouped OpenPose body-only keypoints for pose_evidence. Connect OpenPosePreprocessor POSE_KEYPOINT output."},
                ),
                "pose_keypoint_dw": (
                    "POSE_KEYPOINT",
                    {"tooltip": "Grouped DW-Pose body-only keypoints for pose_evidence. Connect DWPreprocessor POSE_KEYPOINT output."},
                ),
                "aux_densepose": (
                    "IMAGE",
                    {"tooltip": "DensePose support image for per-joint support facts (optional)."},
                ),
            },
        }

    def execute(
        self,
        model: Any,
        vae: Any,
        positive: Any,
        negative: Any,
        latent_image: Any,
        image: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        lora_name: str        = _NO_LORA_BASELINE,
        lora_strength: float  = 1.0,
        positive_prompt_hint: str = "",
        negative_prompt_hint: str = "",
        workflow_ref_json: str = "",
        clip_vision_1: Any    = None,
        clip_vision_2: Any    = None,
        mask_face: Any           = None,
        mask_main_subject: Any   = None,
        mask_skin: Any           = None,
        mask_clothing: Any       = None,
        mask_hair: Any           = None,
        aux_depth: Any           = None,
        aux_normal: Any          = None,
        aux_edge: Any            = None,
        pose_keypoint_openpose: Any = None,
        pose_keypoint_dw: Any       = None,
        aux_densepose: Any          = None,
    ) -> tuple[Any, Any]:
        from core.time_ids import now_iso
        from extractor.sources import extract_all
        from extractor.extract import run as _extract_run

        timestamp = now_iso()
        treasurer = self._get_treasurer()
        workflow_ref: dict[str, Any] | None = None
        if workflow_ref_json.strip():
            try:
                parsed = json.loads(workflow_ref_json)
                if isinstance(parsed, dict):
                    workflow_ref = parsed
                else:
                    diagnostics.emit(
                        "WARN", "EXTRACTOR.WORKFLOW_REF_PARSE_FAIL", _WHERE,
                        "workflow_ref_json parsed to a non-dict value; ignoring",
                    )
            except Exception as exc:
                diagnostics.emit(
                    "WARN", "EXTRACTOR.WORKFLOW_REF_PARSE_FAIL", _WHERE,
                    f"workflow_ref_json parse failed: {exc}",
                )

        # Normalize lora_name: "(baseline)" sentinel or empty string → None
        _lora_name: str | None  = (
            None if (not lora_name or lora_name == _NO_LORA_BASELINE) else lora_name
        )
        _lora_str: float | None = None if _lora_name is None else lora_strength

        # ---- Phase 1: ComfyUI objects → plain Python ----
        result = extract_all(
            model         = model,
            vae           = vae,
            positive      = positive,
            negative      = negative,
            latent        = latent_image,
            image         = image,
            seed          = seed,
            steps         = steps,
            cfg           = float(cfg),
            sampler_name  = sampler_name,
            scheduler     = scheduler,
            denoise       = float(denoise),
            lora_name     = _lora_name,
            lora_strength = _lora_str,
            positive_prompt_hint = positive_prompt_hint,
            negative_prompt_hint = negative_prompt_hint,
            clip_vision_1 = clip_vision_1,
            clip_vision_2 = clip_vision_2,
            mask_face          = mask_face,
            mask_main_subject  = mask_main_subject,
            mask_skin          = mask_skin,
            mask_clothing      = mask_clothing,
            mask_hair          = mask_hair,
            aux_depth  = aux_depth,
            aux_normal = aux_normal,
            aux_edge   = aux_edge,
        )

        # ---- Phase 2: Stage + commit all asset blobs ----
        valued_refs = _commit_all_assets(treasurer, result)

        # ---- Phase 2.5: Build pose_evidence domain (optional) ----
        extra_domains: dict = {}
        if pose_keypoint_openpose is not None or pose_keypoint_dw is not None:
            from comfyui.replay_enrichment import tensor_image_to_hwc_float, tensor_mask_to_hw_float
            from extractor.pose_evidence import build_pose_evidence
            if mask_main_subject is None:
                diagnostics.emit(
                    "WARN", "EXTRACTOR.POSE_EVIDENCE_FAIL", _WHERE,
                    "pose_evidence skipped: mask_main_subject not connected",
                )
            else:
                try:
                    _image_arr     = tensor_image_to_hwc_float(image)
                    _mask_arr      = tensor_mask_to_hw_float(mask_main_subject)
                    _densepose_arr = tensor_image_to_hwc_float(aux_densepose) if aux_densepose is not None else None
                    _face_raw      = result.face_analysis_raw
                    _face_dict     = _face_raw if isinstance(_face_raw, dict) and _face_raw.get("status") != "not_available" else None
                    _pe = build_pose_evidence(
                        image_arr=_image_arr,
                        main_subject_mask_arr=_mask_arr,
                        openpose_keypoint_data=pose_keypoint_openpose,
                        dw_keypoint_data=pose_keypoint_dw,
                        densepose_arr=_densepose_arr,
                        face_analysis=_face_dict,
                    )
                    if _pe:
                        extra_domains["pose_evidence"] = _pe
                except Exception as exc:
                    diagnostics.emit(
                        "WARN", "EXTRACTOR.POSE_EVIDENCE_FAIL", _WHERE,
                        f"build_pose_evidence raised: {exc}",
                    )

        # ---- Phase 3: Assemble candidates + Bouncer → DataBank ----
        try:
            _extract_run(
                result,
                valued_refs,
                treasurer,
                timestamp,
                extra_domains=extra_domains,
                workflow_ref=workflow_ref,
            )
        except Exception as exc:
            diagnostics.emit(
                "ERROR", "EXTRACTOR.ASSET_COMMIT_FAIL", _WHERE,
                f"extract.run raised unexpectedly: {exc}",
            )

        return (latent_image, image)   # passthrough


# ---------------------------------------------------------------------------
# SampleGuard node
# ---------------------------------------------------------------------------


class SampleGuard:
    """Pre-flight guard: halts workflow if this exact sample already exists in DB.

    Sits inline between source nodes and KSampler.  Computes method/eval/sample
    hashes from the same inputs the Extractor uses (no pixel extraction, no writes),
    queries the DB read-only, and either:

      - Passes all inputs through unchanged if the sample is new.
      - Raises a ComfyUI error (red node, halted workflow) if the sample exists.

    Method/Eval hits are logged as INFO diagnostics only — they are expected for
    most runs (same method + LoRA, new seed or strength).

    Hash computation failures → WARN + pass through (fail-open).  The Extractor
    downstream will catch real vital-field problems.

    Wiring (replaces the direct connection from source nodes to KSampler):
        source nodes → SampleGuard → KSampler
    """

    CATEGORY     = "lora_eval"
    FUNCTION     = "execute"
    RETURN_TYPES = ("MODEL", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("model", "vae", "positive", "negative", "latent", "seed")

    # ---- Read-only DB singleton (lazy, class-level) ----
    _treasurer: Any = None

    @classmethod
    def _get_treasurer(cls) -> Any:
        if cls._treasurer is None:
            from databank.treasurer import open_treasurer
            cls._treasurer = open_treasurer(read_only=True)
        return cls._treasurer

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        import comfy.samplers
        import folder_paths

        lora_files   = folder_paths.get_filename_list("loras")
        lora_choices = [_NO_LORA_BASELINE] + lora_files

        return {
            "required": {
                "model":        ("MODEL",),
                "vae":          ("VAE",),
                "positive":     ("CONDITIONING",),
                "negative":     ("CONDITIONING",),
                "latent":       ("LATENT",),
                "seed":         ("INT",   {"default": 0,
                                           "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 10000}),
                "cfg":          ("FLOAT", {"default": 7.0,
                                           "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 1.0,
                                           "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lora_name":     (lora_choices,),
                "lora_strength": ("FLOAT", {"default": 1.0,
                                            "min": 0.0, "max": 2.0, "step": 0.1,
                                            "tooltip": "LoRA conditioning strength (from LoraLoader node). 0.0 = baseline (no LoRA applied); valid LoRA range 0.1–2.0."}),
            },
        }

    def execute(
        self,
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
        lora_name: str       = _NO_LORA_BASELINE,
        lora_strength: float = 1.0,
    ) -> tuple:
        from extractor.sources import compute_hashes_only

        _lora_name: str | None   = (
            None if (not lora_name or lora_name == _NO_LORA_BASELINE) else lora_name
        )
        _lora_str: float | None  = None if _lora_name is None else lora_strength

        # ---- Phase 1: compute hashes (no pixel extraction, no writes) ----
        try:
            hashes = compute_hashes_only(
                model=model, vae=vae, positive=positive, negative=negative,
                latent=latent, seed=seed, steps=steps, cfg=float(cfg),
                sampler_name=sampler_name, scheduler=scheduler,
                denoise=float(denoise), lora_name=_lora_name, lora_strength=_lora_str,
            )
        except Exception as exc:
            diagnostics.emit(
                "WARN", "GUARD.HASH_FAIL", _WHERE,
                f"hash computation raised unexpectedly; passing through: {exc}",
            )
            return (model, vae, positive, negative, latent, seed)

        method_hash = hashes["method_hash"]
        eval_hash   = hashes["eval_hash"]
        sample_hash = hashes["sample_hash"]

        # ---- Phase 2: DB read-only existence check ----
        sample_rec = None
        try:
            treasurer = self._get_treasurer()

            method_rec = treasurer.get_method(method_hash)
            if method_rec is not None:
                diagnostics.emit(
                    "INFO", "GUARD.METHOD_EXISTS", _WHERE,
                    f"method already in DB: {method_hash[:16]}",
                )

            eval_rec = treasurer.get_eval(eval_hash)
            if eval_rec is not None:
                diagnostics.emit(
                    "INFO", "GUARD.EVAL_EXISTS", _WHERE,
                    f"eval already in DB: {eval_hash[:16]}",
                )

            sample_rec = treasurer.get_sample(sample_hash)

        except Exception as exc:
            diagnostics.emit(
                "WARN", "GUARD.DB_ERROR", _WHERE,
                f"DB query failed; passing through: {exc}",
            )
            return (model, vae, positive, negative, latent, seed)

        # ---- Phase 3: halt if sample already exists ----
        if sample_rec is not None:
            diagnostics.emit(
                "WARN", "GUARD.SAMPLE_EXISTS", _WHERE,
                f"sample already exists: {sample_hash[:16]}",
                seed=seed, lora_strength=_lora_str,
            )
            raise Exception(
                f"[LoRA Eval Guard] Sample already exists "
                f"(hash: {sample_hash[:16]}…). "
                f"seed={seed}, lora_strength={_lora_str}. "
                f"Use a different seed or strength, or remove this guard."
            )

        return (model, vae, positive, negative, latent, seed)


# ---------------------------------------------------------------------------
# Asset staging helpers (module-level private)
# ---------------------------------------------------------------------------


def _commit_all_assets(treasurer: Treasurer, result: Any) -> dict:
    """Stage and commit all binary asset blobs from an ExtractionResult.

    Returns a valued_refs dict keyed by semantic name.  Each value is either
    a clean ValueRef dict or a make_null() Invalid wrapper on failure.

    Never raises.
    """
    valued_refs: dict = {}

    def _commit(data: bytes | None, asset_type: str, fmt: str, key: str) -> None:
        valued_refs[key] = _store_asset_blob(treasurer, data, asset_type, fmt)

    # ---- Main image (canonical float32 .npy for re-ingest integrity) ----
    _commit(result.image_npy_bytes, "image",     "npy", "image")

    # ---- Luminance ----
    _commit(result.lum_npy_bytes,   "luminance", "png", "luminance")

    # ---- CLIP Vision slots (global embedding + last_hidden_state per model) ----
    for idx, slot in enumerate(result.clip_vision_slots):
        _commit(
            slot.get("global_embedding_bytes"),
            "embedding", "npy",
            f"clip_slot_{idx}_global",
        )
        _commit(
            slot.get("last_hidden_state_bytes"),
            "embedding", "npy",
            f"clip_slot_{idx}_lhs",
        )

    # ---- Masks ----
    for mask_name, mask_data in result.mask_bytes.items():
        _commit(mask_data, "mask", "png", f"mask_{mask_name}")

    # ---- Face embeddings (best face = index 0, sorted by det_score) ----
    if isinstance(result.face_analysis_raw, dict):
        for i, face in enumerate(result.face_analysis_raw.get("faces", [])):
            for attr, key_suffix in (
                ("embedding_array",        "embedding"),
                ("normed_embedding_array", "normed_embedding"),
            ):
                arr = face.get(attr)
                if arr is None:
                    continue
                try:
                    from extractor.primitives import ndarray_to_npy_bytes as _npy_bytes
                    _commit(_npy_bytes(arr), "embedding", "npy", f"face_{i}_{key_suffix}")
                except Exception as exc:
                    diagnostics.emit(
                        "ERROR", "EXTRACTOR.ASSET_COMMIT_FAIL", _WHERE,
                        f"face embedding npy_bytes failed: {exc}",
                    )
                    valued_refs[f"face_{i}_{key_suffix}"] = make_null(
                        "EXTRACTOR.ASSET_COMMIT_FAIL"
                    )

    # ---- Aux images ----
    for aux_name, aux_data in result.aux_image_bytes.items():
        _commit(aux_data, "aux_image", "png", f"aux_{aux_name}")

    return valued_refs


def _store_asset_blob(
    treasurer: Treasurer,
    data: bytes | None,
    asset_type: str,
    fmt: str,
) -> dict:
    """Store one asset blob through the Treasurer door.

    Returns a clean ValueRef dict on success, or a make_null() Invalid wrapper
    on failure.  Never raises.
    """
    if data is None:
        return make_null("EXTRACTOR.ASSET_COMMIT_FAIL")
    try:
        return treasurer.store_asset_blob(data, asset_type, fmt)
    except Exception as exc:
        diagnostics.emit(
            "ERROR", "EXTRACTOR.ASSET_COMMIT_FAIL", _WHERE,
            str(exc),
            asset_type=asset_type,
            fmt=fmt,
        )
        return make_null("EXTRACTOR.ASSET_COMMIT_FAIL")


# ---------------------------------------------------------------------------
# Node class registry
# All node classes must be listed here for ComfyUI to discover them.
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "LoraEvalLabProbe":    LabProbe,
    "LoraEvalExtractor":   Extractor,
    "LoraEvalSampleGuard": SampleGuard,
    "LoraEvalSampleReplayLoad": SampleReplayLoad,
    "LoraEvalSampleNeedsMeasurementGuard": SampleNeedsMeasurementGuard,
    "LoraEvalReplayEnricher": ReplayEnricher,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LoraEvalLabProbe":    "LoRA Eval — Lab Probe",
    "LoraEvalExtractor":   "LoRA Eval — Extractor",
    "LoraEvalSampleGuard": "LoRA Eval — Sample Guard",
    "LoraEvalSampleReplayLoad": "LoRA Eval — Sample Replay Load",
    "LoraEvalSampleNeedsMeasurementGuard": "LoRA Eval — Sample Needs Measurement Guard",
    "LoraEvalReplayEnricher": "LoRA Eval — Replay Enricher",
}
