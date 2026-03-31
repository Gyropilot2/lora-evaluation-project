"""lab/probe.py — on-site Lab ComfyUI node for field discovery.

The Lab Probe is a PROSPECTING tool. It observes inputs at workflow execution
time, extracts everything it can as portable primitives, and writes a JSON
dump file for offline analysis by field_catalog.py.

Goal: exhaust the extractable signal space. Capture too much rather than too
little. Failures on individual fields are recorded, not raised — the probe
must never crash a workflow.

Input design (v0.9.6):
  LATENT + VAE     — probe decodes the latent internally. VAE identity is hashed
                     for run identity. Decoded IMAGE is returned for preview.
                     NOTE (V2): architecture will change to accept IMAGE + VAE
                     directly; internal VAEDecode will be removed. Lab left as-is.
  CONDITIONING     — positive (required) and negative (optional) conditioning
                     tensors. We hash the tensors, not the raw text.
  CLIP_VISION × 2 — two optional CLIP_VISION model slots. Encoding done
                     internally. Output stored as {model_hash[:16]: {...}} dict —
                     slot index is discarded (wiring convenience only).
                     slot_1: SigLIP / SigLIP2 (no projection, 1024 spatial tokens)
                     slot_2: ViT-L or ViT-H (projected image_embeds [768/1024])
                     Per each active mask: patch pool from spatial features of the
                     GLOBAL encode (no re-encode). Stored as patch_pools per mask.
                     Masked re-encode (image*mask → encode) NOT done — masks are
                     applied to stored embeddings at comparison time.
  MASK × 5        — optional masks from external segmentation (white=subject,
                     black=rest). Slots: face | main_subject | skin | clothing | hair.
                     Stored as ValueRefs only — raw mask tensors as .npy files.
                     Derived stats (pixel_stats, edge_density) are computed at
                     comparison time from stored image + mask, not pre-baked here.
  INSIGHTFACE      — boolean toggle (default True). Auto-resolves AntelopeV2 from
                     ComfyUI models/insightface/. Runs face detection and ArcFace
                     identity embedding on the decoded image.
  AUX IMAGES × 4  — optional preprocessor outputs from external nodes.
                     Pixel stats + content hash captured for structural drift analysis.
                     Slots: depth (DepthAnythingV2) | pose (DW-Pose) |
                            normal (DSINE) | edge (LineArt-Edge).
  LUMINANCE MAP    — computed internally from decoded image (BT.709 coefficients).
                     Stored as a .npy attachment alongside the JSON dump.
                     Enables masked luminance analysis at comparison time without
                     reloading the full image.

Baseline handling:
  Auto-detected from model.patches. No session_tag. Pairing is content-based
  in field_catalog.py.

ComfyUI dependency: YES. No production DB access — dumps only.
"""

from __future__ import annotations

import struct
import uuid
from pathlib import Path
from typing import Any

from core.diagnostics import emit
from core.hashing import (
    hash_bytes,
    hash_tensor,
    hash_safetensors_content,
    method_hash as _compute_method_hash,
    eval_hash as _compute_eval_hash,
    sample_hash as _compute_sample_hash,
)
from core.paths import get_path
from core.time_ids import now_iso
from lab.dump_writer import write_attachment, write_dump

_WHERE = "lab.probe"
_PROBE_VERSION = "0.9.6"
_NO_LORA_SENTINEL = "(none)"

def _read_evidence_version() -> str:
    import json
    try:
        schema = get_path("project_root") / "contracts" / "evidence.schema.json"
        return json.loads(schema.read_text(encoding="utf-8")).get("_meta", {}).get("version", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"

_EVIDENCE_VERSION = _read_evidence_version()

# Module-level InsightFace app cache — expensive to load, free to reuse.
_insightface_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def _try(fn: Any) -> dict[str, Any]:
    """Call fn(); wrap any exception into a structured error record."""
    try:
        return {"status": "ok", "value": fn()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": type(exc).__name__, "detail": str(exc)}


def _make_valueref(
    asset_type: str,
    content_hash: str,
    path: str | None,
    dtype: str | None = None,
    shape: list | None = None,
) -> dict[str, Any]:
    """Build a ValueRef portable content-addressed asset reference."""
    vr: dict[str, Any] = {
        "valueref_version": 1,
        "kind": "asset_ref",
        "asset_type": asset_type,
        "format": "npy",
        "content_hash": {"algo": "blake3", "digest": content_hash},
        "path": path,
    }
    if dtype is not None:
        vr["dtype"] = dtype
    if shape is not None:
        vr["shape"] = shape
    return vr


def _invalid(status: str, reason_code: str = "", detail: str = "") -> dict[str, Any]:
    """Build an Invalid wrapper for a field that cannot hold a valid value."""
    return {"status": status, "reason_code": reason_code, "detail": detail}


def _fast_file_hash(path: "Path") -> str:
    """BLAKE3 hash of first 64 KB + last 64 KB of a file.

    Fast stable fingerprint that works for any file format (GGUF, .pt, .ckpt, etc.)
    without loading the full file into memory. Two large models with identical
    head+tail are astronomically unlikely to be the same model.
    """
    CHUNK = 65536  # 64 KB
    with open(path, "rb") as f:
        head = f.read(CHUNK)
        size = path.stat().st_size
        if size > CHUNK:
            f.seek(max(0, size - CHUNK))
            tail = f.read(CHUNK)
        else:
            tail = b""
    return hash_bytes(head + tail)


# ---------------------------------------------------------------------------
# Generic model weight hasher
# ---------------------------------------------------------------------------


def _hash_model_weights(model_obj: Any) -> dict[str, Any]:
    """Hash all named parameter tensors of a PyTorch model for stable identity.

    Unwrap priority: first_stage_model (VAE) → model (CLIP) → self.
    """
    import blake3 as _blake3

    h = _blake3.blake3()
    class_name = type(model_obj).__name__
    param_count = 0

    try:
        inner = (
            getattr(model_obj, "first_stage_model", None)
            or getattr(model_obj, "model", None)
            or model_obj
        )
        if hasattr(inner, "named_parameters"):
            for name, param in sorted(inner.named_parameters()):
                h.update(name.encode("utf-8"))
                h.update(param.data.cpu().float().contiguous().numpy().tobytes())
                param_count += param.numel()
        return {
            "class": class_name,
            "inner_class": type(inner).__name__,
            "model_hash": h.hexdigest(),
            "param_count": param_count,
        }
    except Exception as exc:  # noqa: BLE001
        return {"class": class_name, "model_hash": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Latent
# ---------------------------------------------------------------------------


def _extract_latent(latent: Any) -> dict[str, Any]:
    """Extract latent tensor metadata. Asset writing is a V2 concern."""
    out: dict[str, Any] = {}
    t = latent.get("samples") if isinstance(latent, dict) else latent
    if t is None:
        return {"status": "error", "detail": "no 'samples' key in latent dict"}

    out["shape"] = _try(lambda: list(t.shape))
    out["dtype"] = _try(lambda: str(t.dtype))
    out["stats"] = _try(lambda: {
        "min": float(t.float().min()),
        "max": float(t.float().max()),
        "mean": float(t.float().mean()),
        "std": float(t.float().std()),
    })
    out["content_hash"] = _try(lambda: hash_tensor(
        str(t.dtype), tuple(t.shape), t.cpu().float().contiguous().numpy().tobytes()
    ))
    out["asset_note"] = "V2: store as ValueRef (.npy in databank/assets/)"
    return out


# ---------------------------------------------------------------------------
# VAE model
# ---------------------------------------------------------------------------


def _extract_vae_model(vae: Any) -> dict[str, Any]:
    return _try(lambda: _hash_model_weights(vae))


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def _extract_conditioning(cond: Any) -> dict[str, Any]:
    """Hash and describe a ComfyUI CONDITIONING = List[(context_tensor, dict)]."""
    import blake3 as _blake3

    if cond is None:
        return {"status": "not_provided"}
    if not isinstance(cond, (list, tuple)):
        return {"status": "error", "detail": f"unexpected type: {type(cond).__name__}"}

    h = _blake3.blake3()
    out: dict[str, Any] = {"pair_count": len(cond)}
    first_done = False

    for pair in cond:
        if not isinstance(pair, (list, tuple)) or len(pair) < 1:
            continue
        t = pair[0]
        if hasattr(t, "cpu"):
            try:
                h.update(t.cpu().float().contiguous().numpy().tobytes())
                if not first_done:
                    # Claim the slot immediately so a downstream exception
                    # never causes a second tensor to overwrite these stats.
                    first_done = True
                    tf = t.float()
                    out["first_tensor_shape"] = list(t.shape)
                    out["first_tensor_dtype"] = str(t.dtype)
                    out["mean"] = float(tf.mean())
                    try:
                        out["norm"] = float(tf.norm())
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        if len(pair) > 1 and isinstance(pair[1], dict):
            pooled = pair[1].get("pooled_output")
            if pooled is not None and hasattr(pooled, "cpu"):
                try:
                    h.update(pooled.cpu().float().contiguous().numpy().tobytes())
                    out["pooled_shape"] = list(pooled.shape)
                    out["pooled_norm"] = float(pooled.float().norm())
                except Exception:  # noqa: BLE001
                    pass

            # Guidance value (FLUX / SDXL): stored as a float in the extra dict.
            guidance = pair[1].get("guidance")
            if guidance is not None:
                try:
                    out["guidance"] = float(guidance)
                except (TypeError, ValueError):
                    pass

    out["hash"] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# Safetensors metadata reader
# ---------------------------------------------------------------------------


def _read_lora_metadata(lora_name: str) -> dict[str, Any]:
    """Read the JSON metadata header from a safetensors LoRA file."""
    import json as _json

    try:
        import folder_paths

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            return {"status": "error", "detail": "file not found via folder_paths"}

        with open(lora_path, "rb") as f:
            header_len_raw = f.read(8)
            if len(header_len_raw) < 8:
                return {"status": "error", "detail": "file too short"}
            header_len = struct.unpack("<Q", header_len_raw)[0]
            if header_len > 200 * 1024 * 1024:
                return {"status": "error", "detail": f"header implausibly large: {header_len}"}
            header = _json.loads(f.read(header_len).decode("utf-8"))

        metadata = header.get("__metadata__", {})
        display_name = None
        for key in ("ss_output_name", "modelspec.title", "name", "LoRA_name"):
            if key in metadata and metadata[key]:
                display_name = str(metadata[key])
                break

        return {
            "status": "ok",
            "file_path": str(lora_path),
            "raw_metadata": {k: str(v) for k, v in metadata.items()},
            "inferred_display_name": display_name,
            "metadata_key_count": len(metadata),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------


def _extract_model_pipeline(model_patcher: Any) -> dict[str, Any]:
    """Extract ModelPatcher info: base model, LoRA patches, model_options."""
    out: dict[str, Any] = {}
    out["patcher_class"] = _try(lambda: type(model_patcher).__name__)

    def _base_model() -> dict[str, Any]:
        m = model_patcher.model
        info: dict[str, Any] = {"class": type(m).__name__}
        if hasattr(m, "model_config"):
            cfg = m.model_config
            info["config_class"] = type(cfg).__name__
            if hasattr(cfg, "unet_config") and isinstance(cfg.unet_config, dict):
                info["unet_config"] = {
                    k: (v if isinstance(v, (str, int, float, bool, list, type(None))) else str(v))
                    for k, v in cfg.unet_config.items()
                }
            # Use vars() to check instance dict directly — avoids triggering
            # ComfyUI's __getattr__ override which prints a WARNING side-effect
            # before raising AttributeError, causing noise even when hasattr()
            # correctly returns False.
            _cfg_vars: dict = {}
            try:
                _cfg_vars = vars(cfg)
            except TypeError:
                pass
            for attr in ("manual_cast_dtype", "supported_inference_dtypes",
                         "latent_format", "adm_in_channels"):
                if attr in _cfg_vars:
                    val = _cfg_vars[attr]
                    info[attr] = val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
        if hasattr(m, "model_type"):
            info["model_type"] = str(m.model_type)
        if hasattr(m, "adm_channels"):
            info["adm_channels"] = m.adm_channels
        if hasattr(m, "state_dict"):
            try:
                info["state_dict_key_count"] = len(list(m.state_dict().keys()))
            except Exception:  # noqa: BLE001
                pass  # omit if unavailable; state_dict() can be expensive on some models
        return info

    out["base_model"] = _try(_base_model)

    def _patch_analysis() -> dict[str, Any]:
        patches = model_patcher.patches
        if not patches:
            return {"count": 0, "note": "no LoRA patches"}

        keys = sorted(patches.keys())
        layer_components: dict[str, int] = {}
        block_types: dict[str, int] = {}
        attn_types: dict[str, int] = {}

        for k in keys:
            for part in k.split("."):
                if part in ("to_q", "to_k", "to_v", "to_out", "proj_out", "proj_in",
                            "proj", "ff_net", "ff", "norm1", "norm2", "linear1", "linear2"):
                    layer_components[part] = layer_components.get(part, 0) + 1
                if part in ("input_blocks", "middle_block", "output_blocks",
                            "double_blocks", "single_blocks", "final_layer"):
                    block_types[part] = block_types.get(part, 0) + 1
                if part in ("attn1", "attn2", "img_attn", "txt_attn"):
                    attn_types[part] = attn_types.get(part, 0) + 1

        patch_structs: list[dict[str, Any]] = []
        for k in keys[:8]:
            for tup in patches[k][:1]:
                s: dict[str, Any] = {"key": k, "tuple_len": len(tup), "components": []}
                if len(tup) > 0:
                    try:
                        s["strength"] = float(tup[0])
                    except Exception:  # noqa: BLE001
                        s["strength"] = str(type(tup[0]).__name__)
                for i, item in enumerate(tup[1:], start=1):
                    if hasattr(item, "shape") and hasattr(item, "dtype"):
                        s["components"].append({"index": i, "kind": "tensor",
                                                "shape": list(item.shape), "dtype": str(item.dtype)})
                    elif hasattr(item, "weights") and item.weights is not None:
                        w = item.weights
                        comp: dict[str, Any] = {"index": i, "kind": "LoRAAdapter"}
                        try:
                            if hasattr(w[0], "shape"):
                                comp["lora_up_shape"] = list(w[0].shape)
                            if hasattr(w[1], "shape"):
                                comp["lora_down_shape"] = list(w[1].shape)
                            if w[2] is not None:
                                try:
                                    comp["alpha"] = float(w[2])
                                except Exception:  # noqa: BLE001
                                    comp["alpha"] = str(w[2])
                            comp["has_mid"] = w[3] is not None
                            comp["has_dora_scale"] = w[4] is not None
                        except Exception:  # noqa: BLE001
                            pass
                        s["components"].append(comp)
                    elif isinstance(item, (tuple, list)):
                        s["components"].append({"index": i, "kind": "tensor_tuple", "len": len(item),
                                                "elements": [{"shape": list(e.shape), "dtype": str(e.dtype)}
                                                             if hasattr(e, "shape") else {"type": type(e).__name__}
                                                             for e in item]})
                    else:
                        s["components"].append({"index": i, "kind": type(item).__name__})
                patch_structs.append(s)

        import blake3 as _blake3
        h = _blake3.blake3()
        for k in keys:
            h.update(k.encode("utf-8"))
            for tup in patches[k]:
                for item in tup[1:]:
                    if hasattr(item, "cpu"):
                        try:
                            h.update(item.cpu().float().contiguous().numpy().tobytes())
                        except Exception:  # noqa: BLE001
                            pass
                    elif hasattr(item, "weights") and item.weights is not None:
                        for w in item.weights[:2]:
                            if hasattr(w, "cpu"):
                                try:
                                    h.update(w.cpu().float().contiguous().numpy().tobytes())
                                except Exception:  # noqa: BLE001
                                    pass
                    elif isinstance(item, (tuple, list)):
                        for sub in item:
                            if hasattr(sub, "cpu"):
                                try:
                                    h.update(sub.cpu().float().contiguous().numpy().tobytes())
                                except Exception:  # noqa: BLE001
                                    pass

        return {
            "count": len(keys),
            "keys_sample": keys[:20],
            "keys_tail_sample": keys[-5:] if len(keys) > 20 else [],
            "layer_components": layer_components,
            "block_types": block_types,
            "attn_types": attn_types,
            "patch_structs_sample": patch_structs,
            "content_hash": h.hexdigest(),
        }

    out["lora_patches"] = _try(_patch_analysis)

    def _model_options() -> dict[str, Any]:
        opts = model_patcher.model_options
        result: dict[str, Any] = {}
        for k, v in opts.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, dict):
                result[k] = {dk: (dv if isinstance(dv, (str, int, float, bool, type(None))) else type(dv).__name__)
                             for dk, dv in v.items()}
            else:
                result[k] = type(v).__name__
        return result

    out["model_options"] = _try(_model_options)

    def _compute_base_model_content_hash() -> dict[str, Any]:
        """Hash base model weights as a stable in-memory identity fingerprint.

        Strategy:
          Phase 1 — Structural fingerprint: hash all parameter names + shapes + dtypes.
                     This is instant (no data access) and unique to architecture variant.
          Phase 2 — Value anchor: for the first 3 + last 3 parameters (sorted by name),
                     read first 256 elements in native dtype as raw bytes. Distinguishes
                     different trained weights at the same architecture. Wrapped in
                     try/except so GGUF quantized tensors that can't be numpy()'d are
                     skipped gracefully — structural hash alone is still useful.

        Identifies the inner diffusion model by trying common attribute names:
          diffusion_model (ComfyUI standard) → model → self.
        """
        import numpy as np
        import blake3 as _blake3

        m = model_patcher.model
        inner = (
            getattr(m, "diffusion_model", None)
            or getattr(m, "model", None)
            or m
        )
        if not hasattr(inner, "named_parameters"):
            return {"content_hash": None, "error": "no named_parameters", "param_count": 0}

        h = _blake3.blake3()
        params = sorted(inner.named_parameters())  # list of (name, Parameter)
        param_count = len(params)

        # Phase 1: Structural fingerprint — all names + shapes + dtypes (no data access)
        for name, param in params:
            h.update(f"{name}|{list(param.shape)}|{param.dtype}\n".encode())

        # Phase 2: Value anchor — first 256 elements (raw dtype bytes) of first 3 + last 3 params
        anchor = (params[:3] + params[-3:]) if param_count > 6 else params
        for _name, param in anchor:
            try:
                arr = param.data.cpu().contiguous().numpy()
                flat = arr.ravel()
                n = min(256, flat.size)
                raw = flat[:n].view(np.uint8).tobytes()
                h.update(raw)
            except Exception:  # noqa: BLE001
                pass  # GGUF quantized tensors may not support numpy(); skip gracefully

        return {
            "content_hash": h.hexdigest(),
            "param_count": param_count,
            "sample_strategy": "name_shape_dtype_all + value_anchor_first3_last3_256elem",
        }

    out["base_model_content_hash"] = _try(_compute_base_model_content_hash)
    return out


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def _extract_image(image: Any) -> dict[str, Any]:
    """Extract image tensor metadata: shape, dtype, pixel stats, content hash."""
    out: dict[str, Any] = {}
    out["shape"] = _try(lambda: list(image.shape))
    out["dtype"] = _try(lambda: str(image.dtype))
    out["pixel_stats"] = _try(lambda: {
        "min":  float(image.float().min()),
        "max":  float(image.float().max()),
        "mean": float(image.float().mean()),
        "std":  float(image.float().std()),
    })
    out["content_hash"] = _try(lambda: hash_tensor(
        str(image.dtype), tuple(image.shape), image.cpu().contiguous().numpy().tobytes()
    ))
    return out


# ---------------------------------------------------------------------------
# Luminance map
# ---------------------------------------------------------------------------


def _compute_luminance_map(image: Any) -> tuple[Any, dict[str, Any]]:
    """Compute per-pixel luminance from the decoded image using BT.709 coefficients.

    Y = 0.2126 R + 0.7152 G + 0.0722 B

    image: [B, H, W, C] float32 [0,1]

    Returns:
        lum_array:  float32 numpy array [H, W] for the first batch item.
                    Written to disk as .npy via write_attachment.
        stats:      summary dict — mean, std, min, max, shape, dtype.
                    Stored inline in the JSON dump.
    """
    import numpy as np

    # BT.709 coefficients
    r = float(image[0, :, :, 0].float().mean())  # not used — computed per-pixel below
    del r  # silence linter

    tf = image[0].float().cpu()   # [H, W, C]
    lum = (
        tf[:, :, 0] * 0.2126 +
        tf[:, :, 1] * 0.7152 +
        tf[:, :, 2] * 0.0722
    )  # [H, W]

    lum_np = lum.contiguous().numpy().astype("float32")

    stats: dict[str, Any] = {
        "mean":  float(lum.mean()),
        "std":   float(lum.std()),
        "min":   float(lum.min()),
        "max":   float(lum.max()),
        "shape": list(lum.shape),
        "dtype": "float32",
        "coefficients": "BT.709 (0.2126R + 0.7152G + 0.0722B)",
    }
    return lum_np, stats


# ---------------------------------------------------------------------------
# CLIP Vision: embedding stats
# ---------------------------------------------------------------------------


def _extract_clip_vision(clip_output: Any) -> dict[str, Any]:
    """Extract embedding stats from a CLIP_VISION_OUTPUT.

    Detects SigLIP aliasing (image_embeds == last_hidden_state in memory).
    Adds mean_pooled stats for 3D [B, seq_len, dim] embeddings.
    """
    out: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    for attr in ("image_embeds", "last_hidden_state", "penultimate_hidden_states"):
        t = getattr(clip_output, attr, None)
        if t is not None:
            tensors[attr] = t

    is_aliased = False
    if "image_embeds" in tensors and "last_hidden_state" in tensors:
        try:
            is_aliased = tensors["image_embeds"].data_ptr() == tensors["last_hidden_state"].data_ptr()
        except Exception:  # noqa: BLE001
            pass

    out["is_aliased"] = is_aliased
    if is_aliased:
        out["aliasing_note"] = (
            "image_embeds and last_hidden_state are the same tensor. "
            "Typical of SigLIP/SigLIP2 without projection_dim."
        )

    for attr, tensor in tensors.items():
        def _stats(t_: Any = tensor, a_: str = attr) -> dict[str, Any]:
            tf = t_.float()
            info: dict[str, Any] = {
                "shape": list(t_.shape), "dtype": str(t_.dtype),
                "min": float(tf.min()), "max": float(tf.max()), "mean": float(tf.mean()),
            }
            if is_aliased and a_ == "last_hidden_state":
                info["note"] = "aliased_to_image_embeds_siglip_no_projection"
            if is_aliased and a_ == "image_embeds":
                info["note"] = "aliases_last_hidden_state_siglip_no_projection"
            if a_ == "image_embeds" and len(t_.shape) == 3:
                # SigLIP / SigLIP2: 3D [B, seq_len, dim] — mean-pool over sequence axis
                pooled = tf.mean(dim=1)  # → [B, dim]
                info["mean_pooled_shape"] = list(pooled.shape)
                info["mean_pooled_mean"] = float(pooled.mean())
                try:
                    info["mean_pooled_norm"] = float(pooled.norm(dim=-1).mean())
                except Exception:  # noqa: BLE001
                    pass
                # Store the actual pooled embedding array for .npy asset write.
                # Popped and written by the cv_by_model assembly in probe(); not serialised to JSON.
                try:
                    import numpy as np
                    info["mean_pooled_array"] = pooled.cpu().float().numpy()  # float32 [B, dim]
                except Exception:  # noqa: BLE001
                    pass
            elif a_ == "image_embeds" and len(t_.shape) == 2:
                # ViT-L / ViT-H: 2D [B, dim] — already projected, no pooling needed
                info["mean_pooled_shape"] = list(t_.shape)
                try:
                    info["mean_pooled_norm"] = float(tf.norm(dim=-1).mean())
                except Exception:  # noqa: BLE001
                    pass
                try:
                    import numpy as np
                    info["mean_pooled_array"] = tf.cpu().float().numpy()  # float32 [B, dim]
                except Exception:  # noqa: BLE001
                    pass
            if a_ == "last_hidden_state":
                # Capture full spatial tensor for mask-weighted patch pooling at comparison time.
                # SigLIP (is_aliased): [1, 1024, 1152] — 1024 spatial patches.
                # ViT-L (not aliased): [1, 257, 1024] — CLS at index 0 + 256 spatial patches.
                # Stored as last_hidden_state_valueref by the probe loop.
                try:
                    import numpy as np
                    info["last_hidden_state_array"] = t_.cpu().float().numpy()
                except Exception:  # noqa: BLE001
                    pass
            return info

        out[attr] = _try(_stats)

    return out


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _resize_mask_to_image(mask: Any, image: Any) -> Any:
    """Resize mask to match image spatial dimensions if needed.

    mask:  [H, W] or [B, H, W]
    image: [B, H, W, C]
    Returns mask with shape matching image [H, W] or [B, H, W].
    """
    import torch.nn.functional as F

    ih, iw = image.shape[1], image.shape[2]
    if mask.dim() == 2:
        mh, mw = mask.shape
    else:
        mh, mw = mask.shape[-2], mask.shape[-1]

    if mh == ih and mw == iw:
        return mask

    m = mask.float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        m = F.interpolate(m, size=(ih, iw), mode="bilinear", align_corners=False)
        return m.squeeze(0).squeeze(0)
    else:
        orig_dim = m.dim()
        if orig_dim == 3:
            m = m.unsqueeze(1)  # [B, 1, H, W]
        m = F.interpolate(m, size=(ih, iw), mode="bilinear", align_corners=False)
        if orig_dim == 3:
            m = m.squeeze(1)
        return m


def _apply_mask_to_image(image: Any, mask: Any) -> Any:
    """Zero out image pixels outside mask region.

    image: [B, H, W, C] float32
    mask:  [H, W] or [B, H, W] float32 [0,1]
    Returns: [B, H, W, C] float32
    """
    m = _resize_mask_to_image(mask, image)
    if m.dim() == 2:
        m = m.unsqueeze(0)     # [1, H, W]
    m = m.unsqueeze(-1)        # [B, H, W, 1]
    return image * m


def _masked_patch_pool(last_hidden_state: Any, mask: Any) -> dict[str, Any]:
    """Mask-weighted pooling of spatial patch features from last_hidden_state.

    last_hidden_state: [B, N_tokens, dim]
    mask: [H, W] or [B, H, W] float32 [0,1]

    CLS token detection (auto):
      ViT / OpenCLIP style — token 0 is CLS; spatial patches are tokens 1..N.
        Examples: ViT-L/14 (257 tokens → 256 patches = 16×16 ✓)
                  EVA02-L / ViT-H (257 tokens → 256 patches = 16×16 ✓)
      SigLIP / SigLIP2 style — no CLS token; ALL tokens are spatial patches.
        Example:  SigLIP2-SO400M (1024 tokens → 1024 patches = 32×32 ✓)

    Detection logic: try N-1 (CLS skip) first. If not a perfect square, try N.
    Falls back to an error record only if neither is a perfect square.

    Downsamples mask to the patch grid, weights each patch by mask coverage,
    pools to a single [dim] vector.
    """
    import torch
    import torch.nn.functional as F

    total_tokens = last_hidden_state.shape[1]

    # Try with CLS skip (ViT / OpenCLIP: token 0 is CLS, tokens 1..N are spatial)
    n_no_cls = total_tokens - 1
    gs_no_cls = int(n_no_cls ** 0.5)
    has_cls = gs_no_cls * gs_no_cls == n_no_cls

    if has_cls:
        patch_features = last_hidden_state[:, 1:, :]   # [B, N_patches, dim]
    else:
        # No CLS token (SigLIP / SigLIP2 style): all tokens are spatial patches
        n_all = total_tokens
        gs_all = int(n_all ** 0.5)
        if gs_all * gs_all != n_all:
            return {
                "status": "error",
                "detail": (
                    f"non-square patch grid: tried {n_no_cls} (with CLS skip) and "
                    f"{n_all} (without CLS skip) — neither is a perfect square"
                ),
            }
        patch_features = last_hidden_state[:, :, :]   # [B, N_patches, dim]

    B, N, dim = patch_features.shape
    grid_size = int(N ** 0.5)   # guaranteed perfect square at this point

    # Downsample mask to [grid_size, grid_size]
    m = mask.float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif m.dim() == 3:
        m = m.unsqueeze(1)               # [B, 1, H, W]

    m = F.interpolate(m, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    patch_weights = m.squeeze().flatten()   # [N_patches]
    if patch_weights.dim() == 0:
        patch_weights = patch_weights.unsqueeze(0)

    total_weight = float(patch_weights.sum())
    if total_weight < 1e-6:
        return {"status": "empty_mask", "covered_patches": 0, "total_patches": N}

    # Weighted pool [B, N, dim] → [B, dim]
    features = patch_features[0].to(patch_weights.device)  # [N, dim]
    pooled = (features * patch_weights.unsqueeze(-1)).sum(0) / total_weight  # [dim]

    return {
        "covered_patches": int((patch_weights > 0.1).sum()),
        "total_patches": N,
        "coverage_ratio": round(total_weight / N, 4),
        "pooled_norm": float(pooled.norm()),
        "pooled_mean": float(pooled.mean()),
        "has_cls_token": has_cls,
    }


# ---------------------------------------------------------------------------
# CLIP Vision slot: model identity + encode + per-mask analysis
# ---------------------------------------------------------------------------


def _extract_clip_vision_slot(
    cv_model: Any,
    image: Any,
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Hash CLIP_VISION model, run encode_image() globally and per mask.

    masks: {"face": tensor_or_None, "main_subject": tensor_or_None, "skin": tensor_or_None,
             "clothing": tensor_or_None, "hair": tensor_or_None}
    """
    out: dict[str, Any] = {}
    out["model"] = _try(lambda: _hash_model_weights(cv_model))

    # Global encode
    try:
        encoded = cv_model.encode_image(image)
    except Exception as exc:  # noqa: BLE001
        out["output"] = {"status": "error", "detail": str(exc)}
        return out

    out["output"] = _extract_clip_vision(encoded)

    # Per-mask patch pool: use spatial features from the ORIGINAL (global) encode.
    # No re-encode — masks are applied to stored global embeddings at comparison time.
    active_masks = {name: m for name, m in masks.items() if m is not None}
    if not active_masks:
        return out

    lhs = getattr(encoded, "last_hidden_state", None)
    patch_pools: dict[str, Any] = {}
    for mask_name, mask in active_masks.items():
        slot_mask: dict[str, Any] = {}
        if lhs is not None:
            slot_mask["patch_pool"] = _try(lambda m=mask, l=lhs: _masked_patch_pool(l, m))
        patch_pools[mask_name] = slot_mask

    out["patch_pools"] = patch_pools
    return out


# ---------------------------------------------------------------------------
# InsightFace face analysis
# ---------------------------------------------------------------------------


def _get_insightface_app(model_path: str) -> Any:
    """Load and cache a FaceAnalysis app for the given model root path.

    Two calling conventions are accepted:

    1. Path to the model pack directory (parent dir is named "models" AND the
       leaf itself does NOT contain a "models/" subdirectory):
         e.g. C:/path/to/insightface/models/antelopev2
              C:/path/to/insightface/models/buffalo_l
       model_name is inferred from the leaf; root is two levels up.

    2. Path to the InsightFace root (the directory that CONTAINS models/):
         e.g. C:/path/to/insightface
       Scans models/ for available packs; prefers antelopev2, then buffalo_l,
       then the first alphabetically sorted candidate.

    Any model pack is supported — the name is never hardcoded.
    """
    if model_path in _insightface_cache:
        return _insightface_cache[model_path]

    from insightface.app import FaceAnalysis

    p = Path(model_path)

    # Convention 1: path points directly at the model pack directory.
    # Guard: parent must be named "models" AND the leaf must not itself
    # contain a "models/" subdirectory (which would indicate this IS the root,
    # not a pack — e.g. .../models/insightface would incorrectly fire without
    # this guard).
    if p.parent.name == "models" and not (p / "models").is_dir():
        model_name = p.name
        root = str(p.parent.parent)
    else:
        # Convention 2: path is the InsightFace root.
        # Scan the models/ subdirectory and pick the best available pack.
        root = str(p)
        models_dir = p / "models"
        model_name = "buffalo_l"  # safe fallback if scan fails
        if models_dir.is_dir():
            candidates = sorted(d.name for d in models_dir.iterdir() if d.is_dir())
            if candidates:
                # Prefer antelopev2 (most capable), then buffalo_l, then first found.
                for preferred in ("antelopev2", "buffalo_l"):
                    if preferred in candidates:
                        model_name = preferred
                        break
                else:
                    model_name = candidates[0]

    app = FaceAnalysis(
        name=model_name,
        root=root,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _insightface_cache[model_path] = app
    return app


def _extract_insightface(image: Any, model_path: str) -> dict[str, Any]:
    """Run InsightFace face analysis on the decoded image.

    image: ComfyUI IMAGE [B, H, W, C] float32 [0,1]
    model_path: path to InsightFace model root (directory containing models/buffalo_l/)

    Returns detection metadata + ArcFace embedding hash/stats.
    Full 512-dim embedding is NOT stored here (V2 asset vault concern).
    """
    if not model_path or not model_path.strip():
        return {"status": "not_provided"}

    try:
        import numpy as np
        import blake3 as _blake3

        app = _get_insightface_app(model_path.strip())

        # ComfyUI IMAGE [B, H, W, C] float32 → [H, W, C] uint8 BGR
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]

        faces = app.get(img_np)
        if not faces:
            return {"status": "ok", "face_count": 0, "detail": "no face detected"}

        def _extract_single_face(face: Any) -> dict[str, Any]:
            """Extract all storable fields from one InsightFace Face object."""
            emb = face.embedding.astype(np.float32)
            h = _blake3.blake3()
            h.update(emb.tobytes())
            rec: dict[str, Any] = {
                "det_score":       float(face.det_score),
                "bbox":            [float(x) for x in face.bbox.tolist()],
                "embedding_hash":  h.hexdigest(),
                "embedding_norm":  float(np.linalg.norm(emb)),
                "embedding_mean":  float(emb.mean()),
                "embedding_array": emb,  # float32 [512] — written as .npy by _build_face_domain
            }
            # normed_embedding [512] — L2-normalized; preferred for cosine similarity
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                try:
                    ne = face.normed_embedding.astype(np.float32)
                    nh = _blake3.blake3()
                    nh.update(ne.tobytes())
                    rec["normed_embedding_hash"]  = nh.hexdigest()
                    rec["normed_embedding_array"] = ne  # float32 [512]
                except Exception:  # noqa: BLE001
                    pass
            # pose [3]: pitch, yaw, roll in degrees
            if hasattr(face, "pose") and face.pose is not None:
                try:
                    pose_arr = np.array(face.pose, dtype=np.float32).ravel()
                    rec["pose"] = [float(x) for x in pose_arr[:3].tolist()]
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(face, "age") and face.age is not None:
                rec["age"] = int(face.age)
            if hasattr(face, "gender") and face.gender is not None:
                rec["gender"] = int(face.gender)
            if hasattr(face, "sex") and face.sex is not None:
                rec["sex"] = str(face.sex)
            # kps [5, 2]: 5 facial keypoints (eyes×2, nose, mouth corners×2)
            if hasattr(face, "kps") and face.kps is not None:
                try:
                    kps_arr = np.array(face.kps, dtype=np.float32)
                    rec["kps"] = [[float(x), float(y)] for x, y in kps_arr.tolist()]
                except Exception:  # noqa: BLE001
                    pass
            return rec

        # Sort all detected faces by detection score (highest confidence first).
        sorted_faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
        result: dict[str, Any] = {
            "status":     "ok",
            "face_count": len(faces),
            "faces":      [_extract_single_face(f) for f in sorted_faces],
        }
        # Raw face inspection: all attributes of all Face objects (net-fishing).
        # _build_face_domain() reads only specific keys and ignores this.
        result["faces_raw"] = _try(lambda: _inspect_insightface_faces_raw(faces))
        return result

    except ImportError:
        return {"status": "not_available", "detail": "insightface package not installed"}
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or f"{type(exc).__name__} (no message)"
        return {"status": "error", "error_type": type(exc).__name__, "detail": detail}


# ---------------------------------------------------------------------------
# Raw input inspection  (net-fishing — lab discovery only)
# ---------------------------------------------------------------------------


def _inspect_latent_raw(latent: Any) -> dict[str, Any]:
    """Inspect all keys in a ComfyUI LATENT dict.

    A LATENT is a dict with at minimum a 'samples' key containing the latent
    tensor. This catalogs every key present and its metadata so we know the
    full structure — not just 'samples'.
    """
    if not isinstance(latent, dict):
        return {"error": f"unexpected type: {type(latent).__name__}"}

    all_keys = list(latent.keys())
    result: dict[str, Any] = {"all_keys": all_keys}

    for key in all_keys:
        val = latent[key]
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            try:
                entry: dict[str, Any] = {
                    "type": type(val).__name__,
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "device": str(val.device) if hasattr(val, "device") else None,
                }
                try:
                    f = val.float()
                    entry["min"] = float(f.min())
                    entry["max"] = float(f.max())
                    entry["mean"] = float(f.mean())
                except Exception:  # noqa: BLE001
                    pass
                result[key] = entry
            except Exception as exc:  # noqa: BLE001
                result[key] = {"type": type(val).__name__, "error": str(exc)}
        elif isinstance(val, (int, float, bool, str, type(None))):
            result[key] = {"type": type(val).__name__, "value": val}
        else:
            result[key] = {"type": type(val).__name__}

    result["has_noise_mask"] = "noise_mask" in all_keys
    result["has_batch_index"] = "batch_index" in all_keys
    return result


def _inspect_conditioning_raw(cond: Any) -> dict[str, Any]:
    """Inspect all keys in each conditioning pair's metadata dict.

    A CONDITIONING is List[(context_tensor, dict)]. Dumps ALL keys from each
    pair's dict — not just pooled_output and guidance which we currently use.
    """
    if cond is None:
        return {"status": "not_provided"}
    if not isinstance(cond, (list, tuple)):
        return {"error": f"unexpected type: {type(cond).__name__}"}

    result: dict[str, Any] = {"pair_count": len(cond)}
    pairs: list[dict[str, Any]] = []

    for i, pair in enumerate(cond):
        if not isinstance(pair, (list, tuple)) or len(pair) < 1:
            pairs.append({"index": i, "error": "malformed pair"})
            continue

        tensor = pair[0]
        meta = pair[1] if len(pair) > 1 else {}
        pair_info: dict[str, Any] = {"index": i}

        if hasattr(tensor, "shape"):
            try:
                pair_info["tensor"] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device) if hasattr(tensor, "device") else None,
                }
            except Exception as exc:  # noqa: BLE001
                pair_info["tensor"] = {"error": str(exc)}

        if isinstance(meta, dict):
            pair_info["meta_all_keys"] = list(meta.keys())
            meta_inspection: dict[str, Any] = {}
            for k, v in meta.items():
                if isinstance(v, (int, float, bool, str, type(None))):
                    meta_inspection[k] = {"type": type(v).__name__, "value": v}
                elif hasattr(v, "shape") and hasattr(v, "dtype"):
                    try:
                        meta_inspection[k] = {
                            "type": type(v).__name__,
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "device": str(v.device) if hasattr(v, "device") else None,
                        }
                    except Exception as exc:  # noqa: BLE001
                        meta_inspection[k] = {"type": type(v).__name__, "error": str(exc)}
                elif isinstance(v, (list, tuple)):
                    meta_inspection[k] = {"type": type(v).__name__, "len": len(v)}
                elif isinstance(v, dict):
                    meta_inspection[k] = {"type": "dict", "keys": list(v.keys())}
                else:
                    meta_inspection[k] = {"type": type(v).__name__}
            pair_info["meta_inspection"] = meta_inspection
        else:
            pair_info["meta_type"] = type(meta).__name__

        pairs.append(pair_info)

    result["pairs"] = pairs
    return result


def _inspect_vae_raw(vae: Any) -> dict[str, Any]:
    """Inspect VAE object: class hierarchy and common config attributes."""
    result: dict[str, Any] = {"class": type(vae).__name__}

    fsm = getattr(vae, "first_stage_model", None)
    if fsm is not None:
        result["first_stage_model_class"] = type(fsm).__name__
        enc = getattr(fsm, "encoder", None)
        dec = getattr(fsm, "decoder", None)
        if enc is not None:
            result["encoder_class"] = type(enc).__name__
        if dec is not None:
            result["decoder_class"] = type(dec).__name__
        cfg = getattr(fsm, "config", None) or getattr(fsm, "params", None)
        if cfg is not None and hasattr(cfg, "__dict__"):
            result["first_stage_model_config"] = {
                k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                for k, v in vars(cfg).items()
                if not k.startswith("_")
            }

    for attr in ("device", "dtype", "working_dtypes", "upscale_ratio",
                 "downscale_ratio", "latent_channels", "memory_used_mb",
                 "output_device", "offload_device"):
        val = getattr(vae, attr, None)
        if val is not None:
            result[attr] = str(val) if not isinstance(val, (int, float, bool)) else val

    return result


def _inspect_clip_vision_raw(cv: Any) -> dict[str, Any]:
    """Inspect a raw CLIP_VISION object BEFORE encoding — structure discovery only."""
    result: dict[str, Any] = {"class": type(cv).__name__}

    for attr in ("model", "inner_model", "patcher", "clip_model", "visual"):
        inner = getattr(cv, attr, None)
        if inner is not None:
            result[f"{attr}_class"] = type(inner).__name__

    for attr in ("load_device", "offload_device", "current_device"):
        val = getattr(cv, attr, None)
        if val is not None:
            result[attr] = str(val)

    inner_model = getattr(cv, "model", None) or getattr(cv, "inner_model", None)
    if inner_model is not None:
        for attr in ("vision_model", "visual_projection", "config", "image_size", "patch_size"):
            val = getattr(inner_model, attr, None)
            if val is not None:
                if isinstance(val, (str, int, float, bool)):
                    result[f"model.{attr}"] = val
                else:
                    result[f"model.{attr}_class"] = type(val).__name__

    for attr in ("image_size", "patch_size", "output_dim"):
        val = getattr(cv, attr, None)
        if val is not None:
            result[attr] = val if isinstance(val, (int, float, str)) else str(val)

    return result


def _inspect_mask_raw(mask: Any) -> dict[str, Any]:
    """Inspect a ComfyUI MASK tensor: shape, dtype, device, value properties."""
    if not hasattr(mask, "shape"):
        return {"error": f"unexpected type: {type(mask).__name__}"}

    result: dict[str, Any] = {
        "class": type(mask).__name__,
        "shape": list(mask.shape),
        "ndim": mask.dim() if hasattr(mask, "dim") else len(mask.shape),
        "dtype": str(mask.dtype),
        "device": str(mask.device) if hasattr(mask, "device") else None,
    }

    try:
        f = mask.float()
        result["min"] = float(f.min())
        result["max"] = float(f.max())
        result["mean"] = float(f.mean())
        result["coverage"] = float((f > 0.5).float().mean())
    except Exception:  # noqa: BLE001
        pass

    try:
        result["is_binary"] = bool(((mask == 0) | (mask == 1)).all())
    except Exception:  # noqa: BLE001
        pass

    total_pixels = 1
    for s in mask.shape:
        total_pixels *= s
    if total_pixels <= 256 * 256:
        try:
            result["unique_count"] = int(mask.unique().numel())
        except Exception:  # noqa: BLE001
            pass

    return result


def _inspect_insightface_faces_raw(faces: list) -> dict[str, Any]:
    """Inspect ALL attributes of InsightFace Face objects from app.get().

    Documents every attribute — not just what _extract_insightface() cherry-picks.
    Checks presence and shape/dtype of each known attribute so the FIELD_CATALOG
    can record what the model actually provides.
    """
    if not faces:
        return {"face_count": 0}

    result: dict[str, Any] = {"face_count": len(faces)}

    face = max(faces, key=lambda f: float(f.det_score) if hasattr(f, "det_score") else 0.0)
    face_inspection: dict[str, Any] = {"class": type(face).__name__}

    known_attrs = [
        "bbox",             # [x0, y0, x1, y1] — bounding box
        "kps",              # [5, 2] — 5 facial keypoints (eyes, nose, mouth corners)
        "det_score",        # float — detection confidence
        "landmark_3d_68",   # [68, 3] — 68-point 3D landmark
        "landmark_2d_106",  # [106, 2] — 106-point 2D landmark
        "pose",             # array or dict — head pose (pitch, yaw, roll)
        "age",              # float — estimated age
        "gender",           # int — 0=female 1=male
        "embedding",        # [512] — ArcFace embedding
        "normed_embedding", # [512] — L2-normalized embedding
        "sex",              # str — alternative gender representation in some models
        "face_token",       # str — face token in some models
    ]

    for attr in known_attrs:
        val = getattr(face, attr, None)
        if val is None:
            face_inspection[attr] = {"present": False}
        elif hasattr(val, "shape"):
            try:
                face_inspection[attr] = {
                    "present": True,
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "min": float(val.min()),
                    "max": float(val.max()),
                }
            except Exception:  # noqa: BLE001
                face_inspection[attr] = {"present": True, "shape": list(val.shape)}
        elif isinstance(val, (int, float, bool)):
            face_inspection[attr] = {"present": True, "value": val}
        elif isinstance(val, str):
            face_inspection[attr] = {"present": True, "value": val}
        elif isinstance(val, dict):
            face_inspection[attr] = {"present": True, "type": "dict", "keys": list(val.keys())}
        else:
            face_inspection[attr] = {"present": True, "type": type(val).__name__}

    try:
        extra_attrs = [k for k in vars(face) if k not in known_attrs and not k.startswith("_")]
        if extra_attrs:
            face_inspection["extra_attrs"] = extra_attrs
    except Exception:  # noqa: BLE001
        pass

    result["best_face"] = face_inspection

    all_faces: list[dict[str, Any]] = []
    for f in faces:
        entry: dict[str, Any] = {}
        try:
            entry["det_score"] = float(f.det_score)
        except Exception:  # noqa: BLE001
            pass
        try:
            entry["bbox"] = [float(x) for x in f.bbox.tolist()]
        except Exception:  # noqa: BLE001
            pass
        all_faces.append(entry)
    result["all_faces"] = all_faces

    return result


def _inspect_image_raw(image: Any) -> dict[str, Any]:
    """Inspect a decoded ComfyUI IMAGE tensor: shape, dtype, device, channel stats."""
    if not hasattr(image, "shape"):
        return {"error": f"unexpected type: {type(image).__name__}"}

    result: dict[str, Any] = {
        "class": type(image).__name__,
        "shape": list(image.shape),
        "ndim": image.dim() if hasattr(image, "dim") else len(image.shape),
        "dtype": str(image.dtype),
        "device": str(image.device) if hasattr(image, "device") else None,
    }

    try:
        result["memory_mb"] = round(image.numel() * image.element_size() / 1e6, 3)
    except Exception:  # noqa: BLE001
        pass

    try:
        f = image.float()
        result["min"] = float(f.min())
        result["max"] = float(f.max())
        result["mean"] = float(f.mean())
        if f.ndim == 4:
            result["channel_means"] = [float(f[..., c].mean()) for c in range(f.shape[3])]
    except Exception:  # noqa: BLE001
        pass

    return result


# ---------------------------------------------------------------------------
# Evidence domain builders  (probe → 3-object schema mapping)
# ---------------------------------------------------------------------------


def _build_masks_domain(
    masks: dict[str, Any],
    out_dir: Any,
    run_id: str,
) -> dict[str, Any]:
    """Write each connected mask as a ValueRef .npy file.

    Stores raw mask tensors only. Derived statistics (pixel_stats, edge_density)
    are NOT pre-baked — computed at comparison time from stored image + mask.
    This preserves mask utility and avoids domain contamination from image data.
    """
    import numpy as np
    result: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        if mask is None:
            continue
        mask_path: str | None = None
        mask_hash: str | None = None
        mask_shape: list | None = None
        try:
            arr = mask.cpu().float().numpy()
            mask_path = str(write_attachment(out_dir, run_id, f"mask_{mask_name}", arr))
            mask_hash = hash_bytes(arr.tobytes())
            mask_shape = list(arr.shape)
        except Exception:  # noqa: BLE001
            pass

        mask_vr = (
            _make_valueref("mask", mask_hash, mask_path, dtype="float32", shape=mask_shape)
            if mask_hash else _invalid("NULL", "LAB_WRITE_FAIL", "mask .npy write failed")
        )
        result[mask_name] = {"valueref": mask_vr}
    return result


def _build_face_domain(
    face_analysis: dict[str, Any],
    out_dir: Any,
    run_id: str,
) -> dict[str, Any]:
    """Map probe face_analysis to the schema 'face_analysis' measurement domain.

    Returns {face_count, faces: [...]}, where each face record contains
    embedding + normed_embedding ValueRefs, pose JSON scalars, and scalar metadata.
    Faces are sorted by det_score descending (highest confidence first).
    """
    status = face_analysis.get("status", "error")
    if status == "not_provided":
        return _invalid("NOT_APPLICABLE", "LAB_FACE_NOT_REQUESTED", "InsightFace not requested")
    if status in ("not_available", "error"):
        return _invalid("NULL", "LAB_FACE_UNAVAIL", face_analysis.get("detail", ""))

    import numpy as np

    face_count = face_analysis.get("face_count", 0)
    raw_faces  = face_analysis.get("faces", [])

    if face_count == 0 or not raw_faces:
        return {"face_count": 0, "faces": []}

    def _write_face(face_data: dict[str, Any], idx: int) -> dict[str, Any]:
        """Write embedding and normed_embedding .npy, assemble face record."""
        # --- embedding ValueRef ---
        emb_data = face_data.get("embedding_array")
        emb_hash = face_data.get("embedding_hash")
        emb_path: str | None = None
        emb_shape: list | None = None
        if emb_data is not None:
            try:
                arr = np.array(emb_data, dtype=np.float32)
                emb_path  = str(write_attachment(out_dir, run_id, f"face_{idx}_embedding", arr))
                emb_shape = list(arr.shape)
            except Exception:  # noqa: BLE001
                pass
        emb_vr = (
            _make_valueref("embedding", emb_hash, emb_path, dtype="float32", shape=emb_shape)
            if emb_hash else _invalid("NULL", "LAB_COMPUTE_FAIL", "embedding hash unavailable")
        )

        # --- normed_embedding ValueRef ---
        ne_data = face_data.get("normed_embedding_array")
        ne_hash = face_data.get("normed_embedding_hash")
        ne_path: str | None = None
        ne_shape: list | None = None
        if ne_data is not None:
            try:
                ne_arr   = np.array(ne_data, dtype=np.float32)
                ne_path  = str(write_attachment(out_dir, run_id, f"face_{idx}_normed_embedding", ne_arr))
                ne_shape = list(ne_arr.shape)
            except Exception:  # noqa: BLE001
                pass
        ne_vr = (
            _make_valueref("embedding", ne_hash, ne_path, dtype="float32", shape=ne_shape)
            if ne_hash else _invalid("NULL", "LAB_NO_DATA", "normed_embedding not in this pack")
        )

        return {
            "embedding":        {"valueref": emb_vr},
            "normed_embedding": {"valueref": ne_vr},
            "pose":             face_data.get("pose") or _invalid("NULL", "LAB_NO_DATA", "pose not available"),
            "kps":              face_data.get("kps") or _invalid("NULL", "LAB_NO_DATA", "kps not available"),
            "embedding_norm":   face_data.get("embedding_norm"),
            "det_score":        face_data.get("det_score"),
            "bbox":             face_data.get("bbox"),
            "age":              face_data.get("age"),
            "gender":           face_data.get("gender"),
            "sex":              face_data.get("sex"),
        }

    return {
        "face_count": face_count,
        "faces": [_write_face(f, i) for i, f in enumerate(raw_faces)],
    }


def _build_aux_domain(
    aux: dict[str, Any],
    aux_images: dict[str, Any],
    out_dir: Any,
    run_id: str,
) -> dict[str, Any]:
    """Map probe aux dict to the schema 'aux' measurement domain.

    Writes each aux image tensor as a .npy attachment so ValueRef paths are non-null.
    aux_images: raw tensor dict {name: tensor} collected alongside aux metadata.
    """
    import numpy as np

    result: dict[str, Any] = {}
    for aux_name, aux_data in aux.items():
        _ch = aux_data.get("content_hash", {})
        _ch_val = (
            _ch.get("value")
            if isinstance(_ch, dict) and _ch.get("status") == "ok"
            else None
        )
        _ps = aux_data.get("pixel_stats", {})
        _ps_val = (
            _ps.get("value")
            if isinstance(_ps, dict) and _ps.get("status") == "ok"
            else None
        )
        _shp = aux_data.get("shape", {})
        _shp_val = (
            _shp.get("value")
            if isinstance(_shp, dict) and _shp.get("status") == "ok"
            else None
        )
        # Write aux image tensor as .npy attachment.
        aux_path: str | None = None
        tensor = aux_images.get(aux_name)
        if tensor is not None and _ch_val:
            try:
                arr = tensor.cpu().float().numpy()
                aux_path = str(write_attachment(out_dir, run_id, f"aux_{aux_name}", arr))
            except Exception:  # noqa: BLE001
                pass
        aux_vr = (
            _make_valueref("aux", _ch_val, aux_path, shape=_shp_val)
            if _ch_val else _invalid("NULL", "LAB_COMPUTE_FAIL")
        )
        result[aux_name] = {
            "output":      {"valueref": aux_vr},
            "pixel_stats": _ps_val if _ps_val is not None else _invalid("NULL", "LAB_COMPUTE_FAIL"),
        }
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _build_report(
    run_id: str,
    data: dict[str, Any],
    dump_path: str,
) -> str:
    """Build a readable status report from the 3-object Evidence dump."""
    method  = data.get("method", {})
    eval_   = data.get("eval", {})
    sample  = data.get("sample", {})
    lora    = eval_.get("lora", {})
    is_baseline = lora.get("hash") is None

    lines: list[str] = ["=== LoRA Eval — Lab Probe ==="]
    lines.append(f"Run ID   : {run_id}")

    # Identity hashes
    mh = method.get("method_hash", "")
    eh = eval_.get("eval_hash", "")
    sh = sample.get("sample_hash", "")
    lines.append(f"Hashes   : method={mh[:16]}...  eval={eh[:16]}...  sample={sh[:16]}...")

    # Type
    if is_baseline:
        lines.append("Type     : BASELINE  (no LoRA)")
    else:
        lora_name_val = lora.get("name") or "?"
        lora_hash_val = lora.get("hash")
        lh_str = lora_hash_val[:16] if isinstance(lora_hash_val, str) else "?"
        lines.append(f"Type     : LoRA run  name={lora_name_val}  hash={lh_str}...")

    # Base model
    bm = method.get("base_model", {})
    bm_hash = bm.get("hash")
    bm_name = bm.get("name") or "?"
    if isinstance(bm_hash, str):
        lines.append(f"Model    : {bm_name}  hash={bm_hash[:16]}...")
    else:
        lines.append(f"Model    : {bm_name}  (hash unavailable — wire checkpoint_name)")

    # VAE
    vae = method.get("vae_model", {})
    vae_hash = vae.get("hash")
    vae_name = vae.get("name") or "?"
    if isinstance(vae_hash, str):
        lines.append(f"VAE      : {vae_name}  hash={vae_hash[:12]}...")

    # Latent
    lat = method.get("latent", {})
    lw = lat.get("width")
    lh = lat.get("height")
    if isinstance(lw, int) and isinstance(lh, int):
        lines.append(f"Latent   : {lw}×{lh} px  (latent_hash={str(sample.get('latent_hash', ''))[:16]}...)")

    # Image
    img_domain = sample.get("image", {})
    img_ps = img_domain.get("pixel_stats")
    if isinstance(img_ps, dict) and "mean" in img_ps:
        ih = str(sample.get("image_hash", ""))[:16]
        lines.append(
            f"Image    : mean={img_ps.get('mean', 0):.4f}"
            f"  min={img_ps.get('min', 0):.4f}  max={img_ps.get('max', 0):.4f}"
            f"  hash={ih}..."
        )

    # Conditioning
    cond = method.get("conditioning", {})
    pos_hash = cond.get("positive_hash")
    if isinstance(pos_hash, str):
        lines.append(f"Cond+    : hash={pos_hash[:16]}...")
    hint = cond.get("positive_text") or ""
    if hint:
        excerpt = (hint[:50] + "...") if len(hint) > 50 else hint
        lines.append(f"Hint     : \"{excerpt}\"")

    # CLIP Vision — keyed by model_hash[:16]; stored in sample.clip_vision
    cv = sample.get("clip_vision", {})
    if not cv:
        lines.append("CLIP Vision : none connected")
    else:
        for model_key, slot_data in cv.items():
            model_info = slot_data.get("model", {})
            if isinstance(model_info, dict) and model_info.get("status") == "ok":
                mv = model_info["value"]
                cls = mv.get("class", "?")
                aliased = slot_data.get("output", {}).get("is_aliased", "?")
                n_masks = len(slot_data.get("patch_pools", {}))
                lines.append(f"CLIP {model_key[:12]}...: {cls}  aliased={aliased}  masks={n_masks}")
            else:
                lines.append(f"CLIP {model_key[:12]}...: error")

    # Face analysis (schema domain in sample)
    face_domain = sample.get("face_analysis", {})
    if isinstance(face_domain, dict):
        if face_domain.get("status") == "NOT_APPLICABLE":
            lines.append("Face     : not requested")
        elif face_domain.get("status") == "NULL":
            lines.append(f"Face     : unavailable — {face_domain.get('reason_code', '?')}")
        else:
            fc = face_domain.get("face_count", 0)
            if fc == 0:
                lines.append("Face     : no face detected")
            else:
                faces_list = face_domain.get("faces", [])
                best = faces_list[0] if faces_list else {}
                det    = best.get("det_score", 0) or 0
                age    = best.get("age", "?")
                gender = "M" if best.get("gender") == 1 else (
                    "F" if best.get("gender") == 0 else "?")
                lines.append(f"Face     : {fc} face(s)  det={det:.2f}  age={age}  gender={gender}")

    # Masks domain
    masks_domain = sample.get("masks", {})
    if isinstance(masks_domain, dict):
        for mname, mdata in masks_domain.items():
            if not isinstance(mdata, dict):
                continue
            ps = mdata.get("pixel_stats")
            ed = mdata.get("edge_density")
            if isinstance(ps, dict) and "mean" in ps and isinstance(ed, (int, float)):
                lines.append(
                    f"Mask {mname:10s}: pixel_mean={ps['mean']:.4f}  edge_density={ed:.4f}"
                )
            elif ps is not None:
                lines.append(f"Mask {mname}: {ps}")

    # Aux domain
    aux_domain = sample.get("aux", {})
    if isinstance(aux_domain, dict):
        for aname, adata in aux_domain.items():
            ps = adata.get("pixel_stats") if isinstance(adata, dict) else None
            if isinstance(ps, dict) and "mean" in ps:
                m = ps["mean"]
                lines.append(
                    f"Aux {aname:10s}: pixel_mean={m:.4f}"
                    if isinstance(m, float) else f"Aux {aname}: ok"
                )

    # Luminance domain
    lum_domain = sample.get("luminance", {})
    if isinstance(lum_domain, dict):
        lum_stats = lum_domain.get("stats")
        if isinstance(lum_stats, dict) and "mean" in lum_stats:
            lum_vr = lum_domain.get("output", {}).get("valueref", {})
            lum_path = lum_vr.get("path") if isinstance(lum_vr, dict) else None
            lines.append(
                f"Luminance  : mean={lum_stats.get('mean', '?'):.4f}"
                f"  std={lum_stats.get('std', '?'):.4f}"
                f"  file={'ok' if lum_path and 'write_failed' not in lum_path else 'FAIL'}"
            )

    # Settings
    settings = method.get("settings", {})
    lines.append(
        f"Settings : seed={sample.get('seed', '?')}  steps={settings.get('steps', '?')}  "
        f"cfg={settings.get('cfg', '?')}  {settings.get('sampler', '?')}/"
        f"{settings.get('scheduler', '?')}  denoise={settings.get('denoise', '?')}"
    )
    lines.append(f"Dump     : {dump_path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------


class LabProbe:
    """On-site Lab Probe — v0.9.5.

    New in v0.9.5:
      CLIP EMBED FIX  — `out_dir` moved before ClipVision loop; was undefined causing
                        NameError caught silently → LAB_WRITE_FAIL on all
                        `global_embedding_valueref` fields.
      CLIP SPATIAL    — `last_hidden_state_valueref` added to each CLIP Vision slot.
                        SigLIP: [1, 1024, 1152] spatial patches for masked pooling.
                        ViT-L:  [1, 257, 1024] with CLS token (skip index 0 for spatial analysis).
      FACE ARRAY      — `face_analysis` now stores ALL detected faces (sorted by
                        det_score desc) instead of only the best face. `_build_face_domain`
                        returns {face_count, faces: [...]}.
      NORMED EMBED    — `normed_embedding` [512] added to each face record as ValueRef
                        .npy. Pre-normalized; preferred for cosine similarity.
      POSE SCALARS    — `pose` [pitch, yaw, roll] added as JSON scalars to each face record.
      AUX .NPY WRITE  — aux images now written as .npy attachments (path was null before).
                        `_build_aux_domain` now accepts raw tensors + out_dir + run_id.
      LATENT CHANNELS — `latent.channels` (samples.shape[1]) replaces `latent.type` (was
                        always null). Added to method_hash components. 16 = Flux/SD3,
                        4 = SD1.5/SDXL.

    New in v0.9.3:
      BASE MODEL HASH — base_model_hash now derived from the loaded model weights in memory
                        via _extract_model_pipeline().base_model_content_hash. No file path
                        or folder_paths lookup needed. Works for safetensors, GGUF, .pt, .ckpt.
                        Strategy: hash all param names+shapes+dtypes (structural) + first 256
                        raw-dtype bytes from first 3 + last 3 params (value anchor).
                        checkpoint_name input kept for display name only.
      CLIP EMBEDDINGS — mean-pooled CLIP embedding vectors now written as .npy assets.
                        For SigLIP (3D image_embeds): mean over seq_len dim → [B, 1152].
                        For ViT-L (2D image_embeds): stored directly → [B, 768].
                        global_embedding_valueref added to each clip_vision slot in the dump.
                        compare.clip_cosine_delta.v1 can now operate on lab dumps.

    New in v0.9.2:
      GGUF BASE HASH  — (superseded by v0.9.3) tried multiple folder_paths keys.
                        Kept _fast_file_hash() as utility; no longer used for base_model_hash.

    New in v0.9.1:
      MODEL PIPELINE  — _model_pipeline now written to dump top-level (_model_pipeline key).
                        base_model.arch populated from pipeline class/model_type/unet_config.
                        lora.hash uses in-memory tensor content_hash (≠ lora.file_hash).
      CONDITIONING    — positive_guidance / negative_guidance extracted from FLUX pair[1] dict.
      FACE EMBED ASSET— embedding_array stored in _extract_insightface() return; .npy written.
      PIXEL STATS STD — image.pixel_stats now includes std field.

    New in v0.9.0:
      EVIDENCE v2     — output restructured from flat dict to 3-object Evidence format
                        (method_record, eval_record, sample_record) matching evidence.schema.json.
                        Identity hashes (method_hash, eval_hash, sample_hash) now computed
                        and stored in the dump.
      NEW INPUTS      — checkpoint_name (STRING): base model filename for base_model_hash.
                        lora_strength (FLOAT): LoRA conditioning strength for sample_hash.
      IMAGE ASSET     — decoded image written as .npy attachment alongside luminance and masks.
      MASKS AS ASSETS — mask tensors written as .npy attachments with ValueRef pointers.

    New in v0.8.3:
      CLIP VISION     — clip_vision_3 slot removed. Two slots only: SigLIP (slot_1)
                        and ViT-L/H (slot_2). ViT-H and ViT-L do the same job at
                        different scales; one projected-embedding slot is sufficient.
      INSIGHTFACE     — path resolution fixed for StabilityMatrix layouts.
                        Now checks folder_paths.get_folder_names_and_paths("insightface")
                        first (respects extra_model_paths.yaml); falls back to
                        models_dir/insightface. Convention-1 guard added to
                        _get_insightface_app: no longer mis-fires when path ends in
                        .../models/insightface (root vs pack distinction).
                        Exception detail now includes error_type; no more empty detail.

    New in v0.8.2:
      PATCH POOL FIX  — _masked_patch_pool now auto-detects CLS token presence.
                        ViT / OpenCLIP (e.g. ViT-L/14, EVA02-L/ViT-H): 257 tokens
                        → skip token 0 (CLS) → 256 spatial patches (16×16 ✓).
                        SigLIP / SigLIP2 (e.g. SigLIP2-SO400M): 1024 tokens,
                        no CLS → use all 1024 spatial patches (32×32 ✓).
                        Previously all SigLIP patch pools returned a "non-square
                        patch grid: 1023 patches" error. Now produces valid results.
                        Return dict includes has_cls_token (bool) for traceability.

    New in v0.8.1:
      INSIGHTFACE     — replaced STRING path input with BOOLEAN toggle.
                        Path auto-resolved: models/insightface/ → prefers
                        antelopev2 > buffalo_l > first found. Default: True.
      AUX IMAGES × 4 — canny slot removed; lineart slot renamed to edge.
                        Final slots: depth | pose | normal | edge.
      TOOLTIPS        — all optional inputs now carry ComfyUI tooltip text
                        describing purpose and recommended preprocessors.

    New in v0.8.0:
      MASK × 5        — expanded from 3 to 5 named slots.
                        face | background | skin | clothing | hair.
                        Polarity: white = subject region, black = rest.
                        body slot removed; skin, clothing, hair added.
      AUX IMAGES × 5 — expanded from 3 to 5 named slots (later trimmed to 4 in v0.8.1).
      LUMINANCE MAP   — per-pixel BT.709 luminance computed internally from
                        decoded image. Written as .npy alongside JSON dump.
                        Stats (mean/std/min/max) stored inline in dump JSON.

    New in v0.7.2 (bug fixes):
      InsightFace model detection generalised — any pack auto-detected.
      state_dict_key_count unwrapped from _try() nesting.
      conditioning first_done guard moved before stats computation.
      _extract_image adds .contiguous() before .numpy().

    New in v0.7.1:
      ClipVision output keyed by model_hash[:16], not slot index.

    New in v0.7.0:
      MASK × 3, INSIGHTFACE, AUX IMAGES × 3.

    Outputs: decoded IMAGE + STRING report.
    Never crashes the workflow — all failures are recorded, not raised.
    """

    CATEGORY = "lora_eval/lab"
    FUNCTION = "probe"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        import comfy.samplers
        import folder_paths

        lora_files = folder_paths.get_filename_list("loras")
        lora_choices = [_NO_LORA_SENTINEL] + lora_files

        return {
            "required": {
                "model":        ("MODEL",),
                "latent":       ("LATENT",),
                "vae":          ("VAE",),
                "positive":     ("CONDITIONING",),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 10000}),
                "cfg":          ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "negative":         ("CONDITIONING",),
                "lora_name":        (lora_choices,),
                # Identity hash inputs — required for canonical method_hash / sample_hash.
                "checkpoint_name":  ("STRING", {"default": "", "multiline": False,
                                                "tooltip": "Base model filename (e.g. 'FLUX.1-dev.safetensors'). Used to compute base_model_hash via hash_safetensors_content. Leave blank to omit from method_hash (hash will still be valid, but base_model_hash = null)."}),
                "lora_strength":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                               "tooltip": "LoRA conditioning strength used in the sampler. Required for sample_hash. Set to 0.0 for baseline (overridden to null)."}),
                # CLIP Vision — model loaded by CLIPVisionLoader, encoded internally.
                # slot_1: semantic similarity  — SigLIP / SigLIP2 (no projection head)
                # slot_2: structural / spatial — DINOv2 (self-supervised, ViT-H scale,
                #          packaged as CLIPVision for IP-Adapter; 257 tokens, 1024-dim projection)
                "clip_vision_1": ("CLIP_VISION", {"tooltip": "Semantic similarity model. Recommended: SigLIP2-SO400M."}),
                "clip_vision_2": ("CLIP_VISION", {"tooltip": "Structural / spatial model. Recommended: DINOv2 (ViT-H, IP-Adapter package)."}),
                # Masks — white = subject region, black = everything else.
                "mask_face":         ("MASK", {"tooltip": "Face region (may include neck). White=face, Black=rest."}),
                "mask_main_subject": ("MASK", {"tooltip": "Main subject / foreground region (white = subject, black = background). Used for mask-based measurements and pose joint overlap."}),
                "mask_skin":         ("MASK", {"tooltip": "Exposed skin (may include face/neck). Excludes clothing and hair."}),
                "mask_clothing":     ("MASK", {"tooltip": "Clothing, fabrics, accessories (bags, jewellery). Excludes bare skin and hair."}),
                "mask_hair":         ("MASK", {"tooltip": "Hair only."}),
                # InsightFace — auto-resolves AntelopeV2 from ComfyUI models/insightface/.
                "load_insightface": ("BOOLEAN",   {"default": True, "tooltip": "Run InsightFace (AntelopeV2) for face detection and ArcFace identity embedding. Auto-resolved from ComfyUI models/insightface/."}),
                # Auxiliary preprocessor outputs — use consistent preprocessors across
                # baseline and LoRA runs so drift is attributable to the LoRA, not the tool.
                "aux_depth":   ("IMAGE",          {"tooltip": "Depth map. Use: DepthAnythingV2-Depth preprocessor."}),
                "aux_normal":  ("IMAGE",          {"tooltip": "Surface normal map. Use: DSINE-NormalMap preprocessor."}),
                "aux_edge":    ("IMAGE",          {"tooltip": "Edge / line art map. Use: LineArt-Edge preprocessor."}),
                # NOTE: aux_pose (raster pose image) removed. Use POSE_KEYPOINT instead.
                # Structured pose evidence — POSE_KEYPOINT outputs from preprocessor nodes.
                "pose_keypoint_openpose": (
                    "POSE_KEYPOINT",
                    {"tooltip": "OpenPose body-only POSE_KEYPOINT output. Connect OpenPosePreprocessor."},
                ),
                "pose_keypoint_dw": (
                    "POSE_KEYPOINT",
                    {"tooltip": "DW-Pose body-only POSE_KEYPOINT output. Connect DWPreprocessor."},
                ),
                "aux_densepose": (
                    "IMAGE",
                    {"tooltip": "DensePose support image for per-joint support facts (optional)."},
                ),
                # Display label — metadata only, not used for hashing
                "prompt_hint": ("STRING",         {"multiline": False, "default": "", "tooltip": "Optional display label. Not used for hashing or pairing — report display only."}),
            },
        }

    def probe(
        self,
        model: Any,
        latent: Any,
        vae: Any,
        positive: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        negative: Any = None,
        lora_name: str = _NO_LORA_SENTINEL,
        checkpoint_name: str = "",
        lora_strength: float = 1.0,
        clip_vision_1: Any = None,
        clip_vision_2: Any = None,
        mask_face: Any = None,
        mask_main_subject: Any = None,
        mask_skin: Any = None,
        mask_clothing: Any = None,
        mask_hair: Any = None,
        load_insightface: bool = True,
        aux_depth: Any = None,
        aux_normal: Any = None,
        aux_edge: Any = None,
        pose_keypoint_openpose: Any = None,
        pose_keypoint_dw: Any = None,
        aux_densepose: Any = None,
        prompt_hint: str = "",
    ) -> tuple[Any, str]:
        import torch

        run_id = str(uuid.uuid4())
        _timestamp = now_iso()

        try:
            is_baseline = not bool(model.patches)
        except Exception:  # noqa: BLE001
            is_baseline = False

        lora_name_stored = None if (not lora_name or lora_name == _NO_LORA_SENTINEL) else lora_name

        # --- Decode latent ---
        image = None
        decode_error = None
        try:
            image = vae.decode(latent["samples"])
        except Exception as exc:  # noqa: BLE001
            decode_error = str(exc)
            emit("WARN", "LAB.DECODE_FAILED", _WHERE,
                 "VAE decode failed; using fallback black image",
                 run_id=run_id, error=decode_error)
            image = torch.zeros([1, 8, 8, 3], dtype=torch.float32)

        # --- Masks dict ---
        masks = {
            "face":         mask_face,
            "main_subject": mask_main_subject,
            "skin":         mask_skin,
            "clothing":     mask_clothing,
            "hair":         mask_hair,
        }

        # --- Resolve output directory (needed for attachments and JSON dump) ---
        # Must be defined before the ClipVision loop which writes .npy embeddings.
        try:
            out_dir = get_path("lab_dumps_root")
        except Exception:  # noqa: BLE001
            out_dir = get_path("project_root") / "data" / "lab_dumps"

        # --- ClipVision — keyed by model_hash[:16], not slot index ---
        # Slot index is a ComfyUI wiring convenience and carries no semantic meaning.
        # Using model_hash as the key means "SigLIP in slot_1 baseline + SigLIP in slot_2
        # LoRA run" still produces matching key paths in field_catalog flat-diff.
        # Duplicate-model guard: if the same model is wired to two slots, keep first only.
        cv_by_model: dict[str, Any] = {}
        for cv_model in [clip_vision_1, clip_vision_2]:
            if cv_model is None:
                continue
            result = _extract_clip_vision_slot(cv_model, image, masks)
            model_hash = ""
            mi = result.get("model", {})
            if isinstance(mi, dict) and mi.get("status") == "ok":
                model_hash = mi["value"].get("model_hash", "")
            key = model_hash[:16] if len(model_hash) >= 16 else f"cv_{len(cv_by_model) + 1}"
            if key in cv_by_model:
                continue  # same model wired to two slots — keep first

            # Write mean-pooled embedding as .npy asset and attach ValueRef to the slot result.
            # The array is at result["output"]["image_embeds"]["value"]["mean_pooled_array"]
            # (set by _extract_clip_vision). Pop it before JSON serialisation; write via
            # write_attachment(). Works for both SigLIP (3D→pooled) and ViT-L (2D projected).
            _emb_arr = None
            try:
                _ie_wrap = result.get("output", {}).get("image_embeds", {})
                if isinstance(_ie_wrap, dict) and _ie_wrap.get("status") == "ok":
                    _emb_arr = _ie_wrap.get("value", {}).pop("mean_pooled_array", None)
            except Exception:  # noqa: BLE001
                pass

            if _emb_arr is not None:
                try:
                    _emb_npy_path = str(write_attachment(
                        out_dir, run_id, f"clip_vision_{key[:8]}_embedding", _emb_arr
                    ))
                    _emb_hash = hash_bytes(_emb_arr.tobytes())
                    result["global_embedding_valueref"] = _make_valueref(
                        "clip_vision_embedding",
                        _emb_hash,
                        _emb_npy_path,
                        dtype="float32",
                        shape=list(_emb_arr.shape),
                    )
                except Exception:  # noqa: BLE001
                    result["global_embedding_valueref"] = _invalid(
                        "NULL", "LAB_WRITE_FAIL", "clip embedding .npy write failed"
                    )
            else:
                result["global_embedding_valueref"] = _invalid(
                    "NULL", "LAB_COMPUTE_FAIL", "mean_pooled_array not produced"
                )

            # Write last_hidden_state spatial features as a separate ValueRef.
            # SigLIP: [1, 1024, 1152] — 1024 spatial patches for mask-weighted pooling.
            # ViT-L:  [1, 257, 1024] — skip token 0 (CLS) for spatial analysis.
            _lhs_arr = None
            try:
                _lhs_wrap = result.get("output", {}).get("last_hidden_state", {})
                if isinstance(_lhs_wrap, dict) and _lhs_wrap.get("status") == "ok":
                    _lhs_arr = _lhs_wrap.get("value", {}).pop("last_hidden_state_array", None)
            except Exception:  # noqa: BLE001
                pass

            if _lhs_arr is not None:
                try:
                    _lhs_npy_path = str(write_attachment(
                        out_dir, run_id, f"clip_vision_{key[:8]}_spatial", _lhs_arr
                    ))
                    _lhs_hash = hash_bytes(_lhs_arr.tobytes())
                    result["last_hidden_state_valueref"] = _make_valueref(
                        "clip_vision_spatial",
                        _lhs_hash,
                        _lhs_npy_path,
                        dtype="float32",
                        shape=list(_lhs_arr.shape),
                    )
                except Exception:  # noqa: BLE001
                    result["last_hidden_state_valueref"] = _invalid(
                        "NULL", "LAB_WRITE_FAIL", "last_hidden_state .npy write failed"
                    )
            else:
                result["last_hidden_state_valueref"] = _invalid(
                    "NULL", "LAB_COMPUTE_FAIL", "last_hidden_state_array not produced"
                )

            cv_by_model[key] = result

        # --- InsightFace ---
        # Auto-resolve the InsightFace root from ComfyUI's registered folder paths.
        # Resolution order:
        #   1. folder_paths.get_folder_names_and_paths("insightface") — registered by
        #      StabilityMatrix / extra_model_paths.yaml; most reliable.
        #   2. folder_paths.models_dir / "insightface" — standard ComfyUI layout fallback.
        # _get_insightface_app() then scans models/ for the best available pack
        # (antelopev2 > buffalo_l > first found).
        if load_insightface:
            try:
                import folder_paths as _fp
                _insightface_root = ""
                # Try registered paths first (StabilityMatrix extra_model_paths.yaml).
                try:
                    _if_entry = _fp.get_folder_names_and_paths("insightface")
                    if _if_entry and _if_entry[0]:
                        _insightface_root = str(Path(_if_entry[0][0]))
                except Exception:  # noqa: BLE001
                    pass
                # Fall back to standard models_dir / insightface.
                if not _insightface_root:
                    _insightface_root = str(Path(_fp.models_dir) / "insightface")
            except Exception:  # noqa: BLE001
                _insightface_root = ""
        else:
            _insightface_root = ""
        face_analysis = _extract_insightface(image, _insightface_root)

        # --- Auxiliary images ---
        aux: dict[str, Any] = {}
        aux_images: dict[str, Any] = {}  # raw tensors for .npy write in _build_aux_domain
        for aux_name, aux_image in [
            ("depth",  aux_depth),
            ("normal", aux_normal),
            ("edge",   aux_edge),
            # "pose" raster aux removed — use POSE_KEYPOINT inputs instead
        ]:
            if aux_image is not None:
                aux[aux_name] = _extract_image(aux_image)
                aux_images[aux_name] = aux_image

        # --- Raw POSE_KEYPOINT capture (discovery only) ---
        # Lab records the raw POSE_KEYPOINT payloads for field discovery.
        # Structured pose_evidence building (build_pose_evidence) is the Extractor's job;
        # lab/ must not import from extractor/ (pipeline boundary law).
        _raw_pose_keypoints: dict[str, Any] = {}
        if pose_keypoint_openpose is not None:
            _raw_pose_keypoints["openpose"] = _try(lambda: pose_keypoint_openpose)
        if pose_keypoint_dw is not None:
            _raw_pose_keypoints["dw_pose"] = _try(lambda: pose_keypoint_dw)

        # --- Luminance map ---
        # Compute per-pixel BT.709 luminance from decoded image; write as .npy attachment.
        # Enables masked luminance analysis at comparison time without reloading the full image.
        luminance_info: dict[str, Any] = {"status": "not_computed"}
        if decode_error is None:
            try:
                import blake3 as _blake3
                lum_array, lum_stats = _compute_luminance_map(image)
                lum_hash = _blake3.blake3(lum_array.tobytes()).hexdigest()
                try:
                    lum_path = write_attachment(out_dir, run_id, "luminance", lum_array)
                    lum_file = str(lum_path)
                except Exception as exc:  # noqa: BLE001
                    lum_file = f"write_failed: {exc}"
                luminance_info = {
                    "content_hash": lum_hash,
                    "file_path": lum_file,
                    "stats": lum_stats,
                }
            except Exception as exc:  # noqa: BLE001
                luminance_info = {"status": "error", "detail": str(exc)}

        # --- Intermediate extractions (needed for hash computation) ---
        _lat_raw   = _extract_latent(latent)
        _img_raw   = (
            _extract_image(image)
            if decode_error is None
            else {"status": "error", "detail": f"VAE decode failed: {decode_error}"}
        )
        _vae_raw   = _extract_vae_model(vae)
        _cond_pos  = _extract_conditioning(positive)
        _cond_neg  = _extract_conditioning(negative)
        _pipeline  = _extract_model_pipeline(model)

        # --- LoRA metadata (informational only) ---
        lora_metadata: dict[str, Any] = {"status": "not_provided"}
        if lora_name_stored:
            lora_metadata = _read_lora_metadata(lora_name_stored)

        # --- Write image as .npy attachment (for ValueRef) ---
        _image_npy_path: str | None = None
        if decode_error is None:
            try:
                import numpy as _np
                _img_arr = image.cpu().float().numpy()
                _image_npy_path = str(write_attachment(out_dir, run_id, "image", _img_arr))
            except Exception:  # noqa: BLE001
                pass

        # --- Identity hash computation ---
        # base_model_hash: derived from the loaded model weights in memory — no file path needed.
        # _extract_model_pipeline() hashes all parameter names+shapes+dtypes (structural) plus
        # the first 256 raw-dtype bytes of the first 3 + last 3 parameters (value anchor).
        # Works for safetensors, GGUF, .pt, .ckpt — whatever ComfyUI loaded.
        # checkpoint_name (if provided) is kept purely as a display name.
        _bm_ch_wrap = _pipeline.get("base_model_content_hash", {})
        _base_model_hash: str | None = (
            _bm_ch_wrap.get("value", {}).get("content_hash")
            if isinstance(_bm_ch_wrap, dict) and _bm_ch_wrap.get("status") == "ok"
            else None
        )
        _base_model_name: str = checkpoint_name.strip() if checkpoint_name else ""

        # vae_model_hash: from _hash_model_weights(vae) result stored in _vae_raw.
        _vae_model_hash: str | None = (
            _vae_raw.get("value", {}).get("model_hash")
            if isinstance(_vae_raw, dict) and _vae_raw.get("status") == "ok"
            else None
        )

        # conditioning hashes: _extract_conditioning() stores hash directly (not in _try wrapper).
        _pos_cond_hash: str | None = _cond_pos.get("hash") if isinstance(_cond_pos, dict) else None
        _neg_cond_hash: str | None = _cond_neg.get("hash") if isinstance(_cond_neg, dict) else None

        # latent pixel dimensions: latent tensor shape is [batch, ch, h, w] in latent space;
        # pixel dims = h*8, w*8 (standard 8× VAE downscale factor).
        _lat_shape_wrap = _lat_raw.get("shape", {}) if isinstance(_lat_raw, dict) else {}
        _lat_shape_val  = (
            _lat_shape_wrap.get("value")
            if isinstance(_lat_shape_wrap, dict) and _lat_shape_wrap.get("status") == "ok"
            else None
        )
        _latent_px_w: int | None = int(_lat_shape_val[3]) * 8 if _lat_shape_val and len(_lat_shape_val) >= 4 else None
        _latent_px_h: int | None = int(_lat_shape_val[2]) * 8 if _lat_shape_val and len(_lat_shape_val) >= 4 else None

        # latent_hash and image_hash: vital integrity fields from content_hash _try() wrappers.
        _lat_ch_wrap  = _lat_raw.get("content_hash", {}) if isinstance(_lat_raw, dict) else {}
        _latent_hash: str | None = (
            _lat_ch_wrap.get("value")
            if isinstance(_lat_ch_wrap, dict) and _lat_ch_wrap.get("status") == "ok"
            else None
        )
        _img_ch_wrap  = _img_raw.get("content_hash", {}) if isinstance(_img_raw, dict) else {}
        _image_hash: str | None = (
            _img_ch_wrap.get("value")
            if isinstance(_img_ch_wrap, dict) and _img_ch_wrap.get("status") == "ok"
            else None
        )

        # lora_hash (file): hash_safetensors_content of the LoRA file bytes; null for baseline.
        _lora_content_hash: str | None = None
        _lora_file_hash:    str | None = None
        if not is_baseline and lora_name_stored:
            try:
                import folder_paths as _fp_lora
                _lpath = _fp_lora.get_full_path("loras", lora_name_stored)
                if _lpath:
                    _lora_content_hash = hash_safetensors_content(_lpath)
                    _lora_file_hash    = hash_bytes(open(_lpath, "rb").read())  # noqa: SIM115
            except Exception:  # noqa: BLE001
                pass  # falls through to Invalid wrapper

        # lora_hash (canonical): in-memory tensor hash from _pipeline.lora_patches.
        # This captures the loaded float32 patch tensors, not the raw file bytes —
        # different from _lora_content_hash. Preferred for eval identity (tensor-stable).
        _lora_patch_info: dict = {}
        if (
            isinstance(_pipeline.get("lora_patches"), dict)
            and _pipeline["lora_patches"].get("status") == "ok"
        ):
            _lora_patch_info = _pipeline["lora_patches"].get("value", {}) or {}
        _lora_patch_hash: str | None = _lora_patch_info.get("content_hash") or None

        # lora_strength_val: null for baseline.
        _lora_strength_val: float | None = None if is_baseline else lora_strength

        # Compute the three canonical identity hashes.
        _pos_guidance: float | None = _cond_pos.get("guidance")
        _neg_guidance: float | None = _cond_neg.get("guidance")
        _method_hash_components: dict[str, Any] = {
            "base_model_hash":                _base_model_hash,
            "model_extras":                   None,  # Lab probe does not extract model_extras; V2 extractor handles this. Null hashes correctly for standard configs.
            "positive_conditioning_hash":     _pos_cond_hash,
            "positive_conditioning_guidance": _pos_guidance,
            "negative_conditioning_hash":     _neg_cond_hash,
            "negative_conditioning_guidance": _neg_guidance,
            "steps":                          steps,
            "denoise":                        denoise,
            "sampler":                        sampler_name,
            "scheduler":                      scheduler,
            "cfg":                            cfg,
            "latent_width":  _latent_px_w,
            "latent_height": _latent_px_h,
            "latent_shape":  list(_lat_shape_val) if _lat_shape_val and len(_lat_shape_val) >= 4 else None,
            "vae_model_hash": _vae_model_hash,
        }
        _the_method_hash = _compute_method_hash(_method_hash_components)
        # Canonical eval identity uses the in-memory lora patch hash (float32 tensors).
        # Falls back to file-bytes hash if pipeline patch hash unavailable.
        _eval_lora_hash  = _lora_patch_hash or _lora_content_hash
        _the_eval_hash   = _compute_eval_hash(_the_method_hash, _eval_lora_hash)
        _the_sample_hash = _compute_sample_hash(_the_eval_hash, seed, _lora_strength_val)

        # --- Build ValueRefs for image and luminance ---
        _img_shape_wrap = _img_raw.get("shape", {}) if isinstance(_img_raw, dict) else {}
        _img_shape_val  = (
            _img_shape_wrap.get("value")
            if isinstance(_img_shape_wrap, dict) and _img_shape_wrap.get("status") == "ok"
            else None
        )
        _img_ps_wrap   = _img_raw.get("pixel_stats", {}) if isinstance(_img_raw, dict) else {}
        _img_ps_val    = (
            _img_ps_wrap.get("value")
            if isinstance(_img_ps_wrap, dict) and _img_ps_wrap.get("status") == "ok"
            else None
        )
        _img_vr = (
            _make_valueref("image", _image_hash, _image_npy_path,
                           dtype="float32", shape=_img_shape_val)
            if _image_hash
            else _invalid("NULL", "LAB_DECODE_FAIL", "image decode failed or hash unavailable")
        )

        _lum_is_str = isinstance(luminance_info.get("content_hash"), str)
        _lum_vr = (
            _make_valueref("luminance", luminance_info["content_hash"],
                           luminance_info.get("file_path"),
                           dtype="float32")
            if _lum_is_str
            else _invalid("NULL", "LAB_COMPUTE_FAIL", "luminance not computed")
        )

        # --- Assemble 3-object Evidence structure ---
        _vae_val = _vae_raw.get("value", {}) if isinstance(_vae_raw, dict) and _vae_raw.get("status") == "ok" else {}

        method_record: dict[str, Any] = {
            "method_hash":       _the_method_hash,
            "extractor_version": f"lab-probe-{_PROBE_VERSION}",
            "timestamp":         _timestamp,
            "is_dirty":          False,
            "base_model": {
                "hash": _base_model_hash or _invalid(
                    "NULL", "LAB_NO_CKPT",
                    "checkpoint_name not provided or file not found"),
                "name": _base_model_name or None,
                # arch: informational — Bouncer will move to extras (not a schema field).
                "arch": (
                    _pipeline["base_model"].get("value", {})
                    if isinstance(_pipeline.get("base_model"), dict)
                       and _pipeline["base_model"].get("status") == "ok"
                    else None
                ),
            },
            "vae_model": {
                "hash": _vae_model_hash or _invalid("NULL", "LAB_HASH_FAIL",
                                                     "VAE weight hash unavailable"),
                "name": _vae_val.get("class"),
            },
            "conditioning": {
                "positive_hash":       _pos_cond_hash or _invalid("NULL", "LAB_HASH_FAIL"),
                "positive_text":       prompt_hint or None,
                "positive_guidance":   _pos_guidance,
                "negative_hash":       _neg_cond_hash or _invalid("NULL", "LAB_HASH_FAIL"),
                "negative_text":       None,
                "negative_guidance":   _neg_guidance,
            },
            "settings": {
                "steps":     steps,
                "denoise":   denoise,
                "sampler":   sampler_name,
                "scheduler": scheduler,
                "cfg":       cfg,
            },
            "latent": {
                "width":    _latent_px_w if _latent_px_w is not None
                            else _invalid("NULL", "LAB_SHAPE_FAIL", "latent shape unavailable"),
                "height":   _latent_px_h if _latent_px_h is not None
                            else _invalid("NULL", "LAB_SHAPE_FAIL", "latent shape unavailable"),
                "shape": list(_lat_shape_val) if _lat_shape_val and len(_lat_shape_val) >= 4 else None,
            },
            "model_extras": None,
        }

        eval_record: dict[str, Any] = {
            "eval_hash":   _the_eval_hash,
            "method_hash": _the_method_hash,
            "timestamp":   _timestamp,
            "is_dirty":    False,
            "lora": {
                # hash: in-memory tensor hash (canonical identity); falls back to file hash.
                "hash": (
                    None if is_baseline
                    else (
                        _lora_patch_hash or _lora_content_hash
                        or _invalid("NULL", "LAB_HASH_FAIL", "LoRA hash unavailable")
                    )
                ),
                # file_hash: raw file bytes hash for provenance / integrity cross-check.
                "file_hash": _lora_content_hash if not is_baseline else None,
                "name":      lora_name_stored if not is_baseline else None,
            },
        }

        sample_record: dict[str, Any] = {
            "sample_hash":       _the_sample_hash,
            "eval_hash":         _the_eval_hash,
            "seed":              seed,
            "lora_strength":     _lora_strength_val,
            "latent_hash":       _latent_hash or _invalid("ERROR", "LAB_HASH_FAIL",
                                                           "latent content_hash unavailable"),
            "image_hash":        _image_hash  or _invalid("ERROR", "LAB_HASH_FAIL",
                                                           "image content_hash unavailable"),
            "evidence_version":  _EVIDENCE_VERSION,
            "extractor_version": f"lab-probe-{_PROBE_VERSION}",
            "timestamp":         _timestamp,
            "ingest_status":     "OK",   # Bouncer will override on actual ingest
            "is_dirty":          False,
            # Measurement domains:
            "image": {
                "output":      {"valueref": _img_vr},
                "pixel_stats": _img_ps_val if _img_ps_val is not None
                               else _invalid("NULL", "LAB_DECODE_FAIL"),
            },
            "luminance": {
                "output": {"valueref": _lum_vr},
                "stats":  luminance_info.get("stats") or _invalid("NULL", "LAB_COMPUTE_FAIL"),
            },
            "clip_vision":   cv_by_model,
            "masks":         _build_masks_domain(masks, out_dir, run_id),
            "face_analysis": _build_face_domain(face_analysis, out_dir, run_id),
            "aux":           _build_aux_domain(aux, aux_images, out_dir, run_id),
            "diagnostics":   [],
        }

        # --- Raw input inspection (net-fishing — lab discovery, not Evidence schema) ---
        # Captures ALL keys/attributes from every input type so we know the full
        # structure available from ComfyUI at runtime. Analogous to _model_pipeline.
        _faces_raw_wrap = face_analysis.get("faces_raw") if isinstance(face_analysis, dict) else None

        data: dict[str, Any] = {
            "dump_format":     "evidence_v2",
            "probe_version":   _PROBE_VERSION,
            "_model_pipeline": _pipeline,   # informational; not part of Evidence schema
            "_raw_inputs": {
                "latent":        _try(lambda: _inspect_latent_raw(latent)),
                "vae":           _try(lambda: _inspect_vae_raw(vae)),
                "positive":      _try(lambda: _inspect_conditioning_raw(positive)),
                "negative":      (
                    _try(lambda: _inspect_conditioning_raw(negative))
                    if negative is not None else {"status": "not_provided"}
                ),
                "clip_vision_1": (
                    _try(lambda: _inspect_clip_vision_raw(clip_vision_1))
                    if clip_vision_1 is not None else {"status": "not_provided"}
                ),
                "clip_vision_2": (
                    _try(lambda: _inspect_clip_vision_raw(clip_vision_2))
                    if clip_vision_2 is not None else {"status": "not_provided"}
                ),
                "masks": {
                    "face":       (_try(lambda: _inspect_mask_raw(mask_face))
                                   if mask_face is not None else {"status": "not_provided"}),
                    "main_subject": (_try(lambda: _inspect_mask_raw(mask_main_subject))
                                    if mask_main_subject is not None else {"status": "not_provided"}),
                    "skin":       (_try(lambda: _inspect_mask_raw(mask_skin))
                                   if mask_skin is not None else {"status": "not_provided"}),
                    "clothing":   (_try(lambda: _inspect_mask_raw(mask_clothing))
                                   if mask_clothing is not None else {"status": "not_provided"}),
                    "hair":       (_try(lambda: _inspect_mask_raw(mask_hair))
                                   if mask_hair is not None else {"status": "not_provided"}),
                },
                "insightface_faces_raw": (
                    _faces_raw_wrap
                    if _faces_raw_wrap is not None
                    else ({"status": "not_requested"} if not load_insightface else {"status": "not_computed"})
                ),
                "image_decoded": (
                    _try(lambda: _inspect_image_raw(image))
                    if decode_error is None else {"status": "decode_failed"}
                ),
                # Raw POSE_KEYPOINT discovery data (net-fishing: structure, field names, payload shape)
                "pose_keypoints": _raw_pose_keypoints if _raw_pose_keypoints else {"status": "not_provided"},
            },
            "method":          method_record,
            "eval":            eval_record,
            "sample":          sample_record,
        }

        dump_path = str(out_dir)
        try:
            written = write_dump(out_dir, run_id, data)
            dump_path = str(written)
        except Exception as exc:  # noqa: BLE001
            emit("ERROR", "LAB.PROBE_ERROR", _WHERE,
                 "Probe dump write failed; workflow continues",
                 run_id=run_id, error=str(exc))

        report = _build_report(
            run_id=run_id,
            data=data,
            dump_path=dump_path,
        )
        return (image, report)
