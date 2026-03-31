"""
extractor/primitives.py — conversion functions for ComfyUI → portable primitives.

Converts ComfyUI types to plain Python / numpy:
  - Model objects     → BLAKE3 content hashes (structural + value anchor strategy)
  - Tensors           → numpy bytes for asset storage
  - LATENT/IMAGE      → vital integrity hashes
  - CONDITIONING      → canonical BLAKE3 hash + guidance float
  - LoRA patches      → canonical in-memory hash
  - CLIP_VISION       → numpy arrays for asset storage
  - InsightFace       → face analysis dict with numpy embedding arrays

ComfyUI dependency: YES — all ComfyUI-specific imports are inside function bodies
so this module imports cleanly without ComfyUI installed.

None-returning convention: all hash functions return None on failure (never raise).
Conversion functions (tensor_to_npy_bytes, compute_luminance, PNG encoders)
raise RuntimeError on failure — callers in sources.py catch and wrap as Invalid.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from contracts.validation_errors import make_null
from core import diagnostics
from core.asset_codecs import (
    embedding_to_fp16npy_bytes,
    ndarray_to_png16_bytes,
    ndarray_to_png8_bytes,
)
from core.hashing import hash_bytes, hash_tensor

_WHERE = "extractor.primitives"


# ---------------------------------------------------------------------------
# Base model hash
# ---------------------------------------------------------------------------


def hash_base_model(model_patcher: Any) -> str | None:
    """Hash base model weights from a live ModelPatcher for stable identity.

    Strategy (identical to probe._compute_base_model_content_hash):
      Phase 1 — Structural fingerprint: hash all param names + shapes + dtypes.
                 Instant (no data read) and unique to architecture variant.
      Phase 2 — Value anchor: first 256 elements (raw dtype bytes) of the first
                 3 + last 3 parameters sorted by name. Distinguishes trained weights
                 at the same architecture.

    Identifies the inner diffusion model via:
      diffusion_model (ComfyUI standard) → model → self.

    Returns:
        BLAKE3 hex digest on success, None on failure.
    """
    try:
        import numpy as np
        import blake3 as _blake3

        m = model_patcher.model
        inner = (
            getattr(m, "diffusion_model", None)
            or getattr(m, "model", None)
            or m
        )
        if not hasattr(inner, "named_parameters"):
            return None

        h = _blake3.blake3()
        params = sorted(inner.named_parameters())
        param_count = len(params)

        # Phase 1: structural fingerprint — names + shapes + dtypes (no data access)
        for name, param in params:
            h.update(f"{name}|{list(param.shape)}|{param.dtype}\n".encode())

        # Phase 2: value anchor — first 256 elements of first 3 + last 3 params
        anchor = (params[:3] + params[-3:]) if param_count > 6 else params
        for _name, param in anchor:
            try:
                arr = param.data.cpu().contiguous().numpy()
                flat = arr.ravel()
                n = min(256, flat.size)
                raw = flat[:n].view(np.uint8).tobytes()
                h.update(raw)
            except Exception:  # noqa: BLE001
                pass  # GGUF / quantized tensors may not support numpy(); skip gracefully

        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# VAE model hash
# ---------------------------------------------------------------------------


def hash_vae_model(vae: Any) -> str | None:
    """Hash VAE model weights from a live VAE object.

    Unwrap priority: first_stage_model → model → self.
    Hashes all named_parameters bytes (same as probe._hash_model_weights).

    Returns:
        BLAKE3 hex digest on success, None on failure.
    """
    try:
        import blake3 as _blake3

        inner = (
            getattr(vae, "first_stage_model", None)
            or getattr(vae, "model", None)
            or vae
        )
        if not hasattr(inner, "named_parameters"):
            return None

        h = _blake3.blake3()
        for name, param in sorted(inner.named_parameters()):
            h.update(name.encode("utf-8"))
            h.update(param.data.cpu().float().contiguous().numpy().tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Conditioning hash
# ---------------------------------------------------------------------------


def hash_conditioning(cond: Any) -> tuple[str | None, float | None]:
    """Hash a ComfyUI CONDITIONING = List[(context_tensor, dict)].

    Hashes context_tensor bytes + pooled_output bytes for all pairs (blake3).
    Extracts guidance float from the first pair's metadata dict.

    Returns:
        (content_hash, guidance_float) — either may be None on failure.
    """
    if not isinstance(cond, (list, tuple)) or not cond:
        return None, None

    try:
        import blake3 as _blake3

        h = _blake3.blake3()
        guidance: float | None = None

        for pair in cond:
            if not isinstance(pair, (list, tuple)) or len(pair) < 1:
                continue
            t = pair[0]
            if hasattr(t, "cpu"):
                try:
                    h.update(t.cpu().float().contiguous().numpy().tobytes())
                except Exception:  # noqa: BLE001
                    pass

            if len(pair) > 1 and isinstance(pair[1], dict):
                pooled = pair[1].get("pooled_output")
                if pooled is not None and hasattr(pooled, "cpu"):
                    try:
                        h.update(pooled.cpu().float().contiguous().numpy().tobytes())
                    except Exception:  # noqa: BLE001
                        pass

                if guidance is None:
                    raw_guidance = pair[1].get("guidance")
                    if raw_guidance is not None:
                        try:
                            guidance = float(raw_guidance)
                        except (TypeError, ValueError):
                            pass

        return h.hexdigest(), guidance
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# Latent hash
# ---------------------------------------------------------------------------


def hash_latent(latent_dict: Any) -> tuple[str | None, list | None]:
    """Hash a ComfyUI LATENT dict['samples'] tensor.

    Uses core.hashing.hash_tensor(dtype, shape, raw_bytes).

    Returns:
        (content_hash, shape_as_list) — shape is [B, C, H_latent, W_latent].
        Either may be None on failure.
    """
    try:
        t = latent_dict.get("samples") if isinstance(latent_dict, dict) else latent_dict
        if t is None:
            return None, None
        content_hash = hash_tensor(
            str(t.dtype),
            tuple(t.shape),
            t.cpu().float().contiguous().numpy().tobytes(),
        )
        return content_hash, list(t.shape)
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# Image hash
# ---------------------------------------------------------------------------


def hash_image(image_tensor: Any) -> str | None:
    """Hash a decoded ComfyUI IMAGE tensor [B, H, W, C] float32.

    Uses core.hashing.hash_tensor(dtype, shape, raw_bytes).

    Returns:
        BLAKE3 hex digest on success, None on failure.
    """
    try:
        return hash_tensor(
            str(image_tensor.dtype),
            tuple(image_tensor.shape),
            image_tensor.cpu().contiguous().numpy().tobytes(),
        )
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# LoRA file hashes
# ---------------------------------------------------------------------------


def hash_lora_file(lora_path: str) -> tuple[str | None, str | None]:
    """Hash a LoRA safetensors file from disk.

    Returns:
        (content_hash, file_hash) where:
          content_hash = hash_safetensors_content (full tensor weight bytes)
          file_hash    = hash_bytes (full raw file bytes)
        Either may be None on failure.
    """
    from core.hashing import hash_safetensors_content

    try:
        p = Path(lora_path)
        if not p.is_file():
            return None, None
        file_bytes = p.read_bytes()
        file_hash = hash_bytes(file_bytes)
        content_hash = hash_safetensors_content(lora_path)
        return content_hash, file_hash
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# LoRA in-memory patch hash  (canonical lora.hash for eval_hash)
# ---------------------------------------------------------------------------


def hash_lora_patches(model_patcher: Any) -> str | None:
    """Hash in-memory LoRA patch tensors from a loaded ModelPatcher.

    Iterates model_patcher.patches (sorted by key), hashes all patch tensor bytes.
    This is the canonical lora.hash used for eval_hash computation — it reflects
    the float32 tensors actually applied to the model, not the raw file bytes.

    Returns:
        BLAKE3 hex digest if patches found, None if no patches or failure.
    """
    try:
        patches = model_patcher.patches
        if not patches:
            return None

        import blake3 as _blake3

        h = _blake3.blake3()
        keys = sorted(patches.keys())
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

        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# LoRA path resolution
# ---------------------------------------------------------------------------


def resolve_lora_path(lora_name: str) -> str | None:
    """Resolve a LoRA filename to its full path via ComfyUI folder_paths.

    Returns:
        Absolute path string on success, None if not found.
    """
    try:
        import folder_paths
        path = folder_paths.get_full_path("loras", lora_name)
        return str(path) if path else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# LoRA safetensors metadata reader
# ---------------------------------------------------------------------------


def read_lora_safetensors_metadata(lora_path: str) -> dict[str, Any]:
    """Read the safetensors header metadata from a LoRA file.

    Returns a dict with:
      status: "ok" | "error"
      raw_metadata: dict[str, str]
      inferred_display_name: str | None
      rank: int | None
      network_alpha: float | None
      affects_text_encoder: bool
      metadata_key_count: int
    """
    import json as _json
    import struct as _struct

    try:
        p = Path(lora_path)
        with open(p, "rb") as f:
            header_len_raw = f.read(8)
            if len(header_len_raw) < 8:
                return {"status": "error", "detail": "file too short"}
            header_len = _struct.unpack("<Q", header_len_raw)[0]
            if header_len > 200 * 1024 * 1024:
                return {"status": "error", "detail": f"header implausibly large: {header_len}"}
            header = _json.loads(f.read(header_len).decode("utf-8"))

        metadata = header.get("__metadata__", {})

        # Infer display name from common metadata keys
        display_name: str | None = None
        for key in ("ss_output_name", "modelspec.title", "name", "LoRA_name"):
            if key in metadata and metadata[key]:
                display_name = str(metadata[key])
                break

        # Infer rank from first lora_down weight shape [rank, in_features].
        # lora_down.shape[0] is the actual rank (e.g. 384 for a rank-384 LoRA).
        # lora_up.shape[0] is out_features — do NOT use that.
        rank: int | None = None
        for tensor_key, tensor_meta in header.items():
            if tensor_key == "__metadata__":
                continue
            if "lora_down" in tensor_key and isinstance(tensor_meta, dict):
                shape = tensor_meta.get("shape")
                if isinstance(shape, list) and shape:
                    rank = int(shape[0])
                    break

        # Infer network_alpha from metadata
        network_alpha: float | None = None
        if "ss_network_alpha" in metadata:
            try:
                network_alpha = float(metadata["ss_network_alpha"])
            except (TypeError, ValueError):
                pass

        # Infer target blocks (unique first-segment keys)
        target_blocks: list[str] = sorted({
            k.split(".")[0]
            for k in header.keys()
            if k != "__metadata__" and "." in k
        })

        # Detect text encoder influence
        affects_te = any(
            "text_model" in k or "text_encoder" in k
            for k in header.keys()
            if k != "__metadata__"
        )

        return {
            "status": "ok",
            "file_path": str(lora_path),
            "raw_metadata": {k: str(v) for k, v in metadata.items()},
            "inferred_display_name": display_name,
            "metadata_key_count": len(metadata),
            "rank": rank,
            "network_alpha": network_alpha,
            "target_blocks": target_blocks,
            "affects_text_encoder": affects_te,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Tensor → numpy bytes conversion  (raises RuntimeError on failure)
# ---------------------------------------------------------------------------


def tensor_to_npy_bytes(tensor: Any) -> tuple[bytes, str, list]:
    """Convert a tensor to a canonical .npy payload for asset storage.

    Uses np.save() to produce a proper .npy file (magic bytes + header + data)
    so np.load(allow_pickle=False) can decode it without shape hints.

    Returns:
        (npy_bytes, dtype_str, shape_list)

    Raises:
        RuntimeError on conversion failure.
    """
    try:
        import numpy as np
        arr = tensor.cpu().float().contiguous().numpy()
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue(), "float32", list(arr.shape)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"tensor_to_npy_bytes failed: {exc}") from exc


def image_to_npy_bytes(image_tensor: Any) -> tuple[bytes, str, list]:
    # DEPRECATED: use target-format variants for derived assets; main image stays .npy
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] → canonical .npy payload.

    Extracts the first batch item ([0]) so stored shape is [H, W, C].
    Uses np.save() to produce a proper .npy file (magic bytes + header + data)
    so np.load(allow_pickle=False) can decode it without shape hints.

    Returns:
        (npy_bytes, dtype_str, shape_list)

    Raises:
        RuntimeError on conversion failure.
    """
    try:
        import numpy as np
        arr = image_tensor[0].cpu().float().contiguous().numpy()
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue(), "float32", list(arr.shape)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"image_to_npy_bytes failed: {exc}") from exc


def ndarray_to_npy_bytes(arr: Any) -> bytes:
    # DEPRECATED: use target-format variants for derived assets when applicable
    """Convert a numpy ndarray to a canonical .npy payload for asset storage.

    Companion to tensor_to_npy_bytes() and image_to_npy_bytes() for cases
    where the data is already a numpy ndarray (not a PyTorch tensor).
    Uses np.save() so np.load(allow_pickle=False) can decode it without
    any external shape hints.

    Raises:
        Any exception from np.save() — callers are responsible for catching.
    """
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pixel stats
# ---------------------------------------------------------------------------


def compute_pixel_stats(image_tensor: Any) -> dict[str, Any]:
    """Compute pixel statistics from a decoded IMAGE tensor [B, H, W, C].

    Returns:
        {"min": float, "max": float, "mean": float, "std": float}
        or Invalid wrapper on failure.
    """
    try:
        tf = image_tensor.float()
        return {
            "min":  float(tf.min()),
            "max":  float(tf.max()),
            "mean": float(tf.mean()),
            "std":  float(tf.std()),
        }
    except Exception as exc:  # noqa: BLE001
        return make_null("EXTRACTOR.IMAGE_HASH_FAIL", str(exc))


# ---------------------------------------------------------------------------
# Luminance map  (raises RuntimeError on failure)
# ---------------------------------------------------------------------------


def compute_luminance(image_tensor: Any) -> tuple[bytes, dict[str, Any]]:
    """Compute BT.709 luminance map from a decoded image.

    Y = 0.2126R + 0.7152G + 0.0722B

    image_tensor: [B, H, W, C] float32 [0, 1]

    Returns:
        (lum_png_bytes, stats_dict) where stats contains mean/std/min/max/shape/dtype.

    Raises:
        RuntimeError on computation failure.
    """
    try:
        tf = image_tensor[0].float().cpu()  # [H, W, C]
        lum = (
            tf[:, :, 0] * 0.2126
            + tf[:, :, 1] * 0.7152
            + tf[:, :, 2] * 0.0722
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
        return ndarray_to_png16_bytes(lum_np), stats
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"compute_luminance failed: {exc}") from exc


# ---------------------------------------------------------------------------
# CLIP Vision — model hash
# ---------------------------------------------------------------------------


def hash_clip_model(cv_model: Any) -> str | None:
    """Hash CLIP_VISION model weights for slot keying ([:16] used as dict key).

    Returns:
        BLAKE3 hex digest on success, None on failure.
    """
    try:
        import blake3 as _blake3

        inner = getattr(cv_model, "model", None) or cv_model
        if not hasattr(inner, "named_parameters"):
            return None

        h = _blake3.blake3()
        for name, param in sorted(inner.named_parameters()):
            h.update(name.encode("utf-8"))
            h.update(param.data.cpu().float().contiguous().numpy().tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# CLIP Vision — encode
# ---------------------------------------------------------------------------


def encode_clip_vision(cv_model: Any, image_tensor: Any) -> Any:
    """Run encode_image() on a CLIP_VISION model.

    Returns:
        Raw CLIP_VISION_OUTPUT object (not converted to plain Python).

    Raises:
        Exception on failure (caller catches).
    """
    return cv_model.encode_image(image_tensor)


# ---------------------------------------------------------------------------
# CLIP Vision — extract arrays from output
# ---------------------------------------------------------------------------


def extract_clip_vision_arrays(clip_output: Any) -> dict[str, Any]:
    """Extract numpy arrays and stats from a CLIP_VISION_OUTPUT object.

    Handles both SigLIP (3D image_embeds [B, seq_len, dim], no CLS) and
    ViT-L/H (2D image_embeds [B, dim], projected).

    Returns:
        {
          "global_embedding_array": ndarray|None,   # float32 [B, dim] mean-pooled
          "global_embedding_shape": list|None,
          "last_hidden_state_array": ndarray|None,  # float32 [B, N, dim]
          "last_hidden_state_shape": list|None,
          "is_aliased": bool,
          "stats": dict,   # inline stats for Evidence JSON
        }
    """
    result: dict[str, Any] = {
        "global_embedding_array": None,
        "global_embedding_shape": None,
        "last_hidden_state_array": None,
        "last_hidden_state_shape": None,
        "is_aliased": False,
        "stats": {},
    }

    try:
        ie = getattr(clip_output, "image_embeds", None)
        lhs = getattr(clip_output, "last_hidden_state", None)

        # Detect SigLIP aliasing (same tensor in memory)
        is_aliased = False
        if ie is not None and lhs is not None:
            try:
                is_aliased = ie.data_ptr() == lhs.data_ptr()
            except Exception:  # noqa: BLE001
                pass
        result["is_aliased"] = is_aliased

        # global_embedding: mean-pooled for 3D (SigLIP), direct for 2D (ViT-L/H)
        if ie is not None:
            try:
                tf = ie.float()
                if len(ie.shape) == 3:
                    # SigLIP: [B, seq_len, dim] → mean over seq_len → [B, dim]
                    pooled = tf.mean(dim=1).cpu().numpy().astype("float32")
                else:
                    # ViT-L/H: [B, dim] — already projected
                    pooled = tf.cpu().numpy().astype("float32")
                result["global_embedding_array"] = pooled
                result["global_embedding_shape"] = list(pooled.shape)
            except Exception:  # noqa: BLE001
                pass

        # last_hidden_state spatial features for downstream masked pooling
        if lhs is not None:
            try:
                lhs_arr = lhs.float().cpu().numpy().astype("float32")
                result["last_hidden_state_array"] = lhs_arr
                result["last_hidden_state_shape"] = list(lhs_arr.shape)
            except Exception:  # noqa: BLE001
                pass

        # Inline stats (non-asset JSON data)
        stats: dict[str, Any] = {"is_aliased": is_aliased}
        if ie is not None:
            try:
                tf = ie.float()
                stats["image_embeds_shape"] = list(ie.shape)
                stats["image_embeds_mean"] = float(tf.mean())
            except Exception:  # noqa: BLE001
                pass
        if lhs is not None:
            try:
                stats["last_hidden_state_shape"] = list(lhs.shape)
            except Exception:  # noqa: BLE001
                pass
        result["stats"] = stats

    except Exception:  # noqa: BLE001
        pass

    return result


# ---------------------------------------------------------------------------
# CLIP Vision — mask-weighted patch pool
# ---------------------------------------------------------------------------


def compute_patch_pools(
    last_hidden_state: Any,
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Compute mask-weighted pooling of spatial patch features.

    Identical algorithm to probe._masked_patch_pool. Auto-detects CLS token.

    last_hidden_state: tensor [B, N_tokens, dim]
    masks: {"face": tensor_or_None, "main_subject": tensor_or_None, ...}

    Returns:
        {"face": {"patch_pool": {...}}, "main_subject": {...}, ...}
        Only active (non-None) masks are included.
    """
    patch_pools: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        if mask is None:
            continue
        try:
            patch_pools[mask_name] = {"patch_pool": _masked_patch_pool(last_hidden_state, mask)}
        except Exception as exc:  # noqa: BLE001
            patch_pools[mask_name] = {"patch_pool": {"status": "error", "detail": str(exc)}}
    return patch_pools


def _masked_patch_pool(last_hidden_state: Any, mask: Any) -> dict[str, Any]:
    """Mask-weighted pooling of spatial patch features from last_hidden_state.

    Auto-detects CLS token presence:
      ViT/OpenCLIP (e.g. ViT-L/14): token 0 is CLS; spatial = tokens 1..N.
      SigLIP/SigLIP2: no CLS token; ALL tokens are spatial.
    """
    import torch.nn.functional as F

    total_tokens = last_hidden_state.shape[1]

    n_no_cls = total_tokens - 1
    gs_no_cls = int(n_no_cls ** 0.5)
    has_cls = gs_no_cls * gs_no_cls == n_no_cls

    if has_cls:
        patch_features = last_hidden_state[:, 1:, :]
    else:
        n_all = total_tokens
        gs_all = int(n_all ** 0.5)
        if gs_all * gs_all != n_all:
            return {
                "status": "error",
                "detail": (
                    f"non-square patch grid: tried {n_no_cls} (CLS skip) "
                    f"and {n_all} (no CLS skip)"
                ),
            }
        patch_features = last_hidden_state[:, :, :]

    B, N, dim = patch_features.shape
    grid_size = int(N ** 0.5)

    m = mask.float()
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(1)

    m = F.interpolate(m, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    patch_weights = m.squeeze().flatten()
    if patch_weights.dim() == 0:
        patch_weights = patch_weights.unsqueeze(0)

    total_weight = float(patch_weights.sum())
    if total_weight < 1e-6:
        return {"status": "empty_mask", "covered_patches": 0, "total_patches": N}

    features = patch_features[0].to(patch_weights.device)
    pooled = (features * patch_weights.unsqueeze(-1)).sum(0) / total_weight

    return {
        "covered_patches": int((patch_weights > 0.1).sum()),
        "total_patches": N,
        "coverage_ratio": round(total_weight / N, 4),
        "pooled_norm": float(pooled.norm()),
        "pooled_mean": float(pooled.mean()),
        "has_cls_token": has_cls,
    }


# ---------------------------------------------------------------------------
# InsightFace root resolution
# ---------------------------------------------------------------------------


def resolve_insightface_root() -> str:
    """Auto-resolve InsightFace model root from ComfyUI folder_paths.

    Resolution order:
      1. folder_paths.get_folder_names_and_paths("insightface") — registered
         by StabilityMatrix / extra_model_paths.yaml.
      2. folder_paths.models_dir / "insightface" — standard ComfyUI layout.

    Returns:
        Path string on success, empty string if unavailable.
    """
    try:
        import folder_paths as _fp
        from pathlib import Path as _Path

        try:
            entry = _fp.get_folder_names_and_paths("insightface")
            if entry and entry[0]:
                return str(_Path(entry[0][0]))
        except Exception:  # noqa: BLE001
            pass

        return str(_Path(_fp.models_dir) / "insightface")
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# InsightFace face analysis  (module-level cache — expensive to load)
# ---------------------------------------------------------------------------

_insightface_cache: dict[str, Any] = {}


def run_insightface(image_tensor: Any, insightface_root: str) -> dict[str, Any]:
    """Run InsightFace face analysis on a decoded image.

    image_tensor: ComfyUI IMAGE [B, H, W, C] float32 [0, 1]
    insightface_root: path to InsightFace model root (contains models/ directory)

    Returns:
        {status, face_count, faces: [face_dict, ...]}
        Never raises — returns error dict on failure.
    """
    if not insightface_root or not insightface_root.strip():
        return {"status": "not_provided"}

    try:
        import numpy as np
        import blake3 as _blake3

        app = _get_insightface_app(insightface_root.strip())
        img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        faces = app.get(img_np)

        if not faces:
            return {"status": "ok", "face_count": 0, "detail": "no face detected"}

        def _extract_face(face: Any) -> dict[str, Any]:
            emb = face.embedding.astype(np.float32)
            h = _blake3.blake3()
            h.update(emb.tobytes())
            rec: dict[str, Any] = {
                "det_score":       float(face.det_score),
                "bbox":            [float(x) for x in face.bbox.tolist()],
                "embedding_hash":  h.hexdigest(),
                "embedding_norm":  float(np.linalg.norm(emb)),
                "embedding_mean":  float(emb.mean()),
                "embedding_array": emb,    # float32 [512] — committed as asset by nodes.py
            }
            if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
                try:
                    ne = face.normed_embedding.astype(np.float32)
                    nh = _blake3.blake3()
                    nh.update(ne.tobytes())
                    rec["normed_embedding_hash"]  = nh.hexdigest()
                    rec["normed_embedding_array"] = ne
                except Exception:  # noqa: BLE001
                    pass
            # Fallback: if InsightFace did not expose normed_embedding, compute it ourselves
            # by L2-normalising the raw embedding.  Cosine similarity is invariant to scaling,
            # but storing a unit-norm vector keeps the procedure contract clean.
            if "normed_embedding_array" not in rec:
                norm = float(np.linalg.norm(emb))
                if norm > 0.0:
                    ne = (emb / norm).astype(np.float32)
                    nh = _blake3.blake3()
                    nh.update(ne.tobytes())
                    rec["normed_embedding_hash"]  = nh.hexdigest()
                    rec["normed_embedding_array"] = ne
            if hasattr(face, "pose") and face.pose is not None:
                try:
                    import numpy as _np2
                    pose_arr = _np2.array(face.pose, dtype=np.float32).ravel()
                    rec["pose"] = [float(x) for x in pose_arr[:3].tolist()]
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(face, "age") and face.age is not None:
                rec["age"] = int(face.age)
            if hasattr(face, "gender") and face.gender is not None:
                rec["gender"] = int(face.gender)
            if hasattr(face, "sex") and face.sex is not None:
                rec["sex"] = str(face.sex)
            if hasattr(face, "kps") and face.kps is not None:
                try:
                    import numpy as _np3
                    kps_arr = _np3.array(face.kps, dtype=np.float32)
                    rec["kps"] = [[float(x), float(y)] for x, y in kps_arr.tolist()]
                except Exception:  # noqa: BLE001
                    pass
            return rec

        sorted_faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
        return {
            "status":     "ok",
            "face_count": len(faces),
            "faces":      [_extract_face(f) for f in sorted_faces],
        }

    except ImportError:
        return {"status": "not_available", "detail": "insightface package not installed"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": type(exc).__name__, "detail": str(exc)}


def _get_insightface_app(model_path: str) -> Any:
    """Load and cache InsightFace FaceAnalysis app.

    Same dual-convention logic as probe._get_insightface_app.
    """
    if model_path in _insightface_cache:
        return _insightface_cache[model_path]

    from insightface.app import FaceAnalysis
    from pathlib import Path as _Path

    p = _Path(model_path)

    if p.parent.name == "models" and not (p / "models").is_dir():
        model_name = p.name
        root = str(p.parent.parent)
    else:
        root = str(p)
        models_dir = p / "models"
        model_name = "buffalo_l"
        if models_dir.is_dir():
            candidates = sorted(d.name for d in models_dir.iterdir() if d.is_dir())
            if candidates:
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


# ---------------------------------------------------------------------------
# Model extras
# ---------------------------------------------------------------------------


def extract_model_extras(model_patcher: Any) -> str | None:
    """Hash non-trivial model_options entries from a ModelPatcher.

    Captures non-base, non-LoRA extras attached via model_options — e.g. custom
    CFG wrappers, attention patches added by nodes like FreeU or SAG.

    ModelSamplingFlux directly modifies BaseModel settings (not model_options)
    and therefore does NOT appear here — that is correct behaviour.

    Returns:
        BLAKE3 hex of canonical JSON if non-trivial entries found, None otherwise.
        None is the expected return for standard (no-extra) configurations.
    """
    try:
        opts = model_patcher.model_options
        if not opts:
            return None

        extras: dict[str, Any] = {}
        for k, v in opts.items():
            if isinstance(v, dict):
                non_trivial = {
                    dk: type(dv).__name__
                    for dk, dv in v.items()
                    if not isinstance(dv, (str, int, float, bool, type(None)))
                }
                if non_trivial:
                    extras[k] = non_trivial
            elif not isinstance(v, (str, int, float, bool, type(None))):
                extras[k] = type(v).__name__

        if not extras:
            return None

        import blake3 as _blake3
        from core.json_codec import canonical_json

        serialised = canonical_json(extras).encode("utf-8")
        return _blake3.blake3(serialised).hexdigest()

    except Exception:  # noqa: BLE001
        return None
