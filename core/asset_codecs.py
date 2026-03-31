"""core/asset_codecs.py - shared asset encode/decode helpers.

These helpers decode already-loaded asset bytes into portable numpy arrays.
They do not read files directly and therefore stay independent from the
DataBank asset door.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def ndarray_to_png16_bytes(arr: Any) -> bytes:
    """Encode a single-channel float array in [0, 1] as a 16-bit PNG."""
    out = np.asarray(arr, dtype=np.float32)
    while out.ndim > 2 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    if out.ndim != 2:
        raise ValueError(f"expected HW or HWC(1) array, got shape {list(out.shape)}")

    out_u16 = np.round(np.clip(out, 0.0, 1.0) * 65535.0).astype(np.uint16)
    img = Image.fromarray(out_u16, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ndarray_to_png8_bytes(arr: Any, mode: str) -> bytes:
    """Encode a float array in [0, 1] as an 8-bit PNG."""
    mode_norm = str(mode).upper()
    out = np.asarray(arr, dtype=np.float32)
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]
    out = np.clip(out, 0.0, 1.0)

    if mode_norm == "RGB":
        if out.ndim != 3:
            raise ValueError(f"expected HWC RGB array, got shape {list(out.shape)}")
        if out.shape[2] == 1:
            out = np.repeat(out, 3, axis=2)
        if out.shape[2] < 3:
            raise ValueError(f"expected 3 channels for RGB, got shape {list(out.shape)}")
        out_u8 = np.round(out[:, :, :3] * 255.0).astype(np.uint8)
        img = Image.fromarray(out_u8, mode="RGB")
    elif mode_norm in {"L", "BINARY"}:
        while out.ndim > 2 and out.shape[0] == 1:
            out = out[0]
        if out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]
        if out.ndim != 2:
            raise ValueError(f"expected HW array for mode {mode_norm}, got shape {list(out.shape)}")
        if mode_norm == "BINARY":
            out_u8 = ((out > 0.5).astype(np.uint8) * 255)
        else:
            out_u8 = np.round(out * 255.0).astype(np.uint8)
        img = Image.fromarray(out_u8, mode="L")
    else:
        raise ValueError(f"unsupported PNG mode: {mode}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def embedding_to_fp16npy_bytes(arr: Any) -> bytes:
    """Convert a numeric array to float16 and serialize it as .npy."""
    out = np.asarray(arr, dtype=np.float32).astype(np.float16)
    buf = io.BytesIO()
    np.save(buf, out)
    return buf.getvalue()


def decode_image_asset_bytes(valueref: dict[str, Any], data: bytes) -> np.ndarray:
    """Decode an image-like asset to HWC float32 in [0, 1].

    Supports current `.npy` image assets and future `.png`/ordinary image files.
    Raises ValueError for unsupported formats or malformed payloads.
    """
    fmt = str(valueref.get("format") or "").lower()

    if fmt == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return _to_hwc_float(arr)

    if fmt in {"png", "jpg", "jpeg", "webp", "bmp"}:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return _to_hwc_float(arr)

    raise ValueError(f"unsupported image asset format: {fmt or '(missing)'}")


def decode_mask_asset_bytes(valueref: dict[str, Any], data: bytes) -> np.ndarray:
    """Decode a mask-like asset to HW float32 in [0, 1]."""
    fmt = str(valueref.get("format") or "").lower()

    if fmt == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return _to_hw_float(arr)

    if fmt in {"png", "jpg", "jpeg", "webp", "bmp"}:
        img = Image.open(io.BytesIO(data)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return _to_hw_float(arr)

    raise ValueError(f"unsupported mask asset format: {fmt or '(missing)'}")


def decode_luminance_asset_bytes(valueref: dict[str, Any], data: bytes) -> np.ndarray:
    """Decode a luminance asset to HW float32 in [0, 1]."""
    fmt = str(valueref.get("format") or "").lower()

    if fmt == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return _to_hw_float(arr)

    if fmt == "png":
        img = Image.open(io.BytesIO(data))
        raw = np.asarray(img)
        arr = raw.astype(np.float32)
        if raw.dtype == np.uint16 or img.mode in {"I;16", "I"}:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
        return _to_hw_float(arr)

    if fmt in {"jpg", "jpeg", "webp", "bmp"}:
        img = Image.open(io.BytesIO(data)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return _to_hw_float(arr)

    raise ValueError(f"unsupported luminance asset format: {fmt or '(missing)'}")


def _to_hwc_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 3:
        raise ValueError(f"expected HWC image-like array, got shape {list(out.shape)}")
    if out.shape[2] > 3:
        out = out[:, :, :3]
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    return np.clip(out, 0.0, 1.0)


def _to_hw_float(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    while out.ndim > 2 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    if out.ndim == 3 and out.shape[2] >= 3:
        out = out[:, :, :3].mean(axis=2)
    if out.ndim != 2:
        raise ValueError(f"expected HW mask-like array, got shape {list(out.shape)}")
    return np.clip(out, 0.0, 1.0)
