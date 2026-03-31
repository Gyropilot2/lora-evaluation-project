"""Shared codecs for review-surface image assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def normalize_image_like_array(array: np.ndarray) -> np.ndarray:
    """Normalize common tensor layouts to a Pillow-friendly image array."""
    out = array
    if out.ndim == 3 and out.shape[0] in {1, 3, 4} and out.shape[-1] not in {1, 3, 4}:
        out = np.moveaxis(out, 0, -1)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim not in {2, 3}:
        raise ValueError("unsupported image tensor shape")
    return out


def image_u8_array(array: np.ndarray) -> np.ndarray:
    """Convert a numeric image-like array to uint8 for encoding."""
    normalized = normalize_image_like_array(array)
    if np.issubdtype(normalized.dtype, np.floating):
        finite = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        clipped = np.clip(finite, 0.0, 1.0)
        return (clipped * 255.0).round().astype(np.uint8)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def png_bytes_from_array(array: np.ndarray) -> bytes:
    """Encode an image-like array as PNG bytes."""
    image = Image.fromarray(image_u8_array(array))
    return _png_bytes(image)


def png_bytes_from_npy(path: Path) -> bytes:
    """Load an image-like .npy asset and encode it as PNG bytes."""
    return png_bytes_from_array(np.load(path))


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
