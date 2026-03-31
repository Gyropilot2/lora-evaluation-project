"""
databank/assets.py — internal asset lifecycle helpers for Treasurer backends.

Assets are content-addressed blobs (embeddings, images, etc.) stored on disk.
They are referenced in Evidence via ValueRef objects.

Lifecycle:
  1. stage_blob(data, asset_type, format)
       Writes blob to tmp/ (not yet content-addressed in assets/).
       Returns a ValueRef with the final committed path (not yet on disk there).

  2. commit_blob(valueref)
       Atomically moves the staged blob from tmp/ to its final content-addressed
       path under assets_root/<asset_type>/<hash>.<format>.
       Returns an updated ValueRef (same content, path confirmed on disk).

  3. load_blob(valueref)
       Reads the asset from disk, verifies the content hash.
       Returns bytes on success, or an Invalid wrapper on failure.

This module is implementation detail for DataBank backends. External modules
must talk through `databank.treasurer.Treasurer`, not import these helpers
directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from contracts.validation_errors import make_error
from core.diagnostics import emit
from core.hashing import hash_bytes
from core.paths import get_path


def stage_blob(data: bytes, asset_type: str, fmt: str) -> dict:
    """Write data to the tmp staging area and return a ValueRef."""
    content_hash = hash_bytes(data)
    tmp_root = get_path("tmp_root")
    tmp_root.mkdir(parents=True, exist_ok=True)

    staged_path = tmp_root / f"{content_hash}.{fmt}"
    try:
        staged_path.write_bytes(data)
    except OSError as exc:
        emit(
            "ERROR",
            "ASSET.STAGE_FAIL",
            "databank.assets.stage_blob",
            f"Failed to write staged blob: {exc}",
            asset_type=asset_type,
            fmt=fmt,
        )
        raise

    assets_root = get_path("assets_root")
    project_root = get_path("project_root")
    final_abs = assets_root / asset_type / f"{content_hash}.{fmt}"
    final_rel = final_abs.relative_to(project_root).as_posix()

    return {
        "valueref_version": 1,
        "kind": "asset_ref",
        "asset_type": asset_type,
        "format": fmt,
        "content_hash": {"algo": "blake3", "digest": content_hash},
        "path": final_rel,
        "_staged_path": str(staged_path),
    }


def commit_blob(valueref: dict) -> dict:
    """Atomically move the staged blob to its final content-addressed location."""
    staged_path = Path(valueref["_staged_path"])
    project_root = get_path("project_root")
    final_abs = project_root / valueref["path"]
    final_abs.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.move(str(staged_path), str(final_abs))
    except OSError as exc:
        emit(
            "ERROR",
            "ASSET.WRITE_FAIL",
            "databank.assets.commit_blob",
            f"Failed to commit staged blob: {exc}",
            staged_path=str(staged_path),
            final_path=valueref["path"],
        )
        raise

    return {k: v for k, v in valueref.items() if k != "_staged_path"}


def load_blob(valueref: dict) -> bytes | dict:
    """Load an asset blob from disk and verify its content hash."""
    project_root = get_path("project_root")
    path = project_root / valueref["path"]

    if not path.exists():
        emit(
            "ERROR",
            "ASSET.READ_FAIL",
            "databank.assets.load_blob",
            "Asset file not found",
            path=valueref["path"],
        )
        return make_error(
            reason_code="ASSET.READ_FAIL",
            detail=f"File not found: {valueref['path']}",
        )

    try:
        data = path.read_bytes()
    except OSError as exc:
        emit(
            "ERROR",
            "ASSET.READ_FAIL",
            "databank.assets.load_blob",
            f"Failed to read asset: {exc}",
            path=valueref["path"],
        )
        return make_error(
            reason_code="ASSET.READ_FAIL",
            detail=str(exc),
        )

    expected = valueref["content_hash"]["digest"]
    actual = hash_bytes(data)
    if actual != expected:
        emit(
            "ERROR",
            "ASSET.HASH_MISMATCH",
            "databank.assets.load_blob",
            "Content hash mismatch — asset may be corrupt",
            path=valueref["path"],
            expected=expected[:16],
            actual=actual[:16],
        )
        return make_error(
            reason_code="ASSET.HASH_MISMATCH",
            detail=f"expected {expected[:8]}…, got {actual[:8]}…",
        )

    return data
