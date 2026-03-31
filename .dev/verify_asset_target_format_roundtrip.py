"""Verify Step 4.5 migrated assets against the retained old source blobs.

This script uses the fact that the old `.npy` blobs still remain on disk after
migration. For each migrated asset family, it recomputes the expected target
bytes from the old unreferenced source blobs, matches them by content hash
against the DB-referenced current assets, and then checks numerical fidelity.

Run from project root:
    python .dev/verify_asset_target_format_roundtrip.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.asset_codecs import (
    decode_image_asset_bytes,
    decode_luminance_asset_bytes,
    decode_mask_asset_bytes,
    embedding_to_fp16npy_bytes,
    ndarray_to_png16_bytes,
    ndarray_to_png8_bytes,
)
from core.hashing import hash_bytes
from core.paths import get_path
from databank import assets as databank_assets
from databank.sqlite_backend import SQLiteBackend


def _new_backend() -> SQLiteBackend:
    from core.diagnostics import emit

    return SQLiteBackend(get_path("db_file"), event_emitter=emit, read_only=True)


def _collect_current_refs() -> dict[str, list[dict[str, Any]]]:
    backend = _new_backend()
    out: dict[str, list[dict[str, Any]]] = {
        "luminance": [],
        "mask": [],
        "aux": [],
        "clip": [],
    }
    try:
        for sample in backend.query_samples({}):
            lum = (sample.get("luminance") or {}).get("output")
            if isinstance(lum, dict):
                out["luminance"].append(lum)

            for entry in (sample.get("masks") or {}).values():
                if isinstance(entry, dict) and isinstance(entry.get("output"), dict):
                    out["mask"].append(entry["output"])

            for entry in (sample.get("aux") or {}).values():
                if isinstance(entry, dict) and isinstance(entry.get("output"), dict):
                    out["aux"].append(entry["output"])

            for slot in (sample.get("clip_vision") or {}).values():
                if not isinstance(slot, dict):
                    continue
                for key in ("global_embedding", "last_hidden_state"):
                    ref = slot.get(key)
                    if isinstance(ref, dict):
                        out["clip"].append(ref)
    finally:
        backend.close()
    return out


def _build_old_source_map(
    folder: str,
    current_paths: set[str],
    encoder,
) -> dict[str, Path]:
    root = get_path("project_root")
    asset_dir = get_path("assets_root") / folder
    out: dict[str, Path] = {}
    for old_path in asset_dir.glob("*.npy"):
        rel = old_path.relative_to(root).as_posix()
        if rel in current_paths:
            continue
        arr = np.load(old_path, allow_pickle=False)
        payload = encoder(arr)
        out[hash_bytes(payload)] = old_path
    return out


def _verify_luminance(refs: list[dict[str, Any]], source_map: dict[str, Path]) -> tuple[int, float]:
    worst = 0.0
    checked = 0
    for ref in refs:
        digest = ref["content_hash"]["digest"]
        old_path = source_map.get(digest)
        if old_path is None:
            raise RuntimeError(f"missing old-source match for luminance digest {digest[:16]}")
        old_arr = np.load(old_path, allow_pickle=False).astype(np.float32)
        blob = databank_assets.load_blob(ref)
        new_arr = decode_luminance_asset_bytes(ref, blob)
        err = float(np.abs(old_arr.astype(np.float64) - new_arr.astype(np.float64)).max())
        worst = max(worst, err)
        checked += 1
    return checked, worst


def _verify_masks(refs: list[dict[str, Any]], source_map: dict[str, Path]) -> tuple[int, float]:
    worst = 0.0
    checked = 0
    for ref in refs:
        digest = ref["content_hash"]["digest"]
        old_path = source_map.get(digest)
        if old_path is None:
            raise RuntimeError(f"missing old-source match for mask digest {digest[:16]}")
        old_arr = np.load(old_path, allow_pickle=False).astype(np.float32)
        expected = (old_arr > 0.5).astype(np.float32)
        blob = databank_assets.load_blob(ref)
        new_arr = decode_mask_asset_bytes(ref, blob)
        err = float(np.abs(expected.astype(np.float64) - new_arr.astype(np.float64)).max())
        uniq = np.unique(new_arr)
        if not set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0}):
            raise RuntimeError(f"mask decode produced non-binary values: {uniq[:10]}")
        worst = max(worst, err)
        checked += 1
    return checked, worst


def _verify_aux(refs: list[dict[str, Any]], source_map: dict[str, Path]) -> tuple[int, float]:
    worst = 0.0
    checked = 0
    for ref in refs:
        digest = ref["content_hash"]["digest"]
        old_path = source_map.get(digest)
        if old_path is None:
            raise RuntimeError(f"missing old-source match for aux digest {digest[:16]}")
        old_arr = np.load(old_path, allow_pickle=False).astype(np.float32)
        blob = databank_assets.load_blob(ref)
        new_arr = decode_image_asset_bytes(ref, blob)
        err = float(np.abs(old_arr.astype(np.float64) - new_arr.astype(np.float64)).max())
        worst = max(worst, err)
        checked += 1
    return checked, worst


def _verify_clip(refs: list[dict[str, Any]], source_map: dict[str, Path]) -> tuple[int, float]:
    worst_cos = 1.0
    checked = 0
    for ref in refs:
        digest = ref["content_hash"]["digest"]
        old_path = source_map.get(digest)
        if old_path is None:
            raise RuntimeError(f"missing old-source match for clip digest {digest[:16]}")
        old_arr = np.load(old_path, allow_pickle=False).astype(np.float32)
        blob = databank_assets.load_blob(ref)
        new_arr = np.load(io.BytesIO(blob), allow_pickle=False).astype(np.float32)
        a = old_arr.reshape(-1)
        b = new_arr.reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = 1.0 if denom == 0.0 else float(np.dot(a, b) / denom)
        worst_cos = min(worst_cos, cos)
        checked += 1
    return checked, worst_cos


def main() -> int:
    refs = _collect_current_refs()
    current_paths = {ref["path"] for family in refs.values() for ref in family}

    lum_sources = _build_old_source_map("luminance", current_paths, ndarray_to_png16_bytes)
    mask_sources = _build_old_source_map("mask", current_paths, lambda arr: ndarray_to_png8_bytes(arr, "binary"))
    aux_sources = _build_old_source_map(
        "aux_image",
        current_paths,
        lambda arr: ndarray_to_png8_bytes(arr, "L" if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1) else "RGB"),
    )
    clip_sources = _build_old_source_map("embedding", current_paths, embedding_to_fp16npy_bytes)

    lum_count, lum_err = _verify_luminance(refs["luminance"], lum_sources)
    mask_count, mask_err = _verify_masks(refs["mask"], mask_sources)
    aux_count, aux_err = _verify_aux(refs["aux"], aux_sources)
    clip_count, clip_cos = _verify_clip(refs["clip"], clip_sources)

    print(f"luminance: checked={lum_count} max_err={lum_err:.9f}")
    print(f"masks: checked={mask_count} max_err={mask_err:.9f}")
    print(f"aux: checked={aux_count} max_err={aux_err:.9f}")
    print(f"clip: checked={clip_count} min_cos={clip_cos:.9f}")

    if lum_err >= (1.0 / 65535.0 + 1e-6):
        raise SystemExit("luminance fidelity check failed")
    if aux_err >= (1.0 / 255.0 + 1e-6):
        raise SystemExit("aux fidelity check failed")
    if mask_err != 0.0:
        raise SystemExit("mask binary roundtrip check failed")
    if clip_cos <= 0.9999:
        raise SystemExit("clip float16 cosine check failed")

    print("verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
