"""lab/dump_writer.py — writes structured probe dump files to disk.

Dump files are plain JSON: timestamped, run-ID tagged, and named to be
sortable by time. They are the Lab's primary output artifact — the raw
material for field_catalog.py analysis.

Binary attachments (numpy arrays: luminance maps, etc.) are written as
.npy files alongside the JSON, named with the same timestamp+run_id prefix.

No ComfyUI dependency. No production DB access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.diagnostics import emit
from core.time_ids import now_iso

_WHERE = "lab.dump_writer"


def write_dump(output_dir: Path | str, run_id: str, data: dict[str, Any]) -> Path:
    """Write a probe dump as a timestamped JSON file.

    Args:
        output_dir: Directory to write the dump file into (created if absent).
        run_id:     Identifier used in the filename (first 8 chars used).
        data:       Probe data dict — must be JSON-serialisable.

    Returns:
        Path to the written file.

    Raises:
        OSError: on write failure (after emitting LAB.DUMP_WRITE_FAIL).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize timestamp for Windows-safe filenames: strip TZ suffix, replace : and . with -
    ts_raw = now_iso()
    ts_safe = ts_raw[:19].replace(":", "-")   # "2026-02-21T17-40-00"
    short_id = str(run_id)[:8]
    filename = f"{ts_safe}_{short_id}.json"
    out_path = out_dir / filename

    payload: dict[str, Any] = {
        "_dump_meta": {
            "run_id": run_id,
            "written_at": ts_raw,
        },
        **data,
    }

    try:
        out_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        emit(
            "ERROR",
            "LAB.DUMP_WRITE_FAIL",
            _WHERE,
            "Failed to write probe dump file",
            out_path=str(out_path),
            error=str(exc),
        )
        raise

    emit(
        "INFO",
        "LAB.DUMP_WRITTEN",
        _WHERE,
        "Probe dump written",
        out_path=str(out_path),
        run_id=run_id,
    )
    return out_path


def write_attachment(
    output_dir: Path | str,
    run_id: str,
    name: str,
    array: "np.ndarray",
) -> Path:
    """Write a numpy array as a .npy attachment alongside a probe dump.

    The filename uses the same timestamp+run_id prefix as the JSON dump
    so attachments and their parent dump are trivially co-located.

    Args:
        output_dir: Directory to write into (must already exist or be creatable).
        run_id:     Same run_id used for the companion JSON dump.
        name:       Short label for this attachment (e.g. "luminance").
                    Used in the filename: {ts}_{id}_{name}.npy
        array:      Numpy array to save.

    Returns:
        Path to the written file.

    Raises:
        OSError: on write failure (after emitting LAB.ATTACHMENT_WRITE_FAIL).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_raw = now_iso()
    ts_safe = ts_raw[:19].replace(":", "-")
    short_id = str(run_id)[:8]
    filename = f"{ts_safe}_{short_id}_{name}.npy"
    out_path = out_dir / filename

    try:
        np.save(str(out_path), array)
    except OSError as exc:
        emit(
            "ERROR",
            "LAB.ATTACHMENT_WRITE_FAIL",
            _WHERE,
            "Failed to write probe attachment file",
            out_path=str(out_path),
            name=name,
            error=str(exc),
        )
        raise

    emit(
        "DEBUG",
        "LAB.ATTACHMENT_WRITTEN",
        _WHERE,
        "Probe attachment written",
        out_path=str(out_path),
        name=name,
        run_id=run_id,
    )
    return out_path


def write_png_attachment(
    output_dir: Path | str,
    run_id: str,
    name: str,
    array: "np.ndarray",
) -> Path:
    """Write an image-like array as a PNG attachment alongside a probe dump."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_raw = now_iso()
    ts_safe = ts_raw[:19].replace(":", "-")
    short_id = str(run_id)[:8]
    filename = f"{ts_safe}_{short_id}_{name}.png"
    out_path = out_dir / filename

    try:
        arr = np.asarray(array)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(str(out_path), format="PNG")
    except OSError as exc:
        emit(
            "ERROR",
            "LAB.ATTACHMENT_WRITE_FAIL",
            _WHERE,
            "Failed to write PNG attachment file",
            out_path=str(out_path),
            name=name,
            error=str(exc),
        )
        raise

    emit(
        "DEBUG",
        "LAB.ATTACHMENT_WRITTEN",
        _WHERE,
        "PNG attachment written",
        out_path=str(out_path),
        name=name,
        run_id=run_id,
    )
    return out_path
