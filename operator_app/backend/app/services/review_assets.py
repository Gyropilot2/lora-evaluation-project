"""Prepared review-asset surface for app payloads and routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import command_center as cc

from core.paths import get_path
from operator_app.backend.pose_render import render_pose_overlay
from operator_app.backend.review_asset_codecs import png_bytes_from_npy


_POSE_OVERLAY_SOURCES: tuple[tuple[str, str], ...] = (
    ("openpose_body", "Pose OpenPose"),
    ("dw_body", "Pose DW"),
)


@dataclass(frozen=True)
class ReviewAsset:
    media_type: str
    file_path: Path | None = None
    content: bytes | None = None


def sample_image_components(sample: dict[str, Any] | None) -> list[dict[str, str]]:
    if not sample:
        return []

    components: list[dict[str, str]] = []
    for kind, domain_key, prefix in (
        ("mask", "masks", "Mask"),
        ("aux", "aux", "Aux"),
    ):
        domain = sample.get(domain_key) or {}
        if not isinstance(domain, dict):
            continue
        for key, item in sorted(domain.items()):
            output = (item or {}).get("output") or {}
            path = output.get("path")
            if not path:
                continue
            components.append(
                {
                    "key": f"{kind}:{key}",
                    "kind": kind,
                    "label": f"{prefix} {_title_case_token(str(key))}",
                    "image_path": str(path),
                }
            )

    sample_hash = str(sample.get("sample_hash") or "")
    for source_key, label in pose_overlay_components(sample):
        components.append(
            {
                "key": f"pose_evidence:{source_key}",
                "kind": "pose_evidence",
                "label": label,
                "image_path": build_pose_overlay_path(sample_hash, source_key),
            }
        )

    return components


def pose_overlay_components(sample: dict[str, Any] | None) -> list[tuple[str, str]]:
    pose_evidence = (sample or {}).get("pose_evidence")
    if not isinstance(pose_evidence, dict):
        return []
    return [
        (source_key, label)
        for source_key, label in _POSE_OVERLAY_SOURCES
        if isinstance(pose_evidence.get(source_key), dict)
    ]


def build_pose_overlay_path(sample_hash: str, source_key: str) -> str:
    return f"pose_evidence:{sample_hash}:{source_key}"


def load_review_asset(path: str, preview: bool) -> ReviewAsset:
    if path.startswith("pose_evidence:"):
        return _load_pose_overlay_asset(path)

    requested = _resolve_file_asset_path(path)
    if requested.suffix.lower() == ".npy" and (preview or requested.parent.name == "image"):
        return ReviewAsset(media_type="image/png", content=png_bytes_from_npy(requested))
    return ReviewAsset(media_type=_media_type_for_path(requested), file_path=requested)


def _load_pose_overlay_asset(path: str) -> ReviewAsset:
    sample_hash, source_key = _parse_pose_overlay_path(path)
    treasurer = cc.new_treasurer(read_only=True)
    try:
        raw_sample = treasurer.get_sample(sample_hash)
    finally:
        treasurer.close()

    if raw_sample is None:
        raise FileNotFoundError("sample not found")

    pose_source = ((raw_sample.get("pose_evidence") or {}).get(source_key))
    if not isinstance(pose_source, dict):
        raise FileNotFoundError(f"pose_evidence source '{source_key}' not present")
    return ReviewAsset(media_type="image/png", content=render_pose_overlay(pose_source))


def _parse_pose_overlay_path(path: str) -> tuple[str, str]:
    parts = path.split(":", 2)
    if len(parts) != 3:
        raise ValueError("malformed pose_evidence virtual path")
    _, sample_hash, source_key = parts
    return sample_hash, source_key


def _resolve_file_asset_path(path: str) -> Path:
    project_root = Path(get_path("project_root")).resolve()
    assets_root = Path(get_path("assets_root")).resolve()
    requested = (project_root / path.replace("\\", "/")).resolve()
    try:
        requested.relative_to(assets_root)
    except ValueError as exc:
        raise ValueError("asset path is outside approved asset roots") from exc
    if not requested.is_file():
        raise FileNotFoundError("asset not found")
    return requested


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".npy":
        return "application/octet-stream"
    return "application/octet-stream"


def _title_case_token(value: str) -> str:
    return value.replace("_", " ").strip().title()
