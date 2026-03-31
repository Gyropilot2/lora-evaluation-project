"""
operator_app/backend/pose_render.py — Render a pose_evidence source dict as a standalone PNG overlay.

Used by the operator-app asset endpoint to visualise structured keypoint data on demand.
No ComfyUI dependency; uses only PIL (already required by review_builder.py).

Color convention:
  Green  — joint detected AND inside_main_subject
  Yellow — joint detected but outside_main_subject
  Gray   — joint missing / not detected
"""
from __future__ import annotations

import io
import math
from typing import Any

from PIL import Image, ImageDraw


# Standard COCO-17 body skeleton adjacency pairs (joint names as stored in pose_evidence).
# Pairs with joints not present in the current model's keypoint set are silently skipped.
_SKELETON_PAIRS: list[tuple[str, str]] = [
    ("nose",       "neck"),
    ("neck",       "l_shoulder"),
    ("neck",       "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow",    "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow",    "r_wrist"),
    ("neck",       "l_hip"),
    ("neck",       "r_hip"),
    ("l_hip",      "l_knee"),
    ("l_knee",     "l_ankle"),
    ("r_hip",      "r_knee"),
    ("r_knee",     "r_ankle"),
    ("l_hip",      "r_hip"),    # pelvis crossbar
]

_COLOR_INSIDE    = (80, 220, 80, 255)    # green — supported by MainSubject
_COLOR_OUTSIDE   = (220, 200, 50, 255)   # yellow — outside MainSubject
_COLOR_MISSING   = (80, 80, 80, 180)     # gray — not detected
_COLOR_BONE      = (150, 150, 150, 150)  # gray — skeleton line
_BG_COLOR        = (20, 20, 20, 230)     # dark background
_JOINT_RADIUS    = 5
_BONE_WIDTH      = 2


def _finite_point(joint: dict[str, Any] | None) -> tuple[float, float] | None:
    if not joint:
        return None
    x = joint.get("x")
    y = joint.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return float(x), float(y)


def render_pose_overlay(pose_source: dict[str, Any]) -> bytes:
    """Render a single pose_evidence source dict (openpose_body or dw_body) as PNG bytes.

    Parameters
    ----------
    pose_source : dict
        One source object from pose_evidence — e.g. ``sample["pose_evidence"]["openpose_body"]``.
        Expected keys: ``canvas_width``, ``canvas_height``, ``people`` (list of person dicts).

    Returns
    -------
    bytes
        Raw PNG image bytes.  Returns a small placeholder image on any rendering error.
    """
    try:
        return _render(pose_source)
    except Exception:
        # Fail-safe: return a tiny dark 1×1 PNG so the endpoint never errors
        img = Image.new("RGBA", (1, 1), _BG_COLOR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _render(pose_source: dict[str, Any]) -> bytes:
    w = max(int(pose_source.get("canvas_width") or 512), 1)
    h = max(int(pose_source.get("canvas_height") or 512), 1)

    img = Image.new("RGBA", (w, h), _BG_COLOR)
    draw = ImageDraw.Draw(img)

    for person in (pose_source.get("people") or []):
        joints: dict[str, dict[str, Any]] = person.get("joints") or {}

        # Draw skeleton lines beneath the dots
        for name_a, name_b in _SKELETON_PAIRS:
            ja = joints.get(name_a)
            jb = joints.get(name_b)
            if not ja or not jb:
                continue
            if ja.get("status") == "missing" or jb.get("status") == "missing":
                continue
            point_a = _finite_point(ja)
            point_b = _finite_point(jb)
            if point_a is None or point_b is None:
                continue
            xa, ya = point_a
            xb, yb = point_b
            draw.line([(xa, ya), (xb, yb)], fill=_COLOR_BONE, width=_BONE_WIDTH)

        # Draw joint dots on top
        for _joint_name, jdata in joints.items():
            status = jdata.get("status", "missing")
            if status == "missing":
                color = _COLOR_MISSING
            elif jdata.get("inside_main_subject", False):
                color = _COLOR_INSIDE
            else:
                color = _COLOR_OUTSIDE
            point = _finite_point(jdata)
            if point is None:
                continue
            x, y = point
            r = _JOINT_RADIUS
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
