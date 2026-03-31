"""Structured pose-scene extraction for image-first lab experiments.

This module does not know about ComfyUI nodes, the DB, or lab dump layout.
It accepts plain numpy arrays plus optional supportive evidence and returns a
JSON-safe scene dossier suitable for lab review.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

POSE_JOINT_COLORS: dict[str, tuple[float, float, float]] = {
    # Upper body — left side (warm tones)
    "neck":       (1.0, 0.0,        0.0),        # red — central anchor
    "l_shoulder": (1.0, 2.0 / 3.0, 0.0),
    "r_shoulder": (1.0 / 3.0, 1.0, 0.0),
    "l_elbow":    (1.0, 1.0,        0.0),
    "r_elbow":    (0.0, 1.0,        0.0),
    "l_wrist":    (2.0 / 3.0, 1.0, 0.0),
    "r_wrist":    (0.0, 1.0,        1.0 / 3.0),
    # Lower body (cool tones)
    "l_hip":      (0.0, 1.0,        2.0 / 3.0),
    "r_hip":      (0.0, 1.0 / 3.0, 1.0),
    "l_knee":     (0.0, 1.0,        1.0),
    "r_knee":     (0.0, 2.0 / 3.0, 1.0),
    "l_ankle":    (2.0 / 3.0, 0.0, 1.0),        # mirror of r_ankle (purple family)
    "r_ankle":    (1.0 / 3.0, 0.0, 1.0),
}

POSE_SEGMENTS: list[tuple[str, str]] = [
    ("neck",       "l_shoulder"),
    ("neck",       "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("r_shoulder", "r_elbow"),
    ("l_elbow",    "l_wrist"),
    ("r_elbow",    "r_wrist"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip",      "r_hip"),
    ("l_hip",      "l_knee"),
    ("r_hip",      "r_knee"),
    ("l_knee",     "l_ankle"),
    ("r_knee",     "r_ankle"),
]

CORE_JOINTS = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
_POSE_COLOR_TOL = 0.075
_POSE_NONBLACK_THRESH = 0.05
_DENSEPOSE_BG_THRESH_U8 = 24
_POSE_KEYPOINT_PRESENT_THRESH = 0.05
_POSE_KEYPOINT_MEDIUM_CONF = 0.4
_POSE_KEYPOINT_HIGH_CONF = 0.8
_JOINT_SUPPORT_RADIUS = 3

POSE_KEYPOINT_INDEXES: dict[str, int] = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,
    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
    "r_hip": 8,
    "r_knee": 9,
    "r_ankle": 10,
    "l_hip": 11,
    "l_knee": 12,
    "l_ankle": 13,
    "r_eye": 14,
    "l_eye": 15,
    "r_ear": 16,
    "l_ear": 17,
}


def analyze_pose_scene(
    image_arr: np.ndarray,
    pose_arr: np.ndarray,
    main_subject_mask_arr: np.ndarray,
    *,
    densepose_arr: np.ndarray | None = None,
    face_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe pose-scene dossier from raw image/mask/preprocessor arrays."""
    image = _to_hwc_float(image_arr)
    pose = _to_hwc_float(pose_arr)
    main_subject_mask = _to_hw_float(main_subject_mask_arr)
    main_subject_bool = main_subject_mask > 0.5

    height, width = pose.shape[:2]
    total_pixels = max(1, height * width)

    pose_mask = np.max(pose, axis=2) > _POSE_NONBLACK_THRESH
    pose_labels, pose_components = _label_components(pose_mask)

    densepose_mask = None
    densepose_stats: dict[str, Any]
    if densepose_arr is not None:
        densepose_mask, densepose_stats = _derive_densepose_human_mask(densepose_arr)
    else:
        densepose_stats = {"status": "not_provided"}

    joint_blobs = _extract_joint_blobs(
        pose,
        pose_labels,
        total_pixels,
        main_subject_mask=main_subject_bool,
        densepose_mask=densepose_mask,
    )
    components_summary, candidates = _build_candidates(
        pose_components,
        joint_blobs,
        main_subject_bool,
        densepose_mask=densepose_mask,
        face_analysis=face_analysis,
        image_shape=(height, width),
    )

    scene_flags = _global_scene_flags(
        candidates,
        face_analysis=face_analysis,
        densepose_present=densepose_arr is not None,
    )

    support_observations = {
        "main_subject_mask_stats": _mask_stats(main_subject_bool),
        "densepose_human_mask_stats": densepose_stats,
        "insightface_summary": _summarize_face_analysis(face_analysis),
    }

    strategy_results = _evaluate_strategies(
        candidates,
        global_scene_flags=scene_flags,
        densepose_present=densepose_mask is not None,
    )
    recommended_display_strategy = _recommended_display_strategy(strategy_results)
    recommended_primary_candidate_id = None
    if recommended_display_strategy is not None:
        recommended_primary_candidate_id = strategy_results[recommended_display_strategy].get(
            "primary_candidate_id"
        )

    for candidate in candidates:
        candidate.pop("_ys", None)
        candidate.pop("_xs", None)
    for blob in joint_blobs:
        blob.pop("_ys", None)
        blob.pop("_xs", None)

    pose_observations = {
        "source_kind": "pose_image",
        "pose_shape": [int(height), int(width), int(pose.shape[2])],
        "non_black_component_count": len(pose_components),
        "eligible_candidate_count": len(candidates),
        "joint_blob_count": len(joint_blobs),
        "components": components_summary,
        "joint_blobs": joint_blobs,
        "scene_flags": scene_flags,
    }

    return {
        "pose_observations": pose_observations,
        "support_observations": support_observations,
        "candidates": candidates,
        "strategy_results": strategy_results,
        "recommended_display_strategy": recommended_display_strategy,
        "recommended_primary_candidate_id": recommended_primary_candidate_id,
    }


def analyze_pose_scene_keypoints(
    image_arr: np.ndarray,
    pose_keypoint_data: Any,
    main_subject_mask_arr: np.ndarray,
    *,
    densepose_arr: np.ndarray | None = None,
    face_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe pose-scene dossier from structured people[] keypoints."""
    image = _to_hwc_float(image_arr)
    main_subject_mask = _to_hw_float(main_subject_mask_arr)
    main_subject_bool = main_subject_mask > 0.5

    height, width = image.shape[:2]
    densepose_mask = None
    densepose_stats: dict[str, Any]
    if densepose_arr is not None:
        densepose_mask, densepose_stats = _derive_densepose_human_mask(densepose_arr)
    else:
        densepose_stats = {"status": "not_provided"}

    payload = _normalize_pose_keypoint_payload(pose_keypoint_data, image_shape=(height, width))
    people_summary, candidates = _build_structured_candidates(
        payload,
        main_subject_bool,
        densepose_mask=densepose_mask,
        face_analysis=face_analysis,
        image_shape=(height, width),
    )

    scene_flags = _global_scene_flags(
        candidates,
        face_analysis=face_analysis,
        densepose_present=densepose_arr is not None,
    )

    support_observations = {
        "main_subject_mask_stats": _mask_stats(main_subject_bool),
        "densepose_human_mask_stats": densepose_stats,
        "insightface_summary": _summarize_face_analysis(face_analysis),
    }

    strategy_results = _evaluate_strategies(
        candidates,
        global_scene_flags=scene_flags,
        densepose_present=densepose_mask is not None,
    )
    recommended_display_strategy = _recommended_display_strategy(strategy_results)
    recommended_primary_candidate_id = None
    if recommended_display_strategy is not None:
        recommended_primary_candidate_id = strategy_results[recommended_display_strategy].get(
            "primary_candidate_id"
        )

    pose_observations = {
        "source_kind": "structured_keypoints",
        "canvas_height": int(payload["canvas_height"]),
        "canvas_width": int(payload["canvas_width"]),
        "raw_people_count": int(len(payload["people"])),
        "eligible_candidate_count": len(candidates),
        "people": people_summary,
        "scene_flags": scene_flags,
    }

    return {
        "pose_observations": pose_observations,
        "support_observations": support_observations,
        "candidates": candidates,
        "strategy_results": strategy_results,
        "recommended_display_strategy": recommended_display_strategy,
        "recommended_primary_candidate_id": recommended_primary_candidate_id,
    }


def render_pose_scene_overlay(
    image_arr: np.ndarray,
    scene: dict[str, Any],
    *,
    overlay_label: str | None = None,
) -> np.ndarray:
    """Render a candidate overlay on top of the original image."""
    image = _to_hwc_float(image_arr)
    base = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(base)

    recommended_strategy = scene.get("recommended_display_strategy")
    primary_id = scene.get("recommended_primary_candidate_id")
    title = overlay_label or "pose scene"
    if recommended_strategy is not None:
        title = f"{title} / {recommended_strategy} / {primary_id or 'abstain'}"
    title_width = max(212, min(420, 10 + (7 * len(title))))
    draw.rectangle((4, 4, title_width, 22), fill=(10, 18, 24))
    draw.text((8, 7), title, fill=(150, 220, 255))

    for idx, candidate in enumerate(scene.get("candidates", [])):
        cid = candidate.get("candidate_id", f"P{idx}")
        bbox = candidate.get("bbox") or [0, 0, 0, 0]
        is_primary = primary_id is not None and cid == primary_id
        color = (90, 225, 190) if is_primary else (255, 177, 82)
        if "selection_ineligible_tiny_fragment" in (candidate.get("flags") or []):
            color = (130, 130, 130)

        x0, y0, x1, y1 = [int(v) for v in bbox]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.text((x0 + 4, max(0, y0 - 12)), cid, fill=color)

        joints = candidate.get("joints") or {}
        for j0, j1 in POSE_SEGMENTS:
            p0 = joints.get(j0) or {}
            p1 = joints.get(j1) or {}
            if p0.get("x") is None or p1.get("x") is None:
                continue
            segment_color = color
            if p0.get("inside_main_subject") is False or p1.get("inside_main_subject") is False:
                segment_color = (255, 120, 80)
            draw.line(
                (
                    int(round(p0["x"])),
                    int(round(p0["y"])),
                    int(round(p1["x"])),
                    int(round(p1["y"])),
                ),
                fill=segment_color,
                width=3 if is_primary else 2,
            )

        for joint in joints.values():
            x = joint.get("x")
            y = joint.get("y")
            if x is None or y is None:
                continue
            r = 4 if is_primary else 3
            if joint.get("inside_main_subject") is False:
                fill = (255, 90, 90)
            elif joint.get("confidence_band") == "low":
                fill = (255, 170, 80)
            else:
                fill = color
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    return np.asarray(base).astype(np.float32) / 255.0


def _normalize_pose_keypoint_payload(
    pose_keypoint_data: Any,
    *,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    payload = pose_keypoint_data
    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, list):
        if not payload:
            raise ValueError("POSE_KEYPOINT payload was an empty list")
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            payload = next((item for item in payload if isinstance(item, dict) and "people" in item), None)
            if payload is None:
                raise ValueError("POSE_KEYPOINT list did not contain a scene dict with people[]")

    if not isinstance(payload, dict):
        raise ValueError(f"POSE_KEYPOINT payload must be dict/list/JSON string, got {type(payload).__name__}")

    people = payload.get("people")
    if not isinstance(people, list):
        raise ValueError("POSE_KEYPOINT payload is missing people[]")

    return {
        "people": people,
        "canvas_height": int(payload.get("canvas_height") or image_shape[0]),
        "canvas_width": int(payload.get("canvas_width") or image_shape[1]),
    }


def _build_structured_candidates(
    payload: dict[str, Any],
    main_subject_mask: np.ndarray,
    *,
    densepose_mask: np.ndarray | None,
    face_analysis: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = image_shape
    total_pixels = max(1, height * width)
    people_summary: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for person_index, person in enumerate(payload["people"]):
        joints, coord_mode = _extract_structured_person_joints(
            person,
            canvas_width=int(payload["canvas_width"]),
            canvas_height=int(payload["canvas_height"]),
            image_shape=image_shape,
        )
        recognized_joint_tags = sorted(
            tag for tag in POSE_JOINT_COLORS if (joints.get(tag) or {}).get("x") is not None
        )
        recognized_joint_count = len(recognized_joint_tags)
        core_joint_count = sum(1 for tag in CORE_JOINTS if (joints.get(tag) or {}).get("x") is not None)
        present_joint_count = sum(
            1
            for joint in joints.values()
            if isinstance(joint, dict) and joint.get("x") is not None
        )
        sample_ys, sample_xs = _structured_candidate_support_coords(joints, image_shape=image_shape)
        pixel_count = int(sample_ys.size)
        bbox = _structured_candidate_bbox(joints, image_shape=image_shape)
        area_fraction = _bbox_area_fraction(bbox, image_shape=image_shape)
        main_overlap = _mask_overlap_from_coords(main_subject_mask, sample_ys, sample_xs)
        dense_overlap = (
            _mask_overlap_from_coords(densepose_mask, sample_ys, sample_xs)
            if densepose_mask is not None
            else None
        )
        face_support = _candidate_face_support(bbox, face_analysis) if bbox is not None else 0.0

        selection_eligible, ignore_reason = _candidate_eligibility(
            pixel_count=pixel_count,
            recognized_joint_count=recognized_joint_count,
            core_joint_count=core_joint_count,
            main_overlap=main_overlap,
            dense_overlap=dense_overlap,
        )

        person_summary = {
            "person_index": int(person_index),
            "bbox": bbox,
            "pixel_count": pixel_count,
            "area_fraction": round(float(area_fraction), 6),
            "recognized_joint_tags": recognized_joint_tags,
            "recognized_joint_count": recognized_joint_count,
            "present_joint_count": present_joint_count,
            "coord_mode": coord_mode,
            "selection_eligible": bool(selection_eligible),
        }
        if ignore_reason is not None:
            person_summary["ignore_reason"] = ignore_reason
        people_summary.append(person_summary)

        if not selection_eligible or bbox is None:
            continue

        flags: list[str] = []
        if recognized_joint_count < 4 or core_joint_count < 2:
            flags.append("pose_fragmented")
        if core_joint_count < len(CORE_JOINTS):
            flags.append("missing_core_joints")
        if any(
            joint.get("confidence_band") == "low"
            for joint in joints.values()
            if isinstance(joint, dict) and joint.get("x") is not None
        ):
            flags.append("low_confidence_joints")
        support_summary = _annotate_joint_support(
            joints,
            main_subject_mask=main_subject_mask,
            densepose_mask=densepose_mask,
        )
        if support_summary["outside_main_subject_joint_count"] > 0:
            flags.append("joints_outside_main_subject")
        if support_summary["outside_densepose_joint_count"] > 0:
            flags.append("joints_outside_densepose")

        candidates.append(
            {
                "candidate_id": f"P{len(candidates)}",
                "source_person_index": int(person_index),
                "bbox": bbox,
                "component_bbox": bbox,
                "centroid": _structured_candidate_centroid(joints),
                "pixel_count": pixel_count,
                "area_fraction": round(float(area_fraction), 6),
                "recognized_joint_tags": recognized_joint_tags,
                "recognized_joint_count": recognized_joint_count,
                "core_joint_count": core_joint_count,
                "coord_mode": coord_mode,
                "duplicate_joint_group_count": 0,
                "main_subject_overlap": main_overlap,
                "densepose_overlap": dense_overlap,
                "face_support": face_support,
                "flags": sorted(set(flags)),
                "joints": joints,
                **support_summary,
                "selection_features": {
                    "recognized_joint_norm": round(recognized_joint_count / max(1, len(POSE_JOINT_COLORS)), 4),
                    "core_joint_norm": round(core_joint_count / max(1, len(CORE_JOINTS)), 4),
                    "area_score": round(min(area_fraction / 0.18, 1.0), 4),
                    "main_subject_overlap": main_overlap,
                    "densepose_overlap": dense_overlap,
                    "face_support": face_support,
                },
            }
        )

    return people_summary, candidates


def _extract_structured_person_joints(
    person: Any,
    *,
    canvas_width: int,
    canvas_height: int,
    image_shape: tuple[int, int],
) -> tuple[dict[str, dict[str, Any]], str]:
    flat = _pose_keypoint_triplets(person)
    coord_mode = _structured_coord_mode(flat)
    target_height, target_width = image_shape
    width_scale = float(canvas_width or target_width)
    height_scale = float(canvas_height or target_height)

    joints: dict[str, dict[str, Any]] = {}
    for joint_tag in POSE_JOINT_COLORS:
        joints[joint_tag] = {
            "x": None,
            "y": None,
            "confidence_band": "none",
            "status": "missing",
            "source_confidence": 0.0,
        }

    for joint_tag, idx in POSE_KEYPOINT_INDEXES.items():
        if idx * 3 + 2 >= len(flat):
            continue
        x = float(flat[idx * 3 + 0])
        y = float(flat[idx * 3 + 1])
        conf = float(flat[idx * 3 + 2])
        if conf <= _POSE_KEYPOINT_PRESENT_THRESH:
            continue

        if coord_mode == "normalized":
            x *= width_scale
            y *= height_scale

        band = "high" if conf >= _POSE_KEYPOINT_HIGH_CONF else "medium"
        if conf < _POSE_KEYPOINT_MEDIUM_CONF:
            band = "low"

        if joint_tag in joints:
            joints[joint_tag] = {
                "x": x,
                "y": y,
                "confidence_band": band,
                "status": "ok" if band != "low" else "low_confidence",
                "source_confidence": round(conf, 4),
            }

    return joints, coord_mode


def _structured_coord_mode(flat: list[float]) -> str:
    coords = [float(v) for i, v in enumerate(flat) if i % 3 != 2]
    nonzero = [v for v in coords if abs(v) > 1e-9]
    if nonzero and max(nonzero) <= 1.5 and min(nonzero) >= 0.0:
        return "normalized"
    return "pixel"


def _pose_keypoint_triplets(person: Any) -> list[float]:
    if not isinstance(person, dict):
        return []
    values = person.get("pose_keypoints_2d")
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return []


def _structured_candidate_support_coords(
    joints: dict[str, dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    points: set[tuple[int, int]] = set()

    for joint in joints.values():
        x = joint.get("x")
        y = joint.get("y")
        if x is None or y is None:
            continue
        points.add((int(round(y)), int(round(x))))

    for start_tag, end_tag in POSE_SEGMENTS:
        start = joints.get(start_tag) or {}
        end = joints.get(end_tag) or {}
        if start.get("x") is None or end.get("x") is None:
            continue
        for y, x in _sample_line_points(start["x"], start["y"], end["x"], end["y"]):
            points.add((y, x))

    if not points:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    ys = np.asarray([pt[0] for pt in points], dtype=np.int32)
    xs = np.asarray([pt[1] for pt in points], dtype=np.int32)
    ys = np.clip(ys, 0, max(0, height - 1))
    xs = np.clip(xs, 0, max(0, width - 1))
    return ys, xs


def _sample_line_points(x0: float, y0: float, x1: float, y1: float) -> list[tuple[int, int]]:
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1.0)) + 1
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)
    return [(int(round(y)), int(round(x))) for x, y in zip(xs.tolist(), ys.tolist())]


def _structured_candidate_bbox(
    joints: dict[str, dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> list[int] | None:
    xs = [float(joint["x"]) for joint in joints.values() if joint.get("x") is not None]
    ys = [float(joint["y"]) for joint in joints.values() if joint.get("y") is not None]
    if not xs or not ys:
        return None

    width = image_shape[1]
    height = image_shape[0]
    pad = max(6, int(round(0.02 * max(width, height))))
    x0 = max(0, int(np.floor(min(xs))) - pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    x1 = min(width - 1, int(np.ceil(max(xs))) + pad)
    y1 = min(height - 1, int(np.ceil(max(ys))) + pad)
    return [x0, y0, x1, y1]


def _structured_candidate_centroid(joints: dict[str, dict[str, Any]]) -> list[float]:
    xs = [float(joint["x"]) for joint in joints.values() if joint.get("x") is not None]
    ys = [float(joint["y"]) for joint in joints.values() if joint.get("y") is not None]
    if not xs or not ys:
        return [0.0, 0.0]
    return [round(float(np.mean(xs)), 4), round(float(np.mean(ys)), 4)]


def _bbox_area_fraction(
    bbox: list[int] | None,
    *,
    image_shape: tuple[int, int],
) -> float:
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    return float(area / max(1, image_shape[0] * image_shape[1]))


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
        # Uploaded masks often arrive as ordinary RGB images even when they are
        # semantically binary. Collapse them here so the experiment can stay
        # image-first instead of depending on ComfyUI MASK-specific wiring.
        out = out[:, :, :3].mean(axis=2)
    if out.ndim != 2:
        raise ValueError(f"expected HW mask-like array, got shape {list(out.shape)}")
    return np.clip(out, 0.0, 1.0)


def _exact_color_mask(arr: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    target = np.asarray(color, dtype=np.float32)
    return np.all(np.abs(arr - target[None, None, :]) <= _POSE_COLOR_TOL, axis=2)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components: list[dict[str, Any]] = []
    comp_id = 0
    ys, xs = np.where(mask)

    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if labels[start_y, start_x] != 0:
            continue

        comp_id += 1
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        labels[start_y, start_x] = comp_id
        comp_ys: list[int] = []
        comp_xs: list[int] = []

        while queue:
            cy, cx = queue.popleft()
            comp_ys.append(cy)
            comp_xs.append(cx)

            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if not mask[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = comp_id
                    queue.append((ny, nx))

        ys_arr = np.asarray(comp_ys, dtype=np.int32)
        xs_arr = np.asarray(comp_xs, dtype=np.int32)
        components.append(
            {
                "component_id": comp_id,
                "pixel_count": int(len(comp_ys)),
                "bbox": _bbox_from_coords(xs_arr, ys_arr),
                "centroid": _centroid_from_coords(xs_arr, ys_arr),
                "ys": ys_arr,
                "xs": xs_arr,
            }
        )

    return labels, components


def _bbox_from_coords(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _centroid_from_coords(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    return [float(xs.mean()), float(ys.mean())]


def _extract_joint_blobs(
    pose: np.ndarray,
    pose_labels: np.ndarray,
    total_pixels: int,
    *,
    main_subject_mask: np.ndarray,
    densepose_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    blobs: list[dict[str, Any]] = []
    for joint_tag, color in POSE_JOINT_COLORS.items():
        joint_mask = _exact_color_mask(pose, color)
        _, joint_components = _label_components(joint_mask)
        for idx, comp in enumerate(joint_components):
            owner_component_id = _majority_label(
                pose_labels[comp["ys"], comp["xs"]]
            )
            blobs.append(
                {
                    "blob_id": f"{joint_tag}_{idx}",
                    "joint_tag": joint_tag,
                    "owner_component_id": owner_component_id,
                    "pixel_count": int(comp["pixel_count"]),
                    "area_fraction": round(float(comp["pixel_count"] / total_pixels), 6),
                    "bbox": comp["bbox"],
                    "centroid": comp["centroid"],
                    "main_subject_overlap": _mask_overlap_from_coords(
                        main_subject_mask,
                        comp["ys"],
                        comp["xs"],
                    ),
                    "densepose_overlap": _mask_overlap_from_coords(
                        densepose_mask,
                        comp["ys"],
                        comp["xs"],
                    ),
                    "_ys": comp["ys"],
                    "_xs": comp["xs"],
                }
            )
    return blobs


def _majority_label(values: np.ndarray) -> int | None:
    vals = values[values > 0]
    if vals.size == 0:
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _build_candidates(
    pose_components: list[dict[str, Any]],
    joint_blobs: list[dict[str, Any]],
    main_subject_mask: np.ndarray,
    *,
    densepose_mask: np.ndarray | None,
    face_analysis: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blobs_by_component: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for blob in joint_blobs:
        owner = blob.get("owner_component_id")
        if owner is None:
            continue
        blobs_by_component.setdefault(owner, {}).setdefault(blob["joint_tag"], []).append(blob)

    component_summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    height, width = image_shape
    total_pixels = max(1, height * width)

    for comp in pose_components:
        comp_id = int(comp["component_id"])
        joint_map = blobs_by_component.get(comp_id, {})
        recognized_joint_tags = sorted(joint_map.keys())
        recognized_joint_count = len(recognized_joint_tags)
        core_joint_count = sum(1 for tag in CORE_JOINTS if tag in joint_map)
        main_overlap = _mask_overlap_from_coords(main_subject_mask, comp["ys"], comp["xs"])
        dense_overlap = (
            _mask_overlap_from_coords(densepose_mask, comp["ys"], comp["xs"])
            if densepose_mask is not None
            else None
        )
        face_support = _candidate_face_support(comp["bbox"], face_analysis)

        selection_eligible, ignore_reason = _candidate_eligibility(
            pixel_count=int(comp["pixel_count"]),
            recognized_joint_count=recognized_joint_count,
            core_joint_count=core_joint_count,
            main_overlap=main_overlap,
            dense_overlap=dense_overlap,
        )

        component_summary = {
            "component_id": comp_id,
            "pixel_count": int(comp["pixel_count"]),
            "bbox": comp["bbox"],
            "centroid": comp["centroid"],
            "recognized_joint_count": recognized_joint_count,
            "selection_eligible": selection_eligible,
        }
        if not selection_eligible and ignore_reason is not None:
            component_summary["ignore_reason"] = ignore_reason
        component_summaries.append(component_summary)

        if not selection_eligible:
            continue

        flags: list[str] = []
        joints: dict[str, dict[str, Any]] = {}
        duplicate_groups = 0
        for joint_tag in POSE_JOINT_COLORS:
            blobs = joint_map.get(joint_tag, [])
            if not blobs:
                joints[joint_tag] = {
                    "x": None,
                    "y": None,
                    "confidence_band": "none",
                    "status": "missing",
                }
                continue

            if len(blobs) == 1:
                blob = blobs[0]
                joints[joint_tag] = {
                    "x": float(blob["centroid"][0]),
                    "y": float(blob["centroid"][1]),
                    "confidence_band": "high",
                    "status": "ok",
                }
                continue

            duplicate_groups += 1
            blob = _select_joint_blob(blobs)
            joints[joint_tag] = {
                "x": float(blob["centroid"][0]),
                "y": float(blob["centroid"][1]),
                "confidence_band": "low",
                "status": "duplicate",
            }

        if duplicate_groups > 0:
            flags.append("duplicate_joint_instances")
        if recognized_joint_count < 4 or core_joint_count < 2:
            flags.append("pose_fragmented")
        if core_joint_count < len(CORE_JOINTS):
            flags.append("missing_core_joints")
        support_summary = _annotate_joint_support(
            joints,
            main_subject_mask=main_subject_mask,
            densepose_mask=densepose_mask,
        )
        if support_summary["outside_main_subject_joint_count"] > 0:
            flags.append("joints_outside_main_subject")
        if support_summary["outside_densepose_joint_count"] > 0:
            flags.append("joints_outside_densepose")

        display_bbox = _candidate_display_bbox(
            ys=comp["ys"],
            xs=comp["xs"],
            main_subject_mask=main_subject_mask,
            densepose_mask=densepose_mask,
            main_overlap=main_overlap,
            dense_overlap=dense_overlap,
            fallback_bbox=comp["bbox"],
        )

        candidates.append(
            {
                "candidate_id": f"P{len(candidates)}",
                "source_component_id": comp_id,
                "bbox": display_bbox,
                "component_bbox": comp["bbox"],
                "centroid": comp["centroid"],
                "pixel_count": int(comp["pixel_count"]),
                "area_fraction": round(float(comp["pixel_count"] / total_pixels), 6),
                "recognized_joint_tags": recognized_joint_tags,
                "recognized_joint_count": recognized_joint_count,
                "core_joint_count": core_joint_count,
                "duplicate_joint_group_count": duplicate_groups,
                "main_subject_overlap": main_overlap,
                "densepose_overlap": dense_overlap,
                "face_support": face_support,
                "flags": sorted(set(flags)),
                "joints": joints,
                **support_summary,
                "selection_features": {
                    "recognized_joint_norm": round(recognized_joint_count / max(1, len(POSE_JOINT_COLORS)), 4),
                    "core_joint_norm": round(core_joint_count / max(1, len(CORE_JOINTS)), 4),
                    "area_score": round(min((comp["pixel_count"] / total_pixels) / 0.18, 1.0), 4),
                    "main_subject_overlap": main_overlap,
                    "densepose_overlap": dense_overlap,
                    "face_support": face_support,
                },
                "_ys": comp["ys"],
                "_xs": comp["xs"],
            }
        )

    return component_summaries, candidates


def _derive_densepose_human_mask(densepose_arr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    densepose = _to_hwc_float(densepose_arr)
    densepose_u8 = (densepose * 255.0).round().astype(np.uint8)
    border = np.concatenate(
        [
            densepose_u8[0, :, :],
            densepose_u8[-1, :, :],
            densepose_u8[:, 0, :],
            densepose_u8[:, -1, :],
        ],
        axis=0,
    )
    colors, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    bg_u8 = colors[np.argmax(counts)]
    dist = np.max(np.abs(densepose_u8.astype(np.int16) - bg_u8[None, None, :].astype(np.int16)), axis=2)
    human_mask = dist >= _DENSEPOSE_BG_THRESH_U8
    stats = _mask_stats(human_mask)
    stats["status"] = "ok"
    stats["background_color_rgb"] = [int(v) for v in bg_u8.tolist()]
    stats["threshold_u8"] = _DENSEPOSE_BG_THRESH_U8
    return human_mask, stats


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    total = int(mask.size)
    count = int(ys.size)
    out: dict[str, Any] = {
        "pixel_count": count,
        "coverage": round(float(count / max(1, total)), 6),
    }
    if count:
        out["bbox"] = _bbox_from_coords(xs, ys)
        out["centroid"] = _centroid_from_coords(xs, ys)
    else:
        out["bbox"] = None
        out["centroid"] = None
    return out


def _mask_overlap_from_coords(mask: np.ndarray | None, ys: np.ndarray, xs: np.ndarray) -> float | None:
    if mask is None or ys.size == 0:
        return None
    return round(float(np.mean(mask[ys, xs])), 6)


def _point_mask_support(
    mask: np.ndarray | None,
    x: float | None,
    y: float | None,
    *,
    radius: int = _JOINT_SUPPORT_RADIUS,
) -> float | None:
    if mask is None or x is None or y is None:
        return None
    height, width = mask.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return None
    y0 = max(0, yi - radius)
    y1 = min(height, yi + radius + 1)
    x0 = max(0, xi - radius)
    x1 = min(width, xi + radius + 1)
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return round(float(np.mean(patch)), 6)


def _annotate_joint_support(
    joints: dict[str, dict[str, Any]],
    *,
    main_subject_mask: np.ndarray,
    densepose_mask: np.ndarray | None,
) -> dict[str, Any]:
    outside_main: list[str] = []
    outside_main_core: list[str] = []
    outside_dense: list[str] = []
    missing: list[str] = []
    present = 0
    inside_main = 0
    inside_dense = 0

    for joint_tag, joint in joints.items():
        x = joint.get("x")
        y = joint.get("y")
        if x is None or y is None:
            joint["main_subject_support"] = None
            joint["inside_main_subject"] = None
            joint["densepose_support"] = None
            joint["inside_densepose"] = None
            missing.append(joint_tag)
            continue

        present += 1
        main_support = _point_mask_support(main_subject_mask, x, y)
        dense_support = _point_mask_support(densepose_mask, x, y) if densepose_mask is not None else None
        inside_main_subject = bool(main_support is not None and main_support >= 0.5)
        inside_densepose = None if dense_support is None else bool(dense_support >= 0.5)

        joint["main_subject_support"] = main_support
        joint["inside_main_subject"] = inside_main_subject
        joint["densepose_support"] = dense_support
        joint["inside_densepose"] = inside_densepose

        if inside_main_subject:
            inside_main += 1
        else:
            outside_main.append(joint_tag)
            if joint_tag in CORE_JOINTS:
                outside_main_core.append(joint_tag)

        if inside_densepose is True:
            inside_dense += 1
        elif inside_densepose is False:
            outside_dense.append(joint_tag)

    return {
        "present_joint_count": int(present),
        "missing_joints": missing,
        "missing_joint_count": int(len(missing)),
        "inside_main_subject_joint_count": int(inside_main),
        "outside_main_subject_joint_count": int(len(outside_main)),
        "outside_main_subject_joints": outside_main,
        "outside_main_subject_core_joints": outside_main_core,
        "inside_densepose_joint_count": int(inside_dense),
        "outside_densepose_joint_count": int(len(outside_dense)),
        "outside_densepose_joints": outside_dense,
    }


def _candidate_eligibility(
    *,
    pixel_count: int,
    recognized_joint_count: int,
    core_joint_count: int,
    main_overlap: float | None,
    dense_overlap: float | None,
) -> tuple[bool, str | None]:
    support_overlap = max(float(main_overlap or 0.0), float(dense_overlap or 0.0))
    if pixel_count < 24 and recognized_joint_count < 2:
        return False, "tiny_fragment"
    if recognized_joint_count < 2:
        return False, "insufficient_joints"
    if core_joint_count == 0 and recognized_joint_count < 4:
        return False, "insufficient_structure"
    if support_overlap < 0.02 and recognized_joint_count < 3:
        return False, "unsupported_fragment"
    return True, None


def _select_joint_blob(blobs: list[dict[str, Any]]) -> dict[str, Any]:
    max_pixels = max(float(blob.get("pixel_count") or 0.0) for blob in blobs)

    def _score(blob: dict[str, Any]) -> tuple[float, float]:
        subject = float(blob.get("main_subject_overlap") or 0.0)
        dense = float(blob.get("densepose_overlap") or 0.0)
        pixel_score = float(blob.get("pixel_count") or 0.0) / max(1.0, max_pixels)
        support_score = (1.35 * subject) + (0.95 * dense) + (0.10 * pixel_score)
        return (support_score, float(blob.get("pixel_count") or 0.0))

    return max(blobs, key=_score)


def _candidate_display_bbox(
    *,
    ys: np.ndarray,
    xs: np.ndarray,
    main_subject_mask: np.ndarray,
    densepose_mask: np.ndarray | None,
    main_overlap: float | None,
    dense_overlap: float | None,
    fallback_bbox: list[int],
) -> list[int]:
    if float(main_overlap or 0.0) >= 0.08:
        subject_pixels = main_subject_mask[ys, xs]
        if np.any(subject_pixels):
            return _bbox_from_coords(xs[subject_pixels], ys[subject_pixels])
    if densepose_mask is not None and float(dense_overlap or 0.0) >= 0.08:
        dense_pixels = densepose_mask[ys, xs]
        if np.any(dense_pixels):
            return _bbox_from_coords(xs[dense_pixels], ys[dense_pixels])
    return fallback_bbox


def _candidate_face_support(candidate_bbox: list[int], face_analysis: dict[str, Any] | None) -> float | None:
    summary = _summarize_face_analysis(face_analysis)
    if summary.get("status") != "ok":
        return None
    scores: list[float] = []
    x0, y0, x1, y1 = [float(v) for v in candidate_bbox]
    for face in summary.get("faces", []):
        bbox = face.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        fx0, fy0, fx1, fy1 = [float(v) for v in bbox]
        cx = (fx0 + fx1) / 2.0
        cy = (fy0 + fy1) / 2.0
        center_inside = float(x0 <= cx <= x1 and y0 <= cy <= y1)
        inter_w = max(0.0, min(x1, fx1) - max(x0, fx0))
        inter_h = max(0.0, min(y1, fy1) - max(y0, fy0))
        face_area = max(1e-6, (fx1 - fx0) * (fy1 - fy0))
        overlap = (inter_w * inter_h) / face_area
        scores.append(max(center_inside, overlap))
    if not scores:
        return 0.0
    return round(float(max(scores)), 6)


def _summarize_face_analysis(face_analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(face_analysis, dict):
        return {"status": "not_available"}

    status = str(face_analysis.get("status") or "not_available")
    if status != "ok":
        return {"status": status, "face_count": int(face_analysis.get("face_count") or 0)}

    out_faces: list[dict[str, Any]] = []
    for face in face_analysis.get("faces", []) or []:
        out_faces.append(
            {
                "bbox": [float(v) for v in (face.get("bbox") or [])[:4]],
                "det_score": float(face.get("det_score") or 0.0),
                "pose": [float(v) for v in (face.get("pose") or [])[:3]],
            }
        )
    return {
        "status": "ok",
        "face_count": int(face_analysis.get("face_count") or len(out_faces)),
        "faces": out_faces,
    }


def _global_scene_flags(
    candidates: list[dict[str, Any]],
    *,
    face_analysis: dict[str, Any] | None,
    densepose_present: bool,
) -> list[str]:
    flags: set[str] = set()
    if not candidates:
        flags.add("no_pose_components")
        return sorted(flags)
    if len(candidates) > 1:
        flags.add("multiple_candidates")
    if not densepose_present:
        flags.add("densepose_missing")
    if max((c["main_subject_overlap"] or 0.0) for c in candidates) < 0.10:
        flags.add("low_main_subject_support")
    if any("pose_fragmented" in (c.get("flags") or []) for c in candidates):
        flags.add("pose_fragmented")
    if any("duplicate_joint_instances" in (c.get("flags") or []) for c in candidates):
        flags.add("duplicate_joint_instances")

    face_summary = _summarize_face_analysis(face_analysis)
    if face_summary.get("status") == "ok" and int(face_summary.get("face_count") or 0) == 0:
        flags.add("no_face_detected")

    return sorted(flags)


def _evaluate_strategies(
    candidates: list[dict[str, Any]],
    *,
    global_scene_flags: list[str],
    densepose_present: bool,
) -> dict[str, dict[str, Any]]:
    results = {
        "pose_cc_only": _evaluate_strategy(
            "pose_cc_only",
            candidates,
            global_scene_flags=global_scene_flags,
            densepose_present=densepose_present,
        ),
        "pose_plus_subject": _evaluate_strategy(
            "pose_plus_subject",
            candidates,
            global_scene_flags=global_scene_flags,
            densepose_present=densepose_present,
        ),
        "pose_plus_subject_dense": _evaluate_strategy(
            "pose_plus_subject_dense",
            candidates,
            global_scene_flags=global_scene_flags,
            densepose_present=densepose_present,
        ),
        "strict_subject_dense": _evaluate_strategy(
            "strict_subject_dense",
            candidates,
            global_scene_flags=global_scene_flags,
            densepose_present=densepose_present,
        ),
    }
    return results


def _evaluate_strategy(
    strategy_name: str,
    candidates: list[dict[str, Any]],
    *,
    global_scene_flags: list[str],
    densepose_present: bool,
) -> dict[str, Any]:
    if not candidates:
        return {
            "primary_candidate_id": None,
            "candidate_scores": [],
            "selection_margin": None,
            "confidence_band": "none",
            "confidence_score": 0.0,
            "scene_flags": sorted(set(global_scene_flags) | {"no_pose_components"}),
            "selection_reason": "No eligible pose candidates were available.",
        }

    score_rows = []
    for candidate in candidates:
        features = candidate.get("selection_features") or {}
        recognized = float(features.get("recognized_joint_norm") or 0.0)
        core = float(features.get("core_joint_norm") or 0.0)
        area = float(features.get("area_score") or 0.0)
        subject = float(candidate.get("main_subject_overlap") or 0.0)
        dense = float(candidate.get("densepose_overlap") or 0.0)
        face = float(candidate.get("face_support") or 0.0)

        if strategy_name == "pose_cc_only":
            score = (
                0.50 * recognized
                + 0.30 * core
                + 0.15 * area
                + 0.05 * face
            )
        elif strategy_name == "pose_plus_subject":
            score = (
                0.45 * subject
                + 0.25 * recognized
                + 0.15 * core
                + 0.10 * area
                + 0.05 * face
            )
        elif strategy_name == "pose_plus_subject_dense":
            if densepose_present:
                score = (
                    0.35 * subject
                    + 0.30 * dense
                    + 0.15 * recognized
                    + 0.10 * core
                    + 0.05 * area
                    + 0.05 * face
                )
            else:
                score = (
                    0.55 * subject
                    + 0.20 * recognized
                    + 0.10 * core
                    + 0.10 * area
                    + 0.05 * face
                )
        else:
            if densepose_present:
                score = (
                    0.40 * subject
                    + 0.35 * dense
                    + 0.10 * recognized
                    + 0.10 * core
                    + 0.05 * face
                )
            else:
                score = (
                    0.60 * subject
                    + 0.15 * recognized
                    + 0.10 * core
                    + 0.15 * face
                )

        if "duplicate_joint_instances" in (candidate.get("flags") or []):
            score -= 0.08
        if "pose_fragmented" in (candidate.get("flags") or []):
            score -= 0.08

        score_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": round(float(max(0.0, min(score, 1.0))), 4),
                "main_subject_overlap": subject,
                "densepose_overlap": None if candidate.get("densepose_overlap") is None else round(dense, 4),
                "recognized_joint_norm": round(recognized, 4),
                "core_joint_norm": round(core, 4),
                "area_score": round(area, 4),
                "face_support": round(face, 4) if candidate.get("face_support") is not None else None,
            }
        )

    score_rows.sort(key=lambda row: row["score"], reverse=True)
    top = score_rows[0]
    second = score_rows[1] if len(score_rows) > 1 else None
    margin = top["score"] - (second["score"] if second else 0.0)

    scene_flags = set(global_scene_flags)
    if strategy_name in {"pose_plus_subject_dense", "strict_subject_dense"} and not densepose_present:
        scene_flags.add("densepose_missing")

    primary_candidate_id: str | None = top["candidate_id"]
    if strategy_name == "strict_subject_dense":
        subject_ok = top["main_subject_overlap"] >= 0.15
        dense_ok = (top["densepose_overlap"] or 0.0) >= 0.10 if densepose_present else top["main_subject_overlap"] >= 0.35
        if top["score"] < 0.55 or margin < 0.10 or not subject_ok or not dense_ok:
            primary_candidate_id = None
    else:
        if len(score_rows) > 1 and margin < 0.05:
            primary_candidate_id = None
        elif top["score"] < 0.25:
            primary_candidate_id = None

    if primary_candidate_id is None:
        scene_flags.add("ambiguous_primary")
        return {
            "primary_candidate_id": None,
            "candidate_scores": score_rows,
            "selection_margin": round(float(margin), 4),
            "confidence_band": "none",
            "confidence_score": 0.0,
            "scene_flags": sorted(scene_flags),
            "selection_reason": (
                f"Abstained: top candidate {top['candidate_id']} score={top['score']:.2f}, "
                f"margin={margin:.2f} did not clear the strategy threshold."
            ),
        }

    candidate = next(item for item in candidates if item["candidate_id"] == primary_candidate_id)
    confidence_band = _confidence_band(top, margin, candidate)
    confidence_score = {"high": 0.9, "medium": 0.6, "low": 0.3, "none": 0.0}[confidence_band]

    reason_bits = [f"selected {primary_candidate_id}"]
    if strategy_name != "pose_cc_only":
        reason_bits.append(f"main_subject={top['main_subject_overlap']:.2f}")
    if strategy_name in {"pose_plus_subject_dense", "strict_subject_dense"} and top["densepose_overlap"] is not None:
        reason_bits.append(f"densepose={top['densepose_overlap']:.2f}")
    reason_bits.append(f"margin={margin:.2f}")

    return {
        "primary_candidate_id": primary_candidate_id,
        "candidate_scores": score_rows,
        "selection_margin": round(float(margin), 4),
        "confidence_band": confidence_band,
        "confidence_score": confidence_score,
        "scene_flags": sorted(scene_flags),
        "selection_reason": ", ".join(reason_bits),
    }


def _confidence_band(top: dict[str, Any], margin: float, candidate: dict[str, Any]) -> str:
    strong_support = max(
        float(top.get("main_subject_overlap") or 0.0),
        float(top.get("densepose_overlap") or 0.0),
    )
    if (
        top["score"] >= 0.75
        and margin >= 0.18
        and (strong_support >= 0.25 or len(candidate.get("recognized_joint_tags") or []) >= 6)
        and candidate.get("core_joint_count", 0) >= 2
    ):
        band = "high"
    elif top["score"] >= 0.50 and margin >= 0.08 and candidate.get("recognized_joint_count", 0) >= 3:
        band = "medium"
    else:
        band = "low"

    penalties = 0
    flags = set(candidate.get("flags") or [])
    if "duplicate_joint_instances" in flags:
        penalties += 1
    if "pose_fragmented" in flags:
        penalties += 1
    if penalties == 0:
        return band
    ordered = ["none", "low", "medium", "high"]
    band_index = ordered.index(band)
    return ordered[max(0, band_index - penalties)]


def _recommended_display_strategy(strategy_results: dict[str, dict[str, Any]]) -> str | None:
    dense_first = strategy_results.get("pose_plus_subject_dense") or {}
    if dense_first.get("primary_candidate_id") is not None:
        return "pose_plus_subject_dense"
    subject_next = strategy_results.get("pose_plus_subject") or {}
    if subject_next.get("primary_candidate_id") is not None:
        return "pose_plus_subject"
    return None
