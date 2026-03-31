"""Structured pose_evidence builders for production storage."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from extractor.pose_scene import analyze_pose_scene_keypoints


def build_pose_evidence(
    image_arr: np.ndarray,
    main_subject_mask_arr: np.ndarray,
    *,
    openpose_keypoint_data: Any = None,
    dw_keypoint_data: Any = None,
    densepose_arr: np.ndarray | None = None,
    face_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the additive pose_evidence domain for any provided grouped sources."""
    domain: dict[str, Any] = {}

    if openpose_keypoint_data is not None:
        domain["openpose_body"] = build_pose_evidence_source(
            image_arr=image_arr,
            pose_keypoint_data=openpose_keypoint_data,
            source_key="openpose_body",
            processor="openpose",
            mode="body_only",
            main_subject_mask_arr=main_subject_mask_arr,
            densepose_arr=densepose_arr,
            face_analysis=face_analysis,
        )

    if dw_keypoint_data is not None:
        domain["dw_body"] = build_pose_evidence_source(
            image_arr=image_arr,
            pose_keypoint_data=dw_keypoint_data,
            source_key="dw_body",
            processor="dwpose",
            mode="body_only",
            main_subject_mask_arr=main_subject_mask_arr,
            densepose_arr=densepose_arr,
            face_analysis=face_analysis,
        )

    return domain


def build_pose_evidence_source(
    *,
    image_arr: np.ndarray,
    pose_keypoint_data: Any,
    source_key: str,
    processor: str,
    mode: str,
    main_subject_mask_arr: np.ndarray,
    densepose_arr: np.ndarray | None = None,
    face_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one fixed-key pose source object for durable storage."""
    payload = _normalize_pose_keypoint_payload(
        pose_keypoint_data,
        image_shape=(int(image_arr.shape[0]), int(image_arr.shape[1])),
    )
    scene = analyze_pose_scene_keypoints(
        image_arr=image_arr,
        pose_keypoint_data=payload,
        main_subject_mask_arr=main_subject_mask_arr,
        densepose_arr=densepose_arr,
        face_analysis=face_analysis,
    )
    people = [_candidate_to_person(candidate) for candidate in (scene.get("candidates") or [])]
    pose_obs = scene.get("pose_observations") or {}

    return {
        "processor": processor,
        "mode": mode,
        "source_kind": "structured_keypoints",
        "coordinate_space": _payload_coordinate_space(payload),
        "canvas_width": int(payload["canvas_width"]),
        "canvas_height": int(payload["canvas_height"]),
        "raw_people_count": int(len(payload["people"])),
        "eligible_candidate_count": int(pose_obs.get("eligible_candidate_count") or len(people)),
        "scene_flags": list(pose_obs.get("scene_flags") or []),
        "raw_observation": {
            "people": [_json_safe(person) for person in payload["people"]],
        },
        "people": people,
    }


def _candidate_to_person(candidate: dict[str, Any]) -> dict[str, Any]:
    joints = {
        joint_name: {
            "x": _json_safe(joint.get("x")),
            "y": _json_safe(joint.get("y")),
            "status": joint.get("status"),
            "confidence_band": joint.get("confidence_band"),
            "source_confidence": _json_safe(joint.get("source_confidence")),
            "main_subject_support": _json_safe(joint.get("main_subject_support")),
            "inside_main_subject": joint.get("inside_main_subject"),
            "densepose_support": _json_safe(joint.get("densepose_support")),
            "inside_densepose": joint.get("inside_densepose"),
        }
        for joint_name, joint in (candidate.get("joints") or {}).items()
        if isinstance(joint, dict)
    }

    return {
        "person_id": str(candidate.get("candidate_id") or ""),
        "person_index": _json_safe(candidate.get("source_person_index", candidate.get("source_component_id"))),
        "bbox": _json_safe(candidate.get("bbox")),
        "area_fraction": _json_safe(candidate.get("area_fraction")),
        "recognized_joint_tags": list(candidate.get("recognized_joint_tags") or []),
        "recognized_joint_count": int(candidate.get("recognized_joint_count") or 0),
        "core_joint_count": int(candidate.get("core_joint_count") or 0),
        "present_joint_count": int(candidate.get("present_joint_count") or 0),
        "main_subject_overlap": _json_safe(candidate.get("main_subject_overlap")),
        "densepose_overlap": _json_safe(candidate.get("densepose_overlap")),
        "face_support": _json_safe(candidate.get("face_support")),
        "missing_joints": list(candidate.get("missing_joints") or []),
        "outside_main_subject_joints": list(candidate.get("outside_main_subject_joints") or []),
        "outside_main_subject_core_joints": list(candidate.get("outside_main_subject_core_joints") or []),
        "outside_densepose_joints": list(candidate.get("outside_densepose_joints") or []),
        "flags": list(candidate.get("flags") or []),
        "joints": joints,
    }


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


def _payload_coordinate_space(payload: dict[str, Any]) -> str:
    modes: set[str] = set()
    for person in payload.get("people") or []:
        flat = _pose_keypoint_triplets(person)
        if not flat:
            continue
        modes.add(_structured_coord_mode(flat))
    if not modes:
        return "pixel"
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _pose_keypoint_triplets(person: Any) -> list[float]:
    if not isinstance(person, dict):
        return []
    values = person.get("pose_keypoints_2d")
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return []


def _structured_coord_mode(flat: list[float]) -> str:
    coords = [float(v) for i, v in enumerate(flat) if i % 3 != 2]
    nonzero = [v for v in coords if abs(v) > 1e-9]
    if nonzero and max(nonzero) <= 1.5 and min(nonzero) >= 0.0:
        return "normalized"
    return "pixel"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value
