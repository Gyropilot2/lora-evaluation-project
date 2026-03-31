"""
Canonical per-sample interpretation procedures.

This module is the contracts-layer home for small executable rules that operate
on one sample's already-derived facts/metrics. It does not aggregate across
samples and it does not decide which metrics a consumer promotes.

Its job is to answer "what do we do with these metric/fact inputs at sample
time?" for settled review-layer rules such as gating, dropped-state helpers,
and attribution helpers.

Important temporary note:
    The dropped-state helpers below intentionally preserve the current
    pre-§8.22 pose-drop behavior (`pose_dropped = pose score is None`). The
    fuller attribution matrix remains exposed here as a callable for later use,
    but it is not yet wired in as the live drop rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass(frozen=True)
class ProcedureSpec:
    key: str
    description: str
    fn: Callable[..., Any]
    applies_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class IdentityGateResult:
    score: float | None
    pre_gate_score: float | None
    usable: bool
    gate_status: str


def loss_from_baseline_count(
    baseline_count: int | float | None,
    lora_count: int | float | None,
) -> bool | None:
    """Return True when baseline had signal and LoRA lost it, else False/None."""
    if baseline_count is None or baseline_count < 1:
        return None
    return lora_count is None or lora_count == 0


def gate_identity_score(
    score: float | None,
    lora_face_count: int | float | None,
) -> IdentityGateResult:
    """Apply the current face-anchor gate while preserving visibility into what happened."""
    gate_status = "open" if lora_face_count else "gated_no_lora_face"
    gated_score = score if gate_status == "open" else None
    return IdentityGateResult(
        score=gated_score,
        pre_gate_score=score,
        usable=gated_score is not None,
        gate_status=gate_status,
    )


def face_detection_lost(
    baseline_face_count: int | float | None,
    lora_face_count: int | float | None,
) -> bool | None:
    """Current face-loss rule used by the review layer."""
    return loss_from_baseline_count(baseline_face_count, lora_face_count)


def pose_dropped(
    pose_score: float | None,
) -> bool:
    """Current pose-drop rule.

    Conservative extraction: this intentionally matches the current
    `review_builder.py` behavior, where `pose_dropped` means the pose score is
    absent, regardless of whether the deeper attribution is pose failure or
    subject collapse.
    """
    return pose_score is None


POSE_SOURCE_KEYS: dict[str, str] = {
    "openpose": "openpose_body",
    "dw": "dw_body",
}

POSE_ANGLE_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    ("l_shoulder", "l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow", "r_wrist"),
    ("l_hip", "l_knee", "l_ankle"),
    ("r_hip", "r_knee", "r_ankle"),
    ("l_elbow", "l_shoulder", "l_hip"),
    ("r_elbow", "r_shoulder", "r_hip"),
    ("l_shoulder", "l_hip", "l_knee"),
    ("r_shoulder", "r_hip", "r_knee"),
)


def _person0_joints(source: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    if not isinstance(source, dict):
        return None
    people = source.get("people") or []
    if not people or not isinstance(people[0], dict):
        return None
    joints = people[0].get("joints") or {}
    return joints if isinstance(joints, dict) else None


def _joint_present(joint: dict[str, Any] | None) -> bool:
    if not isinstance(joint, dict) or joint.get("status") == "missing":
        return False
    x = joint.get("x")
    y = joint.get("y")
    return isinstance(x, (int, float)) and isinstance(y, (int, float))


def _joint_angle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float | None:
    """Angle at joint B in the triplet A-B-C, in degrees."""
    dx1, dy1 = ax - bx, ay - by
    dx2, dy2 = cx - bx, cy - by
    mag1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
    mag2 = math.sqrt(dx2 ** 2 + dy2 ** 2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    cross = dx1 * dy2 - dy1 * dx2
    dot = dx1 * dx2 + dy1 * dy2
    return math.degrees(math.atan2(cross, dot))


def _triplet_angle(
    joints: dict[str, dict[str, Any]] | None,
    triplet: tuple[str, str, str],
) -> float | None:
    if not isinstance(joints, dict):
        return None
    ja = joints.get(triplet[0])
    jb = joints.get(triplet[1])
    jc = joints.get(triplet[2])
    if not (_joint_present(ja) and _joint_present(jb) and _joint_present(jc)):
        return None
    return _joint_angle(ja["x"], ja["y"], jb["x"], jb["y"], jc["x"], jc["y"])


def _present_joint_names(joints: dict[str, dict[str, Any]] | None) -> set[str]:
    if not isinstance(joints, dict):
        return set()
    return {name for name, joint in joints.items() if _joint_present(joint)}


def _triplet_count(joints: dict[str, dict[str, Any]] | None) -> int | None:
    if not isinstance(joints, dict):
        return None
    return sum(1 for triplet in POSE_ANGLE_TRIPLETS if _triplet_angle(joints, triplet) is not None)


def pose_source_metrics(
    baseline_source: dict[str, Any] | None,
    lora_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one source's explicit person0/joint/triplet pose facts."""
    baseline_joints_map = _person0_joints(baseline_source)
    lora_joints_map = _person0_joints(lora_source)

    baseline_has_person0 = baseline_joints_map is not None
    lora_has_person0 = lora_joints_map is not None
    comparable_person0 = baseline_has_person0 and lora_has_person0

    baseline_joint_names = _present_joint_names(baseline_joints_map)
    lora_joint_names = _present_joint_names(lora_joints_map)
    comparable_joint_names = (
        baseline_joint_names & lora_joint_names
        if comparable_person0 else set()
    )

    baseline_triplets = _triplet_count(baseline_joints_map) if baseline_has_person0 else None
    lora_triplets = _triplet_count(lora_joints_map) if lora_has_person0 else None

    angle_deltas: list[float] = []
    comparable_triplets = None
    if comparable_person0:
        for triplet in POSE_ANGLE_TRIPLETS:
            baseline_angle = _triplet_angle(baseline_joints_map, triplet)
            lora_angle = _triplet_angle(lora_joints_map, triplet)
            if baseline_angle is None or lora_angle is None:
                continue
            angle_deltas.append(abs(((lora_angle - baseline_angle) + 180.0) % 360.0 - 180.0))
        comparable_triplets = len(angle_deltas)

    baseline_joints = len(baseline_joint_names) if baseline_has_person0 else None
    lora_joints = len(lora_joint_names) if lora_has_person0 else None
    comparable_joints = len(comparable_joint_names) if comparable_person0 else None

    reliability = None
    if baseline_joints not in (None, 0) and comparable_joints is not None:
        reliability = round(comparable_joints / baseline_joints, 4)

    angle_drift = round(sum(angle_deltas) / len(angle_deltas), 4) if angle_deltas else None

    return {
        "baseline_has_person0": baseline_has_person0,
        "lora_has_person0": lora_has_person0,
        "comparable_person0": comparable_person0,
        "baseline_joints": baseline_joints,
        "lora_joints": lora_joints,
        "comparable_joints": comparable_joints,
        "baseline_triplets": baseline_triplets,
        "lora_triplets": lora_triplets,
        "comparable_triplets": comparable_triplets,
        "angle_drift": angle_drift,
        "reliability": reliability,
        "dropped": pose_dropped(angle_drift),
    }


def select_pose_source(source_metrics: dict[str, dict[str, Any]]) -> str | None:
    """Pick the winning source by reliability first, lower drift as tie-break."""
    best_source = None
    best_reliability = None
    best_drift = None
    for source_name in ("openpose", "dw"):
        metrics = source_metrics.get(source_name) or {}
        if metrics.get("dropped"):
            continue
        reliability = metrics.get("reliability")
        drift = metrics.get("angle_drift")
        if reliability is None or drift is None:
            continue
        if (
            best_source is None
            or reliability > best_reliability
            or (reliability == best_reliability and drift < best_drift)
        ):
            best_source = source_name
            best_reliability = reliability
            best_drift = drift
    return best_source


def pose_pair_metrics(
    baseline_pose_evidence: dict[str, Any] | None,
    lora_pose_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the full paired pose family from explicit per-source facts."""
    baseline_pose = baseline_pose_evidence or {}
    lora_pose = lora_pose_evidence or {}

    per_source: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {}
    for source_name, source_key in POSE_SOURCE_KEYS.items():
        metrics = pose_source_metrics(
            baseline_pose.get(source_key),
            lora_pose.get(source_key),
        )
        per_source[source_name] = metrics
        for suffix, value in metrics.items():
            out[f"pose_{source_name}_{suffix}"] = value

    selected_source = select_pose_source(per_source)
    out["pose_selected_source"] = selected_source

    if selected_source is None:
        out["pose_angle_drift"] = None
        out["pose_reliability"] = None
    else:
        selected_metrics = per_source[selected_source]
        out["pose_angle_drift"] = selected_metrics.get("angle_drift")
        out["pose_reliability"] = selected_metrics.get("reliability")

    out["pose_dropped"] = pose_dropped(out["pose_angle_drift"])
    return out


def attribution_matrix_label(
    pose_detection_lost_value: bool | None,
    face_detection_lost_value: bool | None,
) -> str | None:
    """Classify the §8.22 pose × identity attribution cell when both axes are known."""
    if pose_detection_lost_value is None or face_detection_lost_value is None:
        return None
    if pose_detection_lost_value and face_detection_lost_value:
        return "subject_collapse"
    if pose_detection_lost_value:
        return "pose_failure"
    if face_detection_lost_value:
        return "identity_failure"
    return "both_ok"


PROCEDURES_REGISTRY: dict[str, ProcedureSpec] = {
    "identity_score_gate": ProcedureSpec(
        key="identity_score_gate",
        description=(
            "Apply the current face-anchor gate to identity package scores while also exposing "
            "the pre-gate value, usable flag, and gate status."
        ),
        applies_to=("identity_region_drift_exp", "identity_region_plus_arcface_exp"),
        fn=gate_identity_score,
    ),
    "pose_dropped": ProcedureSpec(
        key="pose_dropped",
        description="Current drop flag for pose hero metrics; True when the pose score is absent.",
        applies_to=("pose_angle_drift",),
        fn=pose_dropped,
    ),
    "face_detection_lost": ProcedureSpec(
        key="face_detection_lost",
        description="True when baseline had at least one face and the LoRA side has none.",
        applies_to=("face_detection_lost",),
        fn=face_detection_lost,
    ),
    "attribution_matrix_label": ProcedureSpec(
        key="attribution_matrix_label",
        description="§8.22 cross-reference label for pose/identity loss: both_ok, pose_failure, identity_failure, or subject_collapse.",
        fn=attribution_matrix_label,
    ),
}


def get_procedure(key: str) -> ProcedureSpec | None:
    """Return one registered procedure by key."""
    return PROCEDURES_REGISTRY.get(key)


def all_procedures() -> dict[str, ProcedureSpec]:
    """Return a shallow copy of the procedures registry."""
    return dict(PROCEDURES_REGISTRY)
