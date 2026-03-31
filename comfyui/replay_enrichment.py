"""ComfyUI replay/backfill nodes for additive sample enrichment."""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts.validation_errors import is_invalid
from core.asset_codecs import decode_image_asset_bytes, decode_mask_asset_bytes
from core.diagnostics import emit
from databank.treasurer import Treasurer
from extractor.extract import run_replay
from extractor.pose_evidence import build_pose_evidence

_WHERE = "comfyui.replay_enrichment"
_REPLAY_SAMPLE_TYPE = "LEP_REPLAY_SAMPLE"


class SampleReplayLoad:
    """Load an existing sample from DB and expose its stored image for replay."""

    CATEGORY = "lora_eval/replay"
    FUNCTION = "load_sample"
    RETURN_TYPES = ("IMAGE", _REPLAY_SAMPLE_TYPE, "STRING")
    RETURN_NAMES = ("image", "replay_sample", "report")

    _treasurer: Any = None

    @classmethod
    def _get_treasurer(cls) -> Any:
        if cls._treasurer is None:
            from databank.treasurer import open_treasurer

            cls._treasurer = open_treasurer(read_only=True)
        return cls._treasurer

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "sample_hash": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Existing sample_hash to load from the DataBank for replay/backfill.",
                    },
                ),
            }
        }

    def load_sample(self, sample_hash: str) -> tuple[Any, dict[str, Any], str]:
        treasurer = self._get_treasurer()
        clean_hash = (sample_hash or "").strip()
        if not clean_hash:
            raise Exception("[LoRA Eval Replay] sample_hash is required.")

        sample_record = treasurer.get_sample(clean_hash)
        if sample_record is None:
            emit("ERROR", "REPLAY.SAMPLE_NOT_FOUND", _WHERE, "Replay sample not found", sample_hash=clean_hash)
            raise Exception(f"[LoRA Eval Replay] Sample not found: {clean_hash}")

        eval_hash = str(sample_record.get("eval_hash") or "")
        eval_record = treasurer.get_eval(eval_hash)
        if eval_record is None:
            emit("ERROR", "DB.READ_FAIL", _WHERE, "Replay sample is missing its parent Eval", sample_hash=clean_hash, eval_hash=eval_hash)
            raise Exception(f"[LoRA Eval Replay] Parent eval not found for sample {clean_hash}.")

        method_hash = str(eval_record.get("method_hash") or "")
        method_record = treasurer.get_method(method_hash)
        if method_record is None:
            emit("ERROR", "DB.READ_FAIL", _WHERE, "Replay sample is missing its parent Method", sample_hash=clean_hash, method_hash=method_hash)
            raise Exception(f"[LoRA Eval Replay] Parent method not found for sample {clean_hash}.")

        image_arr = _decode_sample_image(treasurer, sample_record)
        replay_bundle = {
            "sample_hash": clean_hash,
            "method_record": method_record,
            "eval_record": eval_record,
            "sample_record": sample_record,
        }
        report = "\n".join(
            [
                "sample_replay_load",
                f"sample_hash: {clean_hash}",
                f"eval_hash: {eval_hash}",
                f"method_hash: {method_hash}",
                f"has_pose_evidence: {bool(sample_record.get('pose_evidence'))}",
            ]
        )
        return (_hwc_float_to_image_tensor(image_arr), replay_bundle, report)


class SampleNeedsMeasurementGuard:
    """Read-only guard that halts when all requested enrichment paths already exist."""

    CATEGORY = "lora_eval/replay"
    FUNCTION = "guard_needed"
    RETURN_TYPES = (_REPLAY_SAMPLE_TYPE, "STRING")
    RETURN_NAMES = ("replay_sample", "report")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "replay_sample": (_REPLAY_SAMPLE_TYPE,),
                "requested_paths": (
                    "STRING",
                    {
                        "default": "pose_evidence.openpose_body\npose_evidence.dw_body",
                        "multiline": True,
                        "tooltip": "One dot-path per line (or comma-separated). Guard halts only when all listed paths are already present.",
                    },
                ),
            }
        }

    def guard_needed(
        self,
        replay_sample: dict[str, Any],
        requested_paths: str,
    ) -> tuple[dict[str, Any], str]:
        sample_record = dict(replay_sample.get("sample_record") or {})
        paths = _parse_requested_paths(requested_paths)
        if not paths:
            return (replay_sample, "sample_needs_measurement_guard\nrequested_paths: none")

        present = [path for path in paths if _record_has_valid_path(sample_record, path)]
        missing = [path for path in paths if path not in present]
        report = "\n".join(
            [
                "sample_needs_measurement_guard",
                f"sample_hash: {replay_sample.get('sample_hash') or sample_record.get('sample_hash') or ''}",
                f"requested: {', '.join(paths)}",
                f"present: {', '.join(present) if present else 'none'}",
                f"missing: {', '.join(missing) if missing else 'none'}",
            ]
        )

        if not missing:
            emit(
                "INFO",
                "REPLAY.REQUEST_ALREADY_PRESENT",
                _WHERE,
                "Replay guard halted because all requested enrichment paths already exist",
                sample_hash=replay_sample.get("sample_hash"),
                requested_paths=paths,
            )
            raise Exception(
                "[LoRA Eval Replay Guard] All requested enrichment paths already exist for "
                f"sample {replay_sample.get('sample_hash')}: {', '.join(paths)}"
            )

        return (replay_sample, report)


class ReplayEnricher:
    """Build additive measurement domains from replay inputs and route through Bouncer."""

    CATEGORY = "lora_eval/replay"
    FUNCTION = "enrich_sample"
    OUTPUT_NODE = True
    RETURN_TYPES = (_REPLAY_SAMPLE_TYPE, "STRING")
    RETURN_NAMES = ("replay_sample", "report")

    _treasurer: Any = None

    @classmethod
    def _get_treasurer(cls) -> Any:
        if cls._treasurer is None:
            from databank.treasurer import open_treasurer

            cls._treasurer = open_treasurer()
        return cls._treasurer

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "replay_sample": (_REPLAY_SAMPLE_TYPE,),
            },
            "optional": {
                "pose_keypoint_openpose": (
                    "POSE_KEYPOINT",
                    {"tooltip": "Grouped OpenPose body-only keypoints for additive pose_evidence."},
                ),
                "pose_keypoint_dw": (
                    "POSE_KEYPOINT",
                    {"tooltip": "Grouped DW-Pose body-only keypoints for additive pose_evidence."},
                ),
                "mask_main_subject": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional MainSubject support mask. When omitted, ReplayEnricher "
                            "falls back to the stored masks.main_subject asset."
                        )
                    },
                ),
                "aux_densepose": (
                    "IMAGE",
                    {"tooltip": "Optional DensePose support image used only for overlap support facts."},
                ),
            },
        }

    def enrich_sample(
        self,
        replay_sample: dict[str, Any],
        pose_keypoint_openpose: Any = None,
        pose_keypoint_dw: Any = None,
        mask_main_subject: Any = None,
        aux_densepose: Any = None,
    ) -> tuple[dict[str, Any], str]:
        if pose_keypoint_openpose is None and pose_keypoint_dw is None:
            raise Exception("[LoRA Eval Replay] Provide at least one grouped pose source.")

        treasurer = self._get_treasurer()
        sample_record = dict(replay_sample.get("sample_record") or {})
        image_arr = _decode_sample_image(treasurer, sample_record)
        if mask_main_subject is not None:
            main_subject_arr = tensor_image_to_hw_float(mask_main_subject)
        else:
            main_subject_arr = _decode_sample_mask(treasurer, sample_record, "main_subject")
            if main_subject_arr is None:
                emit(
                    "ERROR",
                    "REPLAY.MAIN_SUBJECT_MISSING",
                    _WHERE,
                    "Replay enrichment requires a MainSubject mask, but none was provided and the stored masks.main_subject asset is missing",
                    sample_hash=replay_sample.get("sample_hash"),
                )
                raise Exception(
                    "[LoRA Eval Replay] MainSubject mask missing. Provide mask_main_subject or ensure the stored masks.main_subject asset is present."
                )

        densepose_arr = tensor_image_to_hwc_float(aux_densepose) if aux_densepose is not None else None
        face_analysis = sample_record.get("face_analysis")
        if is_invalid(face_analysis):
            face_analysis = None

        pose_evidence = build_pose_evidence(
            image_arr=image_arr,
            main_subject_mask_arr=main_subject_arr,
            openpose_keypoint_data=pose_keypoint_openpose,
            dw_keypoint_data=pose_keypoint_dw,
            densepose_arr=densepose_arr,
            face_analysis=face_analysis if isinstance(face_analysis, dict) else None,
        )
        if not pose_evidence:
            raise Exception("[LoRA Eval Replay] No pose_evidence was produced from the provided inputs.")

        result = run_replay(replay_sample, {"pose_evidence": pose_evidence}, treasurer)
        refreshed = treasurer.get_sample(str(replay_sample.get("sample_hash") or sample_record.get("sample_hash") or ""))
        updated_bundle = {
            "sample_hash": replay_sample.get("sample_hash") or sample_record.get("sample_hash"),
            "method_record": replay_sample.get("method_record"),
            "eval_record": replay_sample.get("eval_record"),
            "sample_record": refreshed or sample_record,
        }

        report = "\n".join(
            [
                "replay_enricher",
                f"sample_hash: {updated_bundle['sample_hash']}",
                f"sources_submitted: {', '.join(sorted(pose_evidence.keys()))}",
                f"bouncer_action: {result.get('action')}",
                f"has_pose_evidence_now: {bool((updated_bundle['sample_record'] or {}).get('pose_evidence'))}",
            ]
        )
        return (updated_bundle, report)


def _decode_sample_image(treasurer: Treasurer, sample_record: dict[str, Any]) -> np.ndarray:
    image_domain = sample_record.get("image")
    if not isinstance(image_domain, dict):
        raise Exception("[LoRA Eval Replay] Stored sample has no image domain.")
    valueref = image_domain.get("output")
    if not isinstance(valueref, dict) or is_invalid(valueref):
        raise Exception("[LoRA Eval Replay] Stored sample image ValueRef is missing or invalid.")
    blob = treasurer.load_asset_blob(valueref)
    if is_invalid(blob):
        raise Exception(f"[LoRA Eval Replay] Could not read stored sample image asset: {blob}")
    try:
        return decode_image_asset_bytes(valueref, blob)
    except Exception as exc:  # noqa: BLE001
        emit(
            "ERROR",
            "REPLAY.IMAGE_DECODE_FAIL",
            _WHERE,
            "Failed to decode replay sample image asset",
            sample_hash=sample_record.get("sample_hash"),
            error=str(exc),
            fmt=valueref.get("format"),
        )
        raise Exception(f"[LoRA Eval Replay] Failed to decode stored sample image: {exc}") from exc


def _decode_sample_mask(
    treasurer: Treasurer,
    sample_record: dict[str, Any],
    mask_name: str,
) -> np.ndarray | None:
    masks = sample_record.get("masks")
    if not isinstance(masks, dict):
        return None
    mask_entry = masks.get(mask_name)
    if not isinstance(mask_entry, dict) or is_invalid(mask_entry):
        return None
    valueref = mask_entry.get("output")
    if not isinstance(valueref, dict) or is_invalid(valueref):
        return None
    blob = treasurer.load_asset_blob(valueref)
    if is_invalid(blob):
        return None
    try:
        return decode_mask_asset_bytes(valueref, blob)
    except Exception:
        return None


def _record_has_valid_path(record: dict[str, Any], path: str) -> bool:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return False
        current = current.get(part)
        if current is None:
            return False
    return not is_invalid(current)


def _parse_requested_paths(raw: str) -> list[str]:
    paths: list[str] = []
    for line in (raw or "").replace(",", "\n").splitlines():
        cleaned = line.strip()
        if cleaned:
            paths.append(cleaned)
    return paths


def tensor_image_to_hwc_float(image_tensor: Any) -> np.ndarray:
    arr = np.asarray(image_tensor[0].detach().cpu().float().numpy(), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected IMAGE tensor first batch item to be HWC, got shape {list(arr.shape)}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return np.clip(arr, 0.0, 1.0)


def tensor_image_to_hw_float(image_tensor: Any) -> np.ndarray:
    arr = tensor_image_to_hwc_float(image_tensor)
    return np.clip(arr[:, :, :3].mean(axis=2), 0.0, 1.0)


def tensor_mask_to_hw_float(mask_tensor: Any) -> np.ndarray:
    """Convert a ComfyUI MASK tensor [B, H, W] → HW float32 numpy array."""
    arr = np.asarray(mask_tensor[0].detach().cpu().float().numpy(), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"expected MASK tensor first batch item to be HW, got shape {list(arr.shape)}"
        )
    return np.clip(arr, 0.0, 1.0)


def _hwc_float_to_image_tensor(arr: np.ndarray) -> Any:
    import torch

    clean = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    return torch.from_numpy(clean).unsqueeze(0)
