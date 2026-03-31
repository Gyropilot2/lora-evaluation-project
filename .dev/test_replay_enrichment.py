""".dev/test_replay_enrichment.py

Replay/backfill enrichment smoke test for additive pose_evidence.

Run:
    python .dev/test_replay_enrichment.py
"""

from __future__ import annotations

import io
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import core.paths as _paths
from core.asset_codecs import decode_image_asset_bytes
from databank import assets as _assets
from databank.sqlite_backend import SQLiteBackend


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _mk_method() -> dict:
    return {
        "method_hash": "method-replay-001",
        "extractor_version": "test.0.1",
        "base_model": {"hash": "base-hash-001"},
        "vae_model": {"hash": "vae-hash-001"},
        "conditioning": {
            "positive_hash": "pos-hash-001",
            "negative_hash": "neg-hash-001",
        },
        "settings": {
            "steps": 20,
            "denoise": 1.0,
            "sampler": "euler",
            "scheduler": "normal",
            "cfg": 7.0,
        },
        "latent": {"width": 64, "height": 64, "shape": [1, 4, 8, 8]},
    }


def _mk_eval() -> dict:
    return {
        "eval_hash": "eval-replay-001",
        "method_hash": "method-replay-001",
        "lora": {"hash": "lora-replay-001"},
    }


def _mk_sample(image_ref: dict, mask_ref: dict) -> dict:
    return {
        "sample_hash": "sample-replay-001",
        "eval_hash": "eval-replay-001",
        "seed": 123,
        "lora_strength": 0.8,
        "latent_hash": "latent-replay-001",
        "image_hash": "image-replay-001",
        "extractor_version": "test.0.1",
        "evidence_version": "2.0.0",
        "timestamp": "2026-03-16T00:00:00Z",
        "ingest_status": "OK",
        "image": {"output": image_ref, "shape": [64, 64, 3]},
        "luminance": {"output": {"_stub": "luminance"}},
        "masks": {
            "main_subject": {
                "output": mask_ref,
            }
        },
    }


def _mk_pose_keypoint() -> list[dict]:
    flat = [0.0] * (18 * 3)

    def _set(idx: int, x: float, y: float, conf: float = 1.0) -> None:
        flat[idx * 3 + 0] = x
        flat[idx * 3 + 1] = y
        flat[idx * 3 + 2] = conf

    _set(2, 12.0, 18.0)  # r_shoulder
    _set(5, 26.0, 18.0)  # l_shoulder
    _set(8, 14.0, 36.0)  # r_hip
    _set(11, 24.0, 36.0)  # l_hip
    _set(3, 10.0, 24.0)  # r_elbow
    _set(6, 28.0, 24.0)  # l_elbow
    return [
        {
            "people": [
                {
                    "pose_keypoints_2d": flat,
                    "face_keypoints_2d": None,
                    "hand_left_keypoints_2d": None,
                    "hand_right_keypoints_2d": None,
                }
            ],
            "canvas_height": 64,
            "canvas_width": 64,
        }
    ]


def _mk_pose_keypoint_dw() -> list[dict]:
    flat = [0.0] * (18 * 3)

    def _set(idx: int, x: float, y: float, conf: float = 1.0) -> None:
        flat[idx * 3 + 0] = x
        flat[idx * 3 + 1] = y
        flat[idx * 3 + 2] = conf

    _set(2, 14.0, 19.0)
    _set(5, 27.0, 19.0)
    _set(8, 15.0, 37.0)
    _set(11, 25.0, 37.0)
    _set(3, 11.0, 24.0)
    _set(6, 29.0, 24.0)
    _set(9, 14.0, 48.0)
    _set(12, 24.0, 48.0)
    return [
        {
            "people": [
                {
                    "pose_keypoints_2d": flat,
                    "face_keypoints_2d": None,
                    "hand_left_keypoints_2d": None,
                    "hand_right_keypoints_2d": None,
                }
            ],
            "canvas_height": 64,
            "canvas_width": 64,
        }
    ]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="replay_enrichment_test_"))
    project_root = tmp / "project"
    assets_root = project_root / "assets"
    contracts_root = project_root / "contracts"
    tmp_root = project_root / "tmp"
    project_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    contracts_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    db_path = project_root / "replay.db"

    shutil.copy2(
        _PROJECT_ROOT / "contracts" / "evidence.schema.json",
        contracts_root / "evidence.schema.json",
    )

    original_get_path = _paths.get_path
    path_map = {
        "project_root": project_root,
        "assets_root": assets_root,
        "tmp_root": tmp_root,
        "db_file": db_path,
    }

    def _patched_get_path(name: str) -> Path:
        if name in path_map:
            return path_map[name]
        return original_get_path(name)

    _paths.get_path = _patched_get_path
    try:
        image_arr = np.zeros((64, 64, 3), dtype=np.float32)
        image_arr[:, :, 0] = 0.2
        image_arr[:, :, 1] = 0.4
        image_arr[:, :, 2] = 0.6
        mask_arr = np.zeros((64, 64), dtype=np.float32)
        mask_arr[10:54, 8:40] = 1.0

        image_ref = _assets.commit_blob(_assets.stage_blob(_npy_bytes(image_arr), "image", "npy"))
        mask_ref = _assets.commit_blob(_assets.stage_blob(_npy_bytes(mask_arr), "mask", "npy"))

        db = SQLiteBackend(db_path)
        db.insert_method(_mk_method())
        db.insert_eval(_mk_eval())
        db.insert_sample(_mk_sample(image_ref, mask_ref), extras={})

        # Codec must already understand future png-style image assets too.
        png_buf = io.BytesIO()
        Image.fromarray((image_arr * 255).astype(np.uint8), mode="RGB").save(png_buf, format="PNG")
        png_decoded = decode_image_asset_bytes({"format": "png"}, png_buf.getvalue())
        _assert(png_decoded.shape == (64, 64, 3), "png decode should produce HWC float image")

        replay_mod = importlib.import_module("comfyui.replay_enrichment")
        replay_mod._hwc_float_to_image_tensor = lambda arr: np.expand_dims(arr, axis=0)
        replay_mod._tensor_image_to_hwc_float = lambda arr: np.asarray(arr, dtype=np.float32)
        replay_mod._tensor_image_to_hw_float = lambda arr: np.asarray(arr, dtype=np.float32)

        replay_mod.SampleReplayLoad._treasurer = SQLiteBackend(db_path, read_only=True)
        replay_mod.ReplayEnricher._treasurer = SQLiteBackend(db_path)

        loader = replay_mod.SampleReplayLoad()
        image_tensor, replay_bundle, report = loader.load_sample("sample-replay-001")
        _assert("sample_hash: sample-replay-001" in report, "loader report should mention sample hash")
        _assert(tuple(image_tensor.shape) == (1, 64, 64, 3), "loader should return IMAGE tensor")

        guard = replay_mod.SampleNeedsMeasurementGuard()
        guarded_bundle, guard_report = guard.guard_needed(
            replay_bundle,
            "pose_evidence.openpose_body\npose_evidence.dw_body",
        )
        _assert(guarded_bundle is replay_bundle, "guard should pass through replay bundle when a path is missing")
        _assert("missing: pose_evidence.openpose_body, pose_evidence.dw_body" in guard_report, "guard should report both pose sources missing before enrichment")

        enricher = replay_mod.ReplayEnricher()
        updated_bundle, enrich_report = enricher.enrich_sample(
            replay_bundle,
            pose_keypoint_openpose=_mk_pose_keypoint(),
        )
        _assert("has_pose_evidence_now: True" in enrich_report, "enricher should report pose_evidence present")
        stored = db.get_sample("sample-replay-001")
        pose_evidence = stored.get("pose_evidence") or {}
        _assert("openpose_body" in pose_evidence, "stored sample should contain openpose_body pose_evidence")
        _assert("masks" in stored and "main_subject" in (stored.get("masks") or {}), "existing mask slots must survive replay enrich")
        _assert(bool((updated_bundle.get("sample_record") or {}).get("pose_evidence")), "updated replay bundle should include refreshed pose_evidence")

        guarded_bundle, guard_report = guard.guard_needed(
            updated_bundle,
            "pose_evidence.openpose_body\npose_evidence.dw_body",
        )
        _assert(guarded_bundle is updated_bundle, "guard should still pass through while dw_body is missing")
        _assert("present: pose_evidence.openpose_body" in guard_report, "guard should see openpose_body after first enrich")
        _assert("missing: pose_evidence.dw_body" in guard_report, "guard should report dw_body missing after first enrich")

        updated_bundle, enrich_report = enricher.enrich_sample(
            updated_bundle,
            pose_keypoint_dw=_mk_pose_keypoint_dw(),
        )
        _assert("sources_submitted: dw_body" in enrich_report, "second enrich should report dw_body submission")
        stored = db.get_sample("sample-replay-001")
        pose_evidence = stored.get("pose_evidence") or {}
        _assert("openpose_body" in pose_evidence, "second enrich must preserve existing openpose_body")
        _assert("dw_body" in pose_evidence, "second enrich should add dw_body")

        try:
            guard.guard_needed(updated_bundle, "pose_evidence.openpose_body\npose_evidence.dw_body")
        except Exception as exc:
            _assert("All requested enrichment paths already exist" in str(exc), "guard halt should explain that requested paths are already present")
        else:
            raise AssertionError("guard should halt once all requested enrichment paths exist")

        print("PASS replay enrichment smoke test")
    finally:
        _paths.get_path = original_get_path


if __name__ == "__main__":
    main()
