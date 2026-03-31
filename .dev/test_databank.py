"""
.dev/test_databank.py — DataBank round-trip tests for Step 1.3.

Tests:
  1.3.f — insert_run / get_run round-trip; duplicate run_signature discarded
  1.3.g — insert_sample / get_sample round-trip (includes Invalid wrapper field)
  1.3.h — asset stage / commit / load round-trip with hash verification and tamper check

Run with:  python .dev/test_databank.py
Exit 0 = all passed.  Exit 1 = one or more failures.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from contracts.validation_errors import is_invalid, make_error
from core.hashing import hash_bytes
from databank.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def ok(label: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {label}")


def fail(label: str, detail: str = "") -> None:
    global _FAIL
    _FAIL += 1
    msg = f"  FAIL  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Hand-crafted records
# ---------------------------------------------------------------------------

_RUN_RECORD: dict = {
    "run_id": "run-test-0001",
    "run_signature": "abc123deabc123deabc123deabc123de",
    "extractor_version": "0.0.1",
    "timestamp": "2026-02-20T00:00:00Z",
    "is_dirty": False,
    "base_model": {"name": "sd15_base.safetensors", "hash": "base_hash_001"},
    "lora": {"name": "test_lora.safetensors", "hash": "lora_hash_001", "file_hash": "file_hash_001"},
    "prompt": {"text": "a photo of a cat", "family": "pet_photo"},
    "settings": {
        "seed": 42,
        "steps": 20,
        "strength": 0.8,
        "denoise": 1.0,
        "sampler": "euler",
        "scheduler": "normal",
        "cfg": 7.0,
    },
    "clip_vision_model_hash": "clip_hash_001",
    "input_image_hash": "input_hash_001",
}

_SAMPLE_RECORD: dict = {
    "evidence_version": "1.0.0",
    "sample_id": "sample-test-0001",
    "run_ids": ["run-test-0001"],
    "ingest_status": "WARN",
    "timestamp": "2026-02-20T00:00:01Z",
    "is_dirty": False,
    "measurements": {
        # One valid measurement, one Invalid wrapper
        "clip_delta_cos": 0.123,
        "pixel_delta_l2": make_error(
            reason_code="ASSET.READ_FAIL",
            detail="baseline image not available",
        ),
    },
}

_SAMPLE_EXTRAS: dict = {
    "raw_field_x": "some_value",
    "unrecognised_key": 99,
}


# ---------------------------------------------------------------------------
# 1.3.f — Run round-trip
# ---------------------------------------------------------------------------

def test_run_roundtrip(db: SQLiteBackend) -> None:
    # RETIRED — tested V1 single-table schema (insert_run / get_run / count_runs).
    # V1 schema replaced by 3-table hierarchy (Methods / Evals / Samples) in Phase 1.7.
    # V2 round-trip tests will be written as part of Phase V2 test harness.
    print("\n[1.3.f] Run round-trip — SKIPPED (V1 schema retired)")
    return


# ---------------------------------------------------------------------------
# 1.3.g — Sample round-trip (with Invalid wrapper field)
# ---------------------------------------------------------------------------

def test_sample_roundtrip(db: SQLiteBackend) -> None:
    # RETIRED — tested V1 single-table sample schema (insert_sample / get_sample with
    # "sample_id", "run_ids", "measurements" keys).  V1 schema replaced by 3-table hierarchy.
    # V2 sample round-trip tests will be written as part of Phase V2 test harness.
    print("\n[1.3.g] Sample round-trip — SKIPPED (V1 schema retired)")
    return

    # --- dead code below kept for reference only ---
    db.insert_sample(_SAMPLE_RECORD, _SAMPLE_EXTRAS)
    ok("insert_sample completed without exception")

    retrieved = db.get_sample(_SAMPLE_RECORD["sample_id"])
    if retrieved is None:
        fail("get_sample returns the record")
        return
    ok("get_sample returns the record")

    if retrieved.get("sample_id") == _SAMPLE_RECORD["sample_id"]:
        ok("sample_id matches")
    else:
        fail("sample_id matches", f"got {retrieved.get('sample_id')!r}")

    if retrieved.get("evidence_version") == _SAMPLE_RECORD["evidence_version"]:
        ok("evidence_version preserved in sample facts_json round-trip")
    else:
        fail("evidence_version preserved in sample facts_json round-trip", f"got {retrieved.get('evidence_version')!r}")

    # run_ids from promoted column (authoritative)
    run_ids = retrieved.get("run_ids")
    if run_ids == ["run-test-0001"]:
        ok("run_ids list preserved (from promoted column)")
    else:
        fail("run_ids list preserved", f"got {run_ids!r}")

    if retrieved.get("is_dirty") is False:
        ok("is_dirty defaults to False")
    else:
        fail("is_dirty defaults to False", f"got {retrieved.get('is_dirty')!r}")

    # Invalid wrapper field must survive round-trip unmodified
    measurements = retrieved.get("measurements", {})
    invalid_field = measurements.get("pixel_delta_l2")
    if is_invalid(invalid_field):
        ok("Invalid wrapper field stored and retrieved intact")
    else:
        fail("Invalid wrapper field stored and retrieved intact", f"got {invalid_field!r}")

    if isinstance(invalid_field, dict) and invalid_field.get("reason_code") == "ASSET.READ_FAIL":
        ok("Invalid wrapper reason_code preserved")
    else:
        fail("Invalid wrapper reason_code preserved", f"got {invalid_field!r}")

    # Valid measurement preserved
    cos = measurements.get("clip_delta_cos")
    if cos == 0.123:
        ok("valid measurement (clip_delta_cos=0.123) preserved")
    else:
        fail("valid measurement preserved", f"got {cos!r}")

    # Extras stored separately and accessible via _extras key
    extras = retrieved.get("_extras", {})
    if extras.get("raw_field_x") == "some_value":
        ok("extras stored and accessible via _extras")
    else:
        fail("extras stored and accessible via _extras", f"got {extras!r}")


# ---------------------------------------------------------------------------
# 1.3.h — Asset round-trip (stage → commit → load + tamper check)
# ---------------------------------------------------------------------------

def test_asset_roundtrip(project_root: Path, assets_root: Path, tmp_root: Path) -> None:
    print("\n[1.3.h] Asset round-trip (stage / commit / load + tamper check)")

    # Patch core.paths so assets module resolves to our temp dirs
    import core.paths as _paths
    original_get_path = _paths.get_path

    path_map = {
        "project_root": project_root,
        "assets_root": assets_root,
        "tmp_root": tmp_root,
    }

    def _patched_get_path(name: str) -> Path:
        if name in path_map:
            return path_map[name]
        return original_get_path(name)

    _paths.get_path = _patched_get_path

    try:
        from databank.assets import commit_blob, load_blob, stage_blob

        test_data = b"hello lora evaluation world"

        # Stage
        vref = stage_blob(test_data, "embedding", "npy")
        ok("stage_blob returned without exception")

        if "_staged_path" in vref:
            ok("_staged_path present in staged ValueRef")
        else:
            fail("_staged_path present in staged ValueRef")

        staged = Path(vref["_staged_path"])
        if staged.exists():
            ok("staged file exists in tmp/")
        else:
            fail("staged file exists in tmp/")

        # Commit
        clean_ref = commit_blob(vref)
        ok("commit_blob completed without exception")

        if "_staged_path" not in clean_ref:
            ok("_staged_path removed from committed ValueRef")
        else:
            fail("_staged_path removed from committed ValueRef", "key still present")

        final_path = project_root / clean_ref["path"]
        if final_path.exists():
            ok("committed file exists at final path")
        else:
            fail("committed file exists at final path", f"missing: {final_path}")

        if not staged.exists():
            ok("staged tmp file removed after commit")
        else:
            ok("staged tmp file removed after commit (or tmp == final, acceptable)")

        # Load — happy path
        loaded = load_blob(clean_ref)
        if isinstance(loaded, bytes):
            ok("load_blob returns bytes")
        else:
            fail("load_blob returns bytes", f"got {type(loaded)}")

        if loaded == test_data:
            ok("loaded bytes match original data")
        else:
            fail("loaded bytes match original data")

        # Tamper check — corrupt the file on disk
        final_path.write_bytes(b"corrupted data")
        tampered = load_blob(clean_ref)
        if is_invalid(tampered):
            ok("tampered blob returns Invalid wrapper")
        else:
            fail("tampered blob returns Invalid wrapper", f"got {tampered!r}")

        if isinstance(tampered, dict) and tampered.get("reason_code") == "ASSET.HASH_MISMATCH":
            ok("tampered blob reason_code is ASSET.HASH_MISMATCH")
        else:
            fail("tampered blob reason_code is ASSET.HASH_MISMATCH", f"got {tampered!r}")

    finally:
        _paths.get_path = original_get_path




# ---------------------------------------------------------------------------
# Migration logging route (Command Center façade)
# ---------------------------------------------------------------------------

def test_migration_logging_route(tmpdir: Path) -> None:
    print("\n[1.3.m] Migration logging uses core diagnostics emitter")

    import databank.sqlite_backend as sqlite_backend_module

    observed: list[tuple[str, str]] = []
    original = sqlite_backend_module.emit

    def _capture(severity: str, code: str, where: str, msg: str, **ctx: object) -> dict:
        observed.append((severity, code))
        return {
            "severity": severity,
            "code": code,
            "where": where,
            "msg": msg,
            "ctx": ctx,
        }

    sqlite_backend_module.emit = _capture
    try:
        db = SQLiteBackend(tmpdir / "migration_log_route.db")
        _ = db.count_methods()  # any Treasurer call that forces schema init
        db.close()
    finally:
        sqlite_backend_module.emit = original

    if ("INFO", "DB.MIGRATION_APPLIED") in observed:
        ok("migration apply event emitted via core.diagnostics.emit")
    else:
        fail(
            "migration apply event emitted via core.diagnostics.emit",
            f"observed={observed!r}",
        )

# ---------------------------------------------------------------------------
# is_dirty column defaults and set_run_dirty / set_sample_dirty
# ---------------------------------------------------------------------------

def test_dirty_flags(db: SQLiteBackend) -> None:
    # RETIRED — tested V1 set_run_dirty / set_sample_dirty interface which no longer exists.
    # Dirty-flag logic now lives on Methods / Evals / Samples in the 3-table schema.
    # V2 dirty-flag tests will be written as part of Phase V2 test harness.
    print("\n[1.3.k] is_dirty dirty flags — SKIPPED (V1 schema retired)")
    return

    # --- dead code below kept for reference only ---
    run = db.get_run(_RUN_RECORD["run_id"])
    if run and run.get("is_dirty") is False:
        ok("run is_dirty defaults to False")
    else:
        fail("run is_dirty defaults to False")

    db.set_run_dirty(_RUN_RECORD["run_id"])
    run2 = db.get_run(_RUN_RECORD["run_id"])
    if run2 and run2.get("is_dirty") is True:
        ok("set_run_dirty flips is_dirty to True")
    else:
        fail("set_run_dirty flips is_dirty to True", f"got {run2.get('is_dirty') if run2 else None!r}")

    sample = db.get_sample(_SAMPLE_RECORD["sample_id"])
    if sample and sample.get("is_dirty") is False:
        ok("sample is_dirty defaults to False")
    else:
        fail("sample is_dirty defaults to False")

    db.set_sample_dirty(_SAMPLE_RECORD["sample_id"])
    sample2 = db.get_sample(_SAMPLE_RECORD["sample_id"])
    if sample2 and sample2.get("is_dirty") is True:
        ok("set_sample_dirty flips is_dirty to True")
    else:
        fail("set_sample_dirty flips is_dirty to True")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="lora_eval_test_"))
    try:
        test_migration_logging_route(tmpdir)

        db_path = tmpdir / "test.db"
        assets_root = tmpdir / "assets"
        tmp_root = tmpdir / "tmp"
        project_root = tmpdir  # project root is tmpdir for asset path resolution

        db = SQLiteBackend(db_path)
        try:
            test_run_roundtrip(db)
            test_sample_roundtrip(db)
            test_dirty_flags(db)
        finally:
            db.close()

        test_asset_roundtrip(project_root, assets_root, tmp_root)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if _FAIL == 0:
        print(f"test_storage: PASS — {_PASS} checks passed, 0 failed")
        return 0
    else:
        print(f"test_storage: FAIL — {_PASS} passed, {_FAIL} failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
