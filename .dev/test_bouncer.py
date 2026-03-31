""".dev/test_bouncer.py

Bouncer (Step 1.7.f) behavioral tests — 3-level dedup model.

Tests cover:
  1.4.f  — Happy path: Method + Eval + Sample inserted, ingest_status OK
  1.4.g  — Duplicate method_hash → idempotent (no duplicate row, dedup continues)
  1.4.h  — DUPLICATE_RUN: same sample_hash + same latent_hash → discard silently
  1.4.h2 — ENRICHED: same sample_hash + same latent_hash + new measurement slot
  1.4.i  — NOT_ZERO_DELTA: same sample_hash + different latent_hash → dirty + loud warn
  1.4.j  — Missing vital fields → hard refuse, nothing written
  1.4.k  — Missing required non-vital nested field → NON_STORABLE_REJECT
  1.4.l  — Unknown key → moved to extras_json
  1.4.m  — Non-dict input → SCHEMA.PARSE_FAIL

Run:
    python .dev/test_bouncer.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bouncer.gate import process
from bouncer.schema_loader import evidence_version
from contracts.validation_errors import is_invalid
from core.time_ids import now_iso
from databank.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Test helper builders
# ---------------------------------------------------------------------------


def _mk_method(method_hash: str = "method-hash-aaa") -> dict:
    """Minimal valid Method record (all required fields present)."""
    return {
        "method_hash": method_hash,
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
        "latent": {"width": 512, "height": 512, "shape": [1, 4, 64, 64]},
    }


def _mk_eval(
    eval_hash: str = "eval-hash-bbb",
    method_hash: str = "method-hash-aaa",
    lora_hash: str = "lora-hash-ccc",
) -> dict:
    """Minimal valid Eval record."""
    return {
        "eval_hash": eval_hash,
        "method_hash": method_hash,
        "lora": {"hash": lora_hash},
    }


def _mk_sample(
    sample_hash: str = "sample-hash-ddd",
    eval_hash: str = "eval-hash-bbb",
    seed: int = 42,
    latent_hash: str = "latent-hash-eee",
    image_hash: str = "image-hash-fff",
) -> dict:
    """Minimal valid Sample record.

    image and luminance are required fields in the schema.
    ValueRef stubs pass validation because "valueref" is an unknown type
    in _coerce_value (falls through to accept-as-is).
    """
    return {
        "sample_hash": sample_hash,
        "eval_hash": eval_hash,
        "seed": seed,
        "lora_strength": 0.8,
        "latent_hash": latent_hash,
        "image_hash": image_hash,
        "extractor_version": "test.0.1",
        # Required measurement domains — minimal stubs accepted as-is.
        "image": {"output": {"_stub": "image"}},
        "luminance": {"output": {"_stub": "luminance"}},
    }


def _mk_sample_with_clip(
    sample_hash: str = "sample-hash-ddd",
    eval_hash: str = "eval-hash-bbb",
    seed: int = 42,
    latent_hash: str = "latent-hash-eee",
    image_hash: str = "image-hash-fff",
) -> dict:
    """Sample record with an additional clip_vision slot (triggers enrichment)."""
    s = _mk_sample(sample_hash, eval_hash, seed, latent_hash, image_hash)
    # Add a new measurement domain not present in the basic sample.
    s["clip_vision"] = {"abcdef1234567890": {"output": {"_stub": "clip"}}}
    return s


def _mk_pose_source(processor: str, *, scene_flags: list[str] | None = None) -> dict:
    source = {
        "processor": processor,
        "mode": "body_only",
        "source_kind": "structured_keypoints",
        "coordinate_space": "pixel",
        "canvas_width": 512,
        "canvas_height": 512,
        "raw_people_count": 1,
        "eligible_candidate_count": 1,
        "raw_observation": {
            "people": [
                {
                    "pose_keypoints_2d": [10.0, 20.0, 1.0, 30.0, 40.0, 1.0],
                    "face_keypoints_2d": None,
                    "hand_left_keypoints_2d": None,
                    "hand_right_keypoints_2d": None,
                }
            ]
        },
        "people": [
            {
                "person_id": "P0",
                "person_index": 0,
                "bbox": [8, 18, 32, 42],
                "area_fraction": 0.01,
                "recognized_joint_tags": ["l_shoulder", "r_shoulder"],
                "recognized_joint_count": 2,
                "core_joint_count": 2,
                "present_joint_count": 2,
                "main_subject_overlap": 1.0,
                "densepose_overlap": 0.8,
                "face_support": 0.0,
                "missing_joints": [],
                "outside_main_subject_joints": [],
                "outside_main_subject_core_joints": [],
                "outside_densepose_joints": [],
                "flags": [],
                "joints": {},
            }
        ],
    }
    if scene_flags:
        source["scene_flags"] = scene_flags
    return source


def _mk_sample_with_pose_evidence(
    source_key: str,
    processor: str,
    *,
    scene_flags: list[str] | None = None,
) -> dict:
    s = _mk_sample()
    s["pose_evidence"] = {
        source_key: _mk_pose_source(processor, scene_flags=scene_flags)
    }
    return s


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _contains_invalid_wrapper(value: object) -> bool:
    if is_invalid(value):
        return True
    if isinstance(value, dict):
        return any(_contains_invalid_wrapper(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_invalid_wrapper(v) for v in value)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bouncer_test_"))
    db = SQLiteBackend(tmp / "bouncer.db")

    # ---- 1.4.f: Happy path — all three records stored and linked ----
    r1 = process(_mk_method(), _mk_eval(), _mk_sample(), db)
    _assert(r1.get("ok") is True, f"happy path should be ok; got {r1}")
    _assert(r1.get("action") == "inserted", f"expected 'inserted', got {r1.get('action')}")
    _assert(r1.get("method_hash") == "method-hash-aaa", "method_hash in result")
    _assert(r1.get("eval_hash") == "eval-hash-bbb", "eval_hash in result")
    _assert(r1.get("sample_hash") == "sample-hash-ddd", "sample_hash in result")

    s1 = db.get_sample("sample-hash-ddd")
    _assert(s1 is not None, "sample should be stored")
    _assert(s1.get("ingest_status") == "OK", f"ingest_status should be OK, got {s1.get('ingest_status')}")
    _assert(s1.get("is_dirty") is False, "sample should not be dirty")
    _assert(s1.get("evidence_version") == (evidence_version() or ""), "evidence_version should be stamped")
    _assert(not _contains_invalid_wrapper(s1), "stored facts must not contain Invalid wrappers")

    m1 = db.get_method("method-hash-aaa")
    _assert(m1 is not None, "method should be stored")
    e1 = db.get_eval("eval-hash-bbb")
    _assert(e1 is not None, "eval should be stored")

    # ---- 1.4.g: Duplicate method_hash → idempotent, no error, dedup continues ----
    # Re-submit same method + same eval + different sample → only new sample stored
    r2 = process(
        _mk_method("method-hash-aaa"),                      # same method
        _mk_eval("eval-hash-bbb", "method-hash-aaa"),       # same eval
        _mk_sample("sample-hash-NEW", "eval-hash-bbb", 99), # new sample
        db,
    )
    _assert(r2.get("ok") is True, f"duplicate method should still succeed; got {r2}")
    _assert(r2.get("action") == "inserted", f"expected inserted, got {r2.get('action')}")
    _assert(db.count_methods() == 1, f"should still be 1 method, got {db.count_methods()}")
    _assert(db.count_evals() == 1, f"should still be 1 eval, got {db.count_evals()}")
    _assert(db.get_sample("sample-hash-NEW") is not None, "new sample should be stored")

    # ---- 1.4.h: DUPLICATE_RUN — same sample_hash + same latent_hash ----
    r3 = process(_mk_method(), _mk_eval(), _mk_sample(), db)
    _assert(r3.get("ok") is True, f"duplicate run should be ok; got {r3}")
    _assert(r3.get("action") == "sample_duplicate_run", f"expected sample_duplicate_run, got {r3.get('action')}")
    # Sample count should not have increased (no new row for same PK)
    _assert(db.count_samples() == 2, f"still 2 samples, got {db.count_samples()}")

    # ---- 1.4.h2: ENRICHED — same sample_hash + same latent_hash + new measurement slot ----
    # Use a backend with a custom event_emitter to capture SAMPLE.ENRICHED from enrich_sample.
    import bouncer.gate as gate_mod
    enriched_diags: list[dict] = []

    def _capture_event(sev, code, where, msg, **ctx):
        from core import diagnostics as _diag
        rec = _diag.emit(sev, code, where, msg, **ctx)
        enriched_diags.append(rec)
        return rec

    db_capture = SQLiteBackend(tmp / "bouncer.db", event_emitter=_capture_event)
    r4 = process(
        _mk_method(),
        _mk_eval(),
        _mk_sample_with_clip(),  # same hashes but adds clip_vision slot
        db_capture,
    )
    orig_emit_gate = gate_mod.diagnostics.emit  # save for later restore

    _assert(r4.get("ok") is True, f"enrichment should be ok; got {r4}")
    _assert(r4.get("action") == "sample_duplicate_run", f"expected sample_duplicate_run, got {r4.get('action')}")
    # Backend emits SAMPLE.ENRICHED if it actually added something.
    _assert(
        any(d.get("code") == "SAMPLE.ENRICHED" for d in enriched_diags),
        "expected SAMPLE.ENRICHED diagnostic from backend after clip_vision enrichment",
    )
    # Verify enrichment actually happened in DB
    s1_enriched = db.get_sample("sample-hash-ddd")
    _assert(s1_enriched is not None, "enriched sample should still exist")
    _assert("clip_vision" in s1_enriched, "clip_vision should be present after enrichment")

    # ---- 1.4.h3: recursive pose_evidence enrichment â€” add one source, then another, then a nested field ----
    r4_pose_open = process(
        _mk_method(),
        _mk_eval(),
        _mk_sample_with_pose_evidence("openpose_body", "openpose"),
        db_capture,
    )
    _assert(r4_pose_open.get("ok") is True, f"pose_evidence openpose enrich should be ok; got {r4_pose_open}")
    s1_pose = db.get_sample("sample-hash-ddd")
    _assert(s1_pose is not None, "sample should still exist after pose_evidence enrichment")
    _assert("pose_evidence" in s1_pose, "pose_evidence should be present after enrichment")
    _assert("openpose_body" in (s1_pose.get("pose_evidence") or {}), "openpose_body should be present after enrichment")

    r4_pose_dw = process(
        _mk_method(),
        _mk_eval(),
        _mk_sample_with_pose_evidence("dw_body", "dwpose"),
        db_capture,
    )
    _assert(r4_pose_dw.get("ok") is True, f"pose_evidence dw enrich should be ok; got {r4_pose_dw}")
    s1_pose_two = db.get_sample("sample-hash-ddd")
    pose_domain = s1_pose_two.get("pose_evidence") or {}
    _assert("openpose_body" in pose_domain, "openpose_body should survive after adding dw_body")
    _assert("dw_body" in pose_domain, "dw_body should be added additively")

    r4_pose_nested = process(
        _mk_method(),
        _mk_eval(),
        _mk_sample_with_pose_evidence("openpose_body", "openpose", scene_flags=["multiple_candidates"]),
        db_capture,
    )
    _assert(r4_pose_nested.get("ok") is True, f"nested pose_evidence enrich should be ok; got {r4_pose_nested}")
    s1_pose_three = db.get_sample("sample-hash-ddd")
    openpose_pose = ((s1_pose_three.get("pose_evidence") or {}).get("openpose_body") or {})
    _assert(
        openpose_pose.get("scene_flags") == ["multiple_candidates"],
        f"recursive no-clobber should add missing nested scene_flags, got {openpose_pose.get('scene_flags')!r}",
    )
    _assert(
        "raw_observation" in openpose_pose and "people" in openpose_pose["raw_observation"],
        "existing nested pose_evidence data should survive recursive enrich",
    )

    # ---- 1.4.i: NOT_ZERO_DELTA — same sample_hash, different latent_hash ----
    not_zero_diags: list[dict] = []

    def _capture_nzd(sev, code, where, msg, **ctx):
        rec = orig_emit_gate(sev, code, where, msg, **ctx)
        not_zero_diags.append(rec)
        return rec

    gate_mod.diagnostics.emit = _capture_nzd
    try:
        r5 = process(
            _mk_method(),
            _mk_eval(),
            _mk_sample(latent_hash="DIFFERENT-latent-hash"),  # same sample_hash, different latent
            db,
        )
    finally:
        gate_mod.diagnostics.emit = orig_emit_gate

    _assert(r5.get("ok") is True, f"NOT_ZERO_DELTA should return ok=True; got {r5}")
    _assert(r5.get("action") == "not_zero_delta", f"expected not_zero_delta, got {r5.get('action')}")

    # Existing sample should now be dirty.
    s1_after = db.get_sample("sample-hash-ddd")
    _assert(s1_after is not None, "original sample still exists")
    _assert(s1_after.get("is_dirty") is True, "original sample should be dirty after NOT_ZERO_DELTA")
    _assert(
        any(e.get("code") == "SAMPLE.NOT_ZERO_DELTA" for e in (s1_after.get("_errors") or [])),
        "NOT_ZERO_DELTA error record on existing sample",
    )
    # SAMPLE.NOT_ZERO_DELTA should be WARN severity.
    _assert(
        any(
            d.get("severity") == "WARN" and d.get("code") == "SAMPLE.NOT_ZERO_DELTA"
            for d in not_zero_diags
        ),
        "expected loud WARN for SAMPLE.NOT_ZERO_DELTA",
    )

    # ---- 1.4.j: Missing vital fields → hard refuse, nothing written ----
    db2 = SQLiteBackend(tmp / "bouncer2.db")

    # Missing method_hash
    bad_method = _mk_method()
    del bad_method["method_hash"]
    rj = process(bad_method, _mk_eval(), _mk_sample(), db2)
    _assert(rj.get("ok") is False and rj.get("reason") == "vital_missing", f"expected vital_missing; got {rj}")
    _assert(db2.get_method("method-hash-aaa") is None, "method must not be stored")
    _assert(db2.get_sample("sample-hash-ddd") is None, "sample must not be stored")

    # Missing eval_hash
    bad_eval = _mk_eval()
    del bad_eval["eval_hash"]
    rj2 = process(_mk_method(), bad_eval, _mk_sample(), db2)
    _assert(rj2.get("ok") is False and rj2.get("reason") == "vital_missing", f"expected vital_missing for eval_hash; got {rj2}")

    # Missing sample_hash
    bad_sample = _mk_sample()
    del bad_sample["sample_hash"]
    rj3 = process(_mk_method(), _mk_eval(), bad_sample, db2)
    _assert(rj3.get("ok") is False and rj3.get("reason") == "vital_missing", f"expected vital_missing for sample_hash; got {rj3}")

    # Missing seed (integer vital field)
    bad_seed = _mk_sample()
    del bad_seed["seed"]
    rj4 = process(_mk_method(), _mk_eval(), bad_seed, db2)
    _assert(rj4.get("ok") is False and rj4.get("reason") == "vital_missing", f"expected vital_missing for seed; got {rj4}")

    # Missing latent_hash
    bad_latent = _mk_sample()
    del bad_latent["latent_hash"]
    rj5 = process(_mk_method(), _mk_eval(), bad_latent, db2)
    _assert(rj5.get("ok") is False and rj5.get("reason") == "vital_missing", f"expected vital_missing for latent_hash; got {rj5}")

    # Missing image_hash
    bad_image = _mk_sample()
    del bad_image["image_hash"]
    rj6 = process(_mk_method(), _mk_eval(), bad_image, db2)
    _assert(rj6.get("ok") is False and rj6.get("reason") == "vital_missing", f"expected vital_missing for image_hash; got {rj6}")

    # Confirm nothing was written
    _assert(db2.count_methods() == 0, "nothing should be written after vital_missing")
    _assert(db2.count_samples() == 0, "nothing should be written after vital_missing")

    # ---- 1.4.k: Missing required non-vital nested field → NON_STORABLE_REJECT ----
    bad_nested = _mk_method()
    del bad_nested["base_model"]["hash"]  # base_model.hash is required (but not vital)
    rk = process(bad_nested, _mk_eval(), _mk_sample(), db2)
    _assert(rk.get("ok") is False, f"expected rejection for missing required field; got {rk}")
    _assert(rk.get("reason") == "non_storable_reject", f"expected non_storable_reject, got {rk.get('reason')}")
    _assert(db2.count_methods() == 0, "nothing stored after non_storable_reject")
    _assert(db2.count_samples() == 0, "nothing stored after non_storable_reject")

    # Invalid enum value → also non_storable_reject
    bad_enum = _mk_sample("sample-x", "eval-hash-bbb", 1, "lat-x", "img-x")
    bad_enum_eval = _mk_eval("eval-hash-bbb", "method-hash-aaa")
    bad_enum["ingest_status"] = "INVALID_VALUE"  # not in enum
    rk2 = process(_mk_method(), bad_enum_eval, bad_enum, db2)
    _assert(rk2.get("ok") is False and rk2.get("reason") == "non_storable_reject", f"invalid enum should reject; got {rk2}")

    # ---- 1.4.l: Unknown key → moved to extras ----
    db3 = SQLiteBackend(tmp / "bouncer3.db")
    sample_with_extra = _mk_sample("sample-z", "eval-hash-bbb", 7, "lat-z", "img-z")
    sample_with_extra["weird_key"] = 999
    process(_mk_method(), _mk_eval(), sample_with_extra, db3)
    sz = db3.get_sample("sample-z")
    _assert(sz is not None, "sample-z should be stored")
    _assert("weird_key" not in sz, "unknown key must not be in facts")
    extras = sz.get("_extras") or {}
    _assert(extras.get("sample", {}).get("weird_key") == 999, "unknown key should be in extras.sample")

    # ---- 1.4.m: Non-dict input → SCHEMA.PARSE_FAIL ----
    db4 = SQLiteBackend(tmp / "bouncer4.db")
    r_parse = process("nope", _mk_eval(), _mk_sample(), db4)
    _assert(r_parse.get("ok") is False and r_parse.get("reason") == "parse_fail", f"expected parse_fail; got {r_parse}")
    _assert(db4.count_samples() == 0, "no writes expected after parse_fail")

    r_parse2 = process(_mk_method(), None, _mk_sample(), db4)
    _assert(r_parse2.get("ok") is False and r_parse2.get("reason") == "parse_fail", f"expected parse_fail for None eval; got {r_parse2}")

    print("All Bouncer tests passed.")


if __name__ == "__main__":
    main()
