"""
.dev/test_scorecard.py — Data Quality Scorecard v1 tests for Step 3.4.l.

Tests:
  3.4.l-1  Empty DB → all zeros / 0.0 / {} defaults.
  3.4.l-2  Mixed ingest statuses → correct counts and rates.
  3.4.l-3  Dirty totals and sample_dirty_rate.
  3.4.l-4  Coverage rates per domain from facts_json.
  3.4.l-5  Diagnostics window counting behavior (command_center level).
  3.4.l-Z  Repeated execution → byte-identical JSON (Zero-Delta).

Run with:  python .dev/test_scorecard.py
Exit 0 = all passed.  Exit 1 = one or more failures.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from databank.health import DataBankHealth
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
# DB construction helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-01T00:00:00Z"


def _make_method(tag: str) -> dict:
    return {
        "method_hash": f"meth_{tag}",
        "timestamp":   _TS,
        "base_model":  {"hash": f"base_{tag}"},
        "vae_model":   {"hash": f"vae_{tag}"},
        "is_dirty":    False,
        "settings": {
            "steps": 20, "denoise": 1.0, "sampler": "euler",
            "scheduler": "normal", "cfg": 7.0,
        },
        "latent": {"width": 512, "height": 512, "shape": [1, 4, 64, 64]},
    }


def _make_eval(tag: str, method_tag: str, lora_hash: str | None = None) -> dict:
    return {
        "eval_hash":   f"eval_{tag}",
        "method_hash": f"meth_{method_tag}",
        "timestamp":   _TS,
        "lora":        {} if lora_hash is None else {"hash": lora_hash},
        "is_dirty":    False,
    }


def _make_sample(
    tag: str,
    eval_tag: str,
    seed: int = 1,
    ingest_status: str = "OK",
    is_dirty: bool = False,
    domains: list[str] | None = None,
    extras: dict | None = None,
) -> tuple[dict, dict]:
    """Return (sample_record, extras_dict) with specified domains in facts_json."""
    facts: dict = {
        "sample_hash":  f"samp_{tag}",
        "eval_hash":    f"eval_{eval_tag}",
        "seed":         seed,
        "latent_hash":  f"lh_{tag}",
        "image_hash":   f"ih_{tag}",
        "ingest_status": ingest_status,
        "timestamp":    _TS,
        "is_dirty":     is_dirty,
    }
    # Add domain top-level keys so coverage can detect them
    for domain in (domains or []):
        facts[domain] = {"present": True}

    return facts, (extras or {})


def _build_empty_db(tmp_dir: Path) -> SQLiteBackend:
    return SQLiteBackend(tmp_dir / "test_scorecard.db")


# ---------------------------------------------------------------------------
# 3.4.l-1 — Empty DB defaults
# ---------------------------------------------------------------------------


def test_empty_db() -> None:
    print("\n[3.4.l-1] Empty DB defaults")
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteBackend(Path(tmp) / "empty.db")
        try:
            h = DataBankHealth(db)
            sc = h.scorecard()

            # counts — all zero
            if sc["counts"] == {"methods": 0, "evals": 0, "samples": 0}:
                ok("counts all zero on empty DB")
            else:
                fail("counts all zero", f"got {sc['counts']!r}")

            # dirty — all zero, rate 0.0
            dirty = sc["dirty"]
            if (dirty["methods"] == dirty["evals"] == dirty["samples"] == dirty["total"] == 0
                    and dirty["sample_dirty_rate"] == 0.0):
                ok("dirty all zero on empty DB")
            else:
                fail("dirty all zero", f"got {dirty!r}")

            # ingest_status — all zero / 0.0
            ist = sc["ingest_status"]
            if (ist["ok_count"] == ist["warn_count"] == ist["error_count"] == 0
                    and ist["ok_rate"] == ist["warn_rate"] == ist["error_rate"] == 0.0):
                ok("ingest_status all zero on empty DB")
            else:
                fail("ingest_status all zero", f"got {ist!r}")

            # coverage — all 0.0
            cov = sc["coverage"]
            expected_domains = {"image", "luminance", "clip_vision", "masks", "face_analysis", "aux"}
            if set(cov.keys()) == expected_domains and all(v == 0.0 for v in cov.values()):
                ok("coverage all 0.0 on empty DB")
            else:
                fail("coverage all 0.0", f"got {cov!r}")

            # extras — empty dict
            if sc["extras"] == {}:
                ok("extras is {} on empty DB")
            else:
                fail("extras is {}", f"got {sc['extras']!r}")

            # Required keys present
            for key in ("counts", "dirty", "ingest_status", "coverage", "extras"):
                if key in sc:
                    ok(f"scorecard has required key '{key}'")
                else:
                    fail(f"scorecard has required key '{key}'")

        finally:
            db.close()


# ---------------------------------------------------------------------------
# 3.4.l-2 — Mixed ingest statuses → correct counts and rates
# ---------------------------------------------------------------------------


def test_ingest_status_counts() -> None:
    print("\n[3.4.l-2] Mixed ingest statuses")
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteBackend(Path(tmp) / "mixed.db")
        try:
            db.insert_method(_make_method("a"))
            db.insert_eval(_make_eval("a", "a"))

            # 4 OK, 2 WARN, 1 ERROR = 7 total
            for i in range(4):
                r, e = _make_sample(f"ok{i}", "a", seed=i, ingest_status="OK")
                db.insert_sample(r, e)
            for i in range(2):
                r, e = _make_sample(f"warn{i}", "a", seed=10+i, ingest_status="WARN")
                db.insert_sample(r, e)
            r, e = _make_sample("err0", "a", seed=20, ingest_status="ERROR")
            db.insert_sample(r, e)

            h = DataBankHealth(db)
            sc = h.scorecard()
            ist = sc["ingest_status"]

            if ist["ok_count"] == 4:
                ok("ok_count == 4")
            else:
                fail("ok_count == 4", f"got {ist['ok_count']!r}")

            if ist["warn_count"] == 2:
                ok("warn_count == 2")
            else:
                fail("warn_count == 2", f"got {ist['warn_count']!r}")

            if ist["error_count"] == 1:
                ok("error_count == 1")
            else:
                fail("error_count == 1", f"got {ist['error_count']!r}")

            expected_ok_rate = 4 / 7
            if abs(ist["ok_rate"] - expected_ok_rate) < 1e-12:
                ok(f"ok_rate == 4/7")
            else:
                fail(f"ok_rate == 4/7", f"got {ist['ok_rate']!r}")

            expected_warn_rate = 2 / 7
            if abs(ist["warn_rate"] - expected_warn_rate) < 1e-12:
                ok("warn_rate == 2/7")
            else:
                fail("warn_rate == 2/7", f"got {ist['warn_rate']!r}")

            expected_err_rate = 1 / 7
            if abs(ist["error_rate"] - expected_err_rate) < 1e-12:
                ok("error_rate == 1/7")
            else:
                fail("error_rate == 1/7", f"got {ist['error_rate']!r}")

            # Rates should sum to 1.0
            rate_sum = ist["ok_rate"] + ist["warn_rate"] + ist["error_rate"]
            if abs(rate_sum - 1.0) < 1e-12:
                ok("ok_rate + warn_rate + error_rate == 1.0")
            else:
                fail("rates sum to 1.0", f"got {rate_sum!r}")

        finally:
            db.close()


# ---------------------------------------------------------------------------
# 3.4.l-3 — Dirty totals and sample_dirty_rate
# ---------------------------------------------------------------------------


def test_dirty_totals() -> None:
    print("\n[3.4.l-3] Dirty totals and sample_dirty_rate")
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteBackend(Path(tmp) / "dirty.db")
        try:
            # 2 methods, 3 evals, 5 samples
            db.insert_method(_make_method("x"))
            db.insert_method(_make_method("y"))
            db.insert_eval(_make_eval("x1", "x"))
            db.insert_eval(_make_eval("x2", "x"))
            db.insert_eval(_make_eval("y1", "y"))

            for i in range(5):
                r, e = _make_sample(f"s{i}", "x1", seed=i)
                db.insert_sample(r, e)

            # Dirty: 1 method, 2 evals, 3 samples
            db.set_method_dirty("meth_x")
            db.set_eval_dirty("eval_x1")
            db.set_eval_dirty("eval_y1")
            db.set_sample_dirty("samp_s0")
            db.set_sample_dirty("samp_s1")
            db.set_sample_dirty("samp_s4")

            h = DataBankHealth(db)
            sc = h.scorecard()
            d = sc["dirty"]

            if d["methods"] == 1:
                ok("dirty methods == 1")
            else:
                fail("dirty methods == 1", f"got {d['methods']!r}")

            if d["evals"] == 2:
                ok("dirty evals == 2")
            else:
                fail("dirty evals == 2", f"got {d['evals']!r}")

            if d["samples"] == 3:
                ok("dirty samples == 3")
            else:
                fail("dirty samples == 3", f"got {d['samples']!r}")

            if d["total"] == 6:
                ok("dirty total == 6 (1+2+3)")
            else:
                fail("dirty total == 6", f"got {d['total']!r}")

            expected_rate = 3 / 5
            if abs(d["sample_dirty_rate"] - expected_rate) < 1e-12:
                ok("sample_dirty_rate == 3/5")
            else:
                fail("sample_dirty_rate == 3/5", f"got {d['sample_dirty_rate']!r}")

            # dirty_count() convenience method should match
            if h.dirty_count() == 6:
                ok("dirty_count() == 6")
            else:
                fail("dirty_count() == 6", f"got {h.dirty_count()!r}")

        finally:
            db.close()


# ---------------------------------------------------------------------------
# 3.4.l-4 — Coverage rates per domain
# ---------------------------------------------------------------------------


def test_coverage() -> None:
    print("\n[3.4.l-4] Coverage rates per domain")
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteBackend(Path(tmp) / "coverage.db")
        try:
            db.insert_method(_make_method("m"))
            db.insert_eval(_make_eval("e", "m"))

            # 4 samples total:
            # sA: image + luminance + clip_vision + face_analysis
            # sB: image + luminance + masks
            # sC: image + luminance
            # sD: image (luminance absent, aux present)
            domains_by_sample = [
                ["image", "luminance", "clip_vision", "face_analysis"],
                ["image", "luminance", "masks"],
                ["image", "luminance"],
                ["image", "aux"],
            ]
            for i, domains in enumerate(domains_by_sample):
                r, ex = _make_sample(f"cov{i}", "e", seed=i, domains=domains)
                db.insert_sample(r, ex)

            h = DataBankHealth(db)
            sc = h.scorecard()
            cov = sc["coverage"]

            # image: all 4 → 1.0
            if cov["image"] == 1.0:
                ok("coverage image == 1.0")
            else:
                fail("coverage image == 1.0", f"got {cov['image']!r}")

            # luminance: 3/4 → 0.75
            if abs(cov["luminance"] - 0.75) < 1e-12:
                ok("coverage luminance == 0.75")
            else:
                fail("coverage luminance == 0.75", f"got {cov['luminance']!r}")

            # clip_vision: 1/4 → 0.25
            if abs(cov["clip_vision"] - 0.25) < 1e-12:
                ok("coverage clip_vision == 0.25")
            else:
                fail("coverage clip_vision == 0.25", f"got {cov['clip_vision']!r}")

            # masks: 1/4 → 0.25
            if abs(cov["masks"] - 0.25) < 1e-12:
                ok("coverage masks == 0.25")
            else:
                fail("coverage masks == 0.25", f"got {cov['masks']!r}")

            # face_analysis: 1/4 → 0.25
            if abs(cov["face_analysis"] - 0.25) < 1e-12:
                ok("coverage face_analysis == 0.25")
            else:
                fail("coverage face_analysis == 0.25", f"got {cov['face_analysis']!r}")

            # aux: 1/4 → 0.25
            if abs(cov["aux"] - 0.25) < 1e-12:
                ok("coverage aux == 0.25")
            else:
                fail("coverage aux == 0.25", f"got {cov['aux']!r}")

        finally:
            db.close()


# ---------------------------------------------------------------------------
# 3.4.l-5 — Diagnostics window counting (command_center level)
# ---------------------------------------------------------------------------


def test_diagnostics_window() -> None:
    print("\n[3.4.l-5] Diagnostics window counting behavior")

    import core.paths as _paths
    original_get_path = _paths.get_path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        log_path = tmp_dir / "logs" / "diagnostics.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a synthetic log: 5 WARN, 3 ERROR, 1 FATAL, 2 DEBUG (DEBUG filtered out)
        log_lines = []
        for i in range(5):
            log_lines.append(json.dumps({
                "severity": "WARN", "code": "SCHEMA.MISSING_FIELD",
                "where": "test", "msg": f"warn {i}", "ts": _TS,
            }))
        for i in range(3):
            log_lines.append(json.dumps({
                "severity": "ERROR", "code": "DB.WRITE_FAIL",
                "where": "test", "msg": f"error {i}", "ts": _TS,
            }))
        log_lines.append(json.dumps({
            "severity": "FATAL", "code": "DB.WRITE_FAIL",
            "where": "test", "msg": "fatal 0", "ts": _TS,
        }))
        # DEBUG entries — should NOT appear in errors() since it filters to WARN/ERROR/FATAL
        for i in range(2):
            log_lines.append(json.dumps({
                "severity": "DEBUG", "code": "GUARD.METHOD_EXISTS",
                "where": "test", "msg": f"debug {i}", "ts": _TS,
            }))
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        # Patch paths so command_center reads our temp log
        path_map = {
            "logs_root": tmp_dir / "logs",
            "db_file":   tmp_dir / "cc_test.db",
        }

        def _patched_get_path(name: str) -> Path:
            if name in path_map:
                return path_map[name]
            return original_get_path(name)

        _paths.get_path = _patched_get_path

        try:
            import command_center as cc

            # health() uses errors(limit=100) internally; all 9 WARN/ERROR/FATAL
            # entries are within the window=100.
            result = cc.health(diagnostics_window=100)
            diag = result["scorecard"]["diagnostics"]

            if diag["warn_count"] == 5:
                ok("diagnostics warn_count == 5")
            else:
                fail("diagnostics warn_count == 5", f"got {diag['warn_count']!r}")

            if diag["error_count"] == 3:
                ok("diagnostics error_count == 3")
            else:
                fail("diagnostics error_count == 3", f"got {diag['error_count']!r}")

            if diag["fatal_count"] == 1:
                ok("diagnostics fatal_count == 1")
            else:
                fail("diagnostics fatal_count == 1", f"got {diag['fatal_count']!r}")

            if diag["window"] == 100:
                ok("diagnostics window == 100")
            else:
                fail("diagnostics window == 100", f"got {diag['window']!r}")

            # Narrow window: limit=3 → last 3 WARN/ERROR/FATAL entries
            result_narrow = cc.health(diagnostics_window=3)
            diag_narrow = result_narrow["scorecard"]["diagnostics"]
            total_narrow = (diag_narrow["warn_count"] + diag_narrow["error_count"]
                            + diag_narrow["fatal_count"])
            if total_narrow <= 3:
                ok(f"narrow window=3 returns at most 3 WARN/ERROR/FATAL entries (got {total_narrow})")
            else:
                fail("narrow window=3 returns at most 3 entries", f"got total {total_narrow}")

            # Legacy diagnostics key still present and backward-compatible
            if "diagnostics" in result and "log_path" in result["diagnostics"]:
                ok("legacy diagnostics.log_path present for backward compatibility")
            else:
                fail("legacy diagnostics.log_path present")

            if "db" in result:
                ok("legacy db key present for backward compatibility")
            else:
                fail("legacy db key present")

        finally:
            _paths.get_path = original_get_path


# ---------------------------------------------------------------------------
# 3.4.l-Z — Repeated execution → byte-identical JSON (Zero-Delta)
# ---------------------------------------------------------------------------


def test_zero_delta() -> None:
    print("\n[3.4.l-Z] Zero-Delta — repeated scorecard() calls are byte-identical")
    with tempfile.TemporaryDirectory() as tmp:
        db = SQLiteBackend(Path(tmp) / "zerodelta.db")
        try:
            db.insert_method(_make_method("zd"))
            db.insert_eval(_make_eval("zd", "zd"))

            for i in range(6):
                status = ["OK", "OK", "OK", "WARN", "ERROR", "OK"][i]
                domains = [["image", "luminance", "clip_vision"],
                           ["image", "luminance"],
                           ["image", "luminance", "face_analysis", "masks"],
                           ["image", "luminance", "aux"],
                           ["image"],
                           ["image", "luminance", "clip_vision"]][i]
                r, e = _make_sample(f"zd{i}", "zd", seed=i, ingest_status=status,
                                    domains=domains,
                                    extras={"debug_flag": True} if i == 0 else {})
                db.insert_sample(r, e)

            db.set_sample_dirty("samp_zd2")

            h = DataBankHealth(db)

            # Serialize three times — must be byte-identical
            s1 = json.dumps(h.scorecard(), ensure_ascii=False, sort_keys=True)
            s2 = json.dumps(h.scorecard(), ensure_ascii=False, sort_keys=True)
            s3 = json.dumps(h.scorecard(), ensure_ascii=False, sort_keys=True)

            if s1 == s2 == s3:
                ok("Three consecutive scorecard() calls produce byte-identical JSON")
            else:
                fail("Three consecutive scorecard() calls produce byte-identical JSON")

            # Spot-check a few values to confirm it's not just empty
            sc = h.scorecard()
            if sc["counts"]["samples"] == 6:
                ok("Zero-Delta: sample count correct (6)")
            else:
                fail("Zero-Delta: sample count correct (6)", f"got {sc['counts']['samples']!r}")

            if sc["dirty"]["samples"] == 1:
                ok("Zero-Delta: dirty samples correct (1)")
            else:
                fail("Zero-Delta: dirty samples correct (1)", f"got {sc['dirty']['samples']!r}")

        finally:
            db.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print(" test_scorecard.py — Scorecard tests (3.4.l)")
    print("=" * 60)

    test_empty_db()
    test_ingest_status_counts()
    test_dirty_totals()
    test_coverage()
    test_diagnostics_window()
    test_zero_delta()

    print()
    print("=" * 60)
    print(f"  Results: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)

    if _FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
