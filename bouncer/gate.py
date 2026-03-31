"""bouncer/gate.py — Evidence gate (Bouncer) — 3-level dedup.

Implements Method → Eval → Sample three-level deduplication:
  - Method: idempotent insert (same method_hash → proceed, no new row)
  - Eval:   idempotent insert (same eval_hash → proceed, no new row)
  - Sample (3 cases + enrichment):
      DUPLICATE_RUN  — same sample_hash + same latent_hash, no new measurement data
                       → discard silently
      ENRICHED       — same sample_hash + same latent_hash, new measurement slots
                       → enrich no-clobber (SAMPLE.ENRICHED emitted by backend)
      NOT_ZERO_DELTA — same sample_hash + different latent_hash
                       → flag existing dirty, warn loudly; incoming NOT stored
      new            — new sample_hash → insert

Schema validation (all three records validated before any DB write):
  - Vital field check: hard refuse if missing or wrong type (nothing written)
  - Non-storable field failures: hard reject entire candidate
  - Unknown keys: moved to per-record-kind nested extras
  - ingest_status set to OK / WARN / ERROR on sample record

This module must not import any DB driver. It operates strictly through the
Treasurer interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.validation_errors import is_invalid
from core import diagnostics
from core.time_ids import now_iso
from databank.treasurer import Treasurer

from bouncer import schema_loader

# ---------------------------------------------------------------------------
# Measurement domain keys — these are enrichable post-hoc.
# Core identity/metadata keys are never enrichable.
# ---------------------------------------------------------------------------

_MEASUREMENT_DOMAINS: frozenset[str] = frozenset({
    "image",
    "luminance",
    "clip_vision",
    "masks",
    "face_analysis",
    "aux",
    "pose_evidence",
    "diagnostics",
})

# Fields that Bouncer autofills if missing (not caller-provided).
_AUTOFILL: dict[str, set[str]] = {
    "method": {"is_dirty", "timestamp"},
    "eval":   {"is_dirty", "timestamp"},
    "sample": {"is_dirty", "ingest_status", "evidence_version", "timestamp"},
}


@dataclass
class _ValidateResult:
    record: dict
    extras: dict
    diags: list[dict]
    warn_count: int
    error_count: int
    failures: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process(
    method_candidate: Any,
    eval_candidate: Any,
    sample_candidate: Any,
    treasurer: Treasurer,
) -> dict:
    """Validate, dedup, and write a three-record Evidence candidate.

    Args:
        method_candidate: Method record dict (from extractor or test harness).
        eval_candidate:   Eval record dict.
        sample_candidate: Sample record dict.
        treasurer:        DataBank Treasurer implementation.

    Returns:
        Summary dict with at least ``ok`` (bool) and, on success, ``action``
        describing what happened:
          "inserted"           — new Method + Eval + Sample stored
          "sample_duplicate_run" — same (sample_hash + latent_hash), no new data
          "sample_enriched"    — same (sample_hash + latent_hash), slots enriched
          "not_zero_delta"     — same sample_hash, different latent_hash; existing dirty
        On failure, ``reason`` gives the rejection code.
    """
    where = "bouncer.gate.process"

    # ------------------------------------------------------------------
    # 1. Parse check — all three candidates must be dicts.
    # ------------------------------------------------------------------
    if (
        not isinstance(method_candidate, dict)
        or not isinstance(eval_candidate, dict)
        or not isinstance(sample_candidate, dict)
    ):
        diagnostics.emit(
            "ERROR",
            "SCHEMA.PARSE_FAIL",
            where,
            "One or more incoming candidates are not dicts",
            method_type=type(method_candidate).__name__,
            eval_type=type(eval_candidate).__name__,
            sample_type=type(sample_candidate).__name__,
        )
        return {"ok": False, "reason": "parse_fail"}

    # ------------------------------------------------------------------
    # 2. Vital field checks (hard refusal — nothing written if any fail).
    # ------------------------------------------------------------------
    method_vitals = schema_loader.vital_fields_for("method_record")
    eval_vitals   = schema_loader.vital_fields_for("eval_record")
    sample_vitals = schema_loader.vital_fields_for("sample_record")

    method_defs = schema_loader.method_field_defs()
    eval_defs   = schema_loader.eval_field_defs()
    sample_defs = schema_loader.sample_field_defs()

    vital_failures: list[str] = []
    for field_name in method_vitals:
        if not _vital_ok(method_candidate, field_name, method_defs):
            vital_failures.append(f"method.{field_name}")
    for field_name in eval_vitals:
        if not _vital_ok(eval_candidate, field_name, eval_defs):
            vital_failures.append(f"eval.{field_name}")
    for field_name in sample_vitals:
        if not _vital_ok(sample_candidate, field_name, sample_defs):
            vital_failures.append(f"sample.{field_name}")

    if vital_failures:
        diagnostics.emit(
            "ERROR",
            "SCHEMA.VITAL_MISSING",
            where,
            "Missing or invalid vital field(s) — refusing entire candidate",
            missing=vital_failures,
        )
        return {"ok": False, "reason": "vital_missing", "missing": vital_failures}

    # ------------------------------------------------------------------
    # 3. Validate + sanitize all three records (no DB writes yet).
    # ------------------------------------------------------------------
    extras: dict = {}

    method_v = _validate_record(
        method_candidate, method_defs, extras,
        record_kind="method", autofill=_AUTOFILL["method"],
    )
    eval_v = _validate_record(
        eval_candidate, eval_defs, extras,
        record_kind="eval", autofill=_AUTOFILL["eval"],
    )
    sample_v = _validate_record(
        sample_candidate, sample_defs, extras,
        record_kind="sample", autofill=_AUTOFILL["sample"],
    )

    # ------------------------------------------------------------------
    # 4. Non-storable failures → hard reject (nothing written).
    # ------------------------------------------------------------------
    all_failures = method_v.failures + eval_v.failures + sample_v.failures
    if all_failures:
        method_hash_hint = method_candidate.get("method_hash")
        sample_hash_hint = sample_candidate.get("sample_hash")
        diagnostics.emit(
            "ERROR",
            "SCHEMA.NON_STORABLE_REJECT",
            where,
            "Candidate rejected — non-storable schema field failures",
            method_hash=method_hash_hint,
            sample_hash=sample_hash_hint,
            failures=all_failures,
        )
        return {
            "ok": False,
            "reason": "non_storable_reject",
            "failure_count": len(all_failures),
        }

    # ------------------------------------------------------------------
    # 5. System stamps.
    # ------------------------------------------------------------------
    method_record = method_v.record
    eval_record   = eval_v.record
    sample_record = sample_v.record

    method_record.setdefault("timestamp", now_iso())
    eval_record.setdefault("timestamp", now_iso())

    # Evidence version: Bouncer stamps the current contract version.
    sample_record["evidence_version"] = schema_loader.evidence_version() or str(
        sample_record.get("evidence_version") or ""
    )
    sample_record.setdefault("timestamp", now_iso())

    # ------------------------------------------------------------------
    # 6. ingest_status — reflects overall validation quality.
    # ------------------------------------------------------------------
    ingest_status = _compute_ingest_status(method_v, eval_v, sample_v)
    sample_record["ingest_status"] = ingest_status

    # ------------------------------------------------------------------
    # 7. is_dirty — set from ERROR ingest_status.
    # ------------------------------------------------------------------
    is_dirty = ingest_status == "ERROR"
    method_record["is_dirty"] = bool(method_record.get("is_dirty") or is_dirty)
    eval_record["is_dirty"]   = bool(eval_record.get("is_dirty")   or is_dirty)
    sample_record["is_dirty"] = bool(sample_record.get("is_dirty") or is_dirty)

    # ------------------------------------------------------------------
    # 8. Method — idempotent insert.
    # ------------------------------------------------------------------
    method_hash: str = method_record["method_hash"]
    treasurer.insert_method(method_record)  # True=new, False=exists; both OK

    # ------------------------------------------------------------------
    # 9. Eval — idempotent insert.
    # ------------------------------------------------------------------
    eval_hash: str = eval_record["eval_hash"]
    treasurer.insert_eval(eval_record)  # True=new, False=exists; both OK

    # ------------------------------------------------------------------
    # 10. Sample — 3-case dedup logic.
    # ------------------------------------------------------------------
    sample_hash: str     = sample_record["sample_hash"]
    latent_hash_incoming = sample_record.get("latent_hash")

    existing_sample = treasurer.get_sample(sample_hash)

    if existing_sample is not None:
        latent_hash_existing = existing_sample.get("latent_hash")

        if latent_hash_existing == latent_hash_incoming:
            # Same latent output — try enrichment (no-clobber).
            # Backend emits SAMPLE.ENRICHED if any slot was actually added.
            # Gate always returns "sample_duplicate_run" regardless — the enrichment
            # diagnostic from the backend is the signal for callers that care.
            measurement_delta = {
                k: v for k, v in sample_record.items() if k in _MEASUREMENT_DOMAINS
            }
            if measurement_delta:
                treasurer.enrich_sample(sample_hash, measurement_delta)
            diagnostics.emit(
                "DEBUG",
                "SAMPLE.DUPLICATE_RUN",
                where,
                "Same sample_hash + same latent_hash (duplicate run); enrichment attempted if measurement data present",
                sample_hash=sample_hash,
                had_measurement_delta=bool(measurement_delta),
            )
            return {
                "ok": True,
                "action": "sample_duplicate_run",
                "sample_hash": sample_hash,
            }
        else:
            # Different latent output for same deterministic inputs — NOT_ZERO_DELTA.
            diag = diagnostics.emit(
                "WARN",
                "SAMPLE.NOT_ZERO_DELTA",
                where,
                "Same sample_hash produced different latent_hash (NOT_ZERO_DELTA); "
                "existing sample flagged dirty; incoming NOT stored",
                sample_hash=sample_hash,
                existing_latent_hash=latent_hash_existing,
                incoming_latent_hash=latent_hash_incoming,
            )
            treasurer.set_sample_dirty(sample_hash)
            treasurer.store_errors(sample_hash, [diag])
            return {
                "ok": True,
                "action": "not_zero_delta",
                "sample_hash": sample_hash,
            }

    # New sample — insert.
    treasurer.insert_sample(sample_record, extras)

    return {
        "ok": True,
        "action": "inserted",
        "method_hash": method_hash,
        "eval_hash": eval_hash,
        "sample_hash": sample_hash,
        "ingest_status": sample_record.get("ingest_status"),
        "is_dirty": sample_record.get("is_dirty"),
    }


# ---------------------------------------------------------------------------
# Vital field check
# ---------------------------------------------------------------------------


def _vital_ok(record: dict, field_name: str, defs: dict) -> bool:
    """Return True if the vital field is present and well-typed."""
    val = record.get(field_name)
    if val is None:
        return False
    fdef = defs.get(field_name, {})
    ftype = fdef.get("type", "string")
    if ftype == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if ftype == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    # Default: non-empty string
    return isinstance(val, str) and bool(val)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_record(
    raw: dict,
    defs: dict[str, dict[str, Any]],
    extras: dict,
    *,
    record_kind: str,
    autofill: set[str],
) -> _ValidateResult:
    """Return sanitized record and diagnostics.

    - Unknown keys are moved into ``extras`` (nested under record_kind)
    - Missing required non-vital fields → NON_STORABLE_REJECT failure
    - Autofill fields: if absent, filled with a system default (no warning)
    """
    where = "bouncer.gate.validate"
    out: dict = {}
    diags: list[dict] = []
    warn_count = 0
    error_count = 0
    failures: list[dict] = []

    # Move unknown keys to extras.
    for k, v in raw.items():
        if k not in defs:
            _extras_put(extras, [record_kind], k, v)
            diags.append(
                diagnostics.emit(
                    "WARN",
                    "SCHEMA.UNKNOWN_KEY",
                    where,
                    "Unknown key moved to extras",
                    record_kind=record_kind,
                    key=k,
                )
            )
            warn_count += 1
        else:
            out[k] = v

    # Fill and validate schema-defined keys.
    for fname, fdef in defs.items():
        if fname not in out:
            if fname in autofill:
                out[fname] = _autofill_value(fname, record_kind)
                continue
            if fdef.get("required"):
                diags.append(
                    diagnostics.emit(
                        "WARN",
                        "SCHEMA.MISSING_FIELD",
                        where,
                        "Required field missing; record rejected",
                        record_kind=record_kind,
                        field=fname,
                    )
                )
                failures.append(
                    {
                        "reason": "missing_required_field",
                        "record_kind": record_kind,
                        "field": fname,
                    }
                )
                error_count += 1
            continue

        # Present — validate type.
        out[fname], wc, ec, field_failures = _coerce_value(
            out[fname],
            fdef,
            extras,
            record_kind=record_kind,
            path=[fname],
        )
        warn_count += wc
        error_count += ec
        failures.extend(field_failures)

    return _ValidateResult(out, extras, diags, warn_count, error_count, failures)


def _autofill_value(fname: str, record_kind: str) -> Any:
    if fname == "is_dirty":
        return False
    if fname == "timestamp":
        return now_iso()
    if record_kind == "sample":
        if fname == "evidence_version":
            return schema_loader.evidence_version()
        if fname == "ingest_status":
            return "OK"
    return None


def _coerce_value(
    val: Any,
    fdef: dict[str, Any],
    extras: dict,
    *,
    record_kind: str,
    path: list[str],
) -> tuple[Any, int, int, list[dict]]:
    """Validate/repair a single field value.

    Returns:
        (new_val, warn_count, error_count, failures)
    """
    where = "bouncer.gate.validate"
    warn_count = 0
    error_count = 0
    failures: list[dict] = []

    # Invalid wrappers in incoming records are never storable.
    if is_invalid(val):
        failures.append(
            {
                "reason": "invalid_wrapper_not_storable",
                "record_kind": record_kind,
                "path": ".".join(path),
            }
        )
        return val, 0, 1, failures

    ftype = fdef.get("type")

    # Enum check.
    if "enum" in fdef and val is not None:
        enum = fdef.get("enum")
        if isinstance(enum, list) and val not in enum:
            diagnostics.emit(
                "WARN",
                "SCHEMA.INVALID_TYPE",
                where,
                "Value not in enum; record rejected",
                record_kind=record_kind,
                path=".".join([record_kind] + path),
                got=val,
                allowed=enum,
            )
            failures.append(
                {
                    "reason": "invalid_enum",
                    "record_kind": record_kind,
                    "path": ".".join(path),
                    "got": val,
                }
            )
            return val, 0, 1, failures

    if ftype == "string":
        if isinstance(val, str):
            return val, 0, 0, failures
        diagnostics.emit(
            "WARN", "SCHEMA.INVALID_TYPE", where, "Expected string; record rejected",
            record_kind=record_kind,
            path=".".join([record_kind] + path),
            got_type=type(val).__name__,
        )
        failures.append(
            {"reason": "invalid_type", "expected": "string", "record_kind": record_kind, "path": ".".join(path)}
        )
        return val, 0, 1, failures

    if ftype == "integer":
        if isinstance(val, bool):
            pass  # bool subclasses int — reject
        elif isinstance(val, int):
            return val, 0, 0, failures
        diagnostics.emit(
            "WARN", "SCHEMA.INVALID_TYPE", where, "Expected integer; record rejected",
            record_kind=record_kind,
            path=".".join([record_kind] + path),
            got_type=type(val).__name__,
        )
        failures.append(
            {"reason": "invalid_type", "expected": "integer", "record_kind": record_kind, "path": ".".join(path)}
        )
        return val, 0, 1, failures

    if ftype == "number":
        if isinstance(val, bool):
            pass
        elif isinstance(val, (int, float)):
            return float(val), 0, 0, failures
        diagnostics.emit(
            "WARN", "SCHEMA.INVALID_TYPE", where, "Expected number; record rejected",
            record_kind=record_kind,
            path=".".join([record_kind] + path),
            got_type=type(val).__name__,
        )
        failures.append(
            {"reason": "invalid_type", "expected": "number", "record_kind": record_kind, "path": ".".join(path)}
        )
        return val, 0, 1, failures

    if ftype == "boolean":
        if isinstance(val, bool):
            return val, 0, 0, failures
        diagnostics.emit(
            "WARN", "SCHEMA.INVALID_TYPE", where, "Expected boolean; record rejected",
            record_kind=record_kind,
            path=".".join([record_kind] + path),
            got_type=type(val).__name__,
        )
        failures.append(
            {"reason": "invalid_type", "expected": "boolean", "record_kind": record_kind, "path": ".".join(path)}
        )
        return val, 0, 1, failures

    if ftype == "array":
        if isinstance(val, list):
            return val, 0, 0, failures
        diagnostics.emit(
            "WARN", "SCHEMA.INVALID_TYPE", where, "Expected array; record rejected",
            record_kind=record_kind,
            path=".".join([record_kind] + path),
            got_type=type(val).__name__,
        )
        failures.append(
            {"reason": "invalid_type", "expected": "array", "record_kind": record_kind, "path": ".".join(path)}
        )
        return val, 0, 1, failures

    if ftype == "object":
        if not isinstance(val, dict):
            diagnostics.emit(
                "WARN", "SCHEMA.INVALID_TYPE", where, "Expected object; record rejected",
                record_kind=record_kind,
                path=".".join([record_kind] + path),
                got_type=type(val).__name__,
            )
            failures.append(
                {"reason": "invalid_type", "expected": "object", "record_kind": record_kind, "path": ".".join(path)}
            )
            return val, 0, 1, failures

        # Only validate nested fields if a fixed-key "fields" schema exists.
        # Dynamic-key fields (clip_vision, masks, aux) use "per_key_shape" — skip deep validation.
        nested_defs = fdef.get("fields")
        if not isinstance(nested_defs, dict):
            return val, 0, 0, failures

        cleaned: dict = {}
        # Unknown nested keys → extras.
        for k, v in val.items():
            if k not in nested_defs:
                _extras_put(extras, [record_kind] + path, k, v)
                diagnostics.emit(
                    "WARN",
                    "SCHEMA.UNKNOWN_KEY",
                    where,
                    "Unknown nested key moved to extras",
                    record_kind=record_kind,
                    path=".".join([record_kind] + path),
                    key=k,
                )
                warn_count += 1
            else:
                cleaned[k] = v

        # Fill / validate required nested keys.
        for nk, nf in nested_defs.items():
            if not isinstance(nf, dict):
                continue  # skip _note_* docs that slipped through
            if nk not in cleaned:
                if nf.get("required"):
                    diagnostics.emit(
                        "WARN",
                        "SCHEMA.MISSING_FIELD",
                        where,
                        "Required nested field missing; record rejected",
                        record_kind=record_kind,
                        path=".".join([record_kind] + path),
                        field=nk,
                    )
                    error_count += 1
                    failures.append(
                        {
                            "reason": "missing_required_field",
                            "record_kind": record_kind,
                            "path": ".".join(path + [nk]),
                        }
                    )
                continue

            cleaned[nk], wc, ec, nested_failures = _coerce_value(
                cleaned[nk], nf, extras,
                record_kind=record_kind,
                path=path + [nk],
            )
            warn_count += wc
            error_count += ec
            failures.extend(nested_failures)

        return cleaned, warn_count, error_count, failures

    # Unknown / unsupported type (e.g. "valueref", "pixel_stats"): accept as-is.
    return val, warn_count, error_count, failures


def _compute_ingest_status(
    method_v: _ValidateResult,
    eval_v: _ValidateResult,
    sample_v: _ValidateResult,
) -> str:
    if method_v.error_count or eval_v.error_count or sample_v.error_count:
        return "ERROR"
    if method_v.warn_count or eval_v.warn_count or sample_v.warn_count:
        return "WARN"
    return "OK"


def _extras_put(extras: dict, prefix: list[str], key: str, value: Any) -> None:
    """Place unknown key into extras under a nested prefix."""
    cur = extras
    for part in prefix:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[key] = value
