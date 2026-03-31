"""bouncer/schema_loader.py — loads and caches the Evidence contract.

Bouncer validates incoming candidate records against the Evidence contract
field definitions.

This module:
  - Loads contracts/evidence.schema.json once (cached)
  - Exposes method/eval/sample field definition dicts
  - Exposes per-record-type vital field lists
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.paths import get_path

_CACHE: dict[str, Any] | None = None


def load_evidence_contract() -> dict[str, Any]:
    """Return the full Evidence contract dict (cached)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    schema_path = _schema_path()
    _CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
    return _CACHE


def evidence_version() -> str:
    """Return Evidence version string from schema meta."""
    schema = load_evidence_contract()
    meta = schema.get("_meta", {})
    return str(meta.get("version") or "")


def vital_fields_for(record_type: str) -> list[str]:
    """Return vital field names for the given record type.

    Args:
        record_type: One of "method_record", "eval_record", "sample_record".

    Returns:
        List of vital field names. Empty list if record_type is unknown.
    """
    schema = load_evidence_contract()
    vf = schema.get("vital_fields")
    if not isinstance(vf, dict):
        return []
    fields = vf.get(record_type)
    if isinstance(fields, list):
        return [str(x) for x in fields]
    return []


def method_field_defs() -> dict[str, dict[str, Any]]:
    """Return field definitions for the Method record."""
    schema = load_evidence_contract()
    raw = schema.get("method_record", {}).get("fields", {}) or {}
    # Filter out _note_* documentation keys (string values, not dict field defs).
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def eval_field_defs() -> dict[str, dict[str, Any]]:
    """Return field definitions for the Eval record."""
    schema = load_evidence_contract()
    raw = schema.get("eval_record", {}).get("fields", {}) or {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def sample_field_defs() -> dict[str, dict[str, Any]]:
    """Return field definitions for the Sample record."""
    schema = load_evidence_contract()
    raw = schema.get("sample_record", {}).get("fields", {}) or {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility; use the typed versions above)
# ---------------------------------------------------------------------------

def vital_fields() -> list[str]:
    """Return flat list of all vital fields across all record types (legacy).

    Deprecated: prefer vital_fields_for(record_type).
    """
    schema = load_evidence_contract()
    vf = schema.get("vital_fields")
    if isinstance(vf, list):
        # Old flat-list schema format.
        return [str(x) for x in vf]
    if isinstance(vf, dict):
        # New per-record-type format: flatten all lists.
        result: list[str] = []
        for fields in vf.values():
            if isinstance(fields, list):
                result.extend(str(x) for x in fields)
        return result
    return []


def run_field_defs() -> dict[str, dict[str, Any]]:
    """Legacy alias — the 'run_record' key no longer exists in v2 schema.

    Returns empty dict. Callers should migrate to method_field_defs().
    """
    schema = load_evidence_contract()
    raw = schema.get("run_record", {}).get("fields", {}) or {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _schema_path() -> Path:
    # contracts/evidence.schema.json lives under project_root
    return get_path("project_root") / "contracts" / "evidence.schema.json"
