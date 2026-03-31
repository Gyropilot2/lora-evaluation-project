"""
contracts/validation_errors.py — Invalid wrapper helpers.

The Invalid wrapper is the canonical representation of a missing, erroneous,
or inapplicable field value in Evidence.  It is a plain dict with a "status" key.

Never use Python None or JSON null for Evidence field values — use these helpers.

Status values (from Glossary):
  NULL           — missing / unknown / empty
  ERROR          — extraction or computation failed (system investigation required)
  NOT_APPLICABLE — concept does not apply to this record

Usage:
    from contracts.validation_errors import make_null, make_error, is_invalid

    value = make_null(reason_code="MISSING_LORA_NAME", detail="LoRA name not provided")
    # → {"status": "NULL", "reason_code": "MISSING_LORA_NAME", "detail": "..."}

    if is_invalid(some_field_value):
        ...
"""

from typing import Any

_VALID_STATUSES = frozenset({"NULL", "ERROR", "NOT_APPLICABLE"})


def make_null(reason_code: str | None = None, detail: str | None = None) -> dict:
    """Return an Invalid wrapper with status=NULL.

    Args:
        reason_code: Optional stable code string (e.g. "MISSING_LORA_NAME").
        detail:      Optional short explanation.
    """
    return _build("NULL", reason_code, detail)


def make_error(reason_code: str | None = None, detail: str | None = None) -> dict:
    """Return an Invalid wrapper with status=ERROR.

    Args:
        reason_code: Optional stable code string (e.g. "HASH_READ_FAIL").
        detail:      Optional short explanation.
    """
    return _build("ERROR", reason_code, detail)


def make_not_applicable(
    reason_code: str | None = None, detail: str | None = None
) -> dict:
    """Return an Invalid wrapper with status=NOT_APPLICABLE.

    Args:
        reason_code: Optional stable code string (e.g. "NO_MASK_PROVIDED").
        detail:      Optional short explanation.
    """
    return _build("NOT_APPLICABLE", reason_code, detail)


def is_invalid(val: Any) -> bool:
    """Return True if val is an Invalid wrapper dict.

    An Invalid wrapper is any dict with a "status" key whose value is one of
    NULL | ERROR | NOT_APPLICABLE.  All other dicts and non-dict values return False.
    """
    if not isinstance(val, dict):
        return False
    return val.get("status") in _VALID_STATUSES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build(status: str, reason_code: str | None, detail: str | None) -> dict:
    record: dict = {"status": status}
    if reason_code is not None:
        record["reason_code"] = reason_code
    if detail is not None:
        record["detail"] = detail
    return record
