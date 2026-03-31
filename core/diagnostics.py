"""
core/diagnostics.py — structured diagnostic emission.

This is the ONLY module permitted to write log output.
All other modules call emit() instead of using print() or logging directly.

Severity levels: DEBUG | INFO | WARN | ERROR | FATAL

Records are written as JSON Lines to the diagnostics log file (logs_root/diagnostics.jsonl)
and echoed to stderr for WARN and above.

Note: log file writing is best-effort and silently skipped on failures.
"""

import json
import sys
from typing import Any

VALID_SEVERITIES = frozenset({"DEBUG", "INFO", "WARN", "ERROR", "FATAL"})


def emit(
    severity: str,
    code: str,
    where: str,
    msg: str,
    **ctx: Any,
) -> dict:
    """Emit a structured diagnostic record.

    Args:
        severity: One of DEBUG | INFO | WARN | ERROR | FATAL.
        code:     Stable dot-separated identifier (e.g. ARCH.CHEAT, SCHEMA.MISSING_FIELD).
                  Must exist in contracts/diagnostic_codes.py once that registry is created.
        where:    Module/class/function string identifying origin (e.g. "bouncer.gate.process").
        msg:      Readable message.
        **ctx:    Arbitrary structured context key/value pairs.

    Returns:
        The structured record dict (for callers that want to inspect it).
    """
    if severity not in VALID_SEVERITIES:
        severity = "WARN"

    record = _build_record(severity, code, where, msg, ctx)
    _write(record)
    return record


def _build_record(
    severity: str,
    code: str,
    where: str,
    msg: str,
    ctx: dict,
) -> dict:
    from core.time_ids import now_iso  # late import to avoid circular risk

    return {
        "severity": severity,
        "code": code,
        "where": where,
        "msg": msg,
        "ctx": ctx,
        "timestamp": now_iso(),
    }


def _write(record: dict) -> None:
    # Echo WARN and above to stderr
    if record["severity"] in ("WARN", "ERROR", "FATAL"):
        line = (
            f"[{record['severity']}] {record['code']} @ {record['where']}: {record['msg']}"
        )
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    # Best-effort write to log file
    try:
        from core.paths import get_path  # late import to avoid circular risk

        log_dir = get_path("logs_root")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "diagnostics.jsonl"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Log file write is best-effort; stderr is the fallback
