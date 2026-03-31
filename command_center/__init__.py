"""command_center — minimal operator surface for CLI access.

This module provides a thin façade that routes operator-oriented queries through
Treasurer/DataBank primitives plus diagnostics log access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from command_center.health import DataBankHealth
from core.diagnostics import emit
from core.paths import get_path
from databank.treasurer import Treasurer, open_treasurer


_DEFAULT_RECENT_LIMIT = 10


def emit_runtime_event(
    severity: str,
    code: str,
    where: str,
    msg: str,
    **ctx: Any,
) -> dict:
    """Command Center façade for runtime diagnostic events."""
    return emit(severity, code, where, msg, **ctx)


def new_treasurer(*, read_only: bool = False) -> Treasurer:
    """Create a Treasurer backend instance for callers that need explicit control."""
    return open_treasurer(
        event_emitter=emit_runtime_event,
        read_only=read_only,
    )


def _new_backend() -> Treasurer:
    return new_treasurer(read_only=False)


def summary(limit: int = _DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
    """Return a compact ingest summary for operator visibility."""
    backend = _new_backend()
    try:
        health = DataBankHealth(backend)
        recent_samples = backend.query_samples(filters={"limit": limit})
        recent_evals = backend.query_evals(filters={"limit": limit})
        return {
            "counts": {
                "methods": health.method_count(),
                "evals": health.eval_count(),
                "samples": health.sample_count(),
                "dirty_total": health.dirty_count(),
            },
            "recent": {
                "samples": recent_samples,
                "evals": recent_evals,
            },
        }
    finally:
        backend.close()


def list_loras(limit: int = 100) -> list[dict[str, Any]]:
    """Return LoRA inventory from the dedicated LoRA catalog table."""
    backend = _new_backend()
    try:
        rows = backend.query_loras(filters={"limit": limit})
        out: list[dict[str, Any]] = []
        for rec in rows:
            lora_hash = rec.get("lora_hash")
            if not isinstance(lora_hash, str) or lora_hash == "":
                continue
            eval_count = backend.count_evals(filters={"lora_hash": lora_hash})
            sample_count = len(backend.list_samples_by_lora_hash(lora_hash))
            out.append(
                {
                    "lora_hash": lora_hash,
                    "name": rec.get("name"),
                    "file_hash": rec.get("file_hash"),
                    "rank": rec.get("rank"),
                    "network_alpha": rec.get("network_alpha"),
                    "target_blocks": rec.get("target_blocks"),
                    "affects_text_encoder": rec.get("affects_text_encoder"),
                    "eval_count": eval_count,
                    "sample_count": sample_count,
                    "is_dirty": rec.get("is_dirty", False),
                }
            )
        return sorted(out, key=lambda x: x["eval_count"], reverse=True)
    finally:
        backend.close()


def get_sample(sample_hash: str) -> dict[str, Any] | None:
    """Return one sample record by hash."""
    backend = _new_backend()
    try:
        return backend.get_sample(sample_hash)
    finally:
        backend.close()


def get_method(method_hash: str) -> dict[str, Any] | None:
    """Return one method record by hash."""
    backend = _new_backend()
    try:
        return backend.get_method(method_hash)
    finally:
        backend.close()


def get_eval(eval_hash: str) -> dict[str, Any] | None:
    """Return one eval record by hash."""
    backend = _new_backend()
    try:
        return backend.get_eval(eval_hash)
    finally:
        backend.close()


def list_evals(
    limit: int = _DEFAULT_RECENT_LIMIT,
    method_hash: str | None = None,
    lora_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Return eval records with optional method/lora filtering."""
    backend = _new_backend()
    try:
        filters: dict[str, Any] = {"limit": limit}
        if method_hash:
            filters["method_hash"] = method_hash
        if lora_hash:
            filters["lora_hash"] = lora_hash
        return backend.query_evals(filters=filters)
    finally:
        backend.close()


def health(diagnostics_window: int = 100) -> dict[str, Any]:
    """Return DataBank and diagnostics health indicators.

    Args:
        diagnostics_window: Number of recent WARN/ERROR/FATAL log entries to
            examine for the scorecard diagnostics section.  Defaults to 100.

    Returns a dict with three top-level keys:
      ``db``          — legacy flat summary (backward-compatible).
      ``diagnostics`` — legacy diagnostics summary (backward-compatible).
      ``scorecard``   — Data Quality Scorecard v1 (structured; see health.py).
    """
    backend = _new_backend()
    try:
        h = DataBankHealth(backend)
        sc = h.scorecard()

        # Add diagnostics section — reads log file; not DataBankHealth's role.
        recent = errors(limit=diagnostics_window)
        sc["diagnostics"] = {
            "warn_count":  sum(1 for e in recent if e.get("severity") == "WARN"),
            "error_count": sum(1 for e in recent if e.get("severity") == "ERROR"),
            "fatal_count": sum(1 for e in recent if e.get("severity") == "FATAL"),
            "window":      diagnostics_window,
        }

        return {
            # --- backward-compatible legacy keys ---
            "db": {
                "methods":           sc["counts"]["methods"],
                "evals":             sc["counts"]["evals"],
                "samples":           sc["counts"]["samples"],
                "dirty_total":       sc["dirty"]["total"],
                "error_rate":        sc["ingest_status"]["error_rate"],
                "extras_frequency":  sc["extras"],
            },
            "diagnostics": {
                "log_path":     str(_diagnostics_path()),
                "recent_errors": sc["diagnostics"]["error_count"]
                                + sc["diagnostics"]["fatal_count"],
            },
            # --- scorecard v1 ---
            "scorecard": sc,
        }
    finally:
        backend.close()


def _diagnostics_path() -> Path:
    return get_path("logs_root") / "diagnostics.jsonl"


def run_batch(
    workflow_path: str | Path | None = None,
    *,
    paths: str | None = None,
    sample_hashes: list[str] | None = None,
    limit: int | None = None,
    include_dirty: bool = False,
    force: bool = False,
    dry_run: bool = False,
    poll: float = 3.0,
) -> int:
    """Run the ComfyUI batch replay runner. Returns exit code (0 = success).

    Canonical CLI entry point: ``python -m command_center.batch replay``.
    This facade is retained for programmatic callers and the Command Center
    public API surface.

    Args:
        workflow_path: Path to replay workflow JSON. Defaults to
            config/workflow_replay_api.json (the canonical location).
        paths: Comma-separated requested replay paths. Defaults to both
            pose_evidence sources (openpose_body, dw_body).
        sample_hashes: Replay only these specific sample hashes.
        limit: Max number of samples after filtering.
        include_dirty: Include dirty samples (default: skip).
        force: Queue even if the sample already has the requested paths.
        dry_run: Print replay plan without sending anything to ComfyUI.
        poll: Polling interval in seconds while waiting for each job.
    """
    from command_center.batch import main as _run_batch

    argv: list[str] = ["replay"]
    if workflow_path is not None:
        argv += ["--workflow", str(workflow_path)]
    if paths is not None:
        argv += ["--paths", paths]
    for sh in sample_hashes or []:
        argv += ["--sample-hash", sh]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if include_dirty:
        argv.append("--include-dirty")
    if force:
        argv.append("--force")
    if dry_run:
        argv.append("--dry-run")
    if poll != 3.0:
        argv += ["--poll", str(poll)]
    return _run_batch(argv)


def list_workflows() -> dict[str, Any]:
    """Return workspace and staging workflow inventory."""
    workspace = get_path("workflows_root")
    staging = get_path("workflows_staging_root")
    ready = sorted(f.stem for f in workspace.glob("*.json")) if workspace.exists() else []
    pending = sorted(f.name for f in staging.glob("*.json")) if staging.exists() else []
    return {
        "ready": ready,
        "pending_onboard": pending,
        "workspace_path": str(workspace),
        "staging_path": str(staging),
    }


def onboard_workflow(filename: str) -> dict[str, Any]:
    """Onboard a raw workflow file from staging into the workspace.

    ``filename`` is resolved relative to ``workflows_staging_root``.
    Returns a result dict with ``ok``, ``workflow_name``, and ``workspace_path``.
    On topology failure returns ``ok=False`` with an ``error`` key instead of raising.
    """
    from command_center.workflow_onboard import onboard

    staging = get_path("workflows_staging_root")
    raw_path = staging / filename
    if not raw_path.exists():
        return {
            "ok": False,
            "error": f"File not found in staging: {raw_path}",
            "staging_path": str(staging),
        }
    try:
        out_path = onboard(raw_path)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "workflow_name": out_path.stem,
        "workspace_path": str(out_path),
    }


def errors(limit: int = _DEFAULT_RECENT_LIMIT) -> list[dict[str, Any]]:
    """Return recent WARN/ERROR/FATAL diagnostic records."""
    path = _diagnostics_path()
    if not path.exists():
        return []

    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("severity") in {"WARN", "ERROR", "FATAL"}:
                out.append(rec)
    if limit <= 0:
        return out
    return out[-limit:]
