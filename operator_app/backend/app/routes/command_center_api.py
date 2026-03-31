"""App backend routes that expose the stable Command Center surface."""

from __future__ import annotations

import contextlib
import io
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

import command_center as cc
from command_center.review_payload import write_review_dump as write_review_export


_SLASH = chr(47)
_PREFIX = f"{_SLASH}cc"
_SUMMARY_ROUTE = f"{_SLASH}summary"
_LIST_LORAS_ROUTE = f"{_SLASH}list-loras"
_SAMPLES_ROUTE = f"{_SLASH}samples"
_METHODS_ROUTE = f"{_SLASH}methods"
_EVALS_ROUTE = f"{_SLASH}evals"
_HEALTH_ROUTE = f"{_SLASH}health"
_ERRORS_ROUTE = f"{_SLASH}errors"
_REVIEW_DUMP_ROUTE = f"{_SLASH}review-dump"
_RUN_BATCH_ROUTE = f"{_SLASH}run-batch"
_LIST_WORKFLOWS_ROUTE = f"{_SLASH}list-workflows"
_ONBOARD_WORKFLOW_ROUTE = f"{_SLASH}onboard-workflow"

# Simple concurrency guard — one real (non-dry) batch run at a time.
_batch_running: bool = False

router = APIRouter(prefix=_PREFIX, tags=["command-center"])


@router.get("")
def command_center_index() -> dict[str, Any]:
    return {
        "surface": "command_center",
        "routes": {
            "summary": f"{_SLASH}api{_PREFIX}{_SUMMARY_ROUTE}",
            "list_loras": f"{_SLASH}api{_PREFIX}{_LIST_LORAS_ROUTE}",
            "get_sample": f"{_SLASH}api{_PREFIX}{_SAMPLES_ROUTE}{_SLASH}{{sample_hash}}",
            "get_method": f"{_SLASH}api{_PREFIX}{_METHODS_ROUTE}{_SLASH}{{method_hash}}",
            "get_eval": f"{_SLASH}api{_PREFIX}{_EVALS_ROUTE}{_SLASH}{{eval_hash}}",
            "list_evals": f"{_SLASH}api{_PREFIX}{_EVALS_ROUTE}",
            "health": f"{_SLASH}api{_PREFIX}{_HEALTH_ROUTE}",
            "errors": f"{_SLASH}api{_PREFIX}{_ERRORS_ROUTE}",
            "review_dump": f"{_SLASH}api{_PREFIX}{_REVIEW_DUMP_ROUTE}",
            "run_batch": f"{_SLASH}api{_PREFIX}{_RUN_BATCH_ROUTE}",
            "list_workflows": f"{_SLASH}api{_PREFIX}{_LIST_WORKFLOWS_ROUTE}",
            "onboard_workflow": f"{_SLASH}api{_PREFIX}{_ONBOARD_WORKFLOW_ROUTE}",
        },
    }


@router.get(_SUMMARY_ROUTE)
def summary(limit: int = Query(default=10, ge=1, le=500)) -> dict[str, Any]:
    return cc.summary(limit=limit)


@router.get(_LIST_LORAS_ROUTE)
def list_loras(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    return cc.list_loras(limit=limit)


@router.get(f"{_SAMPLES_ROUTE}{_SLASH}{{sample_hash}}")
def get_sample(sample_hash: str) -> dict[str, Any] | None:
    sample = cc.get_sample(sample_hash)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found")
    return sample


@router.get(f"{_METHODS_ROUTE}{_SLASH}{{method_hash}}")
def get_method(method_hash: str) -> dict[str, Any] | None:
    method = cc.get_method(method_hash)
    if method is None:
        raise HTTPException(status_code=404, detail="method not found")
    return method


@router.get(f"{_EVALS_ROUTE}{_SLASH}{{eval_hash}}")
def get_eval(eval_hash: str) -> dict[str, Any] | None:
    ev = cc.get_eval(eval_hash)
    if ev is None:
        raise HTTPException(status_code=404, detail="eval not found")
    return ev


@router.get(_EVALS_ROUTE)
def list_evals(
    limit: int = Query(default=10, ge=1, le=1000),
    method_hash: str | None = None,
    lora_hash: str | None = None,
) -> list[dict[str, Any]]:
    return cc.list_evals(limit=limit, method_hash=method_hash, lora_hash=lora_hash)


@router.get(_HEALTH_ROUTE)
def health(diagnostics_window: int = Query(default=100, ge=1, le=2000)) -> dict[str, Any]:
    return cc.health(diagnostics_window=diagnostics_window)


@router.get(_ERRORS_ROUTE)
def errors(limit: int = Query(default=10, ge=1, le=1000)) -> list[dict[str, Any]]:
    return cc.errors(limit=limit)


@router.post(_REVIEW_DUMP_ROUTE)
def write_review_dump() -> dict[str, Any]:
    return write_review_export()


@router.get(_LIST_WORKFLOWS_ROUTE)
def list_workflows() -> dict[str, Any]:
    """Return workspace (ready) and staging (pending onboard) workflow inventories."""
    return cc.list_workflows()


@router.post(_ONBOARD_WORKFLOW_ROUTE)
def onboard_workflow(filename: str = Query(...)) -> dict[str, Any]:
    """Onboard a raw workflow file from staging into the workspace.

    ``filename`` is relative to the staging directory (e.g. ``workflow_api.json``).
    Returns ``{"ok": true, "workflow_name": ..., "workspace_path": ...}`` on success,
    or ``{"ok": false, "error": ...}`` on topology failure or file-not-found.
    """
    return cc.onboard_workflow(filename)


@router.post(_RUN_BATCH_ROUTE)
def run_batch(
    background_tasks: BackgroundTasks,
    dry_run: bool = Query(default=True),
    limit: int | None = Query(default=None, ge=1, le=1000),
    force: bool = Query(default=False),
    include_dirty: bool = Query(default=False),
) -> dict[str, Any]:
    """Queue or dry-run the ComfyUI batch replay runner.

    With dry_run=true (default): runs synchronously, returns the plan output.
    With dry_run=false: queues a background run and returns immediately.
    Only one real batch run may be active at a time (409 if already running).
    """
    global _batch_running  # noqa: PLW0603

    if not dry_run and _batch_running:
        raise HTTPException(status_code=409, detail="A batch replay is already running.")

    if dry_run:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exit_code = cc.run_batch(dry_run=True, limit=limit, force=force, include_dirty=include_dirty)
        output = stdout_buf.getvalue()
        stderr_out = stderr_buf.getvalue()
        if stderr_out.strip():
            output = output + "\n[stderr]\n" + stderr_out
        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "dry_run": True,
            "output": output,
        }

    def _run_background() -> None:
        global _batch_running  # noqa: PLW0603
        try:
            cc.run_batch(dry_run=False, limit=limit, force=force, include_dirty=include_dirty)
        finally:
            _batch_running = False

    _batch_running = True
    background_tasks.add_task(_run_background)
    return {
        "ok": True,
        "exit_code": None,
        "dry_run": False,
        "output": "Batch replay queued. Monitor diagnostics log (errors command) for progress and results.",
    }
