"""
command_center/batch_replay.py — Replay runner internals library.

The canonical CLI entry point for replay is the unified batch orchestrator:

  python -m command_center.batch replay --workflow config/workflow_replay_api.json

This module is an internal library. `command_center/batch.py` imports the
replay-specific functions (`build_replay_plan`, `patch_replay_workflow`,
`parse_requested_paths`, and the node ID constants) directly.

The standalone `main()` at the bottom is a deprecated redirect stub.

ComfyUI HTTP transport has been extracted to `command_center/comfyui_client.py`.

----

Replay drives ComfyUI via its HTTP API to replay stored sample images through the
replay workflow and write additive `pose_evidence` back through
SampleReplayLoad -> SampleNeedsMeasurementGuard -> ReplayEnricher.

- no KSampler
- no latent regeneration
- uses stored sample image assets from the DB
- uses the DB-stored `masks.main_subject` slot as MainSubject fallback

By default the runner targets samples missing either:
  - pose_evidence.openpose_body
  - pose_evidence.dw_body

It skips already-complete samples unless --force is used.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from contracts.validation_errors import is_invalid
from command_center import comfyui_client as _http
from core.diagnostics import emit
from databank.treasurer import Treasurer, open_treasurer


# ---------------------------------------------------------------------------
# Replay workflow node IDs
# ---------------------------------------------------------------------------

N_REPLAY_LOAD = "1"
N_REPLAY_GUARD = "2"
N_REPLAY_ENRICHER = "3"

DEFAULT_REQUESTED_PATHS = [
    "pose_evidence.openpose_body",
    "pose_evidence.dw_body",
]

PATH_TO_INPUT_KEY = {
    "pose_evidence.openpose_body": "pose_keypoint_openpose",
    "pose_evidence.dw_body": "pose_keypoint_dw",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_requested_paths(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_REQUESTED_PATHS)
    items: list[str] = []
    for chunk in str(raw).replace(",", "\n").splitlines():
        clean = chunk.strip()
        if clean:
            items.append(clean)
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def record_has_valid_path(record: dict[str, Any], path: str) -> bool:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return False
        if part not in current:
            return False
        current = current[part]
        if is_invalid(current):
            return False
    return not is_invalid(current)


def missing_paths(record: dict[str, Any], requested_paths: list[str]) -> list[str]:
    return [path for path in requested_paths if not record_has_valid_path(record, path)]


def build_replay_plan(
    treasurer: Treasurer,
    *,
    requested_paths: list[str],
    sample_hashes: list[str] | None,
    limit: int | None,
    include_dirty: bool,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return replay plan rows with per-sample missing path info."""
    stats = {
        "scanned": 0,
        "skipped_dirty": 0,
        "skipped_complete": 0,
        "selected": 0,
        "missing_openpose": 0,
        "missing_dw": 0,
    }

    if sample_hashes:
        records = []
        for sample_hash in sample_hashes:
            rec = treasurer.get_sample(sample_hash)
            if rec is not None:
                records.append(rec)
    else:
        records = treasurer.query_samples({})

    plan: list[dict[str, Any]] = []
    for record in records:
        stats["scanned"] += 1
        if bool(record.get("is_dirty")) and not include_dirty:
            stats["skipped_dirty"] += 1
            continue

        needed = list(requested_paths) if force else missing_paths(record, requested_paths)
        if "pose_evidence.openpose_body" in needed:
            stats["missing_openpose"] += 1
        if "pose_evidence.dw_body" in needed:
            stats["missing_dw"] += 1

        if not needed:
            stats["skipped_complete"] += 1
            continue

        plan.append(
            {
                "sample_hash": str(record.get("sample_hash") or ""),
                "eval_hash": str(record.get("eval_hash") or ""),
                "is_dirty": bool(record.get("is_dirty")),
                "missing_paths": needed,
            }
        )

    plan.sort(key=lambda row: row["sample_hash"])
    if limit is not None:
        plan = plan[: max(0, int(limit))]

    stats["selected"] = len(plan)
    return plan, stats


def patch_replay_workflow(template: dict[str, Any], sample_hash: str, requested_paths: list[str]) -> dict[str, Any]:
    """Patch one replay workflow for one sample and one set of requested paths."""
    wf = copy.deepcopy(template)

    if N_REPLAY_LOAD not in wf or N_REPLAY_GUARD not in wf or N_REPLAY_ENRICHER not in wf:
        raise ValueError("Workflow missing one or more required replay node IDs (1, 2, 3).")

    wf[N_REPLAY_LOAD]["inputs"]["sample_hash"] = sample_hash
    wf[N_REPLAY_GUARD]["inputs"]["requested_paths"] = "\n".join(requested_paths)

    enrich_inputs = wf[N_REPLAY_ENRICHER].setdefault("inputs", {})
    for path, input_key in PATH_TO_INPUT_KEY.items():
        if path not in requested_paths:
            enrich_inputs.pop(input_key, None)

    return wf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Deprecated standalone entry point. Use: python -m command_center.batch replay ..."""
    sys.stderr.write(
        "batch_replay.main() is deprecated.\n"
        "Use: python -m command_center.batch replay ...\n"
    )
    return 1
