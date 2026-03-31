"""databank/patching.py — Maintenance patching utilities for sample facts_json.

These functions are NOT part of the Treasurer interface (which is intentionally
append-only for the production write path). They exist for one-time migrations,
cleanup scripts, and DB maintenance operations that need to modify or remove
existing facts_json fields.

All SQL lives here — callers never touch sqlite3 directly; they call into this
module instead.

Canonical use: scripts in databank/operations/ call remove_facts_domain() or
patch_sample_facts() and handle emit() annotation themselves.

Convention: see databank/operations/CONVENTION.md before writing a new script.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable


def patch_sample_facts(
    db_path: str | Path,
    patcher: Callable[[dict], dict | None],
) -> tuple[int, int]:
    """Apply patcher to every sample's facts_json.

    patcher(facts: dict) -> dict | None
      - Return the modified dict if this record needs updating.
      - Return None to skip (record is already in the desired state).

    Returns (modified_count, skipped_count).

    The entire batch runs in a single transaction; either all updates commit
    or none do.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT sample_hash, facts_json FROM samples").fetchall()
        modified = 0
        skipped = 0
        with conn:
            for row in rows:
                facts = json.loads(row["facts_json"])
                result = patcher(facts)
                if result is None:
                    skipped += 1
                else:
                    conn.execute(
                        "UPDATE samples SET facts_json = ? WHERE sample_hash = ?",
                        (json.dumps(result, ensure_ascii=False), row["sample_hash"]),
                    )
                    modified += 1
        return modified, skipped
    finally:
        conn.close()


def remove_facts_domain(
    db_path: str | Path,
    *key_path: str,
) -> tuple[int, int]:
    """Remove a nested key from all sample facts_json records.

    key_path is a sequence of dict keys forming a path into facts_json.
    e.g. remove_facts_domain(db_path, "aux", "pose")
      removes facts["aux"]["pose"] from every sample that has it.

    Idempotent: samples that do not have the key are skipped.

    Returns (modified_count, skipped_count).
    """
    if not key_path:
        raise ValueError("key_path must have at least one element")

    def _patcher(facts: dict) -> dict | None:
        # Walk to the parent of the target key.
        node = facts
        for key in key_path[:-1]:
            if not isinstance(node, dict) or key not in node:
                return None  # path doesn't exist — skip
            node = node[key]
        leaf_key = key_path[-1]
        if not isinstance(node, dict) or leaf_key not in node:
            return None  # already absent — skip
        del node[leaf_key]
        return facts

    return patch_sample_facts(db_path, _patcher)
