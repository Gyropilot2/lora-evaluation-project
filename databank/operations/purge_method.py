"""Purge a Method and all its child Evals/Samples from the DataBank.

Use this when a run produced bad data — wrong workflow, wrong workflow settings,
test noise, etc. — and you want a clean slate so the same seeds/strengths can
be re-ingested without triggering NOT_ZERO_DELTA conflicts.

Deletes bottom-up (FK constraint order):
    1. samples  WHERE eval_hash IN (evals for this method)
    2. evals    WHERE method_hash = <target>
    3. methods  WHERE method_hash = <target>
    4. loras    WHERE lora_hash NOT IN (any remaining eval) — orphan cleanup

Asset blobs are NOT removed by this script; run prune_unreferenced_blobs.py
afterward if you want to reclaim disk space.

Usage:
    # List all methods so you can identify the hash:
    python databank/operations/purge_method.py --list

    # Dry-run (default): show exactly what would be deleted:
    python databank/operations/purge_method.py --method-hash <hash>
    python databank/operations/purge_method.py --latest

    # Actually delete:
    python databank/operations/purge_method.py --method-hash <hash> --delete
    python databank/operations/purge_method.py --latest --delete

Idempotent: yes (re-running after a successful delete is a safe no-op).

Cleanup: this script is a standing maintenance tool; keep it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.paths as paths
from core.diagnostics import emit
from databank.sqlite_backend import SQLiteBackend

emit(
    "WARN",
    "ARCH.CHEAT",
    "purge_method",
    "Maintenance script that deletes a Method and all child Evals/Samples via direct "
    "SQLiteBackend connection access. Treasurer exposes no delete door; raw SQL "
    "deletion is intentional here and confined to databank/operations/.",
    cleanup="Keep as a standing operator tool; no automatic retirement condition.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_backend(read_only: bool) -> SQLiteBackend:
    return SQLiteBackend(
        paths.get_path("db_file"),
        event_emitter=emit,
        read_only=read_only,
    )


def _list_methods(backend: SQLiteBackend) -> None:
    conn = backend._conn_get()
    rows = conn.execute(
        """
        SELECT
            m.method_hash,
            m.created_at,
            m.is_dirty,
            m.inputs_json,
            COUNT(DISTINCT e.eval_hash)  AS eval_count,
            COUNT(DISTINCT s.sample_hash) AS sample_count
        FROM methods m
        LEFT JOIN evals   e ON e.method_hash  = m.method_hash
        LEFT JOIN samples s ON s.eval_hash    = e.eval_hash
        GROUP BY m.method_hash
        ORDER BY m.created_at DESC
        """
    ).fetchall()

    if not rows:
        print("No methods in DB.")
        return

    print(f"{'HASH':>12}  {'CREATED':>24}  {'DIRTY':>5}  {'EVALS':>5}  {'SAMPLES':>7}  BASE_MODEL")
    print("-" * 100)
    for r in rows:
        inputs = json.loads(r["inputs_json"])
        base_model = inputs.get("base_model") or inputs.get("settings", {}).get("base_model") or "?"
        dirty_flag = "YES" if r["is_dirty"] else "-"
        print(
            f"{r['method_hash'][:12]}  "
            f"{r['created_at']:>24}  "
            f"{dirty_flag:>5}  "
            f"{r['eval_count']:>5}  "
            f"{r['sample_count']:>7}  "
            f"{base_model}"
        )
    print(f"\n{len(rows)} method(s) total.")


def _resolve_method_hash(backend: SQLiteBackend, method_hash: str) -> str | None:
    """Resolve a full or prefix hash. Returns full hash if exactly one match, else None."""
    conn = backend._conn_get()
    rows = conn.execute(
        "SELECT method_hash FROM methods WHERE method_hash LIKE ?",
        (method_hash + "%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]["method_hash"]
    if len(rows) == 0:
        print(f"ERROR: no method found matching prefix {method_hash!r}")
    else:
        print(f"ERROR: prefix {method_hash!r} is ambiguous — matches {len(rows)} methods:")
        for r in rows:
            print(f"  {r['method_hash']}")
    return None


def _resolve_latest(backend: SQLiteBackend) -> str | None:
    conn = backend._conn_get()
    row = conn.execute(
        "SELECT method_hash FROM methods ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        print("ERROR: no methods in DB.")
        return None
    return row["method_hash"]


def _describe_method(backend: SQLiteBackend, method_hash: str) -> dict:
    conn = backend._conn_get()

    method_row = conn.execute(
        "SELECT * FROM methods WHERE method_hash = ?", (method_hash,)
    ).fetchone()

    eval_rows = conn.execute(
        "SELECT eval_hash, lora_hash, is_dirty FROM evals WHERE method_hash = ?",
        (method_hash,),
    ).fetchall()

    eval_hashes = [r["eval_hash"] for r in eval_rows]
    lora_hashes = {r["lora_hash"] for r in eval_rows if r["lora_hash"] is not None}

    sample_count = 0
    if eval_hashes:
        placeholders = ",".join("?" * len(eval_hashes))
        row = conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE eval_hash IN ({placeholders})",
            eval_hashes,
        ).fetchone()
        sample_count = row[0]

    # Which lora_hashes would become orphaned after deletion?
    orphaned_loras: set[str] = set()
    for lh in lora_hashes:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM evals WHERE lora_hash = ? AND method_hash != ?",
            (lh, method_hash),
        ).fetchone()[0]
        if remaining == 0:
            orphaned_loras.add(lh)

    inputs = json.loads(method_row["inputs_json"]) if method_row else {}

    return {
        "method_hash": method_hash,
        "created_at": method_row["created_at"] if method_row else "?",
        "is_dirty": bool(method_row["is_dirty"]) if method_row else False,
        "inputs": inputs,
        "eval_hashes": eval_hashes,
        "lora_hashes": list(lora_hashes),
        "orphaned_lora_hashes": list(orphaned_loras),
        "sample_count": sample_count,
    }


def _print_plan(info: dict) -> None:
    print(f"\nTarget method : {info['method_hash']}")
    print(f"Created       : {info['created_at']}")
    print(f"Dirty         : {info['is_dirty']}")
    base_model = info["inputs"].get("base_model") or "?"
    print(f"Base model    : {base_model}")
    print(f"Evals         : {len(info['eval_hashes'])}")
    print(f"Samples       : {info['sample_count']}")
    print(f"Lora hashes   : {len(info['lora_hashes'])}")
    if info["orphaned_lora_hashes"]:
        print(f"Orphaned loras (will also be deleted): {len(info['orphaned_lora_hashes'])}")
        for lh in info["orphaned_lora_hashes"]:
            print(f"  {lh}")
    else:
        print("Orphaned loras: none (lora_hash refs still used by other methods)")
    print()


# ---------------------------------------------------------------------------
# Core delete logic
# ---------------------------------------------------------------------------

def _purge(backend: SQLiteBackend, method_hash: str, dry_run: bool) -> int:
    info = _describe_method(backend, method_hash)
    if info["method_hash"] not in [method_hash]:
        print("ERROR: method not found after resolution.")
        return 1

    _print_plan(info)

    if dry_run:
        print("DRY-RUN: no changes made. Pass --delete to execute.")
        return 0

    conn = backend._conn_get()
    eval_hashes = info["eval_hashes"]

    with conn:
        # 1. Samples
        if eval_hashes:
            placeholders = ",".join("?" * len(eval_hashes))
            result = conn.execute(
                f"DELETE FROM samples WHERE eval_hash IN ({placeholders})",
                eval_hashes,
            )
            print(f"Deleted {result.rowcount} sample(s).")

        # 2. Evals
        result = conn.execute(
            "DELETE FROM evals WHERE method_hash = ?", (method_hash,)
        )
        print(f"Deleted {result.rowcount} eval(s).")

        # 3. Method
        result = conn.execute(
            "DELETE FROM methods WHERE method_hash = ?", (method_hash,)
        )
        print(f"Deleted {result.rowcount} method(s).")

        # 4. Orphaned loras
        if info["orphaned_lora_hashes"]:
            placeholders = ",".join("?" * len(info["orphaned_lora_hashes"]))
            result = conn.execute(
                f"DELETE FROM loras WHERE lora_hash IN ({placeholders})",
                info["orphaned_lora_hashes"],
            )
            print(f"Deleted {result.rowcount} orphaned lora record(s).")
        else:
            print("No orphaned lora records to delete.")

    print("\nDone. Run prune_unreferenced_blobs.py if you also want to reclaim asset blobs.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purge a Method and all its child Evals/Samples from the DataBank."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--list", action="store_true", help="List all methods and exit.")
    target.add_argument("--method-hash", metavar="HASH", help="Full or prefix hash of the method to purge.")
    target.add_argument("--latest", action="store_true", help="Target the most recently created method.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete. Default is dry-run (show plan only).",
    )
    args = parser.parse_args()

    if args.list:
        backend = _open_backend(read_only=True)
        try:
            _list_methods(backend)
        finally:
            backend.close()
        return 0

    if not args.method_hash and not args.latest:
        parser.error("Provide --method-hash, --latest, or --list.")

    backend = _open_backend(read_only=False)
    try:
        if args.latest:
            method_hash = _resolve_latest(backend)
        else:
            method_hash = _resolve_method_hash(backend, args.method_hash)

        if method_hash is None:
            return 1

        return _purge(backend, method_hash, dry_run=not args.delete)
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
