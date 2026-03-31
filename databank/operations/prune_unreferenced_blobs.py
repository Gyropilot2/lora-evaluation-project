"""Remove asset blobs that are no longer referenced anywhere in the DataBank.

This is the disk-cleanup companion to data-maintenance tools such as
``purge_method.py``. It scans stored JSON records for Methods, Evals, Samples,
and LoRA catalog entries, collects every referenced asset content hash, and
reports or deletes files under ``assets/`` whose digest is no longer reachable
from the current DB state.

Run from project root:
    python databank/operations/prune_unreferenced_blobs.py
    python databank/operations/prune_unreferenced_blobs.py --delete

Idempotent: yes. Once a blob is removed, later runs simply stop listing it.

Cleanup: this script is a standing maintenance tool; keep it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core.paths as paths
from core.diagnostics import emit
from databank.sqlite_backend import SQLiteBackend


emit(
    "WARN",
    "ARCH.CHEAT",
    "prune_unreferenced_blobs",
    "Maintenance script that scans DataBank JSON records for asset refs and removes "
    "asset files whose content hashes are no longer referenced anywhere in the DB.",
    cleanup="Keep as a standing operator tool; use after purges or other data cleanup work.",
)


def _collect_content_hashes(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        if value.get("kind") == "asset_ref":
            content_hash = value.get("content_hash")
            if isinstance(content_hash, dict):
                digest = content_hash.get("digest")
                if isinstance(digest, str) and digest:
                    out.add(digest)
        for child in value.values():
            _collect_content_hashes(child, out)
        return
    if isinstance(value, list):
        for child in value:
            _collect_content_hashes(child, out)


def _collect_table_json_column(
    backend: SQLiteBackend,
    table: str,
    column: str,
    out: set[str],
) -> None:
    conn = backend._conn_get()
    rows = conn.execute(f"SELECT {column} FROM {table}").fetchall()
    for row in rows:
        payload = row[column]
        if not isinstance(payload, str) or not payload:
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        _collect_content_hashes(decoded, out)


def _referenced_content_hashes() -> set[str]:
    backend = SQLiteBackend(
        paths.get_path("db_file"),
        event_emitter=emit,
        read_only=True,
    )
    referenced: set[str] = set()
    try:
        _collect_table_json_column(backend, "methods", "inputs_json", referenced)
        _collect_table_json_column(backend, "evals", "inputs_json", referenced)
        for sample in backend.query_samples({}):
            _collect_content_hashes(sample, referenced)
        for lora in backend.query_loras({}):
            _collect_content_hashes(lora, referenced)
    finally:
        backend.close()
    return referenced


def _all_asset_files() -> list[Path]:
    assets_root = paths.get_path("assets_root")
    return [path for path in assets_root.rglob("*") if path.is_file()]


def _unreferenced_files(referenced_hashes: set[str]) -> list[Path]:
    orphaned: list[Path] = []
    for path in _all_asset_files():
        if path.stem not in referenced_hashes:
            orphaned.append(path)
    return sorted(orphaned)


def _print_summary(title: str, files: list[Path]) -> None:
    project_root = paths.get_path("project_root")
    by_suffix = Counter(path.suffix.lower() or "<no_ext>" for path in files)
    by_asset_type = Counter(path.parent.name for path in files)

    print(title)
    print(f"  unreferenced_blobs={len(files)}")
    print(f"  by_suffix={dict(sorted(by_suffix.items()))}")
    print(f"  by_asset_type={dict(sorted(by_asset_type.items()))}")
    preview = [path.relative_to(project_root).as_posix() for path in files[:20]]
    if preview:
        print("  first_20:")
        for rel in preview:
            print(f"    {rel}")


def run_dry() -> int:
    referenced = _referenced_content_hashes()
    orphans = _unreferenced_files(referenced)
    _print_summary("Dry-run summary", orphans)
    return 0


def run_delete() -> int:
    referenced = _referenced_content_hashes()
    orphans = _unreferenced_files(referenced)
    _print_summary("Delete summary (pre-delete)", orphans)
    deleted = 0
    errors = 0
    for path in orphans:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            errors += 1
            print(f"ERROR delete {path}: {exc}")
    print(f"  deleted={deleted}")
    print(f"  delete_errors={errors}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune unreferenced asset blobs from assets/.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually remove unreferenced blobs. Default is dry-run report only.",
    )
    args = parser.parse_args()

    if args.delete:
        return run_delete()
    return run_dry()


if __name__ == "__main__":
    raise SystemExit(main())
