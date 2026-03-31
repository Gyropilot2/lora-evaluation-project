"""
databank/sqlite_backend.py — SQLite WAL-mode implementation of Treasurer.

This is the ONLY file in the project that imports sqlite3.
All SQL lives here. No other module may construct or execute raw SQL.

Table layout (post-migration v2):
  methods          — one row per Method (generative harness)
  evals            — one row per Eval (Method + LoRA identity)
  samples          — one row per Sample (Eval + seed + lora_strength)
  loras            — flat LoRA metadata catalog keyed by lora_hash
  schema_migrations — version-stamped migration log

WAL mode allows concurrent readers with a single logical writer.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from contracts.validation_errors import is_invalid
from core.diagnostics import emit
from core.time_ids import now_iso
from databank import assets as _assets
from databank.treasurer import Treasurer

# ---------------------------------------------------------------------------
# Schema migrations
# Each entry: (version: int, description: str, sql: str)
# Applied in order; each version runs exactly once.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Initial schema: runs, samples, schema_migrations",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT    NOT NULL,
            description TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id        TEXT    PRIMARY KEY,
            run_signature TEXT    NOT NULL UNIQUE,
            created_at    TEXT    NOT NULL,
            lora_hash     TEXT,
            base_hash     TEXT,
            prompt_family TEXT,
            seed          INTEGER,
            strength      REAL,
            is_dirty      INTEGER NOT NULL DEFAULT 0,
            inputs_json   TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_lora_hash     ON runs (lora_hash);
        CREATE INDEX IF NOT EXISTS idx_runs_run_signature ON runs (run_signature);
        CREATE INDEX IF NOT EXISTS idx_runs_is_dirty      ON runs (is_dirty);

        CREATE TABLE IF NOT EXISTS samples (
            sample_id     TEXT    PRIMARY KEY,
            ingest_status TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            run_ids       TEXT    NOT NULL,
            is_dirty      INTEGER NOT NULL DEFAULT 0,
            facts_json    TEXT    NOT NULL,
            extras_json   TEXT    NOT NULL DEFAULT '{}',
            errors_json   TEXT    NOT NULL DEFAULT '[]'
        );

        CREATE INDEX IF NOT EXISTS idx_samples_ingest_status ON samples (ingest_status);
        CREATE INDEX IF NOT EXISTS idx_samples_is_dirty      ON samples (is_dirty);
        """,
    ),
    (
        2,
        "3-table schema: methods, evals, samples (replaces runs + old samples)",
        """
        DROP TABLE IF EXISTS runs;
        DROP TABLE IF EXISTS samples;

        CREATE TABLE IF NOT EXISTS methods (
            method_hash TEXT    PRIMARY KEY,
            created_at  TEXT    NOT NULL,
            base_hash   TEXT,
            vae_hash    TEXT,
            is_dirty    INTEGER NOT NULL DEFAULT 0,
            inputs_json TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_methods_is_dirty  ON methods (is_dirty);
        CREATE INDEX IF NOT EXISTS idx_methods_base_hash ON methods (base_hash);

        CREATE TABLE IF NOT EXISTS evals (
            eval_hash   TEXT    PRIMARY KEY,
            method_hash TEXT    NOT NULL REFERENCES methods (method_hash),
            lora_hash   TEXT,
            created_at  TEXT    NOT NULL,
            is_dirty    INTEGER NOT NULL DEFAULT 0,
            inputs_json TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_evals_method_hash ON evals (method_hash);
        CREATE INDEX IF NOT EXISTS idx_evals_lora_hash   ON evals (lora_hash);
        CREATE INDEX IF NOT EXISTS idx_evals_is_dirty    ON evals (is_dirty);

        CREATE TABLE IF NOT EXISTS samples (
            sample_hash   TEXT    PRIMARY KEY,
            eval_hash     TEXT    NOT NULL REFERENCES evals (eval_hash),
            seed          INTEGER NOT NULL,
            lora_strength REAL,
            latent_hash   TEXT    NOT NULL,
            image_hash    TEXT    NOT NULL,
            ingest_status TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            is_dirty      INTEGER NOT NULL DEFAULT 0,
            facts_json    TEXT    NOT NULL,
            extras_json   TEXT    NOT NULL DEFAULT '{}',
            errors_json   TEXT    NOT NULL DEFAULT '[]'
        );

        CREATE INDEX IF NOT EXISTS idx_samples_eval_hash     ON samples (eval_hash);
        CREATE INDEX IF NOT EXISTS idx_samples_ingest_status ON samples (ingest_status);
        CREATE INDEX IF NOT EXISTS idx_samples_is_dirty      ON samples (is_dirty);
        """,
    ),
    (
        3,
        "Add LoRA catalog table and backfill from eval inputs_json",
        """
        CREATE TABLE IF NOT EXISTS loras (
            lora_hash             TEXT    PRIMARY KEY,
            file_hash             TEXT,
            name                  TEXT,
            rank                  INTEGER,
            network_alpha         REAL,
            target_blocks_json    TEXT,
            affects_text_encoder  INTEGER,
            raw_metadata_json     TEXT,
            created_at            TEXT    NOT NULL,
            is_dirty              INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_loras_name     ON loras (name);
        CREATE INDEX IF NOT EXISTS idx_loras_is_dirty ON loras (is_dirty);

        INSERT OR IGNORE INTO loras (
            lora_hash, file_hash, name, rank, network_alpha,
            target_blocks_json, affects_text_encoder, raw_metadata_json,
            created_at, is_dirty
        )
        SELECT
            e.lora_hash,
            json_extract(e.inputs_json, '$.lora.file_hash') AS file_hash,
            json_extract(e.inputs_json, '$.lora.name') AS name,
            json_extract(e.inputs_json, '$.lora.rank') AS rank,
            json_extract(e.inputs_json, '$.lora.network_alpha') AS network_alpha,
            json_extract(e.inputs_json, '$.lora.target_blocks') AS target_blocks_json,
            json_extract(e.inputs_json, '$.lora.affects_text_encoder') AS affects_text_encoder,
            json_extract(e.inputs_json, '$.lora.raw_metadata') AS raw_metadata_json,
            e.created_at,
            0
        FROM evals e
        WHERE e.lora_hash IS NOT NULL;
        """,
    ),
]

_CURRENT_VERSION = max(v for v, _, _ in _MIGRATIONS)


# ---------------------------------------------------------------------------
# Scalar extraction helpers (handles Invalid wrappers gracefully)
# ---------------------------------------------------------------------------


def _safe_str(val: object) -> str | None:
    if val is None or is_invalid(val):
        return None
    if isinstance(val, str):
        return val
    return str(val)


def _safe_int(val: object) -> int | None:
    if val is None or is_invalid(val):
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_float(val: object) -> float | None:
    if val is None or is_invalid(val):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Promoted-column extractors
# ---------------------------------------------------------------------------


def _promoted_method_cols(record: dict) -> dict:
    """Extract promoted column values from a Method record dict."""
    base = record.get("base_model") or {}
    vae = record.get("vae_model") or {}
    if is_invalid(base):
        base = {}
    if is_invalid(vae):
        vae = {}
    return {
        "base_hash": _safe_str(base.get("hash")),
        "vae_hash": _safe_str(vae.get("hash")),
    }


def _promoted_eval_cols(record: dict) -> dict:
    """Extract promoted column values from an Eval record dict."""
    lora = record.get("lora") or {}
    if is_invalid(lora):
        lora = {}
    return {
        "lora_hash": _safe_str(lora.get("hash")),
    }


# ---------------------------------------------------------------------------
# Row → dict helpers
# ---------------------------------------------------------------------------


def _row_to_method(row: sqlite3.Row) -> dict:
    """Convert a sqlite3 Row from the methods table to a plain dict.

    is_dirty from the promoted column is authoritative (Bouncer may flip it
    after initial insertion).
    """
    result: dict = json.loads(row["inputs_json"])
    result["is_dirty"] = bool(row["is_dirty"])
    return result


def _row_to_eval(row: sqlite3.Row) -> dict:
    """Convert a sqlite3 Row from the evals table to a plain dict.

    is_dirty from the promoted column is authoritative.
    """
    result: dict = json.loads(row["inputs_json"])
    result["is_dirty"] = bool(row["is_dirty"])
    return result


def _row_to_sample(row: sqlite3.Row) -> dict:
    """Convert a sqlite3 Row from the samples table to a plain dict.

    Vital promoted columns (latent_hash, image_hash, ingest_status, is_dirty)
    take precedence over any stale values in the JSON blob.
    _extras and _errors are merged in from their sidecar columns.
    """
    result: dict = json.loads(row["facts_json"])
    result["latent_hash"] = row["latent_hash"]
    result["image_hash"] = row["image_hash"]
    result["ingest_status"] = row["ingest_status"]
    result["is_dirty"] = bool(row["is_dirty"])
    result["_extras"] = json.loads(row["extras_json"])
    result["_errors"] = json.loads(row["errors_json"])
    return result


def _row_to_lora(row: sqlite3.Row) -> dict:
    """Convert a sqlite3 Row from loras table to plain dict."""
    return {
        "lora_hash": row["lora_hash"],
        "file_hash": row["file_hash"],
        "name": row["name"],
        "rank": row["rank"],
        "network_alpha": row["network_alpha"],
        "target_blocks": json.loads(row["target_blocks_json"]) if row["target_blocks_json"] else None,
        "affects_text_encoder": None if row["affects_text_encoder"] is None else bool(row["affects_text_encoder"]),
        "raw_metadata": json.loads(row["raw_metadata_json"]) if row["raw_metadata_json"] else None,
        "created_at": row["created_at"],
        "is_dirty": bool(row["is_dirty"]),
    }


# ---------------------------------------------------------------------------
# No-clobber merge helper (for enrich_sample)
# ---------------------------------------------------------------------------


def _merge_no_clobber(existing: dict, delta: dict) -> dict:
    """Merge delta into existing using recursive no-clobber rules.

    - Absent in existing, or existing value is_invalid → set it.
    - Existing valid scalar/bool/number/string/list/ValueRef → keep existing.
    - Existing valid dict + incoming valid dict → recurse key-by-key.
    - Lists/arrays are treated as atomic; existing valid list wins.
    """
    return _merge_no_clobber_value(existing, delta)


def _merge_no_clobber_value(existing: object, incoming: object) -> object:
    """Return the no-clobber merge of two values."""
    if is_invalid(existing):
        return incoming
    if is_invalid(incoming):
        return existing

    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged: dict = dict(existing)
        for key, incoming_value in incoming.items():
            if key not in merged:
                merged[key] = incoming_value
                continue
            merged[key] = _merge_no_clobber_value(merged[key], incoming_value)
        return merged

    return existing


# ---------------------------------------------------------------------------
# WHERE clause builders
# ---------------------------------------------------------------------------


def _build_method_where(filters: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if "is_dirty" in filters:
        clauses.append("is_dirty = ?")
        params.append(1 if filters["is_dirty"] else 0)
    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def _build_eval_where(filters: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if "method_hash" in filters:
        clauses.append("method_hash = ?")
        params.append(filters["method_hash"])
    if "lora_hash" in filters:
        if filters["lora_hash"] is None:
            # SQLite: NULL = NULL is NULL (not TRUE). Must use IS NULL.
            clauses.append("lora_hash IS NULL")
        else:
            clauses.append("lora_hash = ?")
            params.append(filters["lora_hash"])
    if "is_dirty" in filters:
        clauses.append("is_dirty = ?")
        params.append(1 if filters["is_dirty"] else 0)
    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def _build_sample_where(filters: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if "eval_hash" in filters:
        clauses.append("eval_hash = ?")
        params.append(filters["eval_hash"])
    if "is_dirty" in filters:
        clauses.append("is_dirty = ?")
        params.append(1 if filters["is_dirty"] else 0)
    if "ingest_status" in filters:
        clauses.append("ingest_status = ?")
        params.append(filters["ingest_status"])
    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def _build_lora_where(filters: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if "lora_hash" in filters:
        clauses.append("lora_hash = ?")
        params.append(filters["lora_hash"])
    if "is_dirty" in filters:
        clauses.append("is_dirty = ?")
        params.append(1 if filters["is_dirty"] else 0)
    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# SQLiteBackend
# ---------------------------------------------------------------------------


class SQLiteBackend(Treasurer):
    """SQLite WAL-mode implementation of Treasurer.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    event_emitter : callable, optional
        Diagnostic emitter. Defaults to core.diagnostics.emit.
    read_only : bool, optional
        When True, opens the database in SQLite URI read-only mode
        (``?mode=ro``).  No migrations are applied, no WAL setup is
        performed, and the connection cannot write.  Use this for all
        analysis / review workloads to guarantee the invariant
        "read-only sessions never write to the DB".  Defaults to False.
    """

    def __init__(
        self,
        db_path: Path,
        event_emitter: Callable[..., dict[str, Any]] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._emit_event = event_emitter or emit
        self._read_only = read_only

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn_get(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        return self._conn  # type: ignore[return-value]

    def _connect(self) -> None:
        if self._read_only:
            self._connect_read_only()
        else:
            self._connect_read_write()

    def _connect_read_write(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        self._conn = conn
        self._apply_migrations()

    def _connect_read_only(self) -> None:
        """Open a read-only connection via SQLite URI mode.

        Does NOT run migrations or configure WAL/journal — those require
        write access.  foreign_keys and busy_timeout are session-level
        no-write PRAGMAs and are set for correctness and reliability.

        Raises sqlite3.OperationalError if the database file does not exist.
        """
        # Build a file URI.  Path.as_uri() handles Windows drive letters
        # (e.g. C:\... → file:///C:/...).
        base_uri = self._db_path.as_uri()       # e.g. file:///C:/data/lora_eval.db
        uri = base_uri + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _apply_migrations(self) -> None:
        """Create schema_migrations table and apply any pending migrations."""
        conn = self._conn  # type: ignore[assignment]
        # Bootstrap: create migrations table first (it may not exist yet)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT    NOT NULL,
                description TEXT    NOT NULL
            )
            """
        )
        conn.commit()

        applied: set[int] = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }

        for version, description, sql in _MIGRATIONS:
            if version in applied:
                continue
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
                (version, now_iso(), description),
            )
            conn.commit()
            self._emit_event(
                "INFO",
                "DB.MIGRATION_APPLIED",
                "databank.sqlite_backend._apply_migrations",
                f"Applied migration v{version}: {description}",
                version=version,
            )

    # ------------------------------------------------------------------
    # Method operations
    # ------------------------------------------------------------------

    def insert_method(self, record: dict) -> bool:
        """Insert a Method record. Returns False (silently) if method_hash exists."""
        conn = self._conn_get()
        method_hash = record.get("method_hash")
        if method_hash is None:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.insert_method",
                "method record missing method_hash — cannot insert",
            )
            return False

        existing = conn.execute(
            "SELECT method_hash FROM methods WHERE method_hash = ? LIMIT 1",
            (method_hash,),
        ).fetchone()
        if existing is not None:
            return False

        promoted = _promoted_method_cols(record)
        is_dirty = 1 if record.get("is_dirty") else 0

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO methods
                        (method_hash, created_at, base_hash, vae_hash, is_dirty, inputs_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        method_hash,
                        record.get("timestamp", now_iso()),
                        promoted["base_hash"],
                        promoted["vae_hash"],
                        is_dirty,
                        json.dumps(record, ensure_ascii=False),
                    ),
                )
        except sqlite3.Error as exc:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.insert_method",
                f"Failed to insert method: {exc}",
                method_hash=method_hash,
            )
            raise
        return True

    def get_method(self, method_hash: str) -> dict | None:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT * FROM methods WHERE method_hash = ?", (method_hash,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_method(row)

    def count_methods(self, filters: dict | None = None) -> int:
        conn = self._conn_get()
        where, params = _build_method_where(filters or {})
        row = conn.execute(f"SELECT COUNT(*) FROM methods{where}", params).fetchone()
        return row[0]

    def find_method_by_workflow_digest(self, digest: str) -> dict | None:
        conn = self._conn_get()
        row = conn.execute(
            """
            SELECT *
            FROM methods
            WHERE json_extract(inputs_json, '$.workflow_ref.asset_type') = ?
              AND json_extract(inputs_json, '$.workflow_ref.content_hash.digest') = ?
            LIMIT 1
            """,
            ("workflow_json", digest),
        ).fetchone()
        if row is None:
            return None
        return _row_to_method(row)

    def set_method_dirty(self, method_hash: str) -> None:
        conn = self._conn_get()
        with conn:
            conn.execute(
                "UPDATE methods SET is_dirty = 1 WHERE method_hash = ?",
                (method_hash,),
            )

    def set_method_workflow_ref(self, method_hash: str, workflow_ref: dict) -> bool:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT inputs_json FROM methods WHERE method_hash = ?",
            (method_hash,),
        ).fetchone()
        if row is None:
            return False

        record = json.loads(row["inputs_json"])
        record["workflow_ref"] = workflow_ref
        with conn:
            conn.execute(
                "UPDATE methods SET inputs_json = ? WHERE method_hash = ?",
                (json.dumps(record, ensure_ascii=False), method_hash),
            )
        return True

    # ------------------------------------------------------------------
    # Eval operations
    # ------------------------------------------------------------------

    def insert_eval(self, record: dict) -> bool:
        """Insert an Eval record. Returns False (silently) if eval_hash exists."""
        conn = self._conn_get()
        eval_hash = record.get("eval_hash")
        if eval_hash is None:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.insert_eval",
                "eval record missing eval_hash — cannot insert",
            )
            return False

        existing = conn.execute(
            "SELECT eval_hash FROM evals WHERE eval_hash = ? LIMIT 1",
            (eval_hash,),
        ).fetchone()
        if existing is not None:
            return False

        promoted = _promoted_eval_cols(record)
        method_hash = record.get("method_hash")
        is_dirty = 1 if record.get("is_dirty") else 0
        lora_obj = record.get("lora") if isinstance(record.get("lora"), dict) else {}

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO evals
                        (eval_hash, method_hash, lora_hash, created_at, is_dirty, inputs_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eval_hash,
                        method_hash,
                        promoted["lora_hash"],
                        record.get("timestamp", now_iso()),
                        is_dirty,
                        json.dumps(record, ensure_ascii=False),
                    ),
                )

                # Maintain LoRA catalog as a first-class entity.
                if promoted["lora_hash"]:
                    target_blocks = lora_obj.get("target_blocks")
                    raw_metadata = lora_obj.get("raw_metadata")
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO loras
                            (lora_hash, file_hash, name, rank, network_alpha,
                             target_blocks_json, affects_text_encoder, raw_metadata_json,
                             created_at, is_dirty)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            promoted["lora_hash"],
                            _safe_str(lora_obj.get("file_hash")),
                            _safe_str(lora_obj.get("name")),
                            _safe_int(lora_obj.get("rank")),
                            _safe_float(lora_obj.get("network_alpha")),
                            json.dumps(target_blocks, ensure_ascii=False) if target_blocks is not None else None,
                            None if lora_obj.get("affects_text_encoder") is None else (1 if bool(lora_obj.get("affects_text_encoder")) else 0),
                            json.dumps(raw_metadata, ensure_ascii=False) if raw_metadata is not None else None,
                            record.get("timestamp", now_iso()),
                            is_dirty,
                        ),
                    )
        except sqlite3.Error as exc:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.insert_eval",
                f"Failed to insert eval: {exc}",
                eval_hash=eval_hash,
            )
            raise
        return True

    def get_eval(self, eval_hash: str) -> dict | None:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT * FROM evals WHERE eval_hash = ?", (eval_hash,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_eval(row)

    def query_evals(self, filters: dict | None = None) -> list[dict]:
        conn = self._conn_get()
        where, params = _build_eval_where(filters or {})
        sql = f"SELECT * FROM evals{where}"
        if (filters or {}).get("limit"):
            sql += " LIMIT ?"
            params.append(int(filters["limit"]))  # type: ignore[index]
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_eval(r) for r in rows]

    def count_evals(self, filters: dict | None = None) -> int:
        conn = self._conn_get()
        where, params = _build_eval_where(filters or {})
        row = conn.execute(f"SELECT COUNT(*) FROM evals{where}", params).fetchone()
        return row[0]

    def set_eval_dirty(self, eval_hash: str) -> None:
        conn = self._conn_get()
        with conn:
            conn.execute(
                "UPDATE evals SET is_dirty = 1 WHERE eval_hash = ?",
                (eval_hash,),
            )

    # ------------------------------------------------------------------
    # LoRA catalog operations
    # ------------------------------------------------------------------

    def query_loras(self, filters: dict | None = None) -> list[dict]:
        conn = self._conn_get()
        where, params = _build_lora_where(filters or {})
        sql = f"SELECT * FROM loras{where}"
        if (filters or {}).get("limit"):
            sql += " LIMIT ?"
            params.append(int(filters["limit"]))  # type: ignore[index]
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_lora(r) for r in rows]

    def count_loras(self, filters: dict | None = None) -> int:
        conn = self._conn_get()
        where, params = _build_lora_where(filters or {})
        row = conn.execute(f"SELECT COUNT(*) FROM loras{where}", params).fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Sample operations
    # ------------------------------------------------------------------

    def insert_sample(self, record: dict, extras: dict) -> None:
        conn = self._conn_get()
        is_dirty = 1 if record.get("is_dirty") else 0
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO samples
                        (sample_hash, eval_hash, seed, lora_strength,
                         latent_hash, image_hash, ingest_status,
                         created_at, is_dirty, facts_json, extras_json, errors_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["sample_hash"],
                        record["eval_hash"],
                        record["seed"],
                        record.get("lora_strength"),  # None → NULL for Baseline
                        record["latent_hash"],
                        record["image_hash"],
                        record.get("ingest_status", "OK"),
                        record.get("timestamp", now_iso()),
                        is_dirty,
                        json.dumps(record, ensure_ascii=False),
                        json.dumps(extras, ensure_ascii=False),
                        "[]",
                    ),
                )
        except sqlite3.Error as exc:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.insert_sample",
                f"Failed to insert sample: {exc}",
                sample_hash=record.get("sample_hash"),
            )
            raise

    def get_sample(self, sample_hash: str) -> dict | None:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT * FROM samples WHERE sample_hash = ?", (sample_hash,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_sample(row)

    def enrich_sample(self, sample_hash: str, measurement_delta: dict) -> None:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT facts_json FROM samples WHERE sample_hash = ?", (sample_hash,)
        ).fetchone()
        if row is None:
            return  # not found — no-op per spec
        existing = json.loads(row["facts_json"])
        merged = _merge_no_clobber(existing, measurement_delta)
        if merged == existing:
            return  # no-op: nothing to enrich (all slots already set with valid values)
        with conn:
            conn.execute(
                "UPDATE samples SET facts_json = ? WHERE sample_hash = ?",
                (json.dumps(merged, ensure_ascii=False), sample_hash),
            )
        self._emit_event(
            "INFO",
            "SAMPLE.ENRICHED",
            "databank.sqlite_backend.enrich_sample",
            "Sample enriched with new measurement slots",
            sample_hash=sample_hash,
        )

    def query_samples(self, filters: dict | None = None) -> list[dict]:
        conn = self._conn_get()
        where, params = _build_sample_where(filters or {})
        sql = f"SELECT * FROM samples{where}"
        if (filters or {}).get("limit"):
            sql += " LIMIT ?"
            params.append(int(filters["limit"]))  # type: ignore[index]
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_sample(r) for r in rows]

    def count_samples(self, filters: dict | None = None) -> int:
        conn = self._conn_get()
        where, params = _build_sample_where(filters or {})
        row = conn.execute(
            f"SELECT COUNT(*) FROM samples{where}", params
        ).fetchone()
        return row[0]

    def list_samples_by_lora_hash(self, lora_hash: str) -> list[dict]:
        conn = self._conn_get()
        rows = conn.execute(
            """
            SELECT s.* FROM samples s
            JOIN evals e ON s.eval_hash = e.eval_hash
            WHERE e.lora_hash = ?
            """,
            (lora_hash,),
        ).fetchall()
        return [_row_to_sample(r) for r in rows]

    def store_errors(self, sample_hash: str, errors: list[dict]) -> None:
        conn = self._conn_get()
        row = conn.execute(
            "SELECT errors_json FROM samples WHERE sample_hash = ?", (sample_hash,)
        ).fetchone()
        if row is None:
            emit(
                "ERROR",
                "DB.WRITE_FAIL",
                "databank.sqlite_backend.store_errors",
                "Sample not found — cannot store errors",
                sample_hash=sample_hash,
            )
            return
        existing: list = json.loads(row["errors_json"])
        existing.extend(errors)
        with conn:
            conn.execute(
                "UPDATE samples SET errors_json = ? WHERE sample_hash = ?",
                (json.dumps(existing, ensure_ascii=False), sample_hash),
            )

    def set_sample_dirty(self, sample_hash: str) -> None:
        conn = self._conn_get()
        with conn:
            conn.execute(
                "UPDATE samples SET is_dirty = 1 WHERE sample_hash = ?",
                (sample_hash,),
            )

    def get_all_extras(self, limit: int | None = None) -> list[dict]:
        conn = self._conn_get()
        sql = "SELECT extras_json FROM samples"
        params: list = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [json.loads(r["extras_json"]) for r in rows]

    def get_all_facts_keys(self) -> list[set[str]]:
        conn = self._conn_get()
        rows = conn.execute("SELECT facts_json FROM samples").fetchall()
        result: list[set[str]] = []
        for row in rows:
            try:
                d = json.loads(row[0])
                result.append(set(d.keys()) if isinstance(d, dict) else set())
            except (json.JSONDecodeError, TypeError):
                result.append(set())
        return result

    def store_asset_blob(self, data: bytes, asset_type: str, fmt: str) -> dict:
        staged = _assets.stage_blob(data, asset_type, fmt)
        return _assets.commit_blob(staged)

    def load_asset_blob(self, valueref: dict) -> bytes | dict:
        return _assets.load_blob(valueref)
