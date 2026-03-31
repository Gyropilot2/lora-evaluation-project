"""databank/treasurer.py — Treasurer interface (DataBank front door).

Treasurer is the ONLY legitimate door into the DataBank subsystem.
All modules that need persistent Evidence access call these methods;
no module outside databank/ executes SQL directly or touches DB drivers.

Three-level hierarchy: Method → Eval → Sample.
Any future backend (Postgres, etc.) must implement this interface so that
switching backends is a localized change inside databank/.

Return type conventions:
- insert_method / insert_eval return True if inserted, False if already exists
  (idempotent — silence is not an error).
- get_* return None if the record is not found.
- get_method  returns inputs_json merged with the promoted is_dirty column.
- get_eval    returns inputs_json merged with the promoted is_dirty column.
- get_sample  returns facts_json merged with is_dirty, latent_hash,
              image_hash (all from promoted columns), plus _extras and _errors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Treasurer(ABC):
    """Abstract base class for all DataBank backends."""

    # ------------------------------------------------------------------
    # Method operations
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_method(self, record: dict) -> bool:
        """Insert a Method record.

        Idempotent: if a record with the same method_hash already exists,
        discard silently and return False.

        Returns:
            True  — record was inserted.
            False — method_hash already exists; discarded silently.
        """

    @abstractmethod
    def get_method(self, method_hash: str) -> dict | None:
        """Return the Method record by method_hash, or None if not found.

        The returned dict merges inputs_json with the promoted is_dirty column.
        """

    @abstractmethod
    def count_methods(self, filters: dict | None = None) -> int:
        """Count Method records matching the given filters.

        Supported filter keys:
            is_dirty (bool) — filter by dirty flag.
        """

    @abstractmethod
    def find_method_by_workflow_digest(self, digest: str) -> dict | None:
        """Return one Method record whose workflow_ref content digest matches.

        This is an orchestration helper for workflow-backed reuse lookup.
        It does not make workflow digest a new identity law; callers still
        treat method_hash as the actual Method identity.
        """

    @abstractmethod
    def set_method_dirty(self, method_hash: str) -> None:
        """Mark a Method record as dirty. Idempotent; cannot be cleared by the system."""

    @abstractmethod
    def set_method_workflow_ref(self, method_hash: str, workflow_ref: dict) -> bool:
        """Attach or replace a Method record's stored workflow_ref.

        Returns:
            True  — method existed and inputs_json was updated.
            False — method_hash was not found.
        """

    # ------------------------------------------------------------------
    # Eval operations
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_eval(self, record: dict) -> bool:
        """Insert an Eval record.

        Idempotent: if a record with the same eval_hash already exists,
        discard silently and return False.

        Returns:
            True  — record was inserted.
            False — eval_hash already exists; discarded silently.
        """

    @abstractmethod
    def get_eval(self, eval_hash: str) -> dict | None:
        """Return the Eval record by eval_hash, or None if not found.

        The returned dict merges inputs_json with the promoted is_dirty column.
        """

    @abstractmethod
    def query_evals(self, filters: dict | None = None) -> list[dict]:
        """Return Eval records matching the given filters.

        Supported filter keys:
            method_hash (str) — filter by parent Method.
            lora_hash (str)   — filter by LoRA content hash.
            is_dirty (bool)   — filter by dirty flag.
            limit (int)       — max records to return.
        """

    @abstractmethod
    def count_evals(self, filters: dict | None = None) -> int:
        """Count Eval records matching the given filters.

        Supported filter keys: method_hash (str), lora_hash (str), is_dirty (bool).
        """

    @abstractmethod
    def set_eval_dirty(self, eval_hash: str) -> None:
        """Mark an Eval record as dirty. Idempotent; cannot be cleared by the system."""

    # ------------------------------------------------------------------
    # LoRA catalog operations
    # ------------------------------------------------------------------

    @abstractmethod
    def query_loras(self, filters: dict | None = None) -> list[dict]:
        """Return LoRA catalog entries.

        Supported filter keys:
            lora_hash (str) — exact hash match.
            is_dirty (bool) — filter by dirty flag.
            limit (int)     — max records to return.
        """

    @abstractmethod
    def count_loras(self, filters: dict | None = None) -> int:
        """Count LoRA catalog entries.

        Supported filter keys: lora_hash (str), is_dirty (bool).
        """

    # ------------------------------------------------------------------
    # Sample operations
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_sample(self, record: dict, extras: dict) -> None:
        """Insert a Sample record and its extras.

        Args:
            record: The full sample dict (facts + metadata).  Stored in
                    facts_json; key fields also extracted to promoted columns.
                    Must contain: sample_hash, eval_hash, seed, latent_hash,
                    image_hash, ingest_status, timestamp.
            extras: The extras dict (goes into extras_json). May be empty.
        """

    @abstractmethod
    def get_sample(self, sample_hash: str) -> dict | None:
        """Return the Sample record by sample_hash, or None if not found.

        The returned dict merges facts_json with promoted columns (authoritative):
            latent_hash (str)   — vital integrity field
            image_hash  (str)   — vital slot-consistency field
            ingest_status (str) — OK | WARN | ERROR
            is_dirty (bool)     — dirty flag
        And sidecar blobs merged as:
            _extras (dict)  — from extras_json
            _errors (list)  — from errors_json
        """

    @abstractmethod
    def enrich_sample(self, sample_hash: str, measurement_delta: dict) -> None:
        """Merge new measurement slots into an existing Sample — no-clobber.

        For each top-level key in measurement_delta:
        - Absent from existing OR is an Invalid wrapper → set it.
        - Both existing and incoming values are non-invalid dicts (e.g. clip_vision,
          masks, aux, pose_evidence) → recursive merge: missing nested keys are added;
          existing valid nested values are kept.
        - Existing has a valid non-dict value (scalar, ValueRef) → skip.

        Emits SAMPLE.ENRICHED (INFO) for each enriched sample.
        Does nothing if sample_hash is not found.
        """

    @abstractmethod
    def query_samples(self, filters: dict | None = None) -> list[dict]:
        """Return Sample records matching the given filters.

        Supported filter keys:
            eval_hash (str)     — filter by parent Eval.
            is_dirty (bool)     — filter by dirty flag.
            ingest_status (str) — filter by OK | WARN | ERROR.
            limit (int)         — max records to return.
        """

    @abstractmethod
    def count_samples(self, filters: dict | None = None) -> int:
        """Count Sample records matching the given filters.

        Supported filter keys: eval_hash (str), is_dirty (bool), ingest_status (str).
        """

    @abstractmethod
    def list_samples_by_lora_hash(self, lora_hash: str) -> list[dict]:
        """Return all Samples belonging to Evals that have the given lora_hash."""

    def find_baseline_sample(self, lora_sample_hash: str) -> dict | None:
        """Resolve the clean baseline sample paired to one LoRA sample.

        Traversal stays inside the DataBank door:
        sample -> eval -> method -> clean baseline eval -> clean baseline samples,
        preferring the same seed and falling back to the first clean baseline sample.

        Returns None when:
        - the sample does not exist,
        - the sample is already a baseline sample,
        - no clean baseline eval exists for the method, or
        - no clean baseline samples exist under that eval.
        """
        sample = self.get_sample(lora_sample_hash)
        if sample is None:
            return None

        eval_hash = sample.get("eval_hash")
        if not eval_hash:
            return None

        eval_rec = self.get_eval(eval_hash)
        if eval_rec is None:
            return None

        lora_obj = eval_rec.get("lora") or {}
        lora_hash = lora_obj.get("hash")
        if lora_hash is None:
            return None

        method_hash = eval_rec.get("method_hash")
        if not method_hash:
            return None

        baseline_evals = self.query_evals(
            filters={"method_hash": method_hash, "lora_hash": None, "is_dirty": False}
        )
        if not baseline_evals:
            return None

        baseline_eval_hash = baseline_evals[0].get("eval_hash")
        if not baseline_eval_hash:
            return None

        baseline_samples = self.query_samples(
            filters={"eval_hash": baseline_eval_hash, "is_dirty": False}
        )
        if not baseline_samples:
            return None

        seed = sample.get("seed")
        if seed is not None:
            for baseline_sample in baseline_samples:
                if baseline_sample.get("seed") == seed:
                    return baseline_sample

        return baseline_samples[0]

    @abstractmethod
    def store_errors(self, sample_hash: str, errors: list[dict]) -> None:
        """Append error records to a Sample's errors column.

        The errors list is appended to any existing errors for that sample.
        """

    @abstractmethod
    def set_sample_dirty(self, sample_hash: str) -> None:
        """Mark a Sample record as dirty. Idempotent; cannot be cleared by the system."""

    @abstractmethod
    def get_all_extras(self, limit: int | None = None) -> list[dict]:
        """Return the extras_json dict for every Sample (used by health.extras_frequency).

        Args:
            limit: If set, return at most this many extras dicts.
        """

    @abstractmethod
    def get_all_facts_keys(self) -> list[set[str]]:
        """Return the set of top-level facts_json keys for every Sample.

        Used by DataBankHealth.scorecard() to compute per-domain coverage rates.
        Returns one set per sample (empty set if facts_json is absent or
        unparseable).  No ordering guarantee.
        """

    @abstractmethod
    def store_asset_blob(self, data: bytes, asset_type: str, fmt: str) -> dict:
        """Store one asset blob and return its committed ValueRef.

        Backends may implement this as a staged write plus final commit under
        the hood, but callers only see the committed ValueRef contract.

        Args:
            data: Raw asset bytes to persist.
            asset_type: Semantic category string for content-addressed layout.
            fmt: File extension without the leading dot.

        Returns:
            A clean committed ValueRef dict.
        """

    @abstractmethod
    def load_asset_blob(self, valueref: dict) -> bytes | dict:
        """Load one asset blob by ValueRef.

        Args:
            valueref: The committed ValueRef dict describing one stored asset.

        Returns:
            bytes on success.
            Invalid wrapper dict on read failure or hash mismatch.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any held resources (connections, file handles, etc.).

        Default implementation is a no-op; backends should override if needed.
        """


def open_treasurer(
    *,
    read_only: bool = False,
    event_emitter: Callable[..., Any] | None = None,
) -> Treasurer:
    """Open the current Treasurer backend through the public Treasurer door."""
    from core.paths import get_path
    from databank.sqlite_backend import SQLiteBackend

    return SQLiteBackend(
        get_path("db_file"),
        event_emitter=event_emitter,
        read_only=read_only,
    )
