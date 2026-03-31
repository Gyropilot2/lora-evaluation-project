"""command_center.health — operator-facing health/scorecard reporting."""

from __future__ import annotations

from databank.treasurer import Treasurer

_COVERAGE_DOMAINS: tuple[str, ...] = (
    "image",
    "luminance",
    "clip_vision",
    "masks",
    "face_analysis",
    "aux",
)


class DataBankHealth:
    """Operator-facing DataBank quality reporter over the Treasurer door."""

    def __init__(self, backend: Treasurer) -> None:
        self._db = backend

    def dirty_count(self) -> int:
        dirty_methods = self._db.count_methods(filters={"is_dirty": True})
        dirty_evals = self._db.count_evals(filters={"is_dirty": True})
        dirty_samples = self._db.count_samples(filters={"is_dirty": True})
        return dirty_methods + dirty_evals + dirty_samples

    def error_rate(self) -> float:
        total = self._db.count_samples()
        if total == 0:
            return 0.0
        error_count = self._db.count_samples(filters={"ingest_status": "ERROR"})
        return error_count / total

    def method_count(self) -> int:
        return self._db.count_methods()

    def eval_count(self) -> int:
        return self._db.count_evals()

    def sample_count(self) -> int:
        return self._db.count_samples()

    def extras_frequency(self) -> dict[str, float]:
        total = self._db.count_samples()
        if total == 0:
            return {}

        all_extras = self._db.get_all_extras()
        key_counts: dict[str, int] = {}
        for extras in all_extras:
            if not isinstance(extras, dict):
                continue
            for key in extras:
                key_counts[key] = key_counts.get(key, 0) + 1

        return {key: count / total for key, count in key_counts.items()}

    def scorecard(self) -> dict:
        total_samples = self._db.count_samples()

        counts = {
            "methods": self._db.count_methods(),
            "evals": self._db.count_evals(),
            "samples": total_samples,
        }

        dirty_methods = self._db.count_methods(filters={"is_dirty": True})
        dirty_evals = self._db.count_evals(filters={"is_dirty": True})
        dirty_samples = self._db.count_samples(filters={"is_dirty": True})
        dirty_total = dirty_methods + dirty_evals + dirty_samples
        dirty = {
            "methods": dirty_methods,
            "evals": dirty_evals,
            "samples": dirty_samples,
            "total": dirty_total,
            "sample_dirty_rate": dirty_samples / total_samples if total_samples > 0 else 0.0,
        }

        ok_count = self._db.count_samples(filters={"ingest_status": "OK"})
        warn_count = self._db.count_samples(filters={"ingest_status": "WARN"})
        error_count = self._db.count_samples(filters={"ingest_status": "ERROR"})
        ingest_status = {
            "ok_count": ok_count,
            "warn_count": warn_count,
            "error_count": error_count,
            "ok_rate": ok_count / total_samples if total_samples > 0 else 0.0,
            "warn_rate": warn_count / total_samples if total_samples > 0 else 0.0,
            "error_rate": error_count / total_samples if total_samples > 0 else 0.0,
        }

        all_keys = self._db.get_all_facts_keys()
        coverage: dict[str, float] = {}
        for domain in _COVERAGE_DOMAINS:
            present = sum(1 for keys in all_keys if domain in keys)
            coverage[domain] = present / total_samples if total_samples > 0 else 0.0

        return {
            "counts": counts,
            "dirty": dirty,
            "ingest_status": ingest_status,
            "coverage": coverage,
            "extras": self.extras_frequency(),
        }
