"""
Canonical review-surface aggregation and computation functions.

Registry owns metric truth and metric-owned anatomy.
Procedures own per-sample interpretation logic.
Recipe owns consumer policy: which metrics to expose, in what order, and which
aggregation strategy applies when rolling sample-level readings upward.
Aggregation owns the actual computation: the functions that reduce, compare, and
combine metric values — both across samples and within a paired baseline/LoRA comparison.

Aggregation does not declare which metrics are featured or in what order.
That is Recipe's job. Aggregation only answers "how do the numbers work."

Landing zone for review computation consolidation:
When sample-time review computation moves out of operator_app, it lands here —
pair assembly, package math, masked region comparisons, and per-sample
procedure application all belong in this file.
"""

from __future__ import annotations

from typing import Any


def numeric_metric(metrics: dict[str, Any], key: str) -> float | None:
    """Read one finite numeric metric value from a metrics dict."""
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number == number and number not in (float("inf"), float("-inf")):
            return number
    return None


def mean_metric(sample_metrics: list[dict[str, Any]], key: str) -> float | None:
    """Arithmetic mean across present numeric sample values."""
    values = [value for metrics in sample_metrics if (value := numeric_metric(metrics, key)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def penalized_mean(sample_metrics: list[dict[str, Any]], key: str) -> float | None:
    """Denominator-penalized mean across the full sample population.

    Dropped samples contribute to the denominator, not the numerator, so the
    aggregate falls as support disappears.
    """
    total_count = len(sample_metrics)
    if total_count == 0:
        return None
    survivors = [value for metrics in sample_metrics if (value := numeric_metric(metrics, key)) is not None]
    if not survivors:
        return None
    return sum(survivors) / total_count


def imputed_mean(sample_metrics: list[dict[str, Any]], key: str) -> float | None:
    """Impute dropped values with survivor mean, then average the full cohort.

    This treats `dropped_fraction` as its own explicit trend channel instead of
    silently punishing the score line for the same missing-support event.

    Equivalent behavior:
    - if survivors exist, return their arithmetic mean
    - if every sample is dropped / non-numeric, return None
    """
    return mean_metric(sample_metrics, key)
