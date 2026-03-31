"""
Canonical review-surface recipe choices.

Registry owns metric truth and metric-owned anatomy.
Procedures own per-sample interpretation logic.
Aggregation owns the actual computation: reduction functions, pair math, package scores.
Recipe owns consumer policy: which metrics this consumer exposes, in what order,
and which named aggregation strategy applies when rolling sample-level readings upward.

Recipe does not redefine what a metric is and does not implement computation.
It selects from already-defined metrics, names the aggregation strategy for each,
and declares the featured hero roster for this consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from contracts.aggregation import imputed_mean, mean_metric, penalized_mean


@dataclass(frozen=True)
class FeaturedMetricSpec:
    key: str
    metric_key: str
    label: str
    aggregate_fn_name: str


AGGREGATION_FUNCTIONS: dict[str, Callable[[list[dict[str, Any]], str], float | None]] = {
    "mean": mean_metric,
    "imputed_mean": imputed_mean,
    "penalized_mean": penalized_mean,
}


FEATURED_HERO_METRICS: list[FeaturedMetricSpec] = [
    FeaturedMetricSpec(
        key="identity",
        metric_key="identity_region_plus_arcface_exp",
        label="Identity",
        aggregate_fn_name="imputed_mean",
    ),
    FeaturedMetricSpec(
        key="pose",
        metric_key="pose_angle_drift",
        label="Pose",
        aggregate_fn_name="imputed_mean",
    ),
    FeaturedMetricSpec(
        key="background",
        metric_key="siglip_cos_bg",
        label="Background",
        aggregate_fn_name="mean",
    ),
    FeaturedMetricSpec(
        key="composition",
        metric_key="bg_depth_diff",
        label="Composition",
        aggregate_fn_name="mean",
    ),
    FeaturedMetricSpec(
        key="global_semantic",
        metric_key="clip_global_cos_dist",
        label="Global Semantic",
        aggregate_fn_name="mean",
    ),
    FeaturedMetricSpec(
        key="clothing",
        metric_key="siglip_cos_cloth",
        label="Clothing",
        aggregate_fn_name="mean",
    ),
]


def get_aggregate_fn(name: str) -> Callable[[list[dict[str, Any]], str], float | None]:
    """Resolve one named aggregation function."""
    try:
        return AGGREGATION_FUNCTIONS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown aggregation function: {name}") from exc
