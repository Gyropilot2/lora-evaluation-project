"""Pure review-surface metadata helpers derived from contracts-owned registries."""

from __future__ import annotations

from contracts.metric_labels import METRIC_LABELS
from contracts.metrics_registry import all_metrics
from contracts.recipe import FEATURED_HERO_METRICS
from contracts.review_transport import HeroMetricDescriptor, MetricMetadata


def build_metric_metadata() -> dict[str, MetricMetadata]:
    out: dict[str, MetricMetadata] = {}
    for key, entry in all_metrics().items():
        out[key] = {
            "key": key,
            "label": METRIC_LABELS.get(key, key),
            "description": entry.get("description"),
            "tier": entry.get("tier"),
            "reliability": entry.get("reliability"),
            "polarity": entry.get("polarity"),
            "value_min": entry.get("value_min"),
            "value_max": entry.get("value_max"),
            "value_type": entry.get("value_type"),
            "reliability_metric_key": entry.get("reliability_metric_key"),
            "dropped_metric_key": entry.get("dropped_metric_key"),
            "component_metric_keys": entry.get("component_metric_keys") or [],
            "peer_metric_keys": entry.get("peer_metric_keys") or [],
            "selection_metric_keys": entry.get("selection_metric_keys") or [],
            "inspection_graph_metric_keys": entry.get("inspection_graph_metric_keys") or [],
        }
    return out


def build_hero_roster(metric_metadata: dict[str, MetricMetadata] | None = None) -> list[HeroMetricDescriptor]:
    metadata = metric_metadata or build_metric_metadata()
    out: list[HeroMetricDescriptor] = []
    for featured in FEATURED_HERO_METRICS:
        promoted = metadata.get(featured.metric_key, {})
        out.append(
            {
                "key": featured.key,
                "metric_key": featured.metric_key,
                "label": featured.label,
                "description": promoted.get("description"),
                "aggregate_fn_name": featured.aggregate_fn_name,
            }
        )
    return out
