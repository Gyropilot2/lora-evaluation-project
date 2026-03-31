"""Shared review-payload shaping for gallery and operator-app consumers."""

from __future__ import annotations

from typing import Any

from contracts.aggregation import numeric_metric
from contracts.review_transport import (
    HeroAggregate,
    MethodSlice,
    MetricMetadata,
    ReviewCounts,
    ReviewPayload,
    SampleSlice,
)
from contracts.review_surface import build_hero_roster, build_metric_metadata
from contracts.recipe import FEATURED_HERO_METRICS, get_aggregate_fn
from command_center.review_builder import build_data
from core.paths import get_path


def summarize_review_payload(data: dict[str, Any]) -> ReviewCounts:
    """Small top-level counts for quick operator and agent sanity checks."""
    methods = data.get("methods", [])
    evals = strengths = samples = 0
    for method in methods:
        for eval_item in method.get("evals", []):
            evals += 1
            if eval_item.get("is_baseline"):
                samples += len(eval_item.get("samples", []))
            else:
                strengths += len(eval_item.get("strengths", []))
                for strength in eval_item.get("strengths", []):
                    samples += len(strength.get("samples", []))
    return {
        "methods": len(methods),
        "evals": evals,
        "strength_groups": strengths,
        "samples": samples,
    }

def _dropped_fraction(
    samples: list[dict[str, Any]],
    score_metric_key: str,
    dropped_metric_key: str | None,
) -> float | None:
    if not samples:
        return None
    dropped_count = 0
    for sample in samples:
        metrics = sample.get("metrics", {})
        if dropped_metric_key:
            raw = metrics.get(dropped_metric_key)
            if raw is True:
                dropped_count += 1
                continue
            if raw is False:
                continue
        if numeric_metric(metrics, score_metric_key) is None:
            dropped_count += 1
    return dropped_count / len(samples)


def _build_hero_metrics(
    samples: list[SampleSlice],
    metric_metadata: dict[str, MetricMetadata],
) -> dict[str, HeroAggregate]:
    out: dict[str, HeroAggregate] = {}
    sample_metric_maps = [sample.get("metrics", {}) for sample in samples]
    for featured in FEATURED_HERO_METRICS:
        metadata = metric_metadata.get(featured.metric_key, {})
        aggregate_fn = get_aggregate_fn(featured.aggregate_fn_name)
        score = aggregate_fn(sample_metric_maps, featured.metric_key)
        reliability_metric_key = metadata.get("reliability_metric_key")
        dropped_metric_key = metadata.get("dropped_metric_key")
        out[featured.key] = {
            "key": featured.key,
            "metric_key": featured.metric_key,
            "label": featured.label,
            "score": score,
            "reliability": aggregate_fn(sample_metric_maps, reliability_metric_key) if reliability_metric_key else None,
            "dropped_fraction": _dropped_fraction(samples, featured.metric_key, dropped_metric_key),
            "sample_count": len(samples),
        }
    return out


def build_lora_review_payload(data: dict[str, Any]) -> ReviewPayload:
    """Produce the slim review artifact consumed by the operator app and the review JSON export."""
    metric_metadata = build_metric_metadata()
    hero_roster = build_hero_roster(metric_metadata)
    out = {
        "kind": "lora_gallery_review",
        "summary": summarize_review_payload(data),
        "hero_roster": hero_roster,
        "metric_metadata": metric_metadata,
        "methods": [],
    }

    for method in data.get("methods", []):
        method_samples: list[SampleSlice] = []
        method_out: MethodSlice = {
            "id": method.get("id"),
            "label": method.get("label"),
            "prompt_text": method.get("prompt_text") or method.get("prompt_hint"),
            "prompt_hint": method.get("prompt_hint"),
            "hero_metrics": {},
            "evals": [],
        }
        for eval_item in method.get("evals", []):
            if eval_item.get("is_baseline"):
                eval_samples: list[SampleSlice] = []
                eval_out = {
                    "id": eval_item.get("id"),
                    "label": eval_item.get("label"),
                    "is_baseline": True,
                    "hero_metrics": {},
                    "samples": [],
                }
                for sample in eval_item.get("samples", []):
                    sample_out: SampleSlice = {
                        "id": sample.get("id"),
                        "seed": sample.get("seed"),
                        "strength": sample.get("strength"),
                        "label": sample.get("label"),
                        "image_path": sample.get("image_path"),
                        "metrics": sample.get("metrics", {}),
                        "hero_metrics": {},
                    }
                    sample_out["hero_metrics"] = _build_hero_metrics([sample_out], metric_metadata)
                    eval_out["samples"].append(sample_out)
                    eval_samples.append(sample_out)
                    method_samples.append(sample_out)
                eval_out["hero_metrics"] = _build_hero_metrics(eval_samples, metric_metadata)
            else:
                eval_samples = []
                eval_out = {
                    "id": eval_item.get("id"),
                    "label": eval_item.get("label"),
                    "lora_hash": eval_item.get("lora_hash"),
                    "is_baseline": False,
                    "hero_metrics": {},
                    "strengths": [],
                }
                for strength in eval_item.get("strengths", []):
                    strength_samples: list[SampleSlice] = []
                    strength_out = {
                        "value": strength.get("value"),
                        "label": strength.get("label"),
                        "hero_metrics": {},
                        "samples": [],
                    }
                    for sample in strength.get("samples", []):
                        sample_out: SampleSlice = {
                            "id": sample.get("id"),
                            "seed": sample.get("seed"),
                            "strength": sample.get("strength"),
                            "label": sample.get("label"),
                            "image_path": sample.get("image_path"),
                            "metrics": sample.get("metrics", {}),
                            "hero_metrics": {},
                        }
                        sample_out["hero_metrics"] = _build_hero_metrics([sample_out], metric_metadata)
                        strength_out["samples"].append(sample_out)
                        strength_samples.append(sample_out)
                        eval_samples.append(sample_out)
                        method_samples.append(sample_out)
                    strength_out["hero_metrics"] = _build_hero_metrics(strength_samples, metric_metadata)
                    eval_out["strengths"].append(strength_out)
                eval_out["hero_metrics"] = _build_hero_metrics(eval_samples, metric_metadata)
            method_out["evals"].append(eval_out)
        method_out["hero_metrics"] = _build_hero_metrics(method_samples, metric_metadata)
        out["methods"].append(method_out)
    return out


def build_review_data(
    treasurer: Any,
    method_prefix: str | None = None,
    *,
    include_images: bool = False,
    jpeg_quality: int = 88,
) -> dict[str, Any]:
    """
    Shared door for review-data assembly.

    The heavy method/eval/sample pairing and review-time metric assembly now
    lives in `command_center.review_builder`, so app consumers and the review
    JSON export can depend on the same neutral assembly layer.
    """
    return build_data(
        treasurer,
        method_prefix=method_prefix,
        include_images=include_images,
        jpeg_quality=jpeg_quality,
    )


def review_dump_path() -> str:
    """Return the canonical review JSON export path."""
    return str(get_path("project_root") / "data" / "exports" / "lora_review.json")


def write_review_dump(
    *,
    method_prefix: str | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Write the review JSON export and return basic metadata."""
    import json
    from pathlib import Path

    import command_center as cc

    target_path = Path(out_path or review_dump_path())
    target_path.parent.mkdir(parents=True, exist_ok=True)

    treasurer = cc.new_treasurer(read_only=True)
    try:
        data = build_review_data(treasurer, method_prefix=method_prefix, include_images=False)
    finally:
        treasurer.close()

    payload = build_lora_review_payload(data)
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    return {
        "kind": "lora_review",
        "path": str(target_path),
        "bytes": target_path.stat().st_size,
        "summary": payload.get("summary", {}),
    }


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Regenerate lora_review.json from the DB.")
    ap.add_argument("--method", default=None,
                    help="Filter by method hash prefix (e.g. c55be75d).")
    ap.add_argument("--out", default=None,
                    help="Output path (default: data/exports/lora_review.json).")
    args = ap.parse_args()

    result = write_review_dump(method_prefix=args.method, out_path=args.out)
    size_kb = result["bytes"] / 1024
    sys.stderr.write(f"Written: {result['path']}  ({size_kb:.0f} KB)\n")
