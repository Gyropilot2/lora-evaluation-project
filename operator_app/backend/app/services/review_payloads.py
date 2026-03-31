"""App-facing review payload assembly built from Command Center review surfaces."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import command_center as cc
from command_center.review_payload import (
    build_lora_review_payload,
    build_review_data,
)
from command_center.review_builder import (
    method_label as review_method_label,
    method_prompt_text as review_method_prompt_text,
    method_prompt_hint as review_method_prompt_hint,
    resolved_lora_label as review_resolved_lora_label,
)
from contracts.review_transport import (
    EvalSlice,
    EvalStrength,
    ImageComponentSlice,
    MethodSlice,
    MethodSummary,
    ReviewPayload,
    ReviewSummaryPayload,
    SampleSlice,
)
from core.paths import get_path
from operator_app.backend.app.services.review_assets import sample_image_components
from contracts.review_surface import build_hero_roster, build_metric_metadata

def _db_stamp() -> int:
    db_file = Path(get_path("db_file"))
    candidates = [
        db_file,
        db_file.with_name(f"{db_file.name}-wal"),
    ]
    stamps: list[int] = []
    for candidate in candidates:
        try:
            stamps.append(candidate.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return max(stamps, default=0)


def _code_stamp() -> int:
    project_root = Path(get_path("project_root"))
    candidates = [
        Path(__file__).resolve(),
        project_root / "command_center" / "review_payload.py",
        project_root / "command_center" / "review_builder.py",
        project_root / "contracts" / "metrics_registry.py",
        project_root / "contracts" / "metric_labels.py",
        project_root / "contracts" / "procedures_registry.py",
        project_root / "contracts" / "recipe.py",
    ]
    stamps: list[int] = []
    for candidate in candidates:
        try:
            stamps.append(candidate.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return max(stamps, default=0)


def _cache_stamp() -> tuple[int, int]:
    return (_db_stamp(), _code_stamp())


def _load_payload(method_prefix: str | None = None) -> ReviewPayload:
    treasurer = cc.new_treasurer(read_only=True)
    try:
        data = build_review_data(treasurer, method_prefix=method_prefix, include_images=False)
    finally:
        treasurer.close()
    return build_lora_review_payload(data)


@lru_cache(maxsize=2)
def _full_payload_cached(cache_stamp: tuple[int, int]) -> ReviewPayload:
    del cache_stamp
    return _load_payload()


def _sample_visible_in_review(sample: dict[str, Any]) -> bool:
    if sample.get("is_dirty"):
        return False
    image = sample.get("image") or {}
    output = image.get("output") if isinstance(image, dict) else {}
    return bool((output or {}).get("path"))


def _build_summary_from_treasurer(treasurer: cc.Treasurer) -> tuple[dict[str, int], list[MethodSummary]]:
    all_evals = treasurer.query_evals()
    evals_by_method: dict[str, list[dict[str, Any]]] = {}
    for eval_rec in all_evals:
        method_hash = eval_rec.get("method_hash")
        if isinstance(method_hash, str) and method_hash:
            evals_by_method.setdefault(method_hash, []).append(eval_rec)

    method_hashes = sorted(
        {
            method_hash
            for method_hash in evals_by_method
        }
    )

    methods: list[MethodSummary] = []
    total_evals = 0
    total_strength_groups = 0
    total_samples = 0

    for method_hash in method_hashes:
        method = treasurer.get_method(method_hash)
        if method is None:
            continue

        baselines_by_seed: dict[int, None] = {}
        lora_groups: dict[tuple[str, str], dict[float | int | None, int]] = {}
        evals_for_method = evals_by_method[method_hash]

        for eval_rec in evals_for_method:
            lora_info = eval_rec.get("lora") or {}
            eval_hash = eval_rec.get("eval_hash")
            if not isinstance(eval_hash, str) or not eval_hash:
                continue

            for sample in treasurer.query_samples(filters={"eval_hash": eval_hash}):
                if not _sample_visible_in_review(sample):
                    continue

                if not lora_info:
                    seed = sample.get("seed")
                    if isinstance(seed, int):
                        baselines_by_seed[seed] = None
                    continue

                lora_hash = lora_info.get("hash", "unknown")
                lora_label = review_resolved_lora_label(lora_info, lora_hash)
                group_key = (lora_hash, lora_label)
                strength = sample.get("lora_strength")
                strength_counts = lora_groups.setdefault(group_key, {})
                strength_counts[strength] = strength_counts.get(strength, 0) + 1

        eval_count = (1 if baselines_by_seed else 0) + len(lora_groups)
        strength_groups = sum(len(strengths) for strengths in lora_groups.values())
        sample_count = len(baselines_by_seed) + sum(
            count
            for strengths in lora_groups.values()
            for count in strengths.values()
        )

        total_evals += eval_count
        total_strength_groups += strength_groups
        total_samples += sample_count
        methods.append(
            {
                "id": method_hash,
                "label": review_method_label(method),
                "prompt_text": review_method_prompt_text(method),
                "prompt_hint": review_method_prompt_hint(method),
                "eval_count": eval_count,
                "strength_groups": strength_groups,
                "sample_count": sample_count,
            }
        )

    return (
        {
            "methods": len(methods),
            "evals": total_evals,
            "strength_groups": total_strength_groups,
            "samples": total_samples,
        },
        methods,
    )


@lru_cache(maxsize=2)
def _summary_payload_cached(cache_stamp: tuple[int, int]) -> ReviewSummaryPayload:
    del cache_stamp
    treasurer = cc.new_treasurer(read_only=True)
    try:
        summary, methods = _build_summary_from_treasurer(treasurer)
    finally:
        treasurer.close()
    metric_metadata = build_metric_metadata()
    return {
        "kind": "lora_gallery_review",
        "summary": summary,
        "hero_roster": build_hero_roster(metric_metadata),
        "metric_metadata": metric_metadata,
        "methods": methods,
    }


def review_summary() -> ReviewSummaryPayload:
    return _summary_payload_cached(_cache_stamp())


def list_methods() -> list[MethodSummary]:
    return _summary_payload_cached(_cache_stamp())["methods"]


@lru_cache(maxsize=8)
def _get_method_slice_cached(cache_stamp: tuple[int, int], method_hash: str) -> MethodSlice | None:
    del cache_stamp
    payload = _load_payload(method_prefix=method_hash)
    for method in payload.get("methods", []):
        if method.get("id") == method_hash:
            return method
    return None


def get_method_slice(method_hash: str) -> MethodSlice | None:
    return _get_method_slice_cached(_cache_stamp(), method_hash)


def get_eval_slice(method_hash: str, eval_id: str) -> EvalSlice | None:
    method = get_method_slice(method_hash)
    if not method:
        return None
    for ev in method.get("evals", []):
        if ev.get("id") == eval_id:
            return ev
    return None


def get_strength_slice(method_hash: str, eval_id: str, strength_value: float) -> EvalStrength | None:
    ev = get_eval_slice(method_hash, eval_id)
    if not ev or ev.get("is_baseline"):
        return None
    for strength in ev.get("strengths", []):
        if float(strength.get("value")) == float(strength_value):
            return strength
    return None


def get_sample_slice(method_hash: str, eval_id: str, sample_hash: str) -> SampleSlice | None:
    ev = get_eval_slice(method_hash, eval_id)
    if not ev:
        return None
    if ev.get("is_baseline"):
        for sample in ev.get("samples", []):
            if sample.get("id") == sample_hash:
                return _enrich_sample_slice(sample)
        return None
    for strength in ev.get("strengths", []):
        for sample in strength.get("samples", []):
            if sample.get("id") == sample_hash:
                return _enrich_sample_slice(sample)
    return None


def _enrich_sample_slice(sample: SampleSlice) -> SampleSlice:
    out = dict(sample)
    treasurer = cc.new_treasurer(read_only=True)
    try:
        raw_sample = treasurer.get_sample(sample.get("id"))
        baseline_sample = None
        if raw_sample:
            eval_hash = raw_sample.get("eval_hash")
            eval_rec = None
            if eval_hash:
                eval_rec = treasurer.get_eval(eval_hash)
            if eval_rec and (eval_rec.get("lora") or {}):
                baseline_sample = treasurer.find_baseline_sample(raw_sample.get("sample_hash"))
    finally:
        treasurer.close()
    baseline_components = {
        component["key"]: component
        for component in sample_image_components(baseline_sample)
    }
    image_components: list[ImageComponentSlice] = []
    for component in sample_image_components(raw_sample):
        paired = dict(component)
        baseline_component = baseline_components.get(component["key"])
        if baseline_component:
            paired["secondary_image_path"] = baseline_component["image_path"]
            paired["secondary_label"] = "Baseline"
        image_components.append(paired)
    out["image_components"] = image_components
    return out


def review_dump_path() -> Path:
    return Path(get_path("project_root")) / "data" / "exports" / "review_dump.json"


def write_review_dump() -> dict[str, Any]:
    payload = _load_payload()
    out_path = review_dump_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return {
        "kind": "review_dump",
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "summary": payload.get("summary", {}),
    }
