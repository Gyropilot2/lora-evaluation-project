"""Canonical review transport shapes shared by operator review surfaces."""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict


MetricScalar: TypeAlias = float | int | str | bool | None
MetricMap: TypeAlias = dict[str, MetricScalar]
HeroMetricMap: TypeAlias = dict[str, "HeroAggregate"]
ReviewKind: TypeAlias = Literal["lora_gallery_review"]


class ReviewCounts(TypedDict):
    methods: int
    evals: int
    strength_groups: int
    samples: int


class MethodSummary(TypedDict):
    id: str
    label: str
    prompt_text: str | None
    prompt_hint: NotRequired[str | None]
    eval_count: int
    strength_groups: int
    sample_count: int


class MetricMetadata(TypedDict):
    key: str
    label: str
    description: str | None
    tier: NotRequired[str | None]
    reliability: NotRequired[str | None]
    polarity: NotRequired[str | None]
    value_min: NotRequired[float | int | None]
    value_max: NotRequired[float | int | None]
    value_type: NotRequired[Literal["float", "int", "bool", "str"] | None]
    reliability_metric_key: NotRequired[str | None]
    dropped_metric_key: NotRequired[str | None]
    component_metric_keys: list[str]
    peer_metric_keys: NotRequired[list[str]]
    selection_metric_keys: NotRequired[list[str]]
    inspection_graph_metric_keys: NotRequired[list[str]]


class HeroAggregate(TypedDict):
    key: str
    metric_key: str
    label: str
    score: float | int | None
    reliability: float | int | None
    dropped_fraction: float | None
    sample_count: int


class ImageComponentSlice(TypedDict):
    key: str
    kind: Literal["mask", "aux", "pose_evidence"]
    label: str
    image_path: str
    secondary_image_path: NotRequired[str]
    secondary_label: NotRequired[str]


class SampleSlice(TypedDict):
    id: str
    seed: NotRequired[int]
    strength: NotRequired[float | int | None]
    label: str
    image_path: NotRequired[str | None]
    image_components: NotRequired[list[ImageComponentSlice]]
    metrics: MetricMap
    hero_metrics: HeroMetricMap


class EvalStrength(TypedDict):
    value: float | int
    label: str
    hero_metrics: HeroMetricMap
    samples: list[SampleSlice]


class EvalSlice(TypedDict):
    id: str
    label: str
    lora_hash: NotRequired[str | None]
    is_baseline: bool
    hero_metrics: HeroMetricMap
    strengths: NotRequired[list[EvalStrength]]
    samples: NotRequired[list[SampleSlice]]


class MethodSlice(TypedDict):
    id: str
    label: str
    prompt_text: str | None
    prompt_hint: NotRequired[str | None]
    hero_metrics: HeroMetricMap
    evals: list[EvalSlice]


class HeroMetricDescriptor(TypedDict):
    key: str
    metric_key: str
    label: str
    description: NotRequired[str | None]
    aggregate_fn_name: NotRequired[str | None]


class ReviewPayload(TypedDict):
    kind: ReviewKind
    summary: ReviewCounts
    hero_roster: list[HeroMetricDescriptor]
    metric_metadata: dict[str, MetricMetadata]
    methods: list[MethodSlice]


class ReviewSummaryPayload(TypedDict):
    kind: ReviewKind
    summary: ReviewCounts
    hero_roster: list[HeroMetricDescriptor]
    metric_metadata: dict[str, MetricMetadata]
    methods: list[MethodSummary]
