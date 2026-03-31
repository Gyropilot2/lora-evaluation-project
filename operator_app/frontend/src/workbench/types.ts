export type WorkspaceTab = "workbench" | "logs";
export type FocusKind = "method" | "eval" | "strength" | "sample";
// Canonical backend transport names live in `contracts/review_transport.py`.

export type MethodSummary = {
  id: string;
  label: string;
  prompt_text?: string | null;
  prompt_hint?: string | null;
  eval_count: number;
  strength_groups: number;
  sample_count: number;
};

export type MetricMap = Record<string, number | string | boolean | null | undefined>;

export type MetricMetadata = {
  key: string;
  label: string;
  description: string;
  tier?: string | null;
  reliability?: string | null;
  polarity?: string | null;
  value_min?: number | null;
  value_max?: number | null;
  value_type?: "float" | "int" | "bool" | "str" | null;
  reliability_metric_key?: string | null;
  dropped_metric_key?: string | null;
  component_metric_keys: string[];
  peer_metric_keys?: string[];
  selection_metric_keys?: string[];
  inspection_graph_metric_keys?: string[];
};

export type HeroAggregate = {
  key: string;
  metric_key: string;
  label: string;
  score: number | null;
  reliability: number | null;
  dropped_fraction: number | null;
  sample_count: number;
};

export type ImageComponentSlice = {
  key: string;
  kind: "mask" | "aux" | "pose_evidence";
  label: string;
  image_path: string;
  secondary_image_path?: string;
  secondary_label?: string;
};

export type SampleSlice = {
  id: string;
  seed?: number;
  strength?: number | null;
  label: string;
  image_path?: string | null;
  image_components?: ImageComponentSlice[];
  metrics: MetricMap;
  hero_metrics: Record<string, HeroAggregate>;
};

export type EvalStrength = {
  value: number;
  label: string;
  hero_metrics: Record<string, HeroAggregate>;
  samples: SampleSlice[];
};

export type EvalSlice = {
  id: string;
  label: string;
  lora_hash?: string | null;
  is_baseline: boolean;
  hero_metrics: Record<string, HeroAggregate>;
  strengths?: EvalStrength[];
  samples?: SampleSlice[];
};

export type MethodSlice = {
  id: string;
  label: string;
  prompt_text?: string | null;
  prompt_hint?: string | null;
  hero_metrics: Record<string, HeroAggregate>;
  evals: EvalSlice[];
};

export type VisibleMethod = {
  summary: MethodSummary;
  evals: EvalSlice[] | null;
};

export type EvalRef = { method: MethodSlice; eval: EvalSlice };
export type StrengthRef = { method: MethodSlice; eval: EvalSlice; strength: EvalStrength };
export type SampleRef = { method: MethodSlice; eval: EvalSlice; sample: SampleSlice; strength?: EvalStrength };

export type ComparisonState = {
  level: FocusKind | null;
  ids: string[];
};

export type FocusedEntity =
  | { kind: "method"; summary: MethodSummary; method?: MethodSlice }
  | { kind: "eval"; method: MethodSlice; eval: EvalSlice }
  | { kind: "strength"; method: MethodSlice; eval: EvalSlice; strength: EvalStrength }
  | { kind: "sample"; method: MethodSlice; eval: EvalSlice; sample: SampleSlice; strength?: EvalStrength }
  | null;

export type HeroMetricDescriptor = {
  key: string;
  metric_key: string;
  label: string;
  description?: string | null;
  aggregate_fn_name?: string | null;
};

export type ReviewCounts = {
  methods: number;
  evals: number;
  strength_groups: number;
  samples: number;
};

export type ReviewSummaryPayload = {
  kind: string;
  summary: ReviewCounts;
  hero_roster: HeroMetricDescriptor[];
  metric_metadata: Record<string, MetricMetadata>;
  methods: MethodSummary[];
};

export type MetricInspectionOption = {
  key: string;
  label: string;
  keyLabel: string;
  count: number;
  isHero: boolean;
  graphMetricKeys: string[];
};
