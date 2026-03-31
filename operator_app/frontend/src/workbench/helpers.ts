import type { FocusedEntity, HeroMetricDescriptor, MethodSlice, MetricMap, MetricMetadata, SampleSlice } from "./types";

export function normalizeText(value: string | null | undefined): string {
  return (value ?? "").toLowerCase();
}

export function metricLabel(
  key: string,
  metricMetadata?: Record<string, MetricMetadata>,
  heroRoster?: HeroMetricDescriptor[],
): string {
  const heroMetric = heroRoster?.find((metric) => metric.key === key);
  if (heroMetric) return heroMetric.label;
  const metadataLabel = metricMetadata?.[key]?.label;
  if (metadataLabel) return metadataLabel;
  return key;
}

export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMetricValue(value: MetricMap[string]): string {
  if (typeof value === "number" && Number.isFinite(value)) return formatMetric(value);
  if (value === true) return "yes";
  if (value === false) return "no";
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

export function numericMetric(metrics: MetricMap, key: string): number | null {
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function strengthKey(evalId: string, value: number): string {
  return `${evalId}::${value}`;
}

export function promptHintForEntity(entity: FocusedEntity): string | null {
  if (!entity) return null;
  if (entity.kind === "method") {
    return entity.method?.prompt_text ?? entity.method?.prompt_hint ?? entity.summary.prompt_text ?? entity.summary.prompt_hint ?? null;
  }
  return entity.method.prompt_text ?? entity.method.prompt_hint ?? null;
}

export function sameSeedLabel(sample: SampleSlice): string {
  return sample.seed === undefined ? sample.label : `seed ${sample.seed}`;
}

export function uniqIds(ids: string[]): string[] {
  return Array.from(new Set(ids));
}

export function samplesForEval(evalItem: { is_baseline: boolean; samples?: SampleSlice[]; strengths?: { samples: SampleSlice[] }[] }): SampleSlice[] {
  return evalItem.is_baseline ? (evalItem.samples ?? []) : (evalItem.strengths ?? []).flatMap((strength) => strength.samples);
}

export function samplesForMethod(method: MethodSlice): SampleSlice[] {
  return method.evals.flatMap((evalItem) => samplesForEval(evalItem));
}

export function sortMetricKeys(
  keys: string[],
  metricMetadata?: Record<string, MetricMetadata>,
  heroRoster?: HeroMetricDescriptor[],
): string[] {
  const heroKeys = new Set((heroRoster ?? []).map((metric) => metric.key));
  return [...keys].sort((left, right) => {
    const leftHero = heroKeys.has(left) ? 0 : 1;
    const rightHero = heroKeys.has(right) ? 0 : 1;
    if (leftHero !== rightHero) return leftHero - rightHero;
    return metricLabel(left, metricMetadata, heroRoster).localeCompare(metricLabel(right, metricMetadata, heroRoster));
  });
}

export function previewSampleForEntity(entity: FocusedEntity): SampleSlice | null {
  if (!entity) return null;
  if (entity.kind === "sample") return entity.sample;
  if (entity.kind === "strength") return entity.strength.samples[0] ?? null;
  if (entity.kind === "eval") {
    if (entity.eval.is_baseline) return entity.eval.samples?.[0] ?? null;
    return entity.eval.strengths?.[0]?.samples?.[0] ?? null;
  }
  return (
    entity.method?.evals.flatMap((evalItem) =>
      evalItem.is_baseline ? (evalItem.samples ?? []) : (evalItem.strengths ?? []).flatMap((strength) => strength.samples),
    )[0] ?? null
  );
}
