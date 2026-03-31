import {
  formatMetric,
  metricLabel,
  normalizeText,
  sortMetricKeys,
} from "./helpers";
import {
  type EvalRef,
  type EvalSlice,
  type EvalStrength,
  type FocusKind,
  type FocusedEntity,
  type HeroMetricDescriptor,
  type MethodSlice,
  type MethodSummary,
  type MetricMetadata,
  type MetricMap,
  type SampleRef,
  type StrengthRef,
  type VisibleMethod,
} from "./types";

export type CompareChip = { id: string; label: string; isFocused?: boolean };
export type FactRow = { label: string; value: string };
export type MetricRow = { key: string; label: string; value: string; description?: string | null };

export function buildFocusedEntity(args: {
  focusedSampleId: string | null;
  focusedStrengthId: string | null;
  focusedEvalId: string | null;
  focusedMethodId: string | null;
  sampleRefs: Map<string, SampleRef>;
  strengthRefs: Map<string, StrengthRef>;
  evalRefs: Map<string, EvalRef>;
  methodsById: Record<string, MethodSummary>;
  methodSlices: Record<string, MethodSlice>;
}): FocusedEntity {
  const {
    focusedSampleId,
    focusedStrengthId,
    focusedEvalId,
    focusedMethodId,
    sampleRefs,
    strengthRefs,
    evalRefs,
    methodsById,
    methodSlices,
  } = args;

  if (focusedSampleId) {
    const ref = sampleRefs.get(focusedSampleId);
    if (ref) return { kind: "sample", ...ref };
  }
  if (focusedStrengthId) {
    const ref = strengthRefs.get(focusedStrengthId);
    if (ref) return { kind: "strength", ...ref };
  }
  if (focusedEvalId) {
    const ref = evalRefs.get(focusedEvalId);
    if (ref) return { kind: "eval", ...ref };
  }
  if (focusedMethodId) {
    const summary = methodsById[focusedMethodId];
    if (summary) return { kind: "method", summary, method: methodSlices[focusedMethodId] };
  }
  return null;
}

export function buildVisibleMethods(args: {
  methods: MethodSummary[];
  methodSlices: Record<string, MethodSlice>;
  searchQuery: string;
}): VisibleMethod[] {
  const { methods, methodSlices, searchQuery } = args;
  const query = normalizeText(searchQuery).trim();
  if (!query) {
    return methods.map((summary) => ({
      summary,
      evals: methodSlices[summary.id]?.evals ?? null,
    }));
  }

  return methods.reduce<VisibleMethod[]>((out, summary) => {
    const methodPrompt = summary.prompt_text ?? summary.prompt_hint ?? "";
    const methodMatch = normalizeText(`${summary.label} ${methodPrompt}`).includes(query);
    const slice = methodSlices[summary.id];
    if (methodMatch) {
      out.push({
        summary,
        evals: slice?.evals ?? null,
      });
      return out;
    }
    if (!slice) return out;

    const evals = slice.evals.reduce<EvalSlice[]>((evalOut, evalItem) => {
      const evalMatch = normalizeText(`${evalItem.label} ${evalItem.lora_hash ?? ""}`).includes(query);
      if (evalItem.is_baseline) {
        const samples = (evalItem.samples ?? []).filter((sample) =>
          normalizeText(`${sample.label} ${sample.seed ?? ""}`).includes(query),
        );
        if (evalMatch || samples.length > 0) {
          evalOut.push({ ...evalItem, samples: samples.length > 0 ? samples : evalItem.samples });
        }
        return evalOut;
      }

      const strengths = (evalItem.strengths ?? []).reduce<EvalStrength[]>((strengthOut, strength) => {
        const strengthMatch = normalizeText(`${strength.label} ${strength.value}`).includes(query);
        const samples = strength.samples.filter((sample) =>
          normalizeText(`${sample.label} ${sample.seed ?? ""} ${sample.strength ?? ""}`).includes(query),
        );
        if (evalMatch || strengthMatch || samples.length > 0) {
          strengthOut.push({ ...strength, samples: samples.length > 0 ? samples : strength.samples });
        }
        return strengthOut;
      }, []);

      if (strengths.length > 0) evalOut.push({ ...evalItem, strengths });
      return evalOut;
    }, []);

    if (evals.length > 0) {
      out.push({
        summary,
        evals,
      });
    }
    return out;
  }, []);
}

export function buildCompareChips(args: {
  level: FocusKind | null;
  selectedMethods: MethodSummary[];
  selectedEvals: EvalRef[];
  selectedStrengths: StrengthRef[];
  selectedSamples: SampleRef[];
  focusedSampleId: string | null;
  strengthKey: (evalId: string, value: number) => string;
}): CompareChip[] {
  const { level, selectedMethods, selectedEvals, selectedStrengths, selectedSamples, focusedSampleId, strengthKey } = args;
  if (level === "method") {
    return selectedMethods.map((method) => ({ id: method.id, label: method.label }));
  }
  if (level === "eval") {
    return selectedEvals.map((item) => ({ id: item.eval.id, label: `${item.method.label} - ${item.eval.label}` }));
  }
  if (level === "strength") {
    return selectedStrengths.map((item) => ({
      id: strengthKey(item.eval.id, item.strength.value),
      label: `${item.method.label} - ${item.eval.label} - ${item.strength.label}`,
    }));
  }
  if (level === "sample") {
    return selectedSamples.map((item) => ({
      id: item.sample.id,
      label: `${item.eval.label} - ${item.sample.label}`,
      isFocused: item.sample.id === focusedSampleId,
    }));
  }
  return [];
}

export function buildFocusedComparisonId(
  entity: FocusedEntity,
  strengthIdFor: (evalId: string, value: number) => string,
): string | null {
  if (!entity) return null;
  if (entity.kind === "method") return entity.summary.id;
  if (entity.kind === "eval") return entity.eval.id;
  if (entity.kind === "strength") return strengthIdFor(entity.eval.id, entity.strength.value);
  return entity.sample.id;
}

export function buildCenterTitle(args: {
  comparisonLevel: FocusKind | null;
  comparisonCount: number;
  focusedEntity: FocusedEntity;
}): string {
  const { comparisonLevel, comparisonCount, focusedEntity } = args;
  if (comparisonLevel) return `Comparing ${comparisonCount} ${comparisonLevel}s`;
  if (focusedEntity?.kind === "sample") return `${focusedEntity.eval.label} - ${focusedEntity.sample.label}`;
  if (focusedEntity?.kind === "strength") return `${focusedEntity.eval.label} - ${focusedEntity.strength.label}`;
  if (focusedEntity?.kind === "eval") return focusedEntity.eval.label;
  if (focusedEntity?.kind === "method") return focusedEntity.summary.label;
  return "Operator workbench";
}

export function buildScopeLine(args: {
  comparisonLevel: FocusKind | null;
  compareChipCount: number;
  focusedEntity: FocusedEntity;
}): string {
  const { comparisonLevel, compareChipCount, focusedEntity } = args;
  if (comparisonLevel) return `${compareChipCount} ${comparisonLevel} items selected`;
  if (focusedEntity?.kind === "sample") {
    return `${focusedEntity.method.label} -> ${focusedEntity.eval.label} -> ${focusedEntity.sample.label}`;
  }
  if (focusedEntity?.kind === "strength") {
    return `${focusedEntity.method.label} -> ${focusedEntity.eval.label} -> ${focusedEntity.strength.label}`;
  }
  if (focusedEntity?.kind === "eval") return `${focusedEntity.method.label} -> ${focusedEntity.eval.label}`;
  if (focusedEntity?.kind === "method") return focusedEntity.summary.label;
  return "No scope selected";
}

export function buildFocusTitle(entity: FocusedEntity): string {
  if (entity?.kind === "sample") return `${entity.eval.label} - ${entity.sample.label}`;
  if (entity?.kind === "strength") return `${entity.eval.label} - ${entity.strength.label}`;
  if (entity?.kind === "eval") return entity.eval.label;
  if (entity?.kind === "method") return entity.summary.label;
  return "No focused item";
}

export function buildFocusScopeLine(entity: FocusedEntity): string {
  if (entity?.kind === "sample") return `${entity.method.label} -> ${entity.eval.label} -> ${entity.sample.label}`;
  if (entity?.kind === "strength") return `${entity.method.label} -> ${entity.eval.label} -> ${entity.strength.label}`;
  if (entity?.kind === "eval") return `${entity.method.label} -> ${entity.eval.label}`;
  if (entity?.kind === "method") return entity.summary.label;
  return "Nothing focused";
}

function metricsForFocusedEntity(entity: FocusedEntity, heroRoster: HeroMetricDescriptor[]): MetricMap | null {
  if (!entity) return null;
  if (entity.kind === "sample") {
    const metrics: MetricMap = {};
    for (const metric of heroRoster) {
      metrics[metric.key] = entity.sample.hero_metrics[metric.key]?.score ?? null;
    }
    return metrics;
  }
  if (entity.kind === "strength") {
    const metrics: MetricMap = {};
    for (const metric of heroRoster) metrics[metric.key] = entity.strength.hero_metrics[metric.key]?.score ?? null;
    return metrics;
  }
  if (entity.kind === "eval") {
    const metrics: MetricMap = {};
    for (const metric of heroRoster) metrics[metric.key] = entity.eval.hero_metrics[metric.key]?.score ?? null;
    return metrics;
  }
  if (entity.kind === "method") {
    const metrics: MetricMap = {};
    for (const metric of heroRoster) metrics[metric.key] = entity.method?.hero_metrics[metric.key]?.score ?? null;
    return metrics;
  }
  return null;
}

export function factRowsForFocusedEntity(entity: FocusedEntity): FactRow[] {
  if (!entity) return [];
  if (entity.kind === "method") {
    return [
      { label: "Method id", value: entity.summary.id },
      { label: "Evals", value: String(entity.summary.eval_count) },
      { label: "Strength groups", value: String(entity.summary.strength_groups) },
      { label: "Samples", value: String(entity.summary.sample_count) },
    ];
  }
  if (entity.kind === "eval") {
    return [
      { label: "Eval id", value: entity.eval.id },
      { label: "LoRA hash", value: entity.eval.lora_hash ?? "-" },
      { label: "Baseline", value: entity.eval.is_baseline ? "yes" : "no" },
      {
        label: "Samples",
        value: String(
          entity.eval.is_baseline
            ? (entity.eval.samples ?? []).length
            : (entity.eval.strengths ?? []).reduce((sum, strength) => sum + strength.samples.length, 0),
        ),
      },
    ];
  }
  if (entity.kind === "strength") {
    return [
      { label: "Strength", value: entity.strength.label },
      { label: "Value", value: String(entity.strength.value) },
      { label: "Eval", value: entity.eval.label },
      { label: "Samples", value: String(entity.strength.samples.length) },
    ];
  }
  return [
    { label: "Sample id", value: entity.sample.id },
    { label: "Seed", value: String(entity.sample.seed ?? "-") },
    { label: "Strength", value: String(entity.sample.strength ?? "-") },
    { label: "Eval", value: entity.eval.label },
  ];
}

export function metricRowsForFocusedEntity(
  entity: FocusedEntity,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): MetricRow[] {
  if (!entity) return [];
  if (entity.kind === "sample") {
    return [];
  }

  const aggregateMetrics = metricsForFocusedEntity(entity, heroRoster) ?? {};
  return heroRoster.map((metric) => ({
    key: metric.key,
    label: metricLabel(metric.key, metricMetadata, heroRoster),
    value: formatMetric(typeof aggregateMetrics[metric.key] === "number" ? (aggregateMetrics[metric.key] as number) : null),
    description: metric.description ?? null,
  }));
}
