import { formatMetricValue, metricLabel, numericMetric } from "./helpers";
import type {
  EvalRef,
  EvalStrength,
  FocusKind,
  FocusedEntity,
  HeroMetricDescriptor,
  MetricInspectionOption,
  MetricMetadata,
  MetricMap,
} from "./types";

export type InspectionPoint = {
  label: string;
  strengthValue: number;
  min: number;
  max: number;
  focusedValue: number | null;
  valueCount: number;
};

export type InspectionScale = {
  min: number;
  max: number;
};

export type InspectionCardModel = {
  key: string;
  label: string;
  metricKey: string;
  description: string | null;
  points: InspectionPoint[];
  scale: InspectionScale;
};

export type InspectionFactModel = {
  key: string;
  label: string;
  description: string | null;
  rows: Array<{ label: string; value: string }>;
};

export type SelectedMetricInspectionModel =
  | {
      kind: "ready";
      poolTag: string;
      methodLabel: string;
      evalCount: number;
      strengthCount: number;
      selectedMetricLabel: string;
      selectedMetricKey: string;
      selectedMetricDescription: string | null;
      selectedCard: InspectionCardModel;
      graphCards: InspectionCardModel[];
      hiddenGraphCount: number;
      facts: InspectionFactModel[];
    }
  | {
      kind: "unsupported";
      selectedMetricLabel: string;
      selectedMetricKey: string;
      reason: string;
    };

type EvalInspectionPool = {
  tag: string;
  methodLabel: string;
  evals: EvalRef[];
  focusedEvalId: string | null;
};

export function buildSelectedMetricInspectionModel(args: {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  focusedEvalId: string | null;
  selectedMetric: string | null;
  inspectionMetricOptions: MetricInspectionOption[];
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
  selectedEvals: EvalRef[];
}): SelectedMetricInspectionModel | null {
  const option = args.inspectionMetricOptions.find((item) => item.key === args.selectedMetric) ?? null;
  if (!option || !args.selectedMetric) return null;

  const pool = resolveEvalInspectionPool({
    comparisonLevel: args.comparisonLevel,
    focusedEntity: args.focusedEntity,
    focusedEvalId: args.focusedEvalId,
    selectedEvals: args.selectedEvals,
  });
  if (!pool) return null;
  if ("reason" in pool) {
    return {
      kind: "unsupported",
      selectedMetricLabel: option.label,
      selectedMetricKey: option.key,
      reason: pool.reason,
    };
  }

  const selectedMetadata = args.metricMetadata[option.key];
  const heroMetric = args.heroRoster.find((hero) => hero.metric_key === option.key) ?? null;
  const selectedCard = buildInspectionCard({
    metricKey: option.key,
    label: option.label,
    description: selectedMetadata?.description ?? null,
    pool,
    metricMetadata: args.metricMetadata,
    valueForStrength: (strength) =>
      heroMetric ? strength.hero_metrics[heroMetric.key]?.score ?? null : meanSampleMetric(strength, option.key),
  });
  if (!selectedCard) {
    return {
      kind: "unsupported",
      selectedMetricLabel: option.label,
      selectedMetricKey: option.key,
      reason: "This metric does not have enough numeric strength signal in the current eval pool to build a trustworthy breakdown yet.",
    };
  }

  const graphCards = option.graphMetricKeys
    .slice(0, 5)
    .map((metricKey) =>
      buildInspectionCard({
        metricKey,
        label: metricLabel(metricKey, args.metricMetadata, args.heroRoster),
        description: args.metricMetadata[metricKey]?.description ?? null,
        pool,
        metricMetadata: args.metricMetadata,
        valueForStrength: (strength) => meanSampleMetric(strength, metricKey),
      }),
    )
    .filter((card): card is InspectionCardModel => Boolean(card));

  const facts = buildInspectionFacts(pool, selectedMetadata, args.metricMetadata, args.heroRoster);

  return {
    kind: "ready",
    poolTag: pool.tag,
    methodLabel: pool.methodLabel,
    evalCount: pool.evals.length,
    strengthCount: uniqueStrengthValues(pool.evals).length,
    selectedMetricLabel: option.label,
    selectedMetricKey: option.key,
    selectedMetricDescription: selectedMetadata?.description ?? null,
    selectedCard,
    graphCards,
    hiddenGraphCount: Math.max(0, option.graphMetricKeys.length - 5),
    facts,
  };
}

function resolveEvalInspectionPool(args: {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  focusedEvalId: string | null;
  selectedEvals: EvalRef[];
}): EvalInspectionPool | { reason: string } | null {
  const { comparisonLevel, focusedEntity, focusedEvalId, selectedEvals } = args;

  if (comparisonLevel === "eval" && selectedEvals.length > 0) {
    const loraEvals = selectedEvals.filter((item) => !item.eval.is_baseline);
    if (loraEvals.length === 0) {
      return { reason: "Selected evals need at least one non-baseline eval with strength data." };
    }
    const methodIds = Array.from(new Set(loraEvals.map((item) => item.method.id)));
    if (methodIds.length !== 1) {
      return { reason: "SelectedMetric inspection is eval-first for one method at a time. Mixed-method eval pools do not share an honest strength axis yet." };
    }
    return {
      tag: "selected evals",
      methodLabel: loraEvals[0].method.label,
      evals: loraEvals,
      focusedEvalId,
    };
  }

  if (comparisonLevel === "method" || comparisonLevel === "strength" || comparisonLevel === "sample") {
    return { reason: "SelectedMetric inspection is eval-first in v1. Clear the current compare pool or switch to same-method eval comparison to light it up." };
  }

  if (focusedEntity?.kind === "eval") {
    if (focusedEntity.eval.is_baseline) {
      return { reason: "Baseline evals do not carry the strength ladder this panel needs." };
    }
    return {
      tag: "focused eval",
      methodLabel: focusedEntity.method.label,
      evals: [{ method: focusedEntity.method, eval: focusedEntity.eval }],
      focusedEvalId: focusedEntity.eval.id,
    };
  }

  if (focusedEntity?.kind === "method" && focusedEntity.method) {
    const evals = focusedEntity.method.evals
      .filter((evalItem) => !evalItem.is_baseline)
      .map((evalItem) => ({ method: focusedEntity.method!, eval: evalItem }));
    if (evals.length === 0) {
      return { reason: "This method does not have non-baseline eval strengths to inspect yet." };
    }
    return {
      tag: "method evals",
      methodLabel: focusedEntity.method.label,
      evals,
      focusedEvalId,
    };
  }

  return null;
}

function buildInspectionCard(args: {
  metricKey: string;
  label: string;
  description: string | null;
  pool: EvalInspectionPool;
  metricMetadata: Record<string, MetricMetadata>;
  valueForStrength: (strength: EvalStrength) => number | null;
}): InspectionCardModel | null {
  const points = buildInspectionPoints(args.pool, args.valueForStrength);
  if (points.length === 0) return null;
  return {
    key: args.metricKey,
    label: args.label,
    metricKey: args.metricKey,
    description: args.description,
    points,
    scale: inspectionScale(points, args.metricMetadata[args.metricKey]),
  };
}

function buildInspectionPoints(
  pool: EvalInspectionPool,
  valueForStrength: (strength: EvalStrength) => number | null,
): InspectionPoint[] {
  const strengthGroups = new Map<number, { label: string; values: number[]; focusedValue: number | null }>();

  for (const item of pool.evals) {
    for (const strength of item.eval.strengths ?? []) {
      const value = valueForStrength(strength);
      if (value === null) continue;
      const existing = strengthGroups.get(strength.value) ?? {
        label: strength.label,
        values: [],
        focusedValue: null,
      };
      existing.values.push(value);
      if (pool.focusedEvalId === item.eval.id) existing.focusedValue = value;
      strengthGroups.set(strength.value, existing);
    }
  }

  return Array.from(strengthGroups.entries())
    .sort(([left], [right]) => left - right)
    .map(([strengthValue, group]) => {
      const sortedValues = group.values.slice().sort((left, right) => left - right);
      return {
        label: group.label,
        strengthValue,
        min: sortedValues[0],
        max: sortedValues[sortedValues.length - 1],
        focusedValue:
          group.focusedValue !== null
            ? group.focusedValue
            : sortedValues.length === 1
              ? sortedValues[0]
              : null,
        valueCount: sortedValues.length,
      };
    });
}

function buildInspectionFacts(
  pool: EvalInspectionPool,
  metadata: MetricMetadata | undefined,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): InspectionFactModel[] {
  const factKeys = (metadata?.selection_metric_keys ?? []).filter((metricKey) => {
    const valueType = metricMetadata[metricKey]?.value_type;
    return valueType === "bool" || valueType === "str";
  });

  return factKeys.flatMap((metricKey) => {
    const rows = uniqueStrengthValues(pool.evals)
      .map((strengthValue) => {
        const strengthLabel =
          pool.evals.flatMap((item) => item.eval.strengths ?? []).find((strength) => strength.value === strengthValue)?.label ??
          `strength ${strengthValue}`;
        const values = pool.evals
          .flatMap((item) => (item.eval.strengths ?? []).filter((strength) => strength.value === strengthValue))
          .flatMap((strength) => strength.samples.map((sample) => sample.metrics[metricKey]))
          .filter((value) => value !== null && value !== undefined && value !== "");
        if (values.length === 0) return null;
        return {
          label: strengthLabel,
          value: summarizeFactValues(values),
        };
      })
      .filter((row): row is { label: string; value: string } => Boolean(row));

    if (rows.length === 0) return [];
    return [
      {
        key: metricKey,
        label: metricLabel(metricKey, metricMetadata, heroRoster),
        description: metricMetadata[metricKey]?.description ?? null,
        rows,
      },
    ];
  });
}

function summarizeFactValues(values: MetricMap[string][]): string {
  const counts = new Map<string, number>();
  for (const value of values) {
    const label = formatMetricValue(value);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([label, count]) => `${label} x${count}`)
    .join(" | ");
}

function uniqueStrengthValues(evals: EvalRef[]): number[] {
  return Array.from(
    new Set(
      evals.flatMap((item) => (item.eval.strengths ?? []).map((strength) => strength.value)),
    ),
  ).sort((left, right) => left - right);
}

function meanSampleMetric(strength: EvalStrength, metricKey: string): number | null {
  const values = strength.samples
    .map((sample) => numericMetric(sample.metrics, metricKey))
    .filter((value): value is number => value !== null);
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function inspectionScale(points: InspectionPoint[], metadata: MetricMetadata | undefined): InspectionScale {
  const values = points.flatMap((point) =>
    [point.min, point.max, point.focusedValue].filter((value): value is number => value !== null),
  );
  if (values.length === 0) {
    const fallbackMin = typeof metadata?.value_min === "number" ? metadata.value_min : 0;
    const fallbackMax = typeof metadata?.value_max === "number" ? metadata.value_max : 1;
    return { min: fallbackMin, max: fallbackMax > fallbackMin ? fallbackMax : fallbackMin + 1 };
  }

  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  if (rawMin === rawMax) {
    const padding = Math.max(Math.abs(rawMin) * 0.2, 0.05);
    return { min: rawMin - padding, max: rawMax + padding };
  }
  const padding = (rawMax - rawMin) * 0.12;
  return { min: rawMin - padding, max: rawMax + padding };
}
