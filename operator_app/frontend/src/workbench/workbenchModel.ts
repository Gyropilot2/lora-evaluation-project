import { useMemo } from "react";

import { promptHintForEntity, strengthKey } from "./helpers";
import { type EvalRef, type FocusedEntity, type MethodSlice, type MethodSummary, type MetricInspectionOption, type SampleRef, type StrengthRef } from "./types";
import { buildCenterTitle, buildCompareChips, buildFocusScopeLine, buildFocusTitle, buildFocusedComparisonId, buildFocusedEntity, buildScopeLine, buildVisibleMethods, factRowsForFocusedEntity, metricRowsForFocusedEntity } from "./viewModel";
import { heroRailGroupsForFocusedEntity, remainingMetricRowsForFocusedEntity } from "./metricRailModel";
import { type ComparisonState, type HeroMetricDescriptor, type MetricMetadata } from "./types";

export type WorkbenchRefs = {
  methodsById: Record<string, MethodSummary>;
  evalRefs: Map<string, EvalRef>;
  strengthRefs: Map<string, StrengthRef>;
  sampleRefs: Map<string, SampleRef>;
};

export function useWorkbenchRefs(methods: MethodSummary[], methodSlices: Record<string, MethodSlice>): WorkbenchRefs {
  const methodsById = useMemo(() => Object.fromEntries(methods.map((method) => [method.id, method])), [methods]);
  const loadedMethods = useMemo(() => Object.values(methodSlices), [methodSlices]);

  const evalRefs = useMemo(() => {
    const out = new Map<string, EvalRef>();
    for (const method of loadedMethods) for (const evalItem of method.evals) out.set(evalItem.id, { method, eval: evalItem });
    return out;
  }, [loadedMethods]);

  const strengthRefs = useMemo(() => {
    const out = new Map<string, StrengthRef>();
    for (const method of loadedMethods) for (const evalItem of method.evals) for (const strength of evalItem.strengths ?? []) out.set(strengthKey(evalItem.id, strength.value), { method, eval: evalItem, strength });
    return out;
  }, [loadedMethods]);

  const sampleRefs = useMemo(() => {
    const out = new Map<string, SampleRef>();
    for (const method of loadedMethods) {
      for (const evalItem of method.evals) {
        if (evalItem.is_baseline) {
          for (const sample of evalItem.samples ?? []) out.set(sample.id, { method, eval: evalItem, sample });
          continue;
        }
        for (const strength of evalItem.strengths ?? []) for (const sample of strength.samples) out.set(sample.id, { method, eval: evalItem, strength, sample });
      }
    }
    return out;
  }, [loadedMethods]);

  return { methodsById, evalRefs, strengthRefs, sampleRefs };
}

export function useWorkbenchDerivedState(args: {
  methods: MethodSummary[];
  methodSlices: Record<string, MethodSlice>;
  searchQuery: string;
  comparison: ComparisonState;
  focusedMethodId: string | null;
  focusedEvalId: string | null;
  focusedStrengthId: string | null;
  focusedSampleId: string | null;
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
  refs: WorkbenchRefs;
}) {
  const focusedEntity = useMemo<FocusedEntity>(() => buildFocusedEntity({
    focusedSampleId: args.focusedSampleId,
    focusedStrengthId: args.focusedStrengthId,
    focusedEvalId: args.focusedEvalId,
    focusedMethodId: args.focusedMethodId,
    sampleRefs: args.refs.sampleRefs,
    strengthRefs: args.refs.strengthRefs,
    evalRefs: args.refs.evalRefs,
    methodsById: args.refs.methodsById,
    methodSlices: args.methodSlices,
  }), [args.focusedEvalId, args.focusedMethodId, args.focusedSampleId, args.focusedStrengthId, args.methodSlices, args.refs]);

  const visibleMethods = useMemo(() => buildVisibleMethods({ methods: args.methods, methodSlices: args.methodSlices, searchQuery: args.searchQuery }), [args.methodSlices, args.methods, args.searchQuery]);
  const selectedMethods = useMemo(() => args.comparison.level === "method" ? args.methods.filter((method) => args.comparison.ids.includes(method.id)) : [], [args.comparison, args.methods]);
  const selectedEvals = useMemo(() => args.comparison.level === "eval" ? args.comparison.ids.map((id) => args.refs.evalRefs.get(id)).filter((value): value is EvalRef => Boolean(value)) : [], [args.comparison, args.refs.evalRefs]);
  const selectedStrengths = useMemo(() => args.comparison.level === "strength" ? args.comparison.ids.map((id) => args.refs.strengthRefs.get(id)).filter((value): value is StrengthRef => Boolean(value)) : [], [args.comparison, args.refs.strengthRefs]);
  const selectedSamples = useMemo(() => args.comparison.level === "sample" ? args.comparison.ids.map((id) => args.refs.sampleRefs.get(id)).filter((value): value is SampleRef => Boolean(value)) : [], [args.comparison, args.refs.sampleRefs]);
  const totalSamples = useMemo(() => args.methods.reduce((sum, method) => sum + method.sample_count, 0), [args.methods]);
  const compareChips = useMemo(() => buildCompareChips({ level: args.comparison.level, selectedMethods, selectedEvals, selectedStrengths, selectedSamples, focusedSampleId: args.focusedSampleId, strengthKey }), [args.comparison.level, args.focusedSampleId, selectedEvals, selectedMethods, selectedSamples, selectedStrengths]);
  const focusedComparisonId = useMemo(() => buildFocusedComparisonId(focusedEntity, strengthKey), [focusedEntity]);
  const promptHint = useMemo(() => promptHintForEntity(focusedEntity), [focusedEntity]);
  const centerTitle = useMemo(() => buildCenterTitle({ comparisonLevel: args.comparison.level, comparisonCount: args.comparison.ids.length, focusedEntity }), [args.comparison.ids.length, args.comparison.level, focusedEntity]);
  const scopeLine = useMemo(() => buildScopeLine({ comparisonLevel: args.comparison.level, compareChipCount: compareChips.length, focusedEntity }), [args.comparison.level, compareChips.length, focusedEntity]);
  const focusTitle = useMemo(() => buildFocusTitle(focusedEntity), [focusedEntity]);
  const focusScopeLine = useMemo(() => buildFocusScopeLine(focusedEntity), [focusedEntity]);
  const factRows = useMemo(() => factRowsForFocusedEntity(focusedEntity), [focusedEntity]);
  const metricRows = useMemo(() => metricRowsForFocusedEntity(focusedEntity, args.metricMetadata, args.heroRoster), [focusedEntity, args.heroRoster, args.metricMetadata]);
  const heroRailGroups = useMemo(() => heroRailGroupsForFocusedEntity(focusedEntity, args.metricMetadata, args.heroRoster), [focusedEntity, args.heroRoster, args.metricMetadata]);
  const remainingMetricRows = useMemo(() => remainingMetricRowsForFocusedEntity(focusedEntity, args.metricMetadata, args.heroRoster), [focusedEntity, args.heroRoster, args.metricMetadata]);
  const inspectionMetricOptions = useMemo(() => buildMetricInspectionOptions(args.metricMetadata, args.heroRoster), [args.heroRoster, args.metricMetadata]);

  return { focusedEntity, visibleMethods, selectedMethods, selectedEvals, selectedStrengths, selectedSamples, totalSamples, compareChips, focusedComparisonId, promptHint, centerTitle, scopeLine, focusTitle, focusScopeLine, factRows, metricRows, heroRailGroups, remainingMetricRows, inspectionMetricOptions };
}

export function buildMetricInspectionOptions(
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): MetricInspectionOption[] {
  const heroByMetricKey = new Map(heroRoster.map((hero, index) => [hero.metric_key, { hero, index }]));
  const numericGraphKeysFor = (metadata: MetricMetadata) =>
    (metadata.inspection_graph_metric_keys ?? []).filter((metricKey) => {
      const valueType = metricMetadata[metricKey]?.value_type;
      return valueType !== "bool" && valueType !== "str";
    });

  return Object.values(metricMetadata)
    .filter((metadata) => (metadata.inspection_graph_metric_keys?.length ?? 0) > 0)
    .map((metadata) => {
      const heroEntry = heroByMetricKey.get(metadata.key);
      const graphMetricKeys = numericGraphKeysFor(metadata);
      return {
        key: metadata.key,
        label: heroEntry?.hero.label ?? metadata.label,
        keyLabel: metadata.key,
        count: graphMetricKeys.length,
        isHero: Boolean(heroEntry),
        graphMetricKeys,
        heroOrder: heroEntry?.index ?? Number.MAX_SAFE_INTEGER,
      };
    })
    .sort((left, right) => {
      if (left.isHero !== right.isHero) return left.isHero ? -1 : 1;
      if (left.count !== right.count) return right.count - left.count;
      if (left.isHero && right.isHero) return left.heroOrder - right.heroOrder;
      return left.label.localeCompare(right.label);
    })
    .map(({ heroOrder: _heroOrder, ...option }) => option);
}
