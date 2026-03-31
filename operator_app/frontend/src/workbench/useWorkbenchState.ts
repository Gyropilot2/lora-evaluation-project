import { useCallback, useEffect } from "react";

import { strengthKey } from "./helpers";
import { useReviewData } from "./useReviewData";
import { useWorkbenchDerivedState, useWorkbenchRefs } from "./workbenchModel";
import { useWorkbenchNavigation } from "./useWorkbenchNavigation";

export function useWorkbenchState() {
  const reviewData = useReviewData();

  const refs = useWorkbenchRefs(reviewData.methods, reviewData.methodSlices);

  const baseNavigation = useWorkbenchNavigation({
    initialMethodId: reviewData.initialMethodId,
    methodSlices: reviewData.methodSlices,
    refs,
    ensureMethodLoaded: reviewData.ensureMethodLoaded,
    queueMethodLoads: reviewData.queueMethodLoads,
  });

  const derived = useWorkbenchDerivedState({
    methods: reviewData.methods,
    methodSlices: reviewData.methodSlices,
    searchQuery: baseNavigation.searchQuery,
    comparison: baseNavigation.comparison,
    focusedMethodId: baseNavigation.focusedMethodId,
    focusedEvalId: baseNavigation.focusedEvalId,
    focusedStrengthId: baseNavigation.focusedStrengthId,
    focusedSampleId: baseNavigation.focusedSampleId,
    heroRoster: reviewData.heroRoster,
    metricMetadata: reviewData.metricMetadata,
    refs,
  });

  useEffect(() => {
    if (!baseNavigation.selectedMetric) return;
    const stillAvailable = derived.inspectionMetricOptions.some((option) => option.key === baseNavigation.selectedMetric);
    if (!stillAvailable) baseNavigation.setSelectedMetric(null);
  }, [baseNavigation.selectedMetric, baseNavigation.setSelectedMetric, derived.inspectionMetricOptions]);

  const toggleFocusedInComparison = useCallback(() => {
    const focusedEntity = derived.focusedEntity;
    if (!focusedEntity) return;
    if (focusedEntity.kind === "method") return baseNavigation.toggleComparison("method", focusedEntity.summary.id);
    if (focusedEntity.kind === "eval") return baseNavigation.toggleComparison("eval", focusedEntity.eval.id);
    if (focusedEntity.kind === "strength") return baseNavigation.toggleComparison("strength", strengthKey(focusedEntity.eval.id, focusedEntity.strength.value));
    baseNavigation.toggleComparison("sample", focusedEntity.sample.id);
  }, [baseNavigation, derived.focusedEntity]);

  const selectSameSeedPeers = useCallback((sampleId?: string) => {
    const source =
      sampleId !== undefined
        ? refs.sampleRefs.get(sampleId)
        : derived.focusedEntity?.kind === "sample"
          ? refs.sampleRefs.get(derived.focusedEntity.sample.id)
          : undefined;
    if (!source || source.eval.is_baseline || source.sample.seed === undefined) return;
    const ids = Array.from(refs.sampleRefs.values())
      .filter((item) => item.method.id === source.method.id && !item.eval.is_baseline && item.sample.seed === source.sample.seed)
      .map((item) => item.sample.id);
    baseNavigation.toggleComparisonGroup("sample", ids);
  }, [baseNavigation, derived.focusedEntity, refs.sampleRefs]);

  const selectSameStrengthPeers = useCallback((sampleId?: string, strengthId?: string) => {
    if (sampleId !== undefined || derived.focusedEntity?.kind === "sample") {
      const source =
        sampleId !== undefined
          ? refs.sampleRefs.get(sampleId)
          : derived.focusedEntity?.kind === "sample"
            ? refs.sampleRefs.get(derived.focusedEntity.sample.id)
            : undefined;
      if (!source || source.eval.is_baseline || source.sample.strength === null || source.sample.strength === undefined) return;
      const ids = Array.from(refs.sampleRefs.values())
        .filter((item) => item.method.id === source.method.id && !item.eval.is_baseline && item.sample.strength === source.sample.strength)
        .map((item) => item.sample.id);
      return baseNavigation.toggleComparisonGroup("sample", ids);
    }
    if (strengthId !== undefined || derived.focusedEntity?.kind === "strength") {
      const source =
        strengthId !== undefined
          ? refs.strengthRefs.get(strengthId)
          : derived.focusedEntity?.kind === "strength"
            ? refs.strengthRefs.get(strengthKey(derived.focusedEntity.eval.id, derived.focusedEntity.strength.value))
            : undefined;
      if (!source) return;
      const ids = Array.from(refs.strengthRefs.values())
        .filter((item) => item.method.id === source.method.id && item.strength.value === source.strength.value)
        .map((item) => strengthKey(item.eval.id, item.strength.value));
      baseNavigation.toggleComparisonGroup("strength", ids);
    }
  }, [baseNavigation, derived.focusedEntity, refs.sampleRefs, refs.strengthRefs]);

  const focusFromChip = useCallback((level: "method" | "eval" | "strength" | "sample", id: string) => {
    if (level === "method") return baseNavigation.focusMethod(id);
    if (level === "eval") {
      const ref = refs.evalRefs.get(id);
      if (ref) baseNavigation.focusEval(ref.method.id, ref.eval.id);
      return;
    }
    if (level === "strength") {
      const ref = refs.strengthRefs.get(id);
      if (ref) baseNavigation.focusStrength(ref.method.id, ref.eval.id, ref.strength.value);
      return;
    }
    const ref = refs.sampleRefs.get(id);
    if (ref) baseNavigation.focusSample(ref.method.id, ref.eval.id, ref.sample.id, ref.sample.strength);
  }, [baseNavigation, refs.evalRefs, refs.sampleRefs, refs.strengthRefs]);

  return {
    activeWorkspace: baseNavigation.activeWorkspace,
    setActiveWorkspace: baseNavigation.setActiveWorkspace,
    methods: reviewData.methods,
    heroRoster: reviewData.heroRoster,
    metricMetadata: reviewData.metricMetadata,
    totalSamples: derived.totalSamples,
    methodSlices: reviewData.methodSlices,
    loadingMessage: reviewData.loadingMessage,
    loadingMethodIds: reviewData.loadingMethodIds,
    error: reviewData.error,
    searchQuery: baseNavigation.searchQuery,
    setSearchQuery: baseNavigation.setSearchQuery,
    expandedMethodIds: baseNavigation.expandedMethodIds,
    expandedEvalIds: baseNavigation.expandedEvalIds,
    expandedStrengthIds: baseNavigation.expandedStrengthIds,
    focusedMethodId: baseNavigation.focusedMethodId,
    focusedEvalId: baseNavigation.focusedEvalId,
    focusedStrengthId: baseNavigation.focusedStrengthId,
    focusedSampleId: baseNavigation.focusedSampleId,
    comparison: baseNavigation.comparison,
    focusedEntity: derived.focusedEntity,
    visibleMethods: derived.visibleMethods,
    selectedMethods: derived.selectedMethods,
    selectedEvals: derived.selectedEvals,
    selectedStrengths: derived.selectedStrengths,
    selectedSamples: derived.selectedSamples,
    sampleRefs: refs.sampleRefs,
    compareChips: derived.compareChips,
    focusedComparisonId: derived.focusedComparisonId,
    selectedMetric: baseNavigation.selectedMetric,
    setSelectedMetric: baseNavigation.setSelectedMetric,
    inspectionMetricOptions: derived.inspectionMetricOptions,
    promptHint: derived.promptHint,
    centerTitle: derived.centerTitle,
    scopeLine: derived.scopeLine,
    focusTitle: derived.focusTitle,
    focusScopeLine: derived.focusScopeLine,
    factRows: derived.factRows,
    metricRows: derived.metricRows,
    heroRailGroups: derived.heroRailGroups,
    remainingMetricRows: derived.remainingMetricRows,
    toggleDisclosure: baseNavigation.toggleDisclosure,
    focusMethod: baseNavigation.focusMethod,
    focusEval: baseNavigation.focusEval,
    focusStrength: baseNavigation.focusStrength,
    focusSample: baseNavigation.focusSample,
    clearComparison: baseNavigation.clearComparison,
    toggleComparison: baseNavigation.toggleComparison,
    removeFromComparison: baseNavigation.removeFromComparison,
    selectAllEvalsForMethod: baseNavigation.selectAllEvalsForMethod,
    selectAllStrengthsForEval: baseNavigation.selectAllStrengthsForEval,
    selectAllSamplesForEval: baseNavigation.selectAllSamplesForEval,
    selectAllSamplesForStrength: baseNavigation.selectAllSamplesForStrength,
    toggleFocusedInComparison,
    selectSameSeedPeers,
    selectSameStrengthPeers,
    focusFromChip,
    sampleEnrichments: reviewData.sampleEnrichments,
    sampleDetails: reviewData.sampleDetails,
    sampleDetailLoadingIds: reviewData.sampleDetailLoadingIds,
    ensureSampleDetailLoaded: reviewData.ensureSampleDetailLoaded,
    enrichmentProgress: reviewData.enrichmentProgress,
  };
}
