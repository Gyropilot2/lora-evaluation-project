import { useCallback, useEffect, useState } from "react";

import { strengthKey, uniqIds } from "./helpers";
import { type ComparisonState, type FocusKind, type MethodSlice, type WorkspaceTab } from "./types";
import type { WorkbenchRefs } from "./workbenchModel";

export function useWorkbenchNavigation(args: {
  initialMethodId: string | null;
  methodSlices: Record<string, MethodSlice>;
  refs: WorkbenchRefs;
  ensureMethodLoaded: (methodId: string) => void;
  queueMethodLoads: (methodIds: string[], priority?: "front" | "back") => void;
}) {
  const { initialMethodId, methodSlices, refs, ensureMethodLoaded, queueMethodLoads } = args;
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceTab>("workbench");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedMethodIds, setExpandedMethodIds] = useState<string[]>([]);
  const [expandedEvalIds, setExpandedEvalIds] = useState<string[]>([]);
  const [expandedStrengthIds, setExpandedStrengthIds] = useState<string[]>([]);
  const [focusedMethodId, setFocusedMethodId] = useState<string | null>(null);
  const [focusedEvalId, setFocusedEvalId] = useState<string | null>(null);
  const [focusedStrengthId, setFocusedStrengthId] = useState<string | null>(null);
  const [focusedSampleId, setFocusedSampleId] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonState>({ level: null, ids: [] });

  useEffect(() => {
    if (!initialMethodId) return;
    setFocusedMethodId((current) => current ?? initialMethodId);
    setExpandedMethodIds((current) => (current.length > 0 ? current : [initialMethodId]));
  }, [initialMethodId]);

  const toggleDisclosure = useCallback((level: "method" | "eval" | "strength", id: string) => {
    if (level === "method") {
      setExpandedMethodIds((current) => {
        if (current.includes(id)) return current.filter((item) => item !== id);
        ensureMethodLoaded(id);
        return [...current, id];
      });
      return;
    }
    if (level === "eval") return setExpandedEvalIds((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));
    setExpandedStrengthIds((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]));
  }, [ensureMethodLoaded]);

  const focusMethod = useCallback((methodId: string) => {
    setFocusedMethodId(methodId); setFocusedEvalId(null); setFocusedStrengthId(null); setFocusedSampleId(null);
    setExpandedMethodIds((current) => (current.includes(methodId) ? current : [...current, methodId]));
    ensureMethodLoaded(methodId);
  }, [ensureMethodLoaded]);

  const focusEval = useCallback((methodId: string, evalId: string) => {
    setFocusedMethodId(methodId); setFocusedEvalId(evalId); setFocusedStrengthId(null); setFocusedSampleId(null);
    setExpandedMethodIds((current) => (current.includes(methodId) ? current : [...current, methodId]));
    setExpandedEvalIds((current) => (current.includes(evalId) ? current : [...current, evalId]));
    ensureMethodLoaded(methodId);
  }, [ensureMethodLoaded]);

  const focusStrength = useCallback((methodId: string, evalId: string, value: number) => {
    const id = strengthKey(evalId, value);
    setFocusedMethodId(methodId); setFocusedEvalId(evalId); setFocusedStrengthId(id); setFocusedSampleId(null);
    setExpandedMethodIds((current) => (current.includes(methodId) ? current : [...current, methodId]));
    setExpandedEvalIds((current) => (current.includes(evalId) ? current : [...current, evalId]));
    setExpandedStrengthIds((current) => (current.includes(id) ? current : [...current, id]));
    ensureMethodLoaded(methodId);
  }, [ensureMethodLoaded]);

  const focusSample = useCallback((methodId: string, evalId: string, sampleId: string, strengthValue?: number | null) => {
    setFocusedMethodId(methodId); setFocusedEvalId(evalId);
    setFocusedStrengthId(strengthValue === null || strengthValue === undefined ? null : strengthKey(evalId, strengthValue));
    setFocusedSampleId(sampleId);
    setExpandedMethodIds((current) => (current.includes(methodId) ? current : [...current, methodId]));
    setExpandedEvalIds((current) => (current.includes(evalId) ? current : [...current, evalId]));
    if (strengthValue !== null && strengthValue !== undefined) setExpandedStrengthIds((current) => (current.includes(strengthKey(evalId, strengthValue)) ? current : [...current, strengthKey(evalId, strengthValue)]));
    ensureMethodLoaded(methodId);
  }, [ensureMethodLoaded]);

  const toggleComparisonGroup = useCallback((level: FocusKind, ids: string[]) => {
    const nextIds = uniqIds(ids);
    if (nextIds.length === 0) return;
    if (level === "method") queueMethodLoads(nextIds, "front");
    setComparison((current) => {
      if (current.level !== level) return { level, ids: nextIds };
      const allSelected = nextIds.every((id) => current.ids.includes(id));
      if (allSelected) {
        const remaining = current.ids.filter((id) => !nextIds.includes(id));
        return remaining.length ? { level, ids: remaining } : { level: null, ids: [] };
      }
      return { level, ids: uniqIds([...current.ids, ...nextIds]) };
    });
  }, [queueMethodLoads]);

  const clearComparison = useCallback(() => setComparison({ level: null, ids: [] }), []);
  const toggleComparison = useCallback((level: FocusKind, id: string) => { if (level === "method") ensureMethodLoaded(id); toggleComparisonGroup(level, [id]); }, [ensureMethodLoaded, toggleComparisonGroup]);
  const removeFromComparison = useCallback((level: FocusKind, id: string) => setComparison((current) => {
    if (current.level !== level) return current;
    const remaining = current.ids.filter((item) => item !== id);
    return remaining.length ? { level, ids: remaining } : { level: null, ids: [] };
  }), []);

  const selectAllEvalsForMethod = useCallback((methodId: string) => {
    const method = methodSlices[methodId];
    if (!method) return;
    toggleComparisonGroup("eval", method.evals.filter((evalItem) => !evalItem.is_baseline).map((evalItem) => evalItem.id));
  }, [methodSlices, toggleComparisonGroup]);

  const selectAllStrengthsForEval = useCallback((evalId: string) => {
    const ref = refs.evalRefs.get(evalId);
    if (!ref || ref.eval.is_baseline) return;
    toggleComparisonGroup("strength", (ref.eval.strengths ?? []).map((strength) => strengthKey(ref.eval.id, strength.value)));
  }, [refs.evalRefs, toggleComparisonGroup]);

  const selectAllSamplesForEval = useCallback((evalId: string) => {
    const ref = refs.evalRefs.get(evalId);
    if (!ref || ref.eval.is_baseline) return;
    toggleComparisonGroup("sample", (ref.eval.strengths ?? []).flatMap((strength) => strength.samples.map((sample) => sample.id)));
  }, [refs.evalRefs, toggleComparisonGroup]);

  const selectAllSamplesForStrength = useCallback((strengthId: string) => {
    const ref = refs.strengthRefs.get(strengthId);
    if (!ref) return;
    toggleComparisonGroup("sample", ref.strength.samples.map((sample) => sample.id));
  }, [refs.strengthRefs, toggleComparisonGroup]);

  return { activeWorkspace, setActiveWorkspace, searchQuery, setSearchQuery, expandedMethodIds, expandedEvalIds, expandedStrengthIds, focusedMethodId, focusedEvalId, focusedStrengthId, focusedSampleId, selectedMetric, setSelectedMetric, comparison, toggleDisclosure, focusMethod, focusEval, focusStrength, focusSample, clearComparison, toggleComparison, removeFromComparison, selectAllEvalsForMethod, selectAllStrengthsForEval, selectAllSamplesForEval, selectAllSamplesForStrength, toggleComparisonGroup };
}
