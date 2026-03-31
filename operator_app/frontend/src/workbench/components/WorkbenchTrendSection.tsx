import "./WorkbenchTrendSection.css";
import { strengthKey } from "../helpers";
import { type StrengthResponseItem } from "./StrengthTrend";
import { MetricTrendTable } from "./MetricTrendTable";
import {
  type EvalRef,
  type FocusKind,
  type FocusedEntity,
  type HeroMetricDescriptor,
  type MetricMetadata,
  type MethodSlice,
  type StrengthRef,
} from "../types";

type WorkbenchTrendSectionProps = {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  heroRoster: HeroMetricDescriptor[];
  focusedEvalId: string | null;
  focusedStrengthId: string | null;
  metricMetadata: Record<string, MetricMetadata>;
  selectedEvals: EvalRef[];
  selectedStrengths: StrengthRef[];
  onFocusEval: (methodId: string, evalId: string) => void;
};

function strengthResponseItemsForSelectedEvals(
  selectedEvals: EvalRef[],
  focusedEvalId: string | null,
  onFocusEval: (methodId: string, evalId: string) => void,
): StrengthResponseItem[] {
  return selectedEvals
    .filter((item) => !item.eval.is_baseline)
    .map((item) => ({
      id: `${item.method.id}-${item.eval.id}`,
      label: `${item.method.label} - ${item.eval.label}`,
      strengths: item.eval.strengths ?? [],
      isFocused: item.eval.id === focusedEvalId,
      onFocus: () => onFocusEval(item.method.id, item.eval.id),
    }));
}

function strengthResponseItemsForMethod(method: MethodSlice, onFocusEval: (methodId: string, evalId: string) => void, focusedEvalId: string | null): StrengthResponseItem[] {
  return method.evals
    .filter((evalItem) => !evalItem.is_baseline)
    .map((evalItem) => ({
      id: evalItem.id,
      label: evalItem.label,
      strengths: evalItem.strengths ?? [],
      isFocused: evalItem.id === focusedEvalId,
      onFocus: () => onFocusEval(method.id, evalItem.id),
    }));
}

function strengthResponseItemsForFocusedEval(
  focusedEval: Extract<FocusedEntity, { kind: "eval" }>,
  onFocusEval: (methodId: string, evalId: string) => void,
): StrengthResponseItem[] {
  return [
    {
      id: focusedEval.eval.id,
      label: focusedEval.eval.label,
      strengths: focusedEval.eval.strengths ?? [],
      isFocused: true,
      onFocus: () => onFocusEval(focusedEval.method.id, focusedEval.eval.id),
    },
  ];
}

function strengthResponseItemsForSelectedStrengths(
  selectedStrengths: StrengthRef[],
  focusedEvalId: string | null,
  focusedStrengthId: string | null,
  onFocusEval: (methodId: string, evalId: string) => void,
): StrengthResponseItem[] {
  return selectedStrengths.reduce<StrengthResponseItem[]>((out, item) => {
    const groupId = `${item.method.id}::${item.eval.id}`;
    const existing = out.find((group) => group.id === groupId);
    if (existing) {
      existing.strengths.push(item.strength);
      if (focusedStrengthId === strengthKey(item.eval.id, item.strength.value) || focusedEvalId === item.eval.id) {
        existing.isFocused = true;
      }
      return out;
    }
    out.push({
      id: groupId,
      label: `${item.method.label} - ${item.eval.label}`,
      strengths: [item.strength],
      isFocused: focusedStrengthId === strengthKey(item.eval.id, item.strength.value) || focusedEvalId === item.eval.id,
      onFocus: () => onFocusEval(item.method.id, item.eval.id),
    });
    return out;
  }, []);
}

export function WorkbenchTrendSection({
  comparisonLevel,
  focusedEntity,
  heroRoster,
  focusedEvalId,
  focusedStrengthId,
  metricMetadata,
  selectedEvals,
  selectedStrengths,
  onFocusEval,
}: WorkbenchTrendSectionProps) {
  if (comparisonLevel === "eval" && selectedEvals.length > 0) {
    const items = strengthResponseItemsForSelectedEvals(selectedEvals, focusedEvalId, onFocusEval);
    if (items.length === 0) return null;
    return (
      <section className="panel">
        <div className="panel-heading">
          <h2>Trend review</h2>
          <span className="panel-tag">selected evals</span>
        </div>
        <MetricTrendTable items={items} heroRoster={heroRoster} metricMetadata={metricMetadata} />
      </section>
    );
  }

  if (focusedEntity?.kind === "method" && focusedEntity.method) {
    const items = strengthResponseItemsForMethod(focusedEntity.method, onFocusEval, focusedEvalId);
    if (items.length === 0) return null;
    return (
      <section className="panel">
        <div className="panel-heading">
          <h2>Trend review</h2>
          <span className="panel-tag">method evals</span>
        </div>
        <MetricTrendTable items={items} heroRoster={heroRoster} metricMetadata={metricMetadata} />
      </section>
    );
  }

  if (focusedEntity?.kind === "eval" && !focusedEntity.eval.is_baseline) {
    const items = strengthResponseItemsForFocusedEval(focusedEntity, onFocusEval);
    return (
      <section className="panel">
        <div className="panel-heading">
          <h2>Trend review</h2>
          <span className="panel-tag">focused eval</span>
        </div>
        <MetricTrendTable items={items} heroRoster={heroRoster} metricMetadata={metricMetadata} />
      </section>
    );
  }

  if (comparisonLevel === "strength" && selectedStrengths.length > 0) {
    const items = strengthResponseItemsForSelectedStrengths(selectedStrengths, focusedEvalId, focusedStrengthId, onFocusEval);
    const allSamples = selectedStrengths.flatMap((item) => item.strength.samples);

    return (
      <section className="panel">
        <div className="panel-heading">
          <h2>Trend review</h2>
          <span className="panel-tag">selected strengths</span>
        </div>
        <p className="compact-copy">
          Trend summaries now track badness, unreliability, and dropped fraction at each selected strength, so the pose line rises as the LoRA struggles and the red drop bars only speak up when it falls out entirely.
        </p>
        <div className="trend-summary-grid">
          <div className="trend-summary-stat">
            <strong>Selected strengths</strong>
            <span>{selectedStrengths.length}</span>
          </div>
          <div className="trend-summary-stat">
            <strong>Samples in scope</strong>
            <span>{allSamples.length}</span>
          </div>
          <div className="trend-summary-stat">
            <strong>Eval groups</strong>
            <span>{items.length}</span>
          </div>
          <div className="trend-summary-stat">
            <strong>Hero channels</strong>
            <span>{heroRoster.length}</span>
          </div>
        </div>
        <MetricTrendTable items={items} heroRoster={heroRoster} metricMetadata={metricMetadata} />
      </section>
    );
  }

  return null;
}
