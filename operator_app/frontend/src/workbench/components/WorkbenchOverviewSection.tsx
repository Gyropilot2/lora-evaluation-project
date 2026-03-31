import { formatMetric, metricLabel, samplesForEval, strengthKey } from "../helpers";
import { type EvalRef, type FocusKind, type FocusedEntity, type HeroMetricDescriptor, type MethodSlice, type MethodSummary, type SampleRef, type StrengthRef } from "../types";

type WorkbenchOverviewSectionProps = {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  heroRoster: HeroMetricDescriptor[];
  methodSlices: Record<string, MethodSlice>;
  selectedMethods: MethodSummary[];
  selectedEvals: EvalRef[];
  selectedStrengths: StrengthRef[];
  selectedSamples: SampleRef[];
  focusedMethodId: string | null;
  focusedEvalId: string | null;
  focusedStrengthId: string | null;
  focusedSampleId: string | null;
  onFocusMethod: (methodId: string) => void;
  onFocusEval: (methodId: string, evalId: string) => void;
  onFocusStrength: (methodId: string, evalId: string, value: number) => void;
  onFocusSample: (methodId: string, evalId: string, sampleId: string, strengthValue?: number | null) => void;
};

function heroScore(entity: { hero_metrics?: Record<string, { score: number | null }> }, key: string): number | null {
  return entity.hero_metrics?.[key]?.score ?? null;
}

function heroHeaderCells(heroRoster: HeroMetricDescriptor[]) {
  return heroRoster.map((metric) => <th key={metric.key}>{metric.label}</th>);
}

function heroValueCells(entity: { hero_metrics?: Record<string, { score: number | null }> }, heroRoster: HeroMetricDescriptor[]) {
  return heroRoster.map((metric) => (
    <td key={metric.key}>{formatMetric(heroScore(entity, metric.key))}</td>
  ));
}

export function WorkbenchOverviewSection({
  comparisonLevel,
  focusedEntity,
  heroRoster,
  methodSlices,
  selectedMethods,
  selectedEvals,
  selectedStrengths,
  selectedSamples,
  focusedMethodId,
  focusedEvalId,
  focusedStrengthId,
  focusedSampleId,
  onFocusMethod,
  onFocusEval,
  onFocusStrength,
  onFocusSample,
}: WorkbenchOverviewSectionProps) {
  if (comparisonLevel === "method" && selectedMethods.length > 0) {
    return (
      <section className="panel">
        <div className="panel-heading">
          <h2>Method comparison</h2>
          <span className="panel-tag">selected set</span>
        </div>
        <table className="data-table">
          <thead><tr><th>Method</th><th>Evals</th><th>Strengths</th><th>Samples</th>{heroHeaderCells(heroRoster)}</tr></thead>
          <tbody>
            {selectedMethods.map((method) => {
              const slice = methodSlices[method.id];
              return (
                <tr key={method.id} className={method.id === focusedMethodId && !focusedEvalId ? "is-focused-row is-clickable-row" : "is-clickable-row"} onClick={() => onFocusMethod(method.id)}>
                  <td>{method.label}</td><td>{method.eval_count}</td><td>{method.strength_groups}</td><td>{method.sample_count}</td>
                  {slice ? heroValueCells(slice, heroRoster) : heroRoster.map((metric) => <td key={metric.key}>-</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    );
  }

  if (comparisonLevel === "eval" && selectedEvals.length > 0) {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Eval comparison</h2><span className="panel-tag">selected set</span></div>
        <table className="data-table">
          <thead><tr><th>Eval</th><th>Method</th><th>Baseline</th><th>Strengths</th><th>Samples</th>{heroHeaderCells(heroRoster)}</tr></thead>
          <tbody>
            {selectedEvals.map((item) => {
              return (
                <tr key={item.eval.id} className={item.eval.id === focusedEvalId && !focusedStrengthId && !focusedSampleId ? "is-focused-row is-clickable-row" : "is-clickable-row"} onClick={() => onFocusEval(item.method.id, item.eval.id)}>
                  <td>{item.eval.label}</td><td>{item.method.label}</td><td>{item.eval.is_baseline ? "yes" : "no"}</td><td>{item.eval.is_baseline ? 0 : (item.eval.strengths ?? []).length}</td><td>{samplesForEval(item.eval).length}</td>
                  {heroValueCells(item.eval, heroRoster)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    );
  }

  if (comparisonLevel === "strength" && selectedStrengths.length > 0) {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Strength comparison</h2><span className="panel-tag">selected set</span></div>
        <table className="data-table">
          <thead><tr><th>Strength</th><th>Eval</th><th>Method</th><th>Samples</th>{heroHeaderCells(heroRoster)}</tr></thead>
          <tbody>
            {selectedStrengths.map((item) => {
              const id = strengthKey(item.eval.id, item.strength.value);
              return (
                <tr key={id} className={id === focusedStrengthId ? "is-focused-row is-clickable-row" : "is-clickable-row"} onClick={() => onFocusStrength(item.method.id, item.eval.id, item.strength.value)}>
                  <td>{item.strength.label}</td><td>{item.eval.label}</td><td>{item.method.label}</td><td>{item.strength.samples.length}</td>
                  {heroValueCells(item.strength, heroRoster)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    );
  }

  if (comparisonLevel === "sample" && selectedSamples.length > 0) {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Sample comparison</h2><span className="panel-tag">selected set</span></div>
        <table className="data-table">
          <thead><tr><th>Sample</th><th>Eval</th><th>Method</th><th>Seed</th><th>Strength</th>{heroHeaderCells(heroRoster)}</tr></thead>
          <tbody>
            {selectedSamples.map((item) => (
              <tr key={item.sample.id} className={item.sample.id === focusedSampleId ? "is-focused-row is-clickable-row" : "is-clickable-row"} onClick={() => onFocusSample(item.method.id, item.eval.id, item.sample.id, item.sample.strength)}>
                <td>{item.sample.label}</td><td>{item.eval.label}</td><td>{item.method.label}</td><td>{item.sample.seed ?? "-"}</td><td>{item.sample.strength ?? "-"}</td>
                {heroValueCells(item.sample, heroRoster)}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    );
  }

  if (focusedEntity?.kind === "method") {
    const method = focusedEntity.method;
    if (!method) return null;
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Method overview</h2><span className="panel-tag">{method.evals.length} evals</span></div>
        <div className="summary-grid">
          <div className="summary-stat"><strong>Evals</strong><span>{focusedEntity.summary.eval_count}</span></div>
          <div className="summary-stat"><strong>Strength groups</strong><span>{focusedEntity.summary.strength_groups}</span></div>
          <div className="summary-stat"><strong>Samples</strong><span>{focusedEntity.summary.sample_count}</span></div>
        </div>
        <table className="data-table">
          <thead><tr><th>Eval</th><th>Baseline</th><th>Strengths</th><th>Samples</th></tr></thead>
          <tbody>
            {method.evals.map((evalItem) => (
              <tr key={evalItem.id} className="is-clickable-row" onClick={() => onFocusEval(method.id, evalItem.id)}>
                <td>{evalItem.label}</td><td>{evalItem.is_baseline ? "yes" : "no"}</td><td>{evalItem.is_baseline ? 0 : (evalItem.strengths ?? []).length}</td>
                <td>{evalItem.is_baseline ? (evalItem.samples ?? []).length : (evalItem.strengths ?? []).reduce((sum, strength) => sum + strength.samples.length, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    );
  }

  if (focusedEntity?.kind === "eval") {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Eval overview</h2><span className="panel-tag">{focusedEntity.eval.is_baseline ? "baseline" : "lora eval"}</span></div>
        {focusedEntity.eval.is_baseline ? (
          <div className="summary-grid">
            <div className="summary-stat"><strong>Samples</strong><span>{(focusedEntity.eval.samples ?? []).length}</span></div>
            <div className="summary-stat"><strong>LoRA hash</strong><span>{focusedEntity.eval.lora_hash ?? "-"}</span></div>
            <div className="summary-stat"><strong>Type</strong><span>Baseline</span></div>
          </div>
        ) : (
          <table className="data-table">
            <thead><tr><th>Strength</th><th>Samples</th>{heroHeaderCells(heroRoster)}</tr></thead>
            <tbody>
              {(focusedEntity.eval.strengths ?? []).map((strength) => (
                <tr key={strength.value} className="is-clickable-row" onClick={() => onFocusStrength(focusedEntity.method.id, focusedEntity.eval.id, strength.value)}>
                  <td>{strength.label}</td><td>{strength.samples.length}</td>{heroValueCells(strength, heroRoster)}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    );
  }

  if (focusedEntity?.kind === "strength") {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Strength overview</h2><span className="panel-tag">{focusedEntity.strength.samples.length} samples</span></div>
        <div className="summary-grid">
          {heroRoster.map((metric) => (
            <div key={metric.key} className="summary-stat"><strong>{metricLabel(metric.key, undefined, heroRoster)}</strong><span>{formatMetric(heroScore(focusedEntity.strength, metric.key))}</span></div>
          ))}
        </div>
      </section>
    );
  }

  if (focusedEntity?.kind === "sample") {
    return (
      <section className="panel">
        <div className="panel-heading"><h2>Sample overview</h2><span className="panel-tag">{focusedEntity.eval.label}</span></div>
        <div className="summary-grid">
          {heroRoster.map((metric) => (
            <div key={metric.key} className="summary-stat"><strong>{metricLabel(metric.key, undefined, heroRoster)}</strong><span>{formatMetric(heroScore(focusedEntity.sample, metric.key))}</span></div>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-heading"><h2>Workbench</h2><span className="panel-tag">waiting for scope</span></div>
      <p className="compact-copy">Click something in the tree to inspect it, or build a compare set to take over the center.</p>
    </section>
  );
}
