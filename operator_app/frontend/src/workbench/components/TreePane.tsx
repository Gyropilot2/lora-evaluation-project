import "./TreePane.css";
import { sameSeedLabel, strengthKey } from "../helpers";
import type { FocusKind, VisibleMethod } from "../types";

type TreePaneProps = {
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  loadingMessage: string;
  error: string | null;
  visibleMethods: VisibleMethod[];
  loadingMethodIds: string[];
  expandedMethodIds: string[];
  expandedEvalIds: string[];
  expandedStrengthIds: string[];
  focusedMethodId: string | null;
  focusedEvalId: string | null;
  focusedStrengthId: string | null;
  focusedSampleId: string | null;
  comparisonIds: string[];
  comparisonLevel: FocusKind | null;
  onToggleFocusedInComparison: () => void;
  onClearComparison: () => void;
  hasFocusedEntity: boolean;
  focusedInComparison: boolean;
  onToggleDisclosure: (level: "method" | "eval" | "strength", id: string) => void;
  onFocusMethod: (methodId: string) => void;
  onFocusEval: (methodId: string, evalId: string) => void;
  onFocusStrength: (methodId: string, evalId: string, value: number) => void;
  onFocusSample: (methodId: string, evalId: string, sampleId: string, strengthValue?: number | null) => void;
  onToggleComparison: (level: FocusKind, id: string) => void;
  onSelectAllEvalsForMethod: (methodId: string) => void;
  onSelectAllStrengthsForEval: (evalId: string) => void;
  onSelectAllSamplesForEval: (evalId: string) => void;
  onSelectAllSamplesForStrength: (strengthId: string) => void;
  onSelectSameSeedPeers: (sampleId?: string) => void;
  onSelectSameStrengthPeers: (sampleId?: string, strengthId?: string) => void;
};

function SidebarActions({
  onToggleFocusedInComparison,
  onClearComparison,
  hasFocusedEntity,
  focusedInComparison,
  hasComparison,
}: {
  onToggleFocusedInComparison: () => void;
  onClearComparison: () => void;
  hasFocusedEntity: boolean;
  focusedInComparison: boolean;
  hasComparison: boolean;
}) {
  return (
    <div className="sidebar-actions">
      <button type="button" onClick={onToggleFocusedInComparison} disabled={!hasFocusedEntity}>
        {focusedInComparison ? "Toggle focused" : "Add focused"}
      </button>
      <button type="button" onClick={onClearComparison} disabled={!hasComparison}>
        Clear compare
      </button>
    </div>
  );
}

export function TreePane({
  searchQuery,
  onSearchQueryChange,
  loadingMessage,
  error,
  visibleMethods,
  loadingMethodIds,
  expandedMethodIds,
  expandedEvalIds,
  expandedStrengthIds,
  focusedMethodId,
  focusedEvalId,
  focusedStrengthId,
  focusedSampleId,
  comparisonIds,
  comparisonLevel,
  onToggleFocusedInComparison,
  onClearComparison,
  hasFocusedEntity,
  focusedInComparison,
  onToggleDisclosure,
  onFocusMethod,
  onFocusEval,
  onFocusStrength,
  onFocusSample,
  onToggleComparison,
  onSelectAllEvalsForMethod,
  onSelectAllStrengthsForEval,
  onSelectAllSamplesForEval,
  onSelectAllSamplesForStrength,
  onSelectSameSeedPeers,
  onSelectSameStrengthPeers,
}: TreePaneProps) {
  function isChecked(id: string): boolean {
    return comparisonIds.includes(id);
  }

  function isCheckboxDisabled(level: FocusKind): boolean {
    return comparisonLevel !== null && comparisonLevel !== level;
  }

  function renderTree() {
    return visibleMethods.map(({ summary, evals }) => {
      const expandedMethod = expandedMethodIds.includes(summary.id);
      const methodLoading = loadingMethodIds.includes(summary.id) && !evals;
      return (
        <div key={summary.id} className="tree-group tree-method">
          <div className="tree-row">
            <input
              type="checkbox"
              checked={isChecked(summary.id)}
              onChange={() => onToggleComparison("method", summary.id)}
              disabled={isCheckboxDisabled("method")}
            />
            <button type="button" className="tree-toggle" onClick={() => onToggleDisclosure("method", summary.id)}>
              {expandedMethod ? "-" : "+"}
            </button>
            <button
              type="button"
              className={`tree-focus${focusedMethodId === summary.id && !focusedEvalId ? " is-focused" : ""}`}
              onClick={() => onFocusMethod(summary.id)}
            >
              {summary.label}
            </button>
            <div className="tree-inline-actions">
              <button type="button" className="tree-mini-action" onClick={() => onSelectAllEvalsForMethod(summary.id)}>
                eval
              </button>
            </div>
          </div>
          {expandedMethod && (
            <div className="tree-branch">
              {methodLoading && <div className="tree-loading">Loading method...</div>}
              {!methodLoading &&
                evals?.map((evalItem) => {
                  const expandedEval = expandedEvalIds.includes(evalItem.id);
                  return (
                    <div key={evalItem.id} className="tree-group">
                      <div className="tree-row">
                        <input
                          type="checkbox"
                          checked={isChecked(evalItem.id)}
                          onChange={() => onToggleComparison("eval", evalItem.id)}
                          disabled={isCheckboxDisabled("eval")}
                        />
                        <button type="button" className="tree-toggle" onClick={() => onToggleDisclosure("eval", evalItem.id)}>
                          {expandedEval ? "-" : "+"}
                        </button>
                        <button
                          type="button"
                          className={`tree-focus${focusedEvalId === evalItem.id && !focusedStrengthId && !focusedSampleId ? " is-focused" : ""}`}
                          onClick={() => onFocusEval(summary.id, evalItem.id)}
                        >
                          {evalItem.label}
                        </button>
                        <div className="tree-inline-actions">
                          {!evalItem.is_baseline && (
                            <button
                              type="button"
                              className="tree-mini-action"
                              onClick={() => onSelectAllStrengthsForEval(evalItem.id)}
                            >
                              str
                            </button>
                          )}
                          {!evalItem.is_baseline && (
                            <button
                              type="button"
                              className="tree-mini-action"
                              onClick={() => onSelectAllSamplesForEval(evalItem.id)}
                            >
                              samp
                            </button>
                          )}
                        </div>
                      </div>
                      {expandedEval && (
                        <div className="tree-branch">
                          {evalItem.is_baseline
                            ? (evalItem.samples ?? []).map((sample) => (
                                <div key={sample.id} className="tree-row">
                                  <input
                                    type="checkbox"
                                    checked={isChecked(sample.id)}
                                    onChange={() => onToggleComparison("sample", sample.id)}
                                    disabled={isCheckboxDisabled("sample")}
                                  />
                                  <span className="tree-indent" />
                                  <button
                                    type="button"
                                    className={`tree-focus${focusedSampleId === sample.id ? " is-focused" : ""}`}
                                    onClick={() => onFocusSample(summary.id, evalItem.id, sample.id, sample.strength)}
                                  >
                                    {sample.label}
                                  </button>
                                </div>
                              ))
                            : (evalItem.strengths ?? []).map((strength) => {
                                const id = strengthKey(evalItem.id, strength.value);
                                const expandedStrength = expandedStrengthIds.includes(id);
                                return (
                                  <div key={id} className="tree-group">
                                    <div className="tree-row">
                                      <input
                                        type="checkbox"
                                        checked={isChecked(id)}
                                        onChange={() => onToggleComparison("strength", id)}
                                        disabled={isCheckboxDisabled("strength")}
                                      />
                                      <button
                                        type="button"
                                        className="tree-toggle"
                                        onClick={() => onToggleDisclosure("strength", id)}
                                      >
                                        {expandedStrength ? "-" : "+"}
                                      </button>
                                      <button
                                        type="button"
                                        className={`tree-focus${focusedStrengthId === id && !focusedSampleId ? " is-focused" : ""}`}
                                        onClick={() => onFocusStrength(summary.id, evalItem.id, strength.value)}
                                      >
                                        {strength.label}
                                      </button>
                                      <div className="tree-inline-actions">
                                        <button type="button" className="tree-mini-action" onClick={() => onSelectAllSamplesForStrength(id)}>
                                          samp
                                        </button>
                                      </div>
                                    </div>
                                    {expandedStrength && (
                                      <div className="tree-branch">
                                        {strength.samples.map((sample) => (
                                          <div key={sample.id} className="tree-row">
                                            <input
                                              type="checkbox"
                                              checked={isChecked(sample.id)}
                                              onChange={() => onToggleComparison("sample", sample.id)}
                                              disabled={isCheckboxDisabled("sample")}
                                            />
                                            <span className="tree-indent" />
                                            <button
                                              type="button"
                                              className={`tree-focus${focusedSampleId === sample.id ? " is-focused" : ""}`}
                                              onClick={() => onFocusSample(summary.id, evalItem.id, sample.id, sample.strength)}
                                            >
                                              {sameSeedLabel(sample)}
                                            </button>
                                            <div className="tree-inline-actions">
                                              <button type="button" className="tree-mini-action" onClick={() => onSelectSameSeedPeers(sample.id)}>
                                                seed
                                              </button>
                                              {sample.strength !== null && sample.strength !== undefined && (
                                                <button type="button" className="tree-mini-action" onClick={() => onSelectSameStrengthPeers(sample.id)}>
                                                  str
                                                </button>
                                              )}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      );
    });
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-toolbar">
        <input
          value={searchQuery}
          onChange={(event) => onSearchQueryChange(event.target.value)}
          placeholder="Search methods, evals, strengths..."
        />
        <SidebarActions
          onToggleFocusedInComparison={onToggleFocusedInComparison}
          onClearComparison={onClearComparison}
          hasFocusedEntity={hasFocusedEntity}
          focusedInComparison={focusedInComparison}
          hasComparison={Boolean(comparisonLevel)}
        />
        {loadingMessage ? <div className="status-copy">{loadingMessage}</div> : null}
        {error ? <div className="status-copy status-error">{error}</div> : null}
      </div>
      <div className="tree">{renderTree()}</div>
    </aside>
  );
}
