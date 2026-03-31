import "./App.css";
import "./workbench/shared.css";
import { CompareTray } from "./workbench/components/CompareTray";
import { LogsWorkspace } from "./workbench/components/LogsWorkspace";
import { MetricRail } from "./workbench/components/MetricRail";
import { SelectedMetricPicker } from "./workbench/components/SelectedMetricPicker";
import { TreePane } from "./workbench/components/TreePane";
import { WorkbenchMain } from "./workbench/components/WorkbenchMain";
import { useWorkbenchState } from "./workbench/useWorkbenchState";

export default function App() {
  const workbench = useWorkbenchState();

  const totalMethods  = workbench.methods.length;
  const loadedMethods = Object.keys(workbench.methodSlices).length;
  const { total: enrichTotal, loaded: enrichLoaded } = workbench.enrichmentProgress;

  const isLoadingMethods = !!workbench.loadingMessage;
  const isLoadingAssets  = enrichTotal > 0 && enrichLoaded < enrichTotal;
  const isLoading        = isLoadingMethods || isLoadingAssets;
  const isBootstrapping  = isLoadingMethods && workbench.methods.length === 0;
  const isEmptyDb        = !isLoading && !workbench.error && workbench.methods.length === 0;

  // Two-phase bar: methods fill the first 50%, enrichment fills the second 50%.
  // Keeping phases separate prevents the bar from going backwards — enrichTotal
  // grows as method slices arrive, so a combined denominator would shrink progress.
  const loadingPct = isLoadingMethods && totalMethods > 0
    ? (loadedMethods / totalMethods) * 50
    : enrichTotal > 0
    ? 50 + (enrichLoaded / enrichTotal) * 50
    : 0;

  const loadingLabel = isLoadingMethods
    ? (workbench.methods.find((m) => m.id === workbench.loadingMethodIds[0])?.label ?? null)
    : isLoadingAssets
    ? `assets ${enrichLoaded}/${enrichTotal}`
    : null;
  const workspaceBadge = workbench.activeWorkspace === "logs" ? "Logs / Ops" : "Workbench";
  const contextBadge = workbench.activeWorkspace === "logs"
    ? "local command surface"
    : workbench.comparison.level
    ? `comparing ${workbench.comparison.ids.length} ${workbench.comparison.level}${workbench.comparison.ids.length === 1 ? "" : "s"}`
    : workbench.focusedEntity
    ? `${workbench.focusedEntity.kind} focus`
    : "review ready";

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <div className="topbar-left">
          <strong>LEP Operator App</strong>
          <span>{workbench.methods.length} methods</span>
          <span>{workbench.totalSamples} samples</span>
        </div>
        <div className="topbar-tabs">
          <button
            type="button"
            className={workbench.activeWorkspace === "workbench" ? "is-active" : ""}
            onClick={() => workbench.setActiveWorkspace("workbench")}
          >
            Workbench
          </button>
          <button
            type="button"
            className={workbench.activeWorkspace === "logs" ? "is-active" : ""}
            onClick={() => workbench.setActiveWorkspace("logs")}
          >
            Logs / Ops
          </button>
        </div>
        {isLoading && (
          <div className="app-loading">
            <span className="app-loading-label">
              {loadingLabel ? `Loading ${loadingLabel}…` : (workbench.loadingMessage || "Loading…")}
            </span>
            <div className="app-loading-track">
              <div className="app-loading-fill" style={{ width: `${loadingPct}%` }} />
            </div>
            {isLoadingMethods && totalMethods > 0 && (
              <span className="app-loading-count">{loadedMethods}/{totalMethods}</span>
            )}
            {isLoadingAssets && !isLoadingMethods && (
              <span className="app-loading-count">{enrichLoaded}/{enrichTotal}</span>
            )}
          </div>
        )}

        <div className="topbar-right">
          <span>{workspaceBadge}</span>
          <span>{contextBadge}</span>
        </div>
      </header>

      {workbench.activeWorkspace === "logs" ? (
        <LogsWorkspace />
      ) : (
        <div className="app-main">
          <TreePane
            searchQuery={workbench.searchQuery}
            onSearchQueryChange={workbench.setSearchQuery}
            loadingMessage={workbench.loadingMessage}
            error={workbench.error}
            visibleMethods={workbench.visibleMethods}
            loadingMethodIds={workbench.loadingMethodIds}
            expandedMethodIds={workbench.expandedMethodIds}
            expandedEvalIds={workbench.expandedEvalIds}
            expandedStrengthIds={workbench.expandedStrengthIds}
            focusedMethodId={workbench.focusedMethodId}
            focusedEvalId={workbench.focusedEvalId}
            focusedStrengthId={workbench.focusedStrengthId}
            focusedSampleId={workbench.focusedSampleId}
            comparisonIds={workbench.comparison.ids}
            comparisonLevel={workbench.comparison.level}
            onToggleFocusedInComparison={workbench.toggleFocusedInComparison}
            onClearComparison={workbench.clearComparison}
            hasFocusedEntity={Boolean(workbench.focusedEntity)}
            focusedInComparison={Boolean(
              workbench.focusedComparisonId && workbench.comparison.ids.includes(workbench.focusedComparisonId),
            )}
            onToggleDisclosure={workbench.toggleDisclosure}
            onFocusMethod={workbench.focusMethod}
            onFocusEval={workbench.focusEval}
            onFocusStrength={workbench.focusStrength}
            onFocusSample={workbench.focusSample}
            onToggleComparison={workbench.toggleComparison}
            onSelectAllEvalsForMethod={workbench.selectAllEvalsForMethod}
            onSelectAllStrengthsForEval={workbench.selectAllStrengthsForEval}
            onSelectAllSamplesForEval={workbench.selectAllSamplesForEval}
            onSelectAllSamplesForStrength={workbench.selectAllSamplesForStrength}
            onSelectSameSeedPeers={workbench.selectSameSeedPeers}
            onSelectSameStrengthPeers={workbench.selectSameStrengthPeers}
          />

          <main className="center-pane">
            <div className="center-topbar">
              <div className="center-topbar-row">
                <div className="topbar-title">
                  <div className="scope-line">{workbench.scopeLine}</div>
                  <strong>{workbench.centerTitle}</strong>
                </div>
                {workbench.inspectionMetricOptions.length > 0 ? (
                  <div className="topbar-metric-slot">
                    <SelectedMetricPicker
                      options={workbench.inspectionMetricOptions}
                      selectedMetric={workbench.selectedMetric}
                      onSelectMetric={workbench.setSelectedMetric}
                    />
                  </div>
                ) : null}
              </div>
              {workbench.promptHint ? <div className="topbar-prompt">Prompt: {workbench.promptHint}</div> : null}
            </div>
            <div className="workspace-canvas">
              <CompareTray
                level={workbench.comparison.level}
                chips={workbench.compareChips}
                onClear={workbench.clearComparison}
                onFocusChip={workbench.focusFromChip}
                onRemoveChip={workbench.removeFromComparison}
              />
              {workbench.error ? (
                <div className="canvas-placeholder canvas-placeholder--error">
                  <strong>Backend error</strong>
                  <p>{workbench.error}</p>
                </div>
              ) : isBootstrapping ? (
                <div className="canvas-placeholder canvas-placeholder--connecting">
                  <div className="canvas-placeholder-spinner" />
                  <span>Connecting to backend…</span>
                </div>
              ) : isEmptyDb ? (
                <div className="canvas-placeholder">
                  <span>No methods found. Run an ingestion pass first.</span>
                </div>
              ) : (
                <WorkbenchMain
                  comparisonLevel={workbench.comparison.level}
                  focusedEntity={workbench.focusedEntity}
                  selectedMetric={workbench.selectedMetric}
                  inspectionMetricOptions={workbench.inspectionMetricOptions}
                  heroRoster={workbench.heroRoster}
                  metricMetadata={workbench.metricMetadata}
                  methodSlices={workbench.methodSlices}
                  selectedMethods={workbench.selectedMethods}
                  selectedEvals={workbench.selectedEvals}
                  selectedStrengths={workbench.selectedStrengths}
                  selectedSamples={workbench.selectedSamples}
                  sampleRefs={workbench.sampleRefs}
                  focusedMethodId={workbench.focusedMethodId}
                  focusedEvalId={workbench.focusedEvalId}
                  focusedStrengthId={workbench.focusedStrengthId}
                  focusedSampleId={workbench.focusedSampleId}
                  onFocusMethod={workbench.focusMethod}
                  onFocusEval={workbench.focusEval}
                  onFocusStrength={workbench.focusStrength}
                  onFocusSample={workbench.focusSample}
                  sampleEnrichments={workbench.sampleEnrichments}
                  ensureSampleDetailLoaded={workbench.ensureSampleDetailLoaded}
                />
              )}
            </div>
          </main>

          <aside className="inspector">
            <MetricRail
              focusedEntity={workbench.focusedEntity}
              focusTitle={workbench.focusTitle}
              focusScopeLine={workbench.focusScopeLine}
              factRows={workbench.factRows}
              metricRows={workbench.metricRows}
              heroGroups={workbench.heroRailGroups}
              remainingMetricRows={workbench.remainingMetricRows}
              sampleDetails={workbench.sampleDetails}
              sampleDetailLoadingIds={workbench.sampleDetailLoadingIds}
              ensureSampleDetailLoaded={workbench.ensureSampleDetailLoaded}
            />
          </aside>
        </div>
      )}
    </div>
  );
}
