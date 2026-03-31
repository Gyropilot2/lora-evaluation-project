import "./WorkbenchMain.css";
import { SelectedMetricInspectionPanel } from "./SelectedMetricInspectionPanel";
import { WorkbenchImageStage } from "./WorkbenchImageStage";
import { WorkbenchOverviewSection } from "./WorkbenchOverviewSection";
import { WorkbenchTrendSection } from "./WorkbenchTrendSection";
import {
  type EvalRef,
  type FocusKind,
  type FocusedEntity,
  type HeroMetricDescriptor,
  type ImageComponentSlice,
  type MetricInspectionOption,
  type MetricMetadata,
  type MethodSlice,
  type MethodSummary,
  type SampleRef,
  type SampleSlice,
  type StrengthRef,
} from "../types";

type WorkbenchMainProps = {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  selectedMetric: string | null;
  inspectionMetricOptions: MetricInspectionOption[];
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
  methodSlices: Record<string, MethodSlice>;
  selectedMethods: MethodSummary[];
  selectedEvals: EvalRef[];
  selectedStrengths: StrengthRef[];
  selectedSamples: SampleRef[];
  sampleRefs: Map<string, SampleRef>;
  focusedMethodId: string | null;
  focusedEvalId: string | null;
  focusedStrengthId: string | null;
  focusedSampleId: string | null;
  onFocusMethod: (methodId: string) => void;
  onFocusEval: (methodId: string, evalId: string) => void;
  onFocusStrength: (methodId: string, evalId: string, value: number) => void;
  onFocusSample: (methodId: string, evalId: string, sampleId: string, strengthValue?: number | null) => void;
  sampleEnrichments: Record<string, ImageComponentSlice[]>;
  ensureSampleDetailLoaded: (methodId: string, evalId: string, sampleId: string) => Promise<SampleSlice | null>;
};

export function WorkbenchMain({
  comparisonLevel,
  focusedEntity,
  selectedMetric,
  inspectionMetricOptions,
  heroRoster,
  metricMetadata,
  methodSlices,
  selectedMethods,
  selectedEvals,
  selectedStrengths,
  selectedSamples,
  sampleRefs,
  focusedMethodId,
  focusedEvalId,
  focusedStrengthId,
  focusedSampleId,
  onFocusMethod,
  onFocusEval,
  onFocusStrength,
  onFocusSample,
  sampleEnrichments,
  ensureSampleDetailLoaded,
}: WorkbenchMainProps) {
  return (
    <>
      <WorkbenchImageStage
        comparisonLevel={comparisonLevel}
        focusedEntity={focusedEntity}
        selectedSamples={selectedSamples}
        sampleRefs={sampleRefs}
        focusedSampleId={focusedSampleId}
        sampleEnrichments={sampleEnrichments}
        ensureSampleDetailLoaded={ensureSampleDetailLoaded}
      />
      <WorkbenchOverviewSection
        comparisonLevel={comparisonLevel}
        focusedEntity={focusedEntity}
        heroRoster={heroRoster}
        methodSlices={methodSlices}
        selectedMethods={selectedMethods}
        selectedEvals={selectedEvals}
        selectedStrengths={selectedStrengths}
        selectedSamples={selectedSamples}
        focusedMethodId={focusedMethodId}
        focusedEvalId={focusedEvalId}
        focusedStrengthId={focusedStrengthId}
        focusedSampleId={focusedSampleId}
        onFocusMethod={onFocusMethod}
        onFocusEval={onFocusEval}
        onFocusStrength={onFocusStrength}
        onFocusSample={onFocusSample}
      />
      <SelectedMetricInspectionPanel
        comparisonLevel={comparisonLevel}
        focusedEntity={focusedEntity}
        focusedEvalId={focusedEvalId}
        selectedMetric={selectedMetric}
        inspectionMetricOptions={inspectionMetricOptions}
        heroRoster={heroRoster}
        metricMetadata={metricMetadata}
        selectedEvals={selectedEvals}
      />
      <WorkbenchTrendSection
        comparisonLevel={comparisonLevel}
        focusedEntity={focusedEntity}
        heroRoster={heroRoster}
        focusedEvalId={focusedEvalId}
        focusedStrengthId={focusedStrengthId}
        metricMetadata={metricMetadata}
        selectedEvals={selectedEvals}
        selectedStrengths={selectedStrengths}
        onFocusEval={onFocusEval}
      />
    </>
  );
}
