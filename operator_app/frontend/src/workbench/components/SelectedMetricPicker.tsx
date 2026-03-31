import { useMemo, useRef } from "react";

import type { MetricInspectionOption } from "../types";

type SelectedMetricPickerProps = {
  options: MetricInspectionOption[];
  selectedMetric: string | null;
  onSelectMetric: (metricKey: string | null) => void;
};

export function SelectedMetricPicker({
  options,
  selectedMetric,
  onSelectMetric,
}: SelectedMetricPickerProps) {
  const detailsRef = useRef<HTMLDetailsElement | null>(null);
  const selectedOption = useMemo(
    () => options.find((option) => option.key === selectedMetric) ?? null,
    [options, selectedMetric],
  );

  const heroOptions = options.filter((option) => option.isHero);
  const otherOptions = options.filter((option) => !option.isHero);

  const choose = (metricKey: string | null) => {
    onSelectMetric(metricKey);
    if (detailsRef.current) detailsRef.current.open = false;
  };

  return (
    <details className="metric-picker" ref={detailsRef}>
      <summary className="metric-picker-trigger">
        <span className="metric-picker-copy">
          <span className="metric-picker-label">Inspect metric</span>
          <strong>{selectedOption?.label ?? "None"}</strong>
        </span>
        {selectedOption ? <span className="metric-picker-count">{selectedOption.count}</span> : null}
      </summary>
      <div className="metric-picker-menu">
        <button
          type="button"
          className={`metric-picker-option${selectedMetric === null ? " is-selected" : ""}`}
          onClick={() => choose(null)}
        >
          <span className="metric-picker-option-copy">
            <strong>None</strong>
            <span>Hide the inspection breakdown panel.</span>
          </span>
        </button>
        {heroOptions.length > 0 ? (
          <>
            <div className="metric-picker-group-label">hero metrics</div>
            {heroOptions.map((option) => (
              <MetricOptionButton
                key={option.key}
                option={option}
                selected={selectedMetric === option.key}
                onSelect={choose}
              />
            ))}
          </>
        ) : null}
        {otherOptions.length > 0 ? (
          <>
            <div className="metric-picker-group-label">other metrics</div>
            {otherOptions.map((option) => (
              <MetricOptionButton
                key={option.key}
                option={option}
                selected={selectedMetric === option.key}
                onSelect={choose}
              />
            ))}
          </>
        ) : null}
      </div>
    </details>
  );
}

function MetricOptionButton({
  option,
  selected,
  onSelect,
}: {
  option: MetricInspectionOption;
  selected: boolean;
  onSelect: (metricKey: string) => void;
}) {
  return (
    <button
      type="button"
      className={`metric-picker-option${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(option.key)}
    >
      <span className="metric-picker-option-copy">
        <strong>{option.label}</strong>
        <span>{option.keyLabel}</span>
      </span>
      <span className="metric-picker-option-count">{option.count}</span>
    </button>
  );
}
