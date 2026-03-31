import "./SelectedMetricInspectionPanel.css";

import { useEffect, useRef, useState } from "react";

import { formatMetric } from "../helpers";
import { buildSelectedMetricInspectionModel, type InspectionCardModel, type InspectionPoint } from "../metricInspectionModel";
import {
  trendGridTicks,
  trendPointX,
  trendPointY,
} from "./StrengthTrend";
import type {
  EvalRef,
  FocusKind,
  FocusedEntity,
  HeroMetricDescriptor,
  MetricInspectionOption,
  MetricMetadata,
} from "../types";

type SelectedMetricInspectionPanelProps = {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  focusedEvalId: string | null;
  selectedMetric: string | null;
  inspectionMetricOptions: MetricInspectionOption[];
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
  selectedEvals: EvalRef[];
};

export function SelectedMetricInspectionPanel({
  comparisonLevel,
  focusedEntity,
  focusedEvalId,
  selectedMetric,
  inspectionMetricOptions,
  heroRoster,
  metricMetadata,
  selectedEvals,
}: SelectedMetricInspectionPanelProps) {
  const model = buildSelectedMetricInspectionModel({
    comparisonLevel,
    focusedEntity,
    focusedEvalId,
    selectedMetric,
    inspectionMetricOptions,
    heroRoster,
    metricMetadata,
    selectedEvals,
  });

  if (!model) return null;

  if (model.kind === "unsupported") {
    return (
      <section className="panel is-provisional-panel">
        <div className="panel-heading">
          <h2>Metric inspection</h2>
          <span className="panel-tag">{model.selectedMetricLabel}</span>
        </div>
        <p className="compact-copy">
          <strong>{model.selectedMetricKey}</strong>
        </p>
        <p className="compact-copy provisional-copy">{model.reason}</p>
      </section>
    );
  }

  return (
    <section className="panel metric-inspection-panel">
      <div className="panel-heading">
        <h2>Metric inspection</h2>
        <span className="panel-tag">{model.poolTag}</span>
      </div>
      <p className="compact-copy metric-inspection-subtitle">
        <strong>{model.selectedMetricLabel}</strong>
        <span>= {model.selectedMetricKey}</span>
      </p>
      {model.selectedMetricDescription ? <p className="compact-copy">{model.selectedMetricDescription}</p> : null}
      <div className="summary-grid metric-inspection-summary">
        <div className="summary-stat">
          <strong>Method</strong>
          <span>{model.methodLabel}</span>
        </div>
        <div className="summary-stat">
          <strong>Evals in scope</strong>
          <span>{model.evalCount}</span>
        </div>
        <div className="summary-stat">
          <strong>Strengths in scope</strong>
          <span>{model.strengthCount}</span>
        </div>
      </div>
      <div className="trend-row metric-inspection-cards">
        <MetricInspectionCard card={model.selectedCard} isPrimary />
        {model.graphCards.map((card) => (
          <MetricInspectionCard key={card.key} card={card} />
        ))}
      </div>
      {model.hiddenGraphCount > 0 ? (
        <p className="compact-copy metric-inspection-note">
          {model.hiddenGraphCount} more registry inspection track{model.hiddenGraphCount === 1 ? "" : "s"} hidden in v1.
        </p>
      ) : null}
      {model.facts.length > 0 ? (
        <div className="metric-inspection-facts">
          {model.facts.map((fact) => (
            <article key={fact.key} className="metric-inspection-fact-card">
              <strong>{fact.label}</strong>
              {fact.description ? <p>{fact.description}</p> : null}
              <div className="metric-inspection-fact-rows">
                {fact.rows.map((row) => (
                  <div key={`${fact.key}-${row.label}`} className="metric-inspection-fact-row">
                    <span>{row.label}</span>
                    <span>{row.value}</span>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function MetricInspectionCard({
  card,
  isPrimary = false,
}: {
  card: InspectionCardModel;
  isPrimary?: boolean;
}) {
  return (
    <article className={`trend-metric-cell metric-inspection-card${isPrimary ? " is-primary" : ""}`}>
      <div className="trend-metric-title">
        <div className="metric-inspection-card-heading">
          <span>{card.label}</span>
          {card.description ? (
            <span
              className="metric-inspection-help"
              title={card.description}
              aria-label={`${card.label} details: ${card.description}`}
            >
              ?
            </span>
          ) : null}
        </div>
        <small>{card.metricKey}</small>
      </div>
      <div className="trend-metric-body metric-inspection-card-body">
        <div className="trend-point-strip is-vertical">
          {card.points.map((point) => (
            <div key={`${card.key}-${point.label}`} className="trend-point-pill metric-inspection-pill">
              <strong>{point.label}</strong>
              <span className="metric-inspection-pill-range">L {formatMetric(point.min)}</span>
              <span className="metric-inspection-pill-range">H {formatMetric(point.max)}</span>
              <span className="metric-inspection-pill-focus">
                {point.focusedValue !== null ? `F ${formatMetric(point.focusedValue)}` : `${point.valueCount} evals`}
              </span>
            </div>
          ))}
        </div>
        <InspectionChartFrame points={card.points} scale={card.scale} />
      </div>
    </article>
  );
}

function InspectionChartFrame({
  points,
  scale,
}: {
  points: InspectionPoint[];
  scale: { min: number; max: number };
}) {
  const minX = Math.min(...points.map((point) => point.strengthValue));
  const maxX = Math.max(...points.map((point) => point.strengthValue));
  const xRange = maxX - minX;
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotSize, setPlotSize] = useState({ width: 96, height: 108 });

  useEffect(() => {
    const element = plotRef.current;
    if (!element) return;

    const updateSize = () => {
      const rect = element.getBoundingClientRect();
      const width = Math.max(48, Math.round(rect.width));
      const height = Math.max(72, Math.round(rect.height));
      setPlotSize((current) => (current.width === width && current.height === height ? current : { width, height }));
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const yGridTicks = trendGridTicks(scale.min, scale.max);

  return (
    <div className="trend-sparkline">
      <div className="trend-chart-bundle metric-inspection-chart">
        <div className="trend-plot-column" ref={plotRef}>
          <InspectionRangePlot
            points={points}
            scale={scale}
            width={plotSize.width}
            height={plotSize.height}
            yGridTicks={yGridTicks}
          />
        </div>
        <div className="trend-labels">
          {points.map((point, index) => {
            const x = trendPointX(point.strengthValue, minX, xRange, plotSize.width);
            const transform =
              xRange === 0
                ? "translateX(-50%)"
                : index === 0
                  ? "translateX(0)"
                  : index === points.length - 1
                    ? "translateX(-100%)"
                    : "translateX(-50%)";
            return (
              <span key={`${point.label}-${index}`} className="trend-label" style={{ left: `${x}px`, transform }}>
                {point.strengthValue.toFixed(1)}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function InspectionRangePlot({
  points,
  scale,
  width,
  height,
  yGridTicks,
}: {
  points: InspectionPoint[];
  scale: { min: number; max: number };
  width: number;
  height: number;
  yGridTicks: number[];
}) {
  const minX = Math.min(...points.map((point) => point.strengthValue));
  const maxX = Math.max(...points.map((point) => point.strengthValue));
  const xRange = maxX - minX;
  const yRange = scale.max - scale.min || 1;
  const envelope = inspectionEnvelope(points, scale.min, yRange, minX, xRange, width, height);
  const focusPoints = points
    .filter((point): point is InspectionPoint & { focusedValue: number } => point.focusedValue !== null)
    .map((point) => ({
      x: trendPointX(point.strengthValue, minX, xRange, width),
      y: trendPointY(point.focusedValue, scale.min, yRange, height),
    }));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {points.map((point, index) => {
        const x = trendPointX(point.strengthValue, minX, xRange, width);
        return <line key={`grid-x-${index}`} className="trend-grid-line is-vertical" x1={x} y1={0} x2={x} y2={height} />;
      })}
      {yGridTicks.map((tick) => {
        const y = trendPointY(tick, scale.min, yRange, height);
        return <line key={`grid-y-${tick}`} className="trend-grid-line is-horizontal" x1={0} y1={y} x2={width} y2={y} />;
      })}
      {envelope.kind === "polygon" ? (
        <>
          <polygon className="metric-inspection-envelope" points={envelope.polygonPoints} />
          <polyline className="metric-inspection-envelope-edge" points={envelope.upperLinePoints} />
          <polyline className="metric-inspection-envelope-edge is-lower" points={envelope.lowerLinePoints} />
        </>
      ) : (
        <rect
          className="metric-inspection-envelope"
          x={envelope.x}
          y={envelope.y}
          width={envelope.width}
          height={envelope.height}
          rx="5"
        />
      )}
      {focusPoints.length > 1 ? (
        <polyline
          className="metric-inspection-focus-line"
          points={focusPoints.map((point) => `${point.x},${Number(point.y.toFixed(2))}`).join(" ")}
        />
      ) : null}
      {focusPoints.map((point, index) => (
        <circle key={`focus-${index}`} className="metric-inspection-focus-dot" cx={point.x} cy={point.y} r="3" />
      ))}
    </svg>
  );
}

function inspectionEnvelope(
  points: InspectionPoint[],
  scaleMin: number,
  yRange: number,
  minX: number,
  xRange: number,
  width: number,
  height: number,
):
  | { kind: "polygon"; polygonPoints: string; upperLinePoints: string; lowerLinePoints: string }
  | { kind: "rect"; x: number; y: number; width: number; height: number } {
  const upper = points.map((point) => ({
    x: trendPointX(point.strengthValue, minX, xRange, width),
    y: trendPointY(point.max, scaleMin, yRange, height),
  }));
  const lower = points.map((point) => ({
    x: trendPointX(point.strengthValue, minX, xRange, width),
    y: trendPointY(point.min, scaleMin, yRange, height),
  }));

  if (upper.length <= 1 || xRange === 0) {
    const x = upper[0]?.x ?? width / 2;
    const yTop = upper[0]?.y ?? height / 2;
    const yBottom = lower[0]?.y ?? yTop;
    return {
      kind: "rect",
      x: x - 9,
      y: Math.min(yTop, yBottom),
      width: 18,
      height: Math.max(5, Math.abs(yBottom - yTop)),
    };
  }

  return {
    kind: "polygon",
    polygonPoints: [
      ...upper.map((point) => `${point.x},${Number(point.y.toFixed(2))}`),
      ...lower
        .slice()
        .reverse()
        .map((point) => `${point.x},${Number(point.y.toFixed(2))}`),
    ].join(" "),
    upperLinePoints: upper.map((point) => `${point.x},${Number(point.y.toFixed(2))}`).join(" "),
    lowerLinePoints: lower.map((point) => `${point.x},${Number(point.y.toFixed(2))}`).join(" "),
  };
}
