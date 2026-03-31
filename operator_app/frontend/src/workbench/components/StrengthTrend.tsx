import { numericMetric } from "../helpers";
import { type EvalStrength, type HeroMetricDescriptor, type MetricMetadata, type SampleSlice } from "../types";

export type TrendPoint = {
  label: string;
  value: number;
  sampleCount: number;
  badness: number | null;
  unreliability: number | null;
  unreliabilityMax: number | null;
  droppedFraction: number | null;
};

export type TrendScale = {
  min: number;
  max: number;
};

export type TrendCeilings = {
  badness: number;
  unreliability: number | null;
  dropped: number;
};

export type StrengthResponseItem = {
  id: string;
  label: string;
  strengths: EvalStrength[];
  isFocused?: boolean;
  onFocus?: (() => void) | null;
};

function metricBadness(
  value: number | null,
  metadata: MetricMetadata | undefined,
): number | null {
  if (value === null) return null;
  const polarity = metadata?.polarity;
  if (polarity === "higher_is_better") {
    if (metadata?.value_max === null || metadata?.value_max === undefined) return null;
    return Math.max(0, metadata.value_max - value);
  }
  if (polarity === "lower_is_better") {
    if (metadata?.value_min !== null && metadata?.value_min !== undefined) {
      return Math.max(0, value - metadata.value_min);
    }
    return value;
  }
  return value;
}

function metricUnreliability(value: number | null, metadata: MetricMetadata | undefined): number | null {
  if (value === null) return null;
  if (metadata?.value_max !== null && metadata?.value_max !== undefined) {
    return Math.max(0, metadata.value_max - value);
  }
  if (metadata?.polarity === "lower_is_better" && metadata?.value_min !== null && metadata?.value_min !== undefined) {
    return Math.max(0, value - metadata.value_min);
  }
  return null;
}

function compactStrengthLabel(label: string, value: number) {
  const normalized = label.trim();
  if (/^strength\b/i.test(normalized)) {
    return normalized.replace(/^strength\b/i, "str");
  }
  if (normalized.length > 0) {
    return normalized;
  }
  return `str ${value.toFixed(1)}`;
}

export function strengthSeries(
  strengths: EvalStrength[],
  metric: HeroMetricDescriptor,
  metricMetadata: Record<string, MetricMetadata>,
): TrendPoint[] {
  return strengths
    .slice()
    .sort((left, right) => left.value - right.value)
    .map((strength) => {
      const aggregate = strength.hero_metrics[metric.key];
      const metadata = aggregate ? metricMetadata[aggregate.metric_key] : undefined;
      const reliabilityMetadata =
        metadata?.reliability_metric_key ? metricMetadata[metadata.reliability_metric_key] : undefined;
      const score = aggregate?.score ?? null;
      const reliability = aggregate?.reliability ?? null;
      const unreliabilityMax = reliabilityMetadata?.value_max ?? null;
      return {
        label: compactStrengthLabel(strength.label, strength.value),
        value: strength.value,
        sampleCount: strength.samples.length,
        badness: metricBadness(score, metadata),
        unreliability: metricUnreliability(reliability, reliabilityMetadata),
        unreliabilityMax,
        droppedFraction: aggregate?.dropped_fraction ?? null,
      };
    });
}

function polylineForValues(
  points: Array<{ xValue: number; value: number }>,
  width: number,
  height: number,
  min: number,
  range: number,
  minX: number,
  xRange: number,
) {
  return points
    .map((point) => {
      const x = trendPointX(point.xValue, minX, xRange, width);
      const y = trendPointY(point.value, min, range, height);
      return `${x},${Number(y.toFixed(2))}`;
    })
    .join(" ");
}

export function trendPointX(value: number, minX: number, xRange: number, width: number) {
  if (xRange === 0) return width / 2;
  return ((value - minX) / xRange) * width;
}

export function trendPointY(value: number, min: number, range: number, height: number) {
  const normalizedY = (value - min) / range;
  return value >= min + range ? 0 : height - normalizedY * height;
}

function niceGridStep(range: number, targetLines = 4) {
  if (range <= 0) return 1;
  const rough = range / targetLines;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  if (normalized <= 1) return magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

export function trendGridTicks(min: number, max: number, targetLines = 4) {
  const step = niceGridStep(max - min, targetLines);
  const ticks: number[] = [];
  const epsilon = step * 0.001;
  for (let tick = Math.ceil(min / step) * step; tick < max - epsilon; tick += step) {
    if (tick <= min + epsilon) continue;
    ticks.push(Number(tick.toFixed(10)));
  }
  return ticks;
}

export function trendScaleForRows(
  items: StrengthResponseItem[],
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
) {
  const out: Record<string, { badness: TrendScale; unreliability: TrendScale }> = {};
  for (const metric of heroRoster) {
    let badnessMax = 0;
    let unreliabilityMax = 0;
    for (const item of items) {
      for (const point of strengthSeries(item.strengths, metric, metricMetadata)) {
        if (point.badness !== null) badnessMax = Math.max(badnessMax, point.badness);
        if (point.unreliability !== null) {
          unreliabilityMax = Math.max(unreliabilityMax, point.unreliabilityMax ?? point.unreliability);
        }
      }
    }
    out[metric.key] = {
      badness: { min: 0, max: badnessMax > 0 ? badnessMax : 1 },
      unreliability: { min: 0, max: unreliabilityMax > 0 ? unreliabilityMax : 1 },
    };
  }
  return out;
}

export function TrendPlot({
  points,
  scale,
  ceilings,
  width,
  height,
  yGridTicks,
}: {
  points: TrendPoint[];
  scale: TrendScale;
  ceilings: TrendCeilings;
  width: number;
  height: number;
  yGridTicks?: number[];
}) {
  const numeric = points
    .map((point, index) => ({ ...point, index }))
    .filter((point) => point.badness !== null || point.unreliability !== null || point.droppedFraction !== null);
  if (numeric.length === 0) {
    return <div className="trend-empty">No trend signal</div>;
  }

  const min = scale.min;
  const badnessRange = ceilings.badness - min || 1;
  const unreliabilityRange = ceilings.unreliability ?? 1;
  const droppedRange = ceilings.dropped || 1;
  const minX = Math.min(...points.map((point) => point.value));
  const maxX = Math.max(...points.map((point) => point.value));
  const xRange = maxX - minX;
  const visibleYGridTicks = yGridTicks ?? trendGridTicks(min, ceilings.badness);
  const badnessPoints = numeric
    .filter((point): point is TrendPoint & { index: number; badness: number } => point.badness !== null)
    .map((point) => ({ index: point.index, xValue: point.value, value: point.badness }));
  const unreliabilityPoints = numeric
    .filter((point): point is TrendPoint & { index: number; unreliability: number } => point.unreliability !== null)
    .map((point) => ({ index: point.index, xValue: point.value, value: point.unreliability }));
  const droppedPoints = numeric
    .filter((point): point is TrendPoint & { index: number; droppedFraction: number } => point.droppedFraction !== null)
    .map((point) => ({ index: point.index, xValue: point.value, value: point.droppedFraction }));
  const badnessPolyline = polylineForValues(badnessPoints, width, height, min, badnessRange, minX, xRange);
  const unreliabilityPolyline = polylineForValues(unreliabilityPoints, width, height, 0, unreliabilityRange, minX, xRange);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {points.map((point, index) => {
        const x = trendPointX(point.value, minX, xRange, width);
        return <line key={`grid-x-${index}`} className="trend-grid-line is-vertical" x1={x} y1={0} x2={x} y2={height} />;
      })}
      {visibleYGridTicks.map((tick) => {
        const y = trendPointY(tick, min, badnessRange, height);
        return <line key={`grid-y-${tick}`} className="trend-grid-line is-horizontal" x1={0} y1={y} x2={width} y2={y} />;
      })}
      {droppedPoints.map((point) => {
        if (point.value <= 0) return null;
        const x = trendPointX(point.xValue, minX, xRange, width);
        const y = trendPointY(point.value, 0, droppedRange, height);
        const barHeight = Math.max(3, height - y);
        return <rect key={`drop-${point.index}`} className="trend-dropped-bar" x={x - 2.5} y={height - barHeight} width="5" height={barHeight} rx="1.5" />;
      })}
      {unreliabilityPoints.length > 0 ? <polyline className="trend-unreliability-line" points={unreliabilityPolyline} /> : null}
      {badnessPoints.length > 0 ? <polyline className="trend-badness-line" points={badnessPolyline} /> : null}
      {badnessPoints.map((point) => {
        const x = trendPointX(point.xValue, minX, xRange, width);
        const y = trendPointY(point.value, min, badnessRange, height);
        return <circle key={`bad-${point.index}`} className="trend-badness-dot" cx={x} cy={y} r="2.8" />;
      })}
      {unreliabilityPoints.map((point) => {
        const x = trendPointX(point.xValue, minX, xRange, width);
        const y = trendPointY(point.value, 0, unreliabilityRange, height);
        return <circle key={`unrel-${point.index}`} className="trend-unreliability-dot" cx={x} cy={y} r="2.25" />;
      })}
    </svg>
  );
}
