import { formatMetric } from "../helpers";
import { TrendChartFrame } from "./TrendChartFrame";
import { type TrendCeilings, type TrendPoint, type TrendScale } from "./StrengthTrend";

export function TrendMetricContent({
  itemId,
  metricKey,
  points,
  scale,
  ceilings,
}: {
  itemId: string;
  metricKey: string;
  points: TrendPoint[];
  scale: TrendScale;
  ceilings: TrendCeilings;
}) {
  return (
    <div className="trend-metric-body">
      <div className="trend-point-strip is-vertical">
        {points.map((point) => (
          <div key={`${itemId}-${metricKey}-${point.label}`} className="trend-point-pill">
            <strong>{point.label}</strong>
            <span className="trend-point-badness">B {formatMetric(point.badness)}</span>
            <span className="trend-point-unreliability">U {formatMetric(point.unreliability)}</span>
            <span className="trend-point-dropped">D {formatMetric(point.droppedFraction)}</span>
          </div>
        ))}
      </div>
      <TrendChartFrame points={points} scale={scale} ceilings={ceilings} />
    </div>
  );
}
