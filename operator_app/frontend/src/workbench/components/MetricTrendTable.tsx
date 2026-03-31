import "./MetricTrendTable.css";
import { type HeroMetricDescriptor, type MetricMetadata } from "../types";
import {
  strengthSeries,
  trendScaleForRows,
  type TrendCeilings,
  type StrengthResponseItem,
} from "./StrengthTrend";
import { TrendMetricContent } from "./TrendMetricContent";

function StrengthTrendCell({
  itemId,
  metric,
  title,
  strengths,
  metricMetadata,
  scale,
  ceilings,
}: {
  itemId: string;
  metric: HeroMetricDescriptor;
  title: string;
  strengths: StrengthResponseItem["strengths"];
  metricMetadata: Record<string, MetricMetadata>;
  scale: { min: number; max: number };
  ceilings: TrendCeilings;
}) {
  const points = strengthSeries(strengths, metric, metricMetadata);

  return (
    <div className="trend-metric-cell">
      <div className="trend-metric-title">{title}</div>
      <TrendMetricContent itemId={itemId} metricKey={metric.key} points={points} scale={scale} ceilings={ceilings} />
      <div className="trend-legend">
        <span className="trend-legend-badness">badness</span>
        <span className="trend-legend-unreliability">unreliability</span>
        <span className="trend-legend-dropped">dropped</span>
      </div>
    </div>
  );
}

function trendCeilingsForMetric(): TrendCeilings {
  return {
    badness: 1,
    unreliability: null,
    dropped: 1,
  };
}

function StrengthResponseItemCard({
  item,
  heroRoster,
  metricMetadata,
  scales,
}: {
  item: StrengthResponseItem;
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
  scales: Record<string, { badness: { min: number; max: number }; unreliability: { min: number; max: number } }>;
}) {
  const trendMetricColumns = heroRoster.map((metric) => ({
    id: metric.key,
    metric,
    title: metric.label,
  }));
  const isClickable = Boolean(item.onFocus);

  return (
    <article
      className={`trend-item${item.isFocused ? " is-focused" : ""}${isClickable ? " is-clickable" : ""}`}
      onClick={item.onFocus ?? undefined}
      onKeyDown={
        item.onFocus
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                item.onFocus?.();
              }
            }
          : undefined
      }
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
    >
      <div className="trend-item-label">
        <strong>{item.label}</strong>
        <span>{item.strengths.length} strengths</span>
      </div>
      <div className="trend-row">
        {trendMetricColumns.map(({ id, metric, title }) => (
          <StrengthTrendCell
            key={`${item.id}-${id}`}
            itemId={item.id}
            metric={metric}
            title={title}
            strengths={item.strengths}
            metricMetadata={metricMetadata}
            scale={scales[metric.key].badness}
            ceilings={{
              ...trendCeilingsForMetric(),
              badness: scales[metric.key].badness.max,
              unreliability: scales[metric.key].unreliability.max,
            }}
          />
        ))}
      </div>
    </article>
  );
}

export function MetricTrendTable({
  items,
  heroRoster,
  metricMetadata,
}: {
  items: StrengthResponseItem[];
  heroRoster: HeroMetricDescriptor[];
  metricMetadata: Record<string, MetricMetadata>;
}) {
  const trendMetricColumns = heroRoster.map((metric) => ({
    id: metric.key,
    metric,
    title: metric.label,
  }));
  const visibleItems = items.filter((item) =>
    trendMetricColumns.some(({ metric }) =>
      strengthSeries(item.strengths, metric, metricMetadata).some(
        (point) => point.badness !== null || point.unreliability !== null || point.droppedFraction !== null,
      ),
    ),
  );

  if (visibleItems.length === 0) return null;

  const scales = trendScaleForRows(items, metricMetadata, heroRoster);

  return (
    <div className="trend-matrix">
      <div className="trend-matrix-body">
        {visibleItems.map((item) => (
          <StrengthResponseItemCard key={item.id} item={item} heroRoster={heroRoster} metricMetadata={metricMetadata} scales={scales} />
        ))}
      </div>
    </div>
  );
}
