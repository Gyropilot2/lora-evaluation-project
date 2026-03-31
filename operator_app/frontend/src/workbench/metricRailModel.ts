import { formatMetric, formatMetricValue, formatPercent, metricLabel, sortMetricKeys } from "./helpers";
import type { FocusedEntity, HeroAggregate, HeroMetricDescriptor, MetricMap, MetricMetadata } from "./types";
import type { MetricRow } from "./viewModel";

export type HeroRailFeedRow = { key: string; label: string; value: string; description?: string | null };
export type HeroRailSlot = { key: string; label: string; value: string; feeds: HeroRailFeedRow[]; description?: string | null };
export type HeroRailGroup = {
  id: string;
  label: string;
  metricKey: string;
  description?: string | null;
  slots: HeroRailSlot[];
};

function heroAggregatesForFocusedEntity(entity: FocusedEntity): Record<string, HeroAggregate> {
  if (!entity) return {};
  if (entity.kind === "sample") return entity.sample.hero_metrics;
  if (entity.kind === "strength") return entity.strength.hero_metrics;
  if (entity.kind === "eval") return entity.eval.hero_metrics;
  return entity.method?.hero_metrics ?? {};
}

function uniqueMetricKeys(keys: Array<string | null | undefined>): string[] {
  return keys.filter((key, index, all): key is string => Boolean(key) && all.indexOf(key) === index);
}

function metricFeedRow(
  metrics: MetricMap,
  key: string,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): HeroRailFeedRow {
  return {
    key,
    label: metricLabel(key, metricMetadata, heroRoster),
    value: formatMetricValue(metrics[key]),
    description: metricMetadata[key]?.description ?? null,
  };
}

function detailSlot(
  slotKey: string,
  label: string,
  keys: string[] | undefined,
  metrics: MetricMap,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): HeroRailSlot | null {
  const feedKeys = uniqueMetricKeys(keys ?? []);
  if (feedKeys.length === 0) return null;
  return {
    key: slotKey,
    label,
    value: "",
    feeds: feedKeys.map((key) => metricFeedRow(metrics, key, metricMetadata, heroRoster)),
  };
}

export function heroRailGroupsForFocusedEntity(
  entity: FocusedEntity,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): HeroRailGroup[] {
  if (!entity || entity.kind !== "sample") return [];

  const metrics = entity.sample.metrics;
  const heroAggregates = heroAggregatesForFocusedEntity(entity);

  return heroRoster.flatMap((descriptor) => {
    const aggregate = heroAggregates[descriptor.key];
    if (!aggregate) return [];

    const metadata = metricMetadata[descriptor.metric_key];
    const slots: HeroRailSlot[] = [
      {
        key: `${aggregate.key}:score`,
        label: "Score",
        value: formatMetric(aggregate.score),
        feeds: [],
        description: metadata?.description ?? descriptor.description ?? null,
      },
    ];

    if (metadata?.reliability_metric_key) {
      const reliabilityMetadata = metricMetadata[metadata.reliability_metric_key];
      slots.push({
        key: `${aggregate.key}:reliability`,
        label: "Reliability",
        value: formatMetric(aggregate.reliability),
        feeds: [metricFeedRow(metrics, metadata.reliability_metric_key, metricMetadata, heroRoster)],
        description: reliabilityMetadata?.description ?? null,
      });
    }

    if (metadata?.dropped_metric_key) {
      const droppedMetadata = metricMetadata[metadata.dropped_metric_key];
      slots.push({
        key: `${aggregate.key}:dropped`,
        label: "Dropped",
        value: formatPercent(aggregate.dropped_fraction),
        feeds: [metricFeedRow(metrics, metadata.dropped_metric_key, metricMetadata, heroRoster)],
        description: droppedMetadata?.description ?? null,
      });
    }

    const selectionSlot = detailSlot(
      `${aggregate.key}:selection`,
      "Selection",
      metadata?.selection_metric_keys,
      metrics,
      metricMetadata,
      heroRoster,
    );
    if (selectionSlot) slots.push(selectionSlot);

    const componentSlot = detailSlot(
      `${aggregate.key}:components`,
      "Components",
      metadata?.component_metric_keys,
      metrics,
      metricMetadata,
      heroRoster,
    );
    if (componentSlot) slots.push(componentSlot);

    const peerSlot = detailSlot(
      `${aggregate.key}:peers`,
      "Peer checks",
      metadata?.peer_metric_keys,
      metrics,
      metricMetadata,
      heroRoster,
    );
    if (peerSlot) slots.push(peerSlot);

    return [
      {
        id: descriptor.key,
        label: descriptor.label,
        metricKey: descriptor.metric_key,
        description: descriptor.description ?? metadata?.description ?? null,
        slots,
      },
    ];
  });
}

export function remainingMetricRowsForFocusedEntity(
  entity: FocusedEntity,
  metricMetadata: Record<string, MetricMetadata>,
  heroRoster: HeroMetricDescriptor[],
): MetricRow[] {
  if (!entity || entity.kind !== "sample") return [];

  const consumedKeys = new Set(
    heroRailGroupsForFocusedEntity(entity, metricMetadata, heroRoster).flatMap((group) =>
      group.slots.flatMap((slot) => slot.feeds.map((feed) => feed.key)),
    ),
  );

  return sortMetricKeys(Object.keys(entity.sample.metrics), metricMetadata, heroRoster)
    .filter((key) => !consumedKeys.has(key))
    .map((key) => ({
      key,
      label: metricLabel(key, metricMetadata, heroRoster),
      value: formatMetricValue(entity.sample.metrics[key]),
      description: metricMetadata[key]?.description ?? null,
    }))
    .filter((row) => row.value !== "-");
}
