import { useEffect } from "react";

import "./MetricRail.css";
import { formatCopySections } from "../clipboard";
import { previewSampleForEntity } from "../helpers";
import { reviewAssetUrl } from "../reviewClient";
import { CopyButton } from "./CopyButton";
import { HoverAssetPreview } from "./HoverAssetPreview";
import { MetricLabelWithHelp } from "./MetricLabelWithHelp";
import { RailKvRow } from "./RailKvRow";
import type { FocusedEntity, SampleSlice } from "../types";
import type { HeroRailGroup } from "../metricRailModel";

type FactRow = { label: string; value: string };
type MetricRow = { key: string; label: string; value: string; description?: string | null };

type MetricRailProps = {
  focusedEntity: FocusedEntity;
  focusTitle: string;
  focusScopeLine: string;
  factRows: FactRow[];
  metricRows: MetricRow[];
  heroGroups: HeroRailGroup[];
  remainingMetricRows: MetricRow[];
  sampleDetails: Record<string, SampleSlice>;
  sampleDetailLoadingIds: string[];
  ensureSampleDetailLoaded: (methodId: string, evalId: string, sampleId: string) => Promise<SampleSlice | null>;
};

function heroRowsForCopy(heroGroups: HeroRailGroup[]): FactRow[] {
  return heroGroups.flatMap((group) => [
    { label: `${group.label} = ${group.metricKey}`, value: "" },
    ...group.slots.flatMap((slot) => [
      { label: slot.label, value: slot.value },
      ...slot.feeds.map((feed) => ({ label: `  ${feed.label}`, value: feed.value })),
    ]),
  ]);
}

function isSampleCopyableFact(label: string): boolean {
  return /(?:id|hash)$/i.test(label.trim());
}

function imageRowsForCopy(imageComponents: SampleSlice["image_components"]): FactRow[] {
  return (imageComponents ?? []).map((component) => ({
    label: component.label,
    value: component.kind,
  }));
}

export function MetricRail({
  focusedEntity,
  focusTitle,
  focusScopeLine,
  factRows,
  metricRows,
  heroGroups,
  remainingMetricRows,
  sampleDetails,
  sampleDetailLoadingIds,
  ensureSampleDetailLoaded,
}: MetricRailProps) {
  const previewSample = previewSampleForEntity(focusedEntity);
  const focusedSampleUrl = reviewAssetUrl(previewSample?.image_path);
  const metricsLabel = "Hero metrics";

  const focusedSampleMethodId = focusedEntity?.kind === "sample" ? focusedEntity.method.id : null;
  const focusedSampleEvalId = focusedEntity?.kind === "sample" ? focusedEntity.eval.id : null;
  const focusedSampleId = focusedEntity?.kind === "sample" ? focusedEntity.sample.id : null;
  const sampleDetail = focusedSampleId ? sampleDetails[focusedSampleId] ?? null : null;
  const sampleDetailLoading = focusedSampleId ? sampleDetailLoadingIds.includes(focusedSampleId) : false;
  const imageComponents = sampleDetail?.image_components ?? [];
  const inspectorSections = focusedEntity?.kind === "sample"
    ? [
        { title: "Focused facts", rows: factRows },
        { title: "Hero metrics", rows: heroRowsForCopy(heroGroups) },
        { title: "Image components", rows: imageRowsForCopy(imageComponents) },
        { title: "Remaining payload items", rows: remainingMetricRows.map((row) => ({ label: row.label, value: row.value })) },
      ]
    : [
        { title: "Focused facts", rows: factRows },
        { title: "Hero metrics", rows: metricRows.map((row) => ({ label: row.label, value: row.value })) },
      ];
  const inspectorCopyText = focusedEntity
    ? formatCopySections({
        title: focusTitle,
        subtitle: focusScopeLine,
        sections: inspectorSections.filter((section) => section.rows.length > 0),
      })
    : "";

  useEffect(() => {
    if (!focusedSampleMethodId || !focusedSampleEvalId || !focusedSampleId) return;
    if (sampleDetail || sampleDetailLoading) return;
    void ensureSampleDetailLoaded(focusedSampleMethodId, focusedSampleEvalId, focusedSampleId);
  }, [
    ensureSampleDetailLoaded,
    focusedSampleEvalId,
    focusedSampleId,
    focusedSampleMethodId,
    sampleDetail,
    sampleDetailLoading,
  ]);

  return (
    <aside className="rail">
      <div className="rail-header-row">
        <div className="rail-header">Metric rail</div>
        <CopyButton
          text={inspectorCopyText}
          label="Copy"
          copiedLabel="Copied"
          title="Copy the focused inspector tables"
        />
      </div>
      {focusedEntity ? (
        <section className="rail-focus-card">
          <div className="rail-focus-header">
            {focusedSampleUrl ? (
              <img className="rail-thumb" src={focusedSampleUrl} alt={previewSample?.label ?? "focused preview"} />
            ) : (
              <div className="rail-thumb rail-thumb-placeholder">no preview</div>
            )}
            <div className="rail-focus-meta">
              <div className="rail-subhead">Focused item</div>
              <div className="rail-focus-title">{focusTitle}</div>
              <div className="summary-line">{focusScopeLine}</div>
            </div>
          </div>
        </section>
      ) : (
        <section className="rail-section">
          <div className="rail-subhead">Focused item</div>
          <div className="summary-line">Nothing focused yet.</div>
        </section>
      )}
      <section className="rail-section">
        <div className="rail-subhead">Focused facts</div>
        <div className="rail-kv-list">
          {factRows.map((row) => (
            <RailKvRow
              key={row.label}
              label={row.label}
              value={row.value}
              copyableValue={focusedEntity?.kind === "sample" && isSampleCopyableFact(row.label)}
            />
          ))}
        </div>
      </section>
      {focusedEntity?.kind === "sample" ? (
        <section className="rail-section">
          <div className="rail-subhead">{metricsLabel}</div>
          <div className="rail-hero-list">
            {heroGroups.map((group) => (
              <div key={group.id} className="rail-hero-group">
                <div className="rail-hero-heading">
                  <MetricLabelWithHelp label={group.label} description={group.description} className="rail-hero-name" />
                  <span className="rail-hero-separator">=</span>
                  <code className="rail-hero-key">{group.metricKey}</code>
                </div>
                <div className="rail-hero-slots">
                  {group.slots.map((slot) => (
                    <div key={slot.key} className="rail-hero-slot">
                      <RailKvRow
                        label={slot.label}
                        value={slot.value}
                        description={slot.description}
                        className="rail-kv-row--slot"
                      />
                      {slot.feeds.map((feed) => (
                        <RailKvRow
                          key={feed.key}
                          label={feed.label}
                          value={feed.value}
                          description={feed.description}
                          className="rail-kv-row--feed"
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <section className="rail-section">
          <div className="rail-subhead">{metricsLabel}</div>
          <div className="rail-kv-list">
            {metricRows.map((row) => (
              <RailKvRow key={row.key} label={row.label} value={row.value} description={row.description} />
            ))}
          </div>
        </section>
      )}
      {focusedEntity?.kind === "sample" ? (
        <section className="rail-section">
          <div className="rail-subhead">Image components</div>
          {sampleDetailLoading ? (
            <div className="summary-line">Loading image components...</div>
          ) : imageComponents.length > 0 ? (
            <ul className="rail-component-list">
              {imageComponents.map((component) => (
                <li key={component.key}>
                  <HoverAssetPreview
                    assetPath={component.image_path}
                    label={component.label}
                    primaryLabel="Sample"
                    secondaryAssetPath={component.secondary_image_path}
                    secondaryLabel={component.secondary_label}
                    className="rail-component-trigger"
                  >
                    {component.label}
                  </HoverAssetPreview>
                </li>
              ))}
            </ul>
          ) : (
            <div className="summary-line">No mask or aux previews available for this sample.</div>
          )}
        </section>
      ) : null}
      {focusedEntity?.kind === "sample" && remainingMetricRows.length > 0 ? (
        <section className="rail-section">
          <div className="rail-subhead">Remaining payload items</div>
          <div className="rail-kv-list">
            {remainingMetricRows.map((row) => (
              <RailKvRow key={row.key} label={row.label} value={row.value} description={row.description} />
            ))}
          </div>
        </section>
      ) : null}
    </aside>
  );
}
