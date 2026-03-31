import { useState, useEffect } from "react";
import { reviewAssetPreviewUrl, reviewAssetUrl } from "../reviewClient";
import type { FocusKind, FocusedEntity, ImageComponentSlice, SampleRef, SampleSlice } from "../types";

type WorkbenchImageStageProps = {
  comparisonLevel: FocusKind | null;
  focusedEntity: FocusedEntity;
  selectedSamples: SampleRef[];
  sampleRefs: Map<string, SampleRef>;
  focusedSampleId: string | null;
  sampleEnrichments: Record<string, ImageComponentSlice[]>;
  ensureSampleDetailLoaded: (methodId: string, evalId: string, sampleId: string) => Promise<SampleSlice | null>;
};

function samplePairForFocus(entity: Extract<FocusedEntity, { kind: "sample" }>) {
  const baselineEval = entity.method.evals.find((evalItem) => evalItem.is_baseline);
  const baselineSample =
    baselineEval?.samples?.find((sample) => sample.seed === entity.sample.seed) ?? baselineEval?.samples?.[0] ?? null;
  if (entity.eval.is_baseline) {
    return baselineSample ? [baselineSample] : [entity.sample];
  }
  return [baselineSample, entity.sample].filter((sample): sample is SampleSlice => Boolean(sample));
}

function ImageCards({
  focusedSampleId,
  samples,
}: {
  focusedSampleId: string | null;
  samples: SampleRef[];
}) {
  if (samples.length === 0) {
    return <div className="image-card-placeholder">No image scope yet.</div>;
  }

  return (
    <div className="image-grid">
      {samples.map((item) => {
        const url = reviewAssetUrl(item.sample.image_path);
        return (
          <article key={item.sample.id} className={`image-card${item.sample.id === focusedSampleId ? " is-focused" : ""}`}>
            <div className="image-card-title">{item.eval.label}</div>
            <div className="image-card-frame">
              {url ? <img src={url} alt={item.sample.label} /> : <div className="image-card-placeholder">No image</div>}
            </div>
            <div className="image-card-meta-row">
              <div className="image-card-label">{item.sample.label}</div>
              <div className="summary-line">seed {item.sample.seed ?? "-"} - strength {item.sample.strength ?? "-"}</div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function FocusedSamplePane({
  focusedEntity,
  focusedSampleId,
  sampleRefs,
  sampleEnrichments,
  ensureSampleDetailLoaded,
}: {
  focusedEntity: Extract<FocusedEntity, { kind: "sample" }>;
  focusedSampleId: string | null;
  sampleRefs: Map<string, SampleRef>;
  sampleEnrichments: Record<string, ImageComponentSlice[]>;
  ensureSampleDetailLoaded: (methodId: string, evalId: string, sampleId: string) => Promise<SampleSlice | null>;
}) {
  const [enrichedComponents, setEnrichedComponents] = useState<ImageComponentSlice[] | null>(null);
  const [isLoadingComponents, setIsLoadingComponents] = useState(false);
  const [activeOverlayKey, setActiveOverlayKey] = useState<string | null>(null);

  const sampleId = focusedEntity.sample.id;
  const methodId = focusedEntity.method.id;
  const evalId = focusedEntity.eval.id;

  useEffect(() => {
    // Cache hit — instant, no fetch needed
    const cached = sampleEnrichments[sampleId];
    if (cached !== undefined) {
      setEnrichedComponents(cached.length > 0 ? cached : null);
      setIsLoadingComponents(false);
      setActiveOverlayKey(null);
      return;
    }
    // Cache miss — show spinner and fetch locally
    setEnrichedComponents(null);
    setIsLoadingComponents(true);
    setActiveOverlayKey(null);
    let cancelled = false;
    void ensureSampleDetailLoaded(methodId, evalId, sampleId)
      .then((data) => {
        if (!cancelled) {
          if (!data) {
            setIsLoadingComponents(false);
            return;
          }
          const components = data.image_components ?? [];
          setEnrichedComponents(components.length > 0 ? components : null);
          setIsLoadingComponents(false);
        }
      })
      .catch(() => {
        if (!cancelled) setIsLoadingComponents(false);
      });
    return () => { cancelled = true; };
  }, [ensureSampleDetailLoaded, evalId, methodId, sampleEnrichments, sampleId]); // intentional: cache hit before shared loader

  const pair = samplePairForFocus(focusedEntity)
    .map((sample) => sampleRefs.get(sample.id))
    .filter((item): item is SampleRef => Boolean(item));

  const activeOverlay = activeOverlayKey
    ? enrichedComponents?.find((c) => c.key === activeOverlayKey) ?? null
    : null;

  return (
    <section className="panel image-stage">
      <div className="panel-heading">
        <h2>Pair stage</h2>
        <span className="panel-tag">focused sample view</span>
      </div>
      {isLoadingComponents && (
        <div className="overlay-picker">
          <span className="overlay-loading">Loading overlays…</span>
        </div>
      )}
      {!isLoadingComponents && enrichedComponents && enrichedComponents.length > 0 && (
        <div className="overlay-picker">
          <button
            className={`overlay-btn${activeOverlayKey === null ? " active" : ""}`}
            onClick={() => setActiveOverlayKey(null)}
          >
            Main
          </button>
          {enrichedComponents.map((comp) => (
            <button
              key={comp.key}
              className={`overlay-btn${activeOverlayKey === comp.key ? " active" : ""}`}
              onClick={() => setActiveOverlayKey(comp.key)}
            >
              {comp.label}
            </button>
          ))}
        </div>
      )}
      <div className="image-grid">
        {pair.map((item) => {
          let imgSrc: string | null;
          if (activeOverlay) {
            const isFocused = item.sample.id === focusedEntity.sample.id;
            imgSrc = reviewAssetPreviewUrl(
              isFocused ? activeOverlay.image_path : (activeOverlay.secondary_image_path ?? null),
            );
          } else {
            imgSrc = reviewAssetUrl(item.sample.image_path);
          }
          return (
            <article
              key={item.sample.id}
              className={`image-card${item.sample.id === focusedSampleId ? " is-focused" : ""}`}
            >
              <div className="image-card-title">{item.eval.label}</div>
              <div className="image-card-frame">
                {imgSrc ? (
                  <img src={imgSrc} alt={item.sample.label} />
                ) : (
                  <div className="image-card-placeholder">No image</div>
                )}
              </div>
              <div className="image-card-meta-row">
                <div className="image-card-label">{item.sample.label}</div>
                <div className="summary-line">
                  seed {item.sample.seed ?? "-"} - strength {item.sample.strength ?? "-"}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function WorkbenchImageStage({
  comparisonLevel,
  focusedEntity,
  selectedSamples,
  sampleRefs,
  focusedSampleId,
  sampleEnrichments,
  ensureSampleDetailLoaded,
}: WorkbenchImageStageProps) {
  if (comparisonLevel === "sample" && selectedSamples.length > 0) {
    return (
      <section className="panel image-stage">
        <div className="panel-heading">
          <h2>Survey stage</h2>
          <span className="panel-tag">center owns images</span>
        </div>
        <ImageCards focusedSampleId={focusedSampleId} samples={selectedSamples} />
      </section>
    );
  }

  if (focusedEntity?.kind === "sample") {
    return (
      <FocusedSamplePane
        focusedEntity={focusedEntity}
        focusedSampleId={focusedSampleId}
        sampleRefs={sampleRefs}
        sampleEnrichments={sampleEnrichments}
        ensureSampleDetailLoaded={ensureSampleDetailLoaded}
      />
    );
  }

  if (focusedEntity?.kind === "strength") {
    const samples = focusedEntity.strength.samples
      .map((sample) => sampleRefs.get(sample.id))
      .filter((item): item is SampleRef => Boolean(item));
    return (
      <section className="panel image-stage">
        <div className="panel-heading">
          <h2>Strength survey</h2>
          <span className="panel-tag">{focusedEntity.strength.samples.length} samples</span>
        </div>
        <ImageCards focusedSampleId={focusedSampleId} samples={samples} />
      </section>
    );
  }

  return null;
}
