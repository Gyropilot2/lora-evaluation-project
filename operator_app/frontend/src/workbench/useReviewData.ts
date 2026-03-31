import { useCallback, useEffect, useRef, useState } from "react";

import { samplesForEval } from "./helpers";
import { fetchMethodSlice, fetchReviewSummary, fetchSampleDetail, reviewAssetUrl } from "./reviewClient";
import type {
  HeroMetricDescriptor,
  ImageComponentSlice,
  MethodSlice,
  MethodSummary,
  MetricMetadata,
  SampleSlice,
} from "./types";

type QueuePriority = "front" | "back";

// Max concurrent method-slice fetches. 2 halves tree load time vs serial with no
// meaningful extra cost (enrichment is capped separately, images use new Image()).
const MAX_SLICE_CONCURRENT = 2;
// Max concurrent enrichment fetches — leaves browser connection pool room for images.
// Browsers allow 6 connections per origin (HTTP/1.1); keeping enrichment ≤ 3 means
// images always have at least 3 connections available.
const MAX_ENRICH_CONCURRENT = 3;

export function useReviewData() {
  const [methods, setMethods] = useState<MethodSummary[]>([]);
  const [heroRoster, setHeroRoster] = useState<HeroMetricDescriptor[]>([]);
  const [metricMetadata, setMetricMetadata] = useState<Record<string, MetricMetadata>>({});
  const [methodSlices, setMethodSlices] = useState<Record<string, MethodSlice>>({});
  const [loadingMessage, setLoadingMessage] = useState("Loading methods...");
  const [loadingMethodIds, setLoadingMethodIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [initialMethodId, setInitialMethodId] = useState<string | null>(null);
  const [hasLoadedMethodList, setHasLoadedMethodList] = useState(false);
  const [queuedMethodIds, setQueuedMethodIds] = useState<string[]>([]);
  const [activeQueuedCount, setActiveQueuedCount] = useState(0);
  const loadingMethodIdsRef = useRef<Set<string>>(new Set());
  const loadedMethodIdsRef = useRef<Set<string>>(new Set());
  const prefetchedAssetUrlsRef = useRef<Set<string>>(new Set());
  const sampleDetailRequestsRef = useRef<Record<string, Promise<SampleSlice | null>>>({});
  const sampleDetailsRef = useRef<Record<string, SampleSlice>>({});

  const [sampleDetails, setSampleDetails] = useState<Record<string, SampleSlice>>({});
  const [sampleDetailLoadingIds, setSampleDetailLoadingIds] = useState<string[]>([]);

  // Enrichment cache: sample ID → component list
  const [sampleEnrichments, setSampleEnrichments] = useState<Record<string, ImageComponentSlice[]>>({});
  const enrichmentTotalRef  = useRef(0);
  const enrichmentLoadedRef = useRef(0);
  const [enrichmentProgress, setEnrichmentProgress] = useState({ total: 0, loaded: 0 });
  const enrichingRef       = useRef<Set<string>>(new Set()); // in-flight
  const enrichedRef        = useRef<Set<string>>(new Set()); // completed
  const enrichTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enrichConcurrentRef = useRef(0);                    // active fetch count
  const enrichQueueRef     = useRef<Array<() => void>>([]);  // pending task queue

  // Throttled progress state updater — avoids one setState per sample
  function bumpEnrichProgress() {
    enrichmentLoadedRef.current++;
    if (!enrichTimerRef.current) {
      enrichTimerRef.current = setTimeout(() => {
        setEnrichmentProgress({
          total: enrichmentTotalRef.current,
          loaded: enrichmentLoadedRef.current,
        });
        enrichTimerRef.current = null;
      }, 80);
    }
  }

  const prefetchMethodImages = useCallback((slice: MethodSlice) => {
    if (typeof Image === "undefined") return;

    const urls = slice.evals
      .flatMap((evalItem) => samplesForEval(evalItem))
      .map((sample) => reviewAssetUrl(sample.image_path))
      .filter((url): url is string => Boolean(url))
      .filter((url, index, list) => list.indexOf(url) === index);

    for (const url of urls) {
      if (prefetchedAssetUrlsRef.current.has(url)) continue;
      prefetchedAssetUrlsRef.current.add(url);
      const image = new Image();
      image.decoding = "async";
      image.src = url;
    }
  }, []);

  // After each method slice loads, background-fetch enrichment for all its samples.
  // Results are cached so FocusedSamplePane can skip the fetch entirely on a hit.
  // Rate-limited to MAX_ENRICH_CONCURRENT concurrent fetches so the browser
  // connection pool isn't saturated and image loads can proceed in parallel.
  const ensureSampleDetailLoaded = useCallback(
    async (methodId: string, evalId: string, sampleId: string): Promise<SampleSlice | null> => {
      if (!methodId || !evalId || !sampleId) return null;

      if (sampleDetailsRef.current[sampleId]) return sampleDetailsRef.current[sampleId];
      const inFlight = sampleDetailRequestsRef.current[sampleId];
      if (inFlight) return inFlight;

      setSampleDetailLoadingIds((current) => (current.includes(sampleId) ? current : [...current, sampleId]));

      const request = fetchSampleDetail(methodId, evalId, sampleId)
        .then((detail) => {
          setSampleDetails((current) => {
            if (current[sampleId]) return current;
            const next = { ...current, [sampleId]: detail };
            sampleDetailsRef.current = next;
            return next;
          });
          return detail;
        })
        .catch(() => null)
        .finally(() => {
          delete sampleDetailRequestsRef.current[sampleId];
          setSampleDetailLoadingIds((current) => current.filter((id) => id !== sampleId));
        });

      sampleDetailRequestsRef.current[sampleId] = request;
      return request;
    },
    [],
  );

  const prefetchSampleEnrichments = useCallback((slice: MethodSlice) => {
    if (typeof Image === "undefined") return; // SSR guard

    const pairs: { evalId: string; sampleId: string }[] = [];
    for (const evalItem of slice.evals) {
      for (const sample of samplesForEval(evalItem)) {
        if (!enrichingRef.current.has(sample.id) && !enrichedRef.current.has(sample.id)) {
          pairs.push({ evalId: evalItem.id, sampleId: sample.id });
        }
      }
    }
    if (pairs.length === 0) return;

    enrichmentTotalRef.current += pairs.length;
    setEnrichmentProgress((prev) => ({ ...prev, total: prev.total + pairs.length }));
    // Drain the queue: start the next task if a slot is available
    function drainEnrichQueue() {
      while (enrichConcurrentRef.current < MAX_ENRICH_CONCURRENT && enrichQueueRef.current.length > 0) {
        const task = enrichQueueRef.current.shift()!;
        enrichConcurrentRef.current++;
        task();
      }
    }

    for (const { evalId, sampleId } of pairs) {
      // Mark as queued immediately so duplicate-detection works across batches
      enrichingRef.current.add(sampleId);

      const task = () => {
        void ensureSampleDetailLoaded(slice.id, evalId, sampleId)
          .then((data) => {
            if (!data) {
              enrichedRef.current.add(sampleId);
              enrichingRef.current.delete(sampleId);
              bumpEnrichProgress();
              return;
            }
            enrichedRef.current.add(sampleId);
            enrichingRef.current.delete(sampleId);
            const components = data.image_components ?? [];
            setSampleEnrichments((current) =>
              current[sampleId] !== undefined ? current : { ...current, [sampleId]: components },
            );
            // Prefetch component images using the existing de-dup ref
            for (const comp of components) {
              for (const rawPath of [comp.image_path, comp.secondary_image_path]) {
                if (!rawPath) continue;
                const url = reviewAssetUrl(rawPath) ?? rawPath;
                if (prefetchedAssetUrlsRef.current.has(url)) continue;
                prefetchedAssetUrlsRef.current.add(url);
                const img = new Image();
                img.decoding = "async";
                img.src = url;
              }
            }
            bumpEnrichProgress();
          })
          .catch(() => {
            enrichedRef.current.add(sampleId);
            enrichingRef.current.delete(sampleId);
            bumpEnrichProgress();
          })
          .finally(() => {
            enrichConcurrentRef.current--;
            drainEnrichQueue();
          });
      };

      enrichQueueRef.current.push(task);
    }

    drainEnrichQueue();
  }, [ensureSampleDetailLoaded]); // refs + setters only — no reactive deps

  const loadMethodSlice = useCallback(async (methodId: string) => {
    if (!methodId || loadingMethodIdsRef.current.has(methodId) || loadedMethodIdsRef.current.has(methodId)) return;

    loadingMethodIdsRef.current.add(methodId);
    setLoadingMethodIds((current) => (current.includes(methodId) ? current : [...current, methodId]));
    try {
      const slice = await fetchMethodSlice(methodId);
      loadedMethodIdsRef.current.add(methodId);
      prefetchMethodImages(slice);
      prefetchSampleEnrichments(slice);
      setMethodSlices((current) => (current[methodId] ? current : { ...current, [methodId]: slice }));
    } catch (err) {
      setError(`Failed to load method slice: ${String(err)}`);
    } finally {
      loadingMethodIdsRef.current.delete(methodId);
      setLoadingMethodIds((current) => current.filter((item) => item !== methodId));
    }
  }, [prefetchMethodImages, prefetchSampleEnrichments]);

  const queueMethodLoads = useCallback((methodIds: string[], priority: QueuePriority = "back") => {
    const pendingIds = methodIds.filter(
      (methodId, index, list) =>
        Boolean(methodId) &&
        list.indexOf(methodId) === index &&
        !loadingMethodIdsRef.current.has(methodId) &&
        !loadedMethodIdsRef.current.has(methodId),
    );
    if (pendingIds.length === 0) return;

    setQueuedMethodIds((current) => {
      const remaining = current.filter((methodId) => !pendingIds.includes(methodId));
      return priority === "front" ? [...pendingIds, ...remaining] : [...remaining, ...pendingIds];
    });
  }, []);

  const ensureMethodLoaded = useCallback(
    (methodId: string) => {
      queueMethodLoads([methodId], "front");
    },
    [queueMethodLoads],
  );

  useEffect(() => {
    let cancelled = false;

    async function loadMethods() {
      try {
        setLoadingMessage("Loading methods...");
        const payload = await fetchReviewSummary();
        const list = payload.methods ?? [];
        if (cancelled) return;

        setMethods(list);
        setHeroRoster(payload.hero_roster ?? []);
        setMetricMetadata(payload.metric_metadata ?? {});
        setHasLoadedMethodList(true);

        if (list.length > 0) {
          setInitialMethodId((current) => current ?? list[0].id);
          setLoadingMessage("Loading tree...");
          queueMethodLoads([list[0].id], "front");
          queueMethodLoads(
            list.slice(1).map((method) => method.id),
            "back",
          );
        } else {
          setLoadingMessage("");
        }
      } catch (err) {
        if (cancelled) return;
        setError(`Failed to load operator app data: ${String(err)}`);
        setLoadingMessage("");
      }
    }

    void loadMethods();
    return () => {
      cancelled = true;
    };
  }, [queueMethodLoads]);

  useEffect(() => {
    if (!hasLoadedMethodList) return;
    if (activeQueuedCount >= MAX_SLICE_CONCURRENT) return;

    const slotsAvailable = MAX_SLICE_CONCURRENT - activeQueuedCount;
    const toStart: string[] = [];
    for (const methodId of queuedMethodIds) {
      if (toStart.length >= slotsAvailable) break;
      if (!loadingMethodIdsRef.current.has(methodId) && !loadedMethodIdsRef.current.has(methodId)) {
        toStart.push(methodId);
      }
    }

    if (toStart.length === 0) {
      if (queuedMethodIds.length > 0) {
        setQueuedMethodIds((current) =>
          current.filter(
            (methodId) =>
              !loadingMethodIdsRef.current.has(methodId) && !loadedMethodIdsRef.current.has(methodId),
          ),
        );
      }
      return;
    }

    setQueuedMethodIds((current) => current.filter((methodId) => !toStart.includes(methodId)));
    setActiveQueuedCount((prev) => prev + toStart.length);
    for (const methodId of toStart) {
      void loadMethodSlice(methodId).finally(() => {
        setActiveQueuedCount((prev) => prev - 1);
      });
    }
  }, [activeQueuedCount, hasLoadedMethodList, loadMethodSlice, queuedMethodIds]);

  useEffect(() => {
    if (!hasLoadedMethodList) return;
    if (methods.length === 0) {
      setLoadingMessage("");
      return;
    }
    if (Object.keys(methodSlices).length === 0 && (activeQueuedCount > 0 || queuedMethodIds.length > 0)) {
      setLoadingMessage("Loading tree...");
      return;
    }
    setLoadingMessage("");
  }, [activeQueuedCount, hasLoadedMethodList, methodSlices, methods.length, queuedMethodIds.length]);

  return {
    methods,
    heroRoster,
    metricMetadata,
    methodSlices,
    loadingMessage,
    loadingMethodIds,
    error,
    initialMethodId,
    ensureMethodLoaded,
    queueMethodLoads,
    sampleEnrichments,
    sampleDetails,
    sampleDetailLoadingIds,
    ensureSampleDetailLoaded,
    enrichmentProgress,
  };
}
