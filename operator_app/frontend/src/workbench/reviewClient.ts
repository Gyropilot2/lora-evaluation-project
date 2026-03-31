import type { MethodSlice, ReviewSummaryPayload, SampleSlice } from "./types";

export function reviewAssetUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return `/api/review/assets?path=${encodeURIComponent(path)}`;
}

export function reviewAssetPreviewUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  return `/api/review/assets?path=${encodeURIComponent(path)}&preview=1`;
}

export async function fetchReviewSummary(init?: RequestInit): Promise<ReviewSummaryPayload> {
  const response = await fetch("/api/review/summary", init);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as ReviewSummaryPayload;
}

export async function fetchMethodSlice(methodId: string, init?: RequestInit): Promise<MethodSlice> {
  const response = await fetch(`/api/review/methods/${encodeURIComponent(methodId)}`, init);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as MethodSlice;
}

export async function fetchSampleDetail(
  methodId: string,
  evalId: string,
  sampleId: string,
  init?: RequestInit,
): Promise<SampleSlice> {
  const response = await fetch(
    `/api/review/methods/${encodeURIComponent(methodId)}/evals/${encodeURIComponent(evalId)}/samples/${encodeURIComponent(sampleId)}`,
    init,
  );
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as SampleSlice;
}
