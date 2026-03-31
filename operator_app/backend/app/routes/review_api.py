"""App-facing review slice routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from operator_app.backend.app.services import review_assets, review_payloads


_SLASH = chr(47)
_PREFIX = f"{_SLASH}review"
_SUMMARY_ROUTE = f"{_SLASH}summary"
_METHODS_ROUTE = f"{_SLASH}methods"
_EVALS_ROUTE = f"{_SLASH}evals"
_STRENGTHS_ROUTE = f"{_SLASH}strengths"
_SAMPLES_ROUTE = f"{_SLASH}samples"
_ASSETS_ROUTE = f"{_SLASH}assets"

router = APIRouter(prefix=_PREFIX, tags=["review"])


@router.get(_SUMMARY_ROUTE)
def review_summary() -> dict[str, Any]:
    return review_payloads.review_summary()


@router.get(_METHODS_ROUTE)
def list_review_methods() -> list[dict[str, Any]]:
    return review_payloads.list_methods()


@router.get(f"{_METHODS_ROUTE}{_SLASH}{{method_hash}}")
def get_method_slice(method_hash: str) -> dict[str, Any]:
    method = review_payloads.get_method_slice(method_hash)
    if method is None:
        raise HTTPException(status_code=404, detail="method slice not found")
    return method


@router.get(f"{_METHODS_ROUTE}{_SLASH}{{method_hash}}{_EVALS_ROUTE}{_SLASH}{{eval_id}}")
def get_eval_slice(method_hash: str, eval_id: str) -> dict[str, Any]:
    ev = review_payloads.get_eval_slice(method_hash, eval_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="eval slice not found")
    return ev


@router.get(
    f"{_METHODS_ROUTE}{_SLASH}{{method_hash}}{_EVALS_ROUTE}{_SLASH}{{eval_id}}"
    f"{_STRENGTHS_ROUTE}{_SLASH}{{strength_value}}"
)
def get_strength_slice(method_hash: str, eval_id: str, strength_value: float) -> dict[str, Any]:
    strength = review_payloads.get_strength_slice(method_hash, eval_id, strength_value)
    if strength is None:
        raise HTTPException(status_code=404, detail="strength slice not found")
    return strength


@router.get(
    f"{_METHODS_ROUTE}{_SLASH}{{method_hash}}{_EVALS_ROUTE}{_SLASH}{{eval_id}}"
    f"{_SAMPLES_ROUTE}{_SLASH}{{sample_hash}}"
)
def get_sample_slice(method_hash: str, eval_id: str, sample_hash: str) -> dict[str, Any]:
    sample = review_payloads.get_sample_slice(method_hash, eval_id, sample_hash)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample slice not found")
    return sample


@router.get(_ASSETS_ROUTE, response_model=None)
def get_review_asset(
    path: str = Query(..., min_length=1),
    preview: bool = Query(False),
) -> Response:
    try:
        asset = review_assets.load_review_asset(path, preview)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    if asset.file_path is not None:
        return FileResponse(asset.file_path, media_type=asset.media_type)
    if asset.content is not None:
        return Response(content=asset.content, media_type=asset.media_type)
    raise HTTPException(status_code=500, detail="review asset resolved to no response body")
