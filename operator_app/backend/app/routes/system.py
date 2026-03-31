"""Launcher-probe routes for the Operator App backend."""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(tags=["system"])
_SLASH = chr(47)
_HEALTHZ_ROUTE = f"{_SLASH}healthz"


@router.get(_HEALTHZ_ROUTE)
def healthz() -> dict[str, object]:
    return {
        "ok": True,
        "purpose": "launcher_probe",
    }
