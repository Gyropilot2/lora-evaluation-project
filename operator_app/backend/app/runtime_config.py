"""Operator App runtime config loader shared by backend surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from core.paths import get_path

_SLASH = chr(47)


@dataclass(frozen=True)
class BackendRuntimeConfig:
    host: str
    port: int
    healthz_path: str


@dataclass(frozen=True)
class FrontendRuntimeConfig:
    host: str
    port: int


@dataclass(frozen=True)
class OperatorAppRuntimeConfig:
    backend: BackendRuntimeConfig
    frontend: FrontendRuntimeConfig
    cors_allowed_origins: tuple[str, ...]


def runtime_config_path() -> Path:
    return Path(get_path("project_root")) / "operator_app" / "runtime_config.json"


def load_runtime_config() -> OperatorAppRuntimeConfig:
    with runtime_config_path().open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    backend_raw = raw.get("backend", {})
    frontend_raw = raw.get("frontend", {})
    return OperatorAppRuntimeConfig(
        backend=BackendRuntimeConfig(
            host=str(backend_raw.get("host", "127.0.0.1")),
            port=int(backend_raw.get("port", 8765)),
            healthz_path=str(backend_raw.get("healthz_path", f"{_SLASH}api{_SLASH}healthz")),
        ),
        frontend=FrontendRuntimeConfig(
            host=str(frontend_raw.get("host", "127.0.0.1")),
            port=int(frontend_raw.get("port", 4173)),
        ),
        cors_allowed_origins=tuple(str(origin) for origin in raw.get("cors_allowed_origins", [])),
    )
