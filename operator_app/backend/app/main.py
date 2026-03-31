"""FastAPI entry point for the local Operator App backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from operator_app.backend.app.runtime_config import load_runtime_config
from operator_app.backend.app.routes import command_center_api
from operator_app.backend.app.routes import review_api
from operator_app.backend.app.routes import system


_SLASH = chr(47)
_API_PREFIX = f"{_SLASH}api"


def create_app() -> FastAPI:
    runtime_config = load_runtime_config()
    app = FastAPI(
        title="LEP Operator App",
        version="0.1.0",
        summary="Local operator lab for review, diagnostics, and controlled actions.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_config.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system.router, prefix=_API_PREFIX)
    app.include_router(command_center_api.router, prefix=_API_PREFIX)
    app.include_router(review_api.router, prefix=_API_PREFIX)
    return app


app = create_app()
