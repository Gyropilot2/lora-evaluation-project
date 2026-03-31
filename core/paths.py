"""
core/paths.py — canonical path resolver.

This is the ONLY module permitted to reference raw path strings.
All other modules call get_path(name) to obtain filesystem locations.

Environment variable overrides follow the pattern: LEP_<KEY_UPPER>
  e.g. LEP_DB_FILE overrides the "db_file" key.
"""

import json
import os
from pathlib import Path

# Project root is two levels up from this file (core/paths.py → core/ → project root)
_PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
_CONFIG_FILE: Path = _PROJECT_ROOT / "config" / "paths.json"

_ENV_PREFIX = "LEP_"

_cache: dict[str, Path] | None = None


def _load() -> dict[str, Path]:
    global _cache
    if _cache is not None:
        return _cache

    with _CONFIG_FILE.open("r", encoding="utf-8") as fh:
        raw: dict[str, str] = json.load(fh)

    resolved: dict[str, Path] = {}
    for key, value in raw.items():
        env_key = _ENV_PREFIX + key.upper()
        override = os.environ.get(env_key)
        if override:
            resolved[key] = Path(override).resolve()
        elif key == "project_root":
            resolved[key] = _PROJECT_ROOT
        else:
            p = Path(value)
            if p.is_absolute():
                resolved[key] = p
            else:
                resolved[key] = (_PROJECT_ROOT / p).resolve()

    _cache = resolved
    return _cache


def get_path(name: str) -> Path:
    """Return the resolved Path for the given config key.

    Raises KeyError if the key is not defined in paths.json.
    Supports environment variable overrides (LEP_<KEY_UPPER>).
    """
    return _load()[name]


def reload() -> None:
    """Invalidate the path cache (useful in tests)."""
    global _cache
    _cache = None
