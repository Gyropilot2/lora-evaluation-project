"""
core/json_codec.py — canonical JSON serialisation.

Canonical JSON is used wherever deterministic serialisation matters:
  - run_signature computation
  - recipe_hash computation
  - content-addressed asset keys

Rules:
  - Keys sorted alphabetically (recursive)
  - No trailing whitespace or unnecessary indentation
  - NaN and Infinity are rejected (not valid JSON)
  - Python datetime objects are serialised as ISO-8601 strings
  - Bytes are rejected (use hashing.hash_bytes and store the hex digest instead)
"""

import json
import math
from datetime import datetime
from typing import Any


class _CanonicalEncoder(json.JSONEncoder):
    """JSON encoder that enforces canonical serialisation rules."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

    def iterencode(self, obj: Any, _one_shot: bool = False):  # type: ignore[override]
        # Reject NaN and Infinity at the top-level encode call
        return super().iterencode(obj, _one_shot=_one_shot)


def _check_no_nan(obj: Any) -> None:
    """Recursively raise ValueError on NaN or Infinity values."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"canonical_json does not accept NaN or Infinity, got: {obj!r}")
    elif isinstance(obj, dict):
        for v in obj.values():
            _check_no_nan(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _check_no_nan(item)


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON string for obj.

    Guarantees:
    - Sorted keys (recursive)
    - No NaN or Infinity
    - Compact (no extra whitespace)
    - Deterministic across calls with identical input

    Raises:
        ValueError: if obj contains NaN or Infinity floats.
        TypeError:  if obj contains non-serialisable types (e.g. raw bytes).
    """
    _check_no_nan(obj)
    return json.dumps(obj, cls=_CanonicalEncoder, sort_keys=True, separators=(",", ":"))


def parse_json(text: str) -> Any:
    """Parse a JSON string. Thin wrapper for symmetry with canonical_json."""
    return json.loads(text)
