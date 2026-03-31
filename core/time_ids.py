"""
core/time_ids.py — timestamp and UUID generation helpers.

All run_id, sample_id, and timestamp generation flows through this module
to ensure consistent formatting across the system.
"""

from datetime import datetime, timezone
import uuid


def new_run_id() -> str:
    """Return a new random UUID4 string for a Run record."""
    return str(uuid.uuid4())


def new_sample_id() -> str:
    """Return a new random UUID4 string for a Sample record.

    Note: in production, sample_id is the BLAKE3 content hash of the sample's
    output measurements, not a random UUID.  This helper is provided for
    scaffolding / testing where a real content hash is not yet available.
    Use core.hashing functions to derive the real sample_id from content.
    """
    return str(uuid.uuid4())


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone offset."""
    return datetime.now(tz=timezone.utc).isoformat()
