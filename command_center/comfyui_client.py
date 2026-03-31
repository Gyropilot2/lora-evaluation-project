"""
command_center/comfyui_client.py — Shared ComfyUI HTTP engine.

Provides the minimal ComfyUI API primitives shared by all batch runners
(batch.py run / rerun / replay). All ComfyUI HTTP surface lives here;
callers never construct URLs or interpret history entries directly.

Public API:
    COMFYUI_HOST         -- base URL (overridable via module attribute)
    check_alive()        -- True when ComfyUI is reachable
    queue_prompt()       -- submit a workflow dict, return prompt_id
    wait_for_completion() -- block until prompt finishes, return history entry
    summarize_status()   -- (ok: bool, label: str) from a history entry dict
"""
from __future__ import annotations

import time
from typing import Any

import requests

from core.diagnostics import emit

# ---------------------------------------------------------------------------
# Endpoint layout
# ---------------------------------------------------------------------------

COMFYUI_HOST = "http://127.0.0.1:8188"

_EP_PROMPT  = "prompt"
_EP_HISTORY = "history"
_EP_STATS   = "system_stats"


def _url(route: str, *segments: str) -> str:
    """Build a full ComfyUI URL from bare route segments."""
    parts = [COMFYUI_HOST, route] + list(segments)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_alive() -> bool:
    """Return True if ComfyUI is reachable at COMFYUI_HOST."""
    try:
        requests.get(_url(_EP_STATS), timeout=5)
        return True
    except requests.RequestException:
        return False


def queue_prompt(workflow: dict[str, Any], client_id: str) -> str:
    """Submit a prompt to ComfyUI and return the prompt_id string.

    Raises RuntimeError when ComfyUI accepts the request but reports an
    application-level error in the response body.  Raises requests exceptions
    on transport failures — callers may choose to catch them.
    """
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(_url(_EP_PROMPT), json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        emit(
            "WARN",
            "COMFYUI.PROMPT.REJECTED",
            "comfyui_client.queue_prompt",
            f"ComfyUI rejected prompt submission",
            error=str(data["error"]),
        )
        raise RuntimeError(f"ComfyUI rejected prompt: {data['error']}")
    return str(data["prompt_id"])


def wait_for_completion(prompt_id: str, poll_secs: float = 3.0) -> dict[str, Any]:
    """Block until ComfyUI reports the prompt as finished.

    Returns the raw history entry dict for the prompt.  The entry is the
    value stored under the prompt_id key in ComfyUI's /history response.
    Returns an empty dict when the entry exists but is not a mapping (should
    not happen in practice).

    Transport failures between polls are swallowed and retried — a single
    network hiccup should not abort a long batch run.
    """
    while True:
        time.sleep(poll_secs)
        try:
            history = requests.get(_url(_EP_HISTORY, prompt_id), timeout=15).json()
        except requests.RequestException:
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            return entry if isinstance(entry, dict) else {}


def summarize_status(entry: dict[str, Any]) -> tuple[bool, str]:
    """Best-effort (ok, label) interpretation of a ComfyUI history entry.

    Returns (True, "completed") on apparent success and (False, <reason>)
    when the entry signals an error.  The check is heuristic — ComfyUI's
    history schema is not formally documented.
    """
    status = entry.get("status")
    if isinstance(status, dict):
        status_str = str(status.get("status_str") or "").strip()
        if status_str and status_str.lower() in {"error", "failed", "failure", "exception"}:
            return False, status_str

    messages = entry.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, list) and msg:
                kind = str(msg[0]).lower()
                if "error" in kind or "exception" in kind:
                    return False, kind

    return True, "completed"
