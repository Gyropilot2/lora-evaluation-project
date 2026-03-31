"""
core/hashing.py — BLAKE3 canonical hash functions.

All hashing in the system flows through this module.
Content-addressed identity (method_hash, eval_hash, sample_hash, asset paths)
uses these functions.

Note: hash_safetensors_content reads model weight bytes directly and may be slow
      for large files; call only when a fresh hash is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import blake3

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  (type hints only; not a hard dependency)


# ---------------------------------------------------------------------------
# Primitive hash functions
# ---------------------------------------------------------------------------


def hash_bytes(data: bytes) -> str:
    """Return the BLAKE3 hex digest of raw bytes."""
    return blake3.blake3(data).hexdigest()


def hash_tensor(dtype: str, shape: tuple, raw_bytes: bytes) -> str:
    """Return a stable BLAKE3 hash of a tensor.

    The hash covers: dtype string + shape tuple (as a canonical prefix) + raw bytes.
    This ensures tensors with different dtypes or shapes but same raw bytes
    are not considered identical.
    """
    import json as _json

    prefix = _json.dumps({"dtype": dtype, "shape": list(shape)}, sort_keys=True).encode()
    h = blake3.blake3()
    h.update(prefix)
    h.update(raw_bytes)
    return h.hexdigest()


def hash_image_pixels(rgba_bytes: bytes, width: int, height: int) -> str:
    """Return a stable BLAKE3 hash of raw image pixel data.

    The hash covers: width + height (as a canonical prefix) + RGBA bytes.
    """
    import json as _json

    prefix = _json.dumps({"width": width, "height": height}, sort_keys=True).encode()
    h = blake3.blake3()
    h.update(prefix)
    h.update(rgba_bytes)
    return h.hexdigest()


def hash_safetensors_content(path: str | Path) -> str:
    """Return a BLAKE3 hash of the content bytes of a safetensors file.

    Reads the full file; does NOT include the file path in the hash so that
    moving the file does not change the content hash.

    Raises:
        FileNotFoundError: if the file does not exist.
        OSError: on read failure.
    """
    p = Path(path)
    data = p.read_bytes()
    return hash_bytes(data)


# ---------------------------------------------------------------------------
# Identity hash functions  (Method → Eval → Sample, three-level hierarchy)
# ---------------------------------------------------------------------------
#
# KEY ORDER IS CANONICAL — do not reorder or insert keys mid-list.
# Changing key order invalidates every existing hash stored in the DB.
# New keys must be appended to the end only, with a schema version bump
# recorded in the Session Log.
#
# Key names use flat snake_case to avoid JSON nesting ambiguity.
# Dot-notation in the Glossary (e.g. "base_model.hash") maps to the
# underscore name here (e.g. "base_model_hash").

_METHOD_HASH_KEYS: list[str] = [
    # Generative harness — everything except LoRA identity, seed, lora_strength.
    # See 01_GLOSSARY.md §Method / HashComponents for the authoritative list.
    "base_model_hash",               # BLAKE3 of base model checkpoint
    "model_extras",                  # fingerprint of non-base non-LoRA extras; null OK
    "positive_conditioning_hash",    # BLAKE3 of positive conditioning tensor(s)
    "positive_conditioning_guidance",  # guidance scale; null OK
    "negative_conditioning_hash",    # BLAKE3 of negative conditioning tensor(s)
    "negative_conditioning_guidance",  # guidance scale; null OK
    "steps",                         # KSampler steps (int)
    "denoise",                       # KSampler denoise (float)
    "sampler",                       # KSampler sampler name (str)
    "scheduler",                     # KSampler scheduler name (str)
    "cfg",                           # KSampler cfg (float)
    "latent_width",                  # output image width (int)
    "latent_height",                 # output image height (int)
    "latent_shape",                  # full latent tensor shape [B,C,H,W] (array); null OK
    "vae_model_hash",                # BLAKE3 of VAE model
]


def method_hash(components: dict) -> str:
    """Return a stable BLAKE3 hex digest identifying a Method record.

    A Method captures the entire generative harness (base model, VAE,
    conditioning, sampler settings, latent dimensions) excluding the LoRA,
    seed, and lora_strength.

    Args:
        components: dict with keys matching ``_METHOD_HASH_KEYS``.  Missing
            keys are serialised as JSON null.  Extra keys are silently ignored.
            ``None`` values are serialised as JSON null — NOT as Invalid wrappers.

    Returns:
        BLAKE3 hex digest string.

    Example::

        h = method_hash({
            "base_model_hash": "blake3:aabb...",
            "vae_model_hash":  "blake3:ccdd...",
            "positive_conditioning_hash": "blake3:aaaa...",
            "negative_conditioning_hash": "blake3:bbbb...",
            "positive_conditioning_guidance": None,
            "negative_conditioning_guidance": None,
            "steps": 28, "denoise": 1.0, "sampler": "euler",
            "scheduler": "normal", "cfg": 7.0,
            "latent_width": 1024, "latent_height": 1024,
            "latent_shape": None, "model_extras": None,
        })
    """
    from core.json_codec import canonical_json  # late import to avoid circular risk

    canonical = {k: components.get(k, None) for k in _METHOD_HASH_KEYS}
    serialised = canonical_json(canonical).encode("utf-8")
    return blake3.blake3(serialised).hexdigest()


def eval_hash(parent_method_hash: str, lora_hash: str | None) -> str:
    """Return a stable BLAKE3 hex digest identifying an Eval record.

    An Eval is the combination of a Method (generative harness) and a LoRA
    identity.  Baseline evals (no LoRA) pass ``lora_hash=None``.

    Args:
        parent_method_hash: BLAKE3 hex digest of the parent Method.
        lora_hash: BLAKE3 content hash of the LoRA weight tensors, or ``None``
            for a Baseline eval (hashes as JSON null).

    Returns:
        BLAKE3 hex digest string.
    """
    from core.json_codec import canonical_json  # late import

    payload = {"method_hash": parent_method_hash, "lora_hash": lora_hash}
    serialised = canonical_json(payload).encode("utf-8")
    return blake3.blake3(serialised).hexdigest()


def sample_hash(parent_eval_hash: str, seed: int, lora_strength: float | None) -> str:
    """Return a stable BLAKE3 hex digest identifying a Sample record.

    A Sample is one generated output: a specific (Eval × seed × lora_strength)
    combination.  Baseline samples (no LoRA) pass ``lora_strength=None``.

    The hash is **deterministic before generation** — you can compute a
    sample's identity from its inputs alone, before running the sampler.

    Args:
        parent_eval_hash: BLAKE3 hex digest of the parent Eval.
        seed: integer KSampler seed (the actual seed used after any
            randomisation, not a "random" flag).
        lora_strength: float strength applied to the LoRA, or ``None`` for
            Baseline samples (hashes as JSON null).

    Returns:
        BLAKE3 hex digest string.
    """
    from core.json_codec import canonical_json  # late import

    payload = {
        "eval_hash": parent_eval_hash,
        "seed": seed,
        "lora_strength": lora_strength,
    }
    serialised = canonical_json(payload).encode("utf-8")
    return blake3.blake3(serialised).hexdigest()


# ---------------------------------------------------------------------------
# Run signature  (DEPRECATED — replaced by method_hash / eval_hash / sample_hash)
# ---------------------------------------------------------------------------
#
# Kept for reference during Step 1.7 infrastructure rewrite.
# Remove once databank/ and bouncer/ have been rewritten (Steps 1.7.c–f).

# KEY ORDER IS CANONICAL — do not reorder or insert keys mid-list.
# Changing key order invalidates all existing run_signatures in the DB.
# New keys must be appended to the end only, with a schema version bump noted in Session Log.
_RUN_SIGNATURE_KEYS = [
    "base_hash",
    "lora_hash",
    "prompt_hash",
    "prompt_family",
    "seed",
    "steps",
    "strength",
    "denoise",
    "sampler",
    "scheduler",
    "cfg",
    "clip_vision_model_hash",
    "input_image_hash",
]


def run_signature(components: dict) -> str:
    """Return a stable BLAKE3 hex string for the given run component dict.

    Only the canonical keys defined in _RUN_SIGNATURE_KEYS are included.
    Missing keys are serialised as JSON null (not the Invalid wrapper).
    Keys not in the canonical list are silently ignored.

    Returns:
        BLAKE3 hex digest string.
    """
    from core.json_codec import canonical_json  # late import to avoid circular risk

    canonical = {k: components.get(k, None) for k in _RUN_SIGNATURE_KEYS}
    serialised = canonical_json(canonical).encode("utf-8")
    return blake3.blake3(serialised).hexdigest()
