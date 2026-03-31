# 08 - Asset and ValueRef Policy

Generated: 2026-03-30
See also: `04_CODE_ORG_AND_DATABANK.md` for repo and DataBank ownership,
`06_EVIDENCE_REFERENCE.md` for stored measurement domains.

This file owns the project rules for ValueRef objects, content-addressed assets,
canonical hashing, storage formats, and asset lifecycle. Use it whenever the
question is how non-JSON measurement material is identified, stored, or
recovered.

---

## What This File Owns

- ValueRef contract
- asset identity and content addressing
- canonical hashing rules
- LoRA safetensors content hashing
- asset lifecycle and garbage-collection policy
- storage-format law and locked format table
- ValueRef versioning

---

## ValueRef Contract

Evidence must remain JSON-serializable. When the source produces non-JSON data,
the system stores a portable ValueRef object instead of the live object.

Example:

```json
{
  "valueref_version": 1,
  "kind": "asset_ref",
  "asset_type": "embedding",
  "format": "npy",
  "content_hash": { "algo": "blake3", "digest": "blake3:..." },
  "path": "assets/embeddings/blake3__....npy",
  "dtype": "float32",
  "shape": [1, 768]
}
```

Rules:
- the DB stores the ValueRef object, not the live tensor or image object
- the asset store holds the blob
- loaders may dereference the ValueRef through the asset subsystem
- small summaries such as `shape`, `dtype`, or compact stats may also be stored
  inline when that avoids opening blobs for ordinary work

ValueRef is a portability contract, not a storage optimization gimmick.

---

## Asset Identity and Content Addressing

Assets are content-addressed. The project stores references to them via
ValueRef and identifies them by canonical content hash.

Core law:
- store originals and references
- do not store derived truth artifacts in place of the underlying signal
- DB rows store references, hashes, and metadata
- committed asset blobs live in the asset store

Normal runtime never auto-deletes committed assets.

---

## Canonical Hashing Policy

Default hash algorithm:
- `blake3`

Canonicalization rules:

- embeddings and tensors
  - hash over `dtype`, `shape`, and raw contiguous row-major bytes
  - avoid file-container-specific hashes for semantic identity

- images
  - hash decoded pixel content in a fixed representation
  - include width, height, and pixel bytes
  - do not hash PNG file bytes directly, since metadata and compression can
    vary without changing the image content

Verification rule:
- when an asset is loaded and re-canonicalized, hash mismatch is an error
- mismatches are not silently accepted

The point of canonical hashing is that "same content" should deduplicate even
when serialization details differ.

---

## Safetensors LoRA Content Hash

For LoRA assets, store both:
- `lora_file_hash` - raw file bytes hash for audit and provenance
- `lora_content_hash` - canonical tensor-content hash for semantic identity

Canonical content-hash rule:
- parse the safetensors file
- iterate tensor keys in deterministic lexicographic order
- hash tensor key, dtype, shape, and raw contiguous tensor bytes
- combine contributions into one BLAKE3 digest

If `lora_content_hash` cannot be computed:
- store explicit error state
- mark the ingest as `ERROR`
- keep `lora_file_hash` when possible

This keeps "same tensors, different file packaging" from being misread as a
different LoRA identity.

---

## Asset Lifecycle and GC

Assets are not written directly into their final committed location.

Staging convention:
- write to `tmp/` first
- compute `content_hash`
- commit the DB record that references the asset
- atomically move or rename into the final content-addressed path under
  `assets/`

Lifecycle rules:
- normal runtime does not auto-delete committed assets
- staged `tmp/` material may be TTL-cleaned
- orphan cleanup for committed assets is explicit and tool-driven
- there are no hidden cascading deletes

This protects against zombie blobs and accidental runtime data loss.

---

## Storage Format Law

Choose the smallest truthful representation that still preserves the intended
downstream question.

Rules:
- do not change a format unless the reason is principled
- use lossless image formats for image-like derived assets when those assets are
  not hash-critical source-of-truth material
- use dtype-reduced tensor formats for tensor-like assets when the reduction is
  semantically safe
- do not rely on readers to "interpret" soft data as binary after the fact

The main image is the exception-heavy case: it is re-ingest source-of-truth
material and must remain exact.

---

## Format Table

| Asset | Format | Reason |
|---|---|---|
| Main image (final render) | float32 `.npy` | Re-ingest source of truth; must roundtrip exactly or hash integrity fails |
| Luminance map | 16-bit grayscale PNG | Derived scalar image; genuine >8-bit precision confirmed |
| Aux depth / normal / edge | 8-bit PNG | Derived surfaces; effective precision confirmed sufficient |
| Masks, including `main_subject` | binary PNG | Derived region-membership surfaces; canonical storage is binary |
| CLIP embeddings | float16 `.npy` | Tensor-like, half-size, no decompression overhead |
| Face embeddings | float32 `.npy` | Identity vectors stay exact and are already small enough that size reduction is not worth the tradeoff |

Additional guidance from settled storage decisions:

- `aux.pose` as a colored visualization is not a durable storage target
- structured `pose_evidence` is preferred over raster pose images when grouped
  keypoints are available upstream
- masked CLIP re-encodes are not stored; masks are applied to stored spatial
  features at comparison time
- mask-derived stats such as per-mask `pixel_stats` or `edge_density` are not
  stored when they can be recomputed losslessly from stored image plus mask

---

## Format Notes By Asset Family

### Main image

The stored main image is not "just another asset." Re-ingest loads this file
directly. Any quantized storage format would introduce rounding and cause
hash mismatch on replay. That is why it stays float32 `.npy`.

### Luminance

Luminance is stored as 16-bit grayscale PNG.

Decode contract:
- read with Pillow mode `I`
- divide by `65535.0`
- restore float32 `[0, 1]`

Do not decode luminance through a generic RGB PNG path.

### Masks

Masks are stored as raw mask surfaces only. The system does not canonically
store derived per-mask image statistics inside the mask domain.

### Aux surfaces

Depth, normal, and edge are stored as derived structural surfaces in compact
image form.

Structured pose facts are stored as structured records rather than committed as
the old colored-dot visualization surface.

### Embeddings

Store CLIP-family embeddings as tensors, not as packed image surrogates.

Use dtype reduction where safe.
Do not add compression whose runtime cost erases the storage win.

---

## ValueRef Versioning

ValueRef objects include a `valueref_version` integer.

Rules:
- version `1` is the current shape
- forward changes to the structure must bump the version
- readers must not silently assume that all ValueRefs share one eternal shape

Versioning keeps ValueRef evolution explicit instead of magical.
