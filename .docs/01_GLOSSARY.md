# Lora Evaluation Project - Glossary & Canonical Terms

This file defines the canonical terms used across the project docs and code.

## Canonical enums

### Invalid value wrapper (`Invalid`)
An Evidence field is either a normal value, or an invalid wrapper:

```json
{ "status": "NULL" | "ERROR" | "NOT_APPLICABLE", "reason_code": "OPTIONAL_CODE", "detail": "OPTIONAL_SHORT_TEXT" }
```

- `NULL`: missing/unknown/empty
- `ERROR`: extraction/computation failed (system investigation required)
- `NOT_APPLICABLE`: concept does not apply to this record

### Ingestion status (`ingest_status`)
A lightweight sample flag:

- `OK` | `WARN` | `ERROR`

Suggested meaning:
- `ERROR` if any field had `{status:"ERROR"}` during ingestion
- `WARN` for non-fatal anomalies (missing optional fields, deprecated metrics, etc.)

## IDs

### `method.hash`
Unique identity of a Method record. BLAKE3 hash of all Method HashComponents (canonical
serialization). Serves as the deduplication key: two methods with the same hash are the same
method. See Method below.

### `eval.hash`
Unique identity of an Eval record. BLAKE3 hash of `(method.hash, lora.hash)`. Two evals with
the same hash are the same eval (same method + same LoRA). See Eval below.

### `sample.hash`
Unique identity of a Sample record. BLAKE3 hash of `(eval.hash, seed, lora_strength)`.
Deterministic before generation - you can predict a sample's identity from inputs alone.
See Sample below and `05_VALIDITY_AND_DIAGNOSTICS.md` for the integrity role of
vital fields such as `latent_hash`.

## Schema terms

### Evidence
Facts-only record. JSON-serializable.
Contains:
- metadata (versions, ids, timestamps)
- generative inputs (`base_model`, `lora`, `vae`, `conditioning`, `settings`)
- measurement domains (`image`, `luminance`, optional `clip_vision` / `masks` / `face_analysis` / `aux`)
- diagnostics (structured warnings/errors)
- `_extras` (unknown/unpromoted keys, quarantined)

### Synthesis
Derived / interpreted outputs assembled from Evidence by the review layer (scores, aggregates, comparisons, etc.).
Synthesis is opinionated, versioned, and traceable back to Evidence + recipes/procedures.

---

## Core records

### Method

A Method centralizes the generative configuration ("harness") used to produce samples.
It captures everything that causally determines an output **except** which LoRA is applied,
the seed, and the LoRA strength (those vary per Eval or Sample).

**Identity:**
- `method.hash` - BLAKE3 hash of all Method HashComponents (canonical serialization).

**HashComponents** (what method.hash is derived from):
- `base_model.hash` - fast-hash of the base model checkpoint (header + 64KB tensor data)
- `model_extras` - fingerprint of any non-base, non-LoRA pipeline extras (e.g. ModelSamplingFlux);
  null/empty is valid and hashes as-is
- `positive_conditioning.hash` - BLAKE3 of the positive conditioning tensor(s)
- `positive_conditioning.guidance` - per-conditioning guidance scale if present; null hashes as-is
- `negative_conditioning.hash` - BLAKE3 of the negative conditioning tensor(s); required
- `negative_conditioning.guidance` - per-conditioning guidance scale if present; null hashes as-is
- `steps` - KSampler steps
- `denoise` - KSampler denoise
- `sampler` - KSampler sampler name
- `scheduler` - KSampler scheduler name
- `cfg` - KSampler cfg
- `latent.width`, `latent.height` - image dimensions inferred from Input Latent
- `latent.shape` - full latent tensor shape [B, C, H_latent, W_latent]; C=4 (SD/SDXL) or C=16 (SD3/Flux)
- `vae_model.hash` - BLAKE3 hash of the VAE model

**Purpose:**
- Allows Bouncer to identify duplicate Methods (same harness -> same method.hash)
- Acts as the top-level identity and integrity root for the runs beneath it

**Convenience index (non-authoritative):**
- `child.eval` - list of eval.hashes under this method; may be stale; can be rebuilt by scanning

### Eval

An Eval groups all Sample records that vary by seed x lora_strength under the same
(method + LoRA) combination. It has exactly one parent Method.

**Identity:**
- `eval.hash` - BLAKE3 hash of `(method.hash, lora.hash)`.

**HashComponents:**
- `method.hash` - parent Method identity
- `lora.hash` - BLAKE3 of LoRA weight tensor content; null/empty for Baseline runs (hashes as-is)

**Purpose:**
- "All samples from LoRA X under harness Y" = all Samples under eval.hash
- Baseline Eval: `lora.hash` is null -> eval.hash = hash(method.hash, null)

**Convenience index (non-authoritative):**
- `strength_index` - map of `lora_strength -> [sample.hash, ...]`; may be stale

### Sample

A Sample is a single generated output (one replicate) produced under a specific Eval,
distinguished by seed and (if a LoRA is present) lora_strength. It is the atomic unit that
stores output identity slots and measured facts/metrics.

**Identity:**
- `sample.hash` - BLAKE3 hash of `(eval.hash, seed, lora_strength)`.
  Deterministic and pre-computation. Null/empty lora_strength (Baseline) hashes as-is.

**Vital fields inside sample record:**
- `eval.hash` - parent Eval identity (foreign key)
- `seed` - actual KSampler seed used
- `lora.strength` - LoRA strength for this sample; null for Baseline
- `latent_hash` - BLAKE3 of the output latent tensor bytes. NOT the identity key, but a
  vital integrity field. If the same sample.hash arrives with a different latent_hash -> NOT_ZERO_DELTA.
- `image_hash` - BLAKE3 of the decoded image tensor bytes. Vital for slot consistency checks.

**Measurement domains:**
- `image` - ValueRef to decoded image asset + pixel_stats inline
- `luminance` - ValueRef to BT.709 luminance map + stats inline
- `clip_vision` - dict keyed by `model_hash[:16]`; per-key: global embed ValueRef + spatial `patch_pools` (per-mask patch coverage, no re-encode)
- `masks` - dict keyed by mask name (`face` / `main_subject` = `MainSubject_Mask` / `skin` / `clothing` / `hair` / ...); ValueRef only. True scene background is derived as `1 - main_subject`.
- `face_analysis` - array of all detected faces sorted by det_score desc; per face: ArcFace embedding ValueRef, normed_embedding ValueRef, kps, pose, bbox, det_score, age, gender
- `aux` - dict keyed by preprocessor label (`depth`, `normal`, `edge`); ValueRef + pixel_stats. `aux.pose` (colored dot visualization PNG) was removed from all DB rows - it no longer exists in any sample record.
- `pose_evidence` - additive Sample sub-object for structured pose storage. Fixed source keys are `openpose_body` and `dw_body`. Each source preserves the raw grouped upstream `people[]` payload plus normalized per-person grouped joints, per-joint detector confidence / missingness, and per-joint / per-person support facts against `MainSubject_Mask` and DensePose-derived human region.

Duplicate handling, enrichment behavior, and dirty-record consequences for
Samples are persistence and validity rules, not glossary material. The
authoritative behavior lives in `05_VALIDITY_AND_DIAGNOSTICS.md`.

## Shape policy (project-wide default)

### Nested domain objects (canonical)
Evidence is **nested**: related fields live together under a parent object.
Example: `lora.name` and `lora.hash` live under `lora`, not as flat `lora_name` / `lora_hash`.

Flattening is allowed only for:
- promoted DB columns (indexes)
- UI labels
- external export formats that require it

### Dynamic-key boundary (strict)
Only `masks.{name}` and `aux.{name}` are open-ended keyspaces.

All other domains are fixed-key contracts and must not accept dynamic slot names.

## ValueRef

### ValueRef (portable representation for non-JSON data)
When a field's value is not JSON (e.g., tensors/embeddings), Evidence stores a ValueRef object:

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

`content_hash.algo` is canonically `blake3`.

## Modules (names as used in the docs)

### Extractor
Consumes raw runtime inputs and produces Evidence facts. No opinions. No scoring.

### DataBank
Owns DB writes/reads. Only module that touches SQL.

### Assets
Owns the blob vault (assets) and retrieval. Only module that opens/reads blob bytes.

### Context
A read-only access facade over Evidence + Assets, used by procedures.

### Procedure
A named Python callable with a narrow, stable responsibility in the review layer.
Detailed review-layer procedure behavior lives in `07_SYNTHESIS_PROVISIONAL.md`.

### Aggregation
A named rule for reducing several lower-level readings into one higher-level value.
Detailed aggregation behavior lives in `07_SYNTHESIS_PROVISIONAL.md`.

### Recipe
A named consumer policy describing what to surface and which aggregation strategy to use.
Detailed recipe behavior lives in `07_SYNTHESIS_PROVISIONAL.md`.

### Review transport
The settled payload family used by the Operator App review surface and the local review dump.
Canonical Python home: `contracts/review_transport.py`. Detailed payload anatomy lives in `07_SYNTHESIS_PROVISIONAL.md`.

### Metric
A validated, registered Signal with a formal definition in `contracts/metrics_registry.py`.

Metrics have stable IDs, units, polarity, reliability status, and a definition hash.
Compound metrics are still metrics. Metric registration answers "what is this
thing?" It does not answer whether a consumer chooses to feature it.

### HeroMetric
A primary evaluation metric package surfaced by the Operator App.
Detailed HeroMetric anatomy and the active hero roster live in `07_SYNTHESIS_PROVISIONAL.md`.

---

### Signal vs Metric
- **Signal** - a computed derived value produced by a Procedure. Not yet validated as meaningful or categorized. All Procedure outputs are Signals until promoted.
- **Metric** - a validated, registered Signal with a formal definition in `contracts/metrics_registry.py`. Metrics have stable IDs, units, polarity, and `metric_def_hash`. Signal -> Metric promotion requires human approval.

### Command Center (CC)
The durable local control and reporting surface.
Detailed ownership and interface layers live in `09_OPERATOR_APP_AND_COMMAND_CENTER.md`.

### Treasurer
The named persistence door through which non-`databank` modules access stored
records. Detailed boundary law lives in `04_CODE_ORG_AND_DATABANK.md`.

### CLI and Operator App

**CLI** - local command surface for repeatable inspection and control tasks.

**Operator App** - local interactive review surface.

Detailed control-surface behavior lives in `09_OPERATOR_APP_AND_COMMAND_CENTER.md`.

## Diagnostics

### Diagnostic `code` (required, enum-like)
Diagnostics are structured records. The `code` field must be a stable identifier
(treat it like an enum, not free text).

Canonical style (recommended):
- Dot-separated, uppercase tokens: `DOMAIN.DETAIL`
  Examples: `SCHEMA.MISSING_FIELD`, `ASSET.WRITE_FAIL`, `ARCH.CHEAT`

The full diagnostics contract and code policy live in
`05_VALIDITY_AND_DIAGNOSTICS.md`.

## Dirty Flag

A persistent suspicion flag on stored records. Detailed dirty-flag policy lives
in `05_VALIDITY_AND_DIAGNOSTICS.md`.

---

## Vital Fields

Fields whose absence is a hard structural error rather than a warning. The
authoritative vital-field policy lives in `05_VALIDITY_AND_DIAGNOSTICS.md`.

## Diagnostic Codes

Stable machine-readable identifiers for structured diagnostics. The active code
policy and examples live in `05_VALIDITY_AND_DIAGNOSTICS.md`.

## Assets

### Staged blobs (`tmp/`)
Large blobs may be written to a staging area before they become committed
assets. The lifecycle policy now lives in `08_ASSET_AND_VALUEREF_POLICY.md`.

### Assets (`assets/`)
Committed assets live under content-addressed paths and are referenced by
ValueRef. The authoritative storage and hashing policy now lives in
`08_ASSET_AND_VALUEREF_POLICY.md`.
