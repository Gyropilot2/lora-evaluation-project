# 05 - Validity and Diagnostics

Generated: 2026-03-30
See also: `01_GLOSSARY.md` for canonical terms and record identities,
`03_SYSTEM_BLUEPRINT.md` for subsystem responsibilities.

This file owns the rules for invalid and partial records, structured
diagnostics, extras handling, schema and registry synchronization, and the
conditions under which record structure may evolve. Use it when the question is
what the system stores or emits when data is incomplete, invalid, noisy, or
changing shape over time.

---

## What This File Owns

- valid versus invalid record policy
- invalid wrapper semantics
- null propagation rules
- dirty-flag and vital-field behavior
- diagnostics contract and code expectations
- `_extras` handling
- schema and registry synchronization
- explicit schema evolution and migration rules

---

## Valid vs Invalid Records

Policy:
- prefer truth over convenience
- never fake values to make a record "valid"
- store anyway when the structure is still representable
- attach explicit errors instead of hiding failure

Valid records:
- Evidence checks pass

Invalid records:
- one or more non-vital Evidence checks fail
- the record remains structurally representable
- missing or unknown values use explicit wrappers instead of silent JSON null
- `_extras` stays attached to the originating sample
- `errors[]` explains what failed

The default posture is store-invalid, not "drop anything imperfect."

---

## Invalid Wrapper Semantics

Evidence values may be absent or invalid. The project represents this in the
same slot as the field using a small status wrapper.

Status meanings:
- `NULL` - missing, unknown, empty, not extractable, or intentionally omitted
- `ERROR` - extraction or computation failed in a way that suggests the system
  may be broken or an assumption was violated
- `NOT_APPLICABLE` - the concept does not apply to this record

Representation rule:

```json
{
  "status": "NULL" | "ERROR" | "NOT_APPLICABLE",
  "reason_code": "OPTIONAL_CODE",
  "detail": "OPTIONAL_SHORT_TEXT"
}
```

Fields are stored as either:
- a normal primitive or ValueRef
- or this wrapper object

There is no silent fallback to invented numbers.

---

## Behavior on Invalid Inputs

General rules:

- Bouncer never drops a parsed record except for structurally unreadable input
  or missing vital fields.
- If the record is valid JSON but violates Evidence on non-vital fields, store
  it with explicit invalid wrappers and `ingest_status` set to `WARN` or
  `ERROR`.
- If a field-level failure can be represented honestly, the record remains
  storable.

Review-layer behavior:

- default: invalid records are excluded from ordinary review output
- debug or health paths may include them explicitly for diagnosis

Field-level failure does not automatically invalidate the entire record.
Structural non-representability does.

---

## Null Propagation

If a derived metric depends on missing, invalid, or unavailable inputs:

- the output is invalid
- the invalid state stays explicit
- explanation tags or reason codes should identify the missing dependency

There are no silent fallbacks.
There are no "make it pass" substitutes.

This rule applies across review assembly as well: if an input is invalid for a
computation, the output remains invalid until the missing support is restored.

---

## Dirty Flags and Vital Fields

### Dirty Flag

`is_dirty` is a persistent suspicion flag, not a transient UI mood.

Rules:
- present on Method, Eval, Sample, and LoRA records
- set when structural integrity failures or serious error conditions are found
- never auto-cleared by the system
- only cleared by explicit human action through the appropriate health tooling

Dirtiness does not propagate upward automatically. A dirty Sample does not make
its Eval dirty unless the triggering condition explicitly warrants it.

Review assembly excludes dirty Samples by default.

### Vital Fields

Vital fields are fields whose absence is a hard error. If a vital field is
missing or invalid, the record is refused entirely and nothing is written.

Method vital fields:
- `method.hash`

Eval vital fields:
- `eval.hash`
- `eval.method.hash`

Sample vital fields:
- `sample.hash`
- `eval.hash`
- `seed`
- `latent_hash`
- `image_hash`

Do not promote a field to vital without explicit review. Vital status is a
structural decision, not just a convenience check.

---

## Diagnostics Contract

Diagnostics are first-class for:
- developer debugging
- visible integrity and coverage warnings
- explicit temporary scaffolding via `ARCH.CHEAT`

Diagnostics are structured records, not free text.

Minimum fields:
- `ts` - ISO timestamp
- `severity` - `DEBUG | INFO | WARN | ERROR | FATAL`
- `code` - stable identifier from a central registry
- `where` - module/class/function tag
- `msg` - short readable message
- `ctx` - small JSON object with relevant context

Recommended fields:
- `count`
- `first_seen`
- `last_seen`
- `fingerprint`

Display example:

```text
[WARN] [bouncer.gate] SCHEMA.MISSING_FIELD -> Required field is NULL/invalid
```

Command Center writes runtime diagnostics to the canonical diagnostics stream.
For `ERROR` and `FATAL` records emitted through the runtime facade, Command
Center also mirrors the same record to its own append-only error log for
operator review.

---

## Diagnostic Codes and Registry Expectations

Diagnostic `code` values are stable identifiers. They are not free text.

Recommended style:
- dot-separated uppercase tokens such as `SCHEMA.MISSING_FIELD`

Known examples:
- `SAMPLE.DUPLICATE_RUN`
- `SAMPLE.NOT_ZERO_DELTA`
- `SAMPLE.SLOT_CONFLICT`
- `SAMPLE.ENRICHED`
- `METHOD.INDEX_STALE`
- `SCHEMA.MISSING_FIELD`
- `SCHEMA.PARSE_FAIL`
- `SCHEMA.UNKNOWN_KEY`
- `SCHEMA.VITAL_MISSING`
- `ASSET.HASH_MISMATCH`
- `ASSET.WRITE_FAIL`
- `DB.WRITE_FAIL`
- `ARCH.CHEAT`

Registry expectations:
- codes come from one central registry
- synonymous duplicates such as `FIELD_MISSING` vs `MISSING_FIELD` are not
  acceptable
- `ARCH.CHEAT` entries must include an explicit cleanup condition

---

## `_extras` Policy

`_extras` stores unknown keys, types, or values attached to the same sample.

Rules:
- never pooled globally by default
- excluded from derivation by default
- attached to the originating sample rather than dropped
- promotion requires recurrence plus explicit review

This is where the project quarantines unpromoted structure without pretending it
already belongs in canon.

---

## Schema and Metric Registry Relationship

- schema defines the DB boundary
- metric registry defines promoted metric keys and metadata
- schema and registry must remain synchronized

Rules:
- no unregistered metric keys outside `_extras`
- `_extras` keys are not promoted by accident
- promotion is explicit and versioned

Additional anti-drift rule:
- if the same metric key is produced with meaningfully changed behavior, it
  must be versioned as a new key or explicitly deprecated with a loud warning

Schema is storage law.
Registry is metric identity law.
Neither should silently drift away from the other.

---

## Schema Evolution and Record Mutability

Evidence is facts-only, but it is not sacred text.

Principles:
- latest truthful taxonomy wins
- records may be rewritten or replaced when doing so improves factual accuracy
  or structural clarity
- there is no silent in-place mutation

Rules:
- promotions from `_extras` into canon keys are explicit
- rewrites and migrations are version-stamped
- review assembly must tolerate mixed record states during transition periods
- if a newly ingested record matches an existing `sample_hash` but produces a
  different `latent_hash`, that is `NOT_ZERO_DELTA`, not a quiet overwrite

This project is accuracy-first, not history-first. But changes still need to be
explicit enough that the system can tell what happened.

---

## Migration Execution Rules

Schema migrations may run against a live DB instance, but only under explicit
tooling.

Rules:
- migrations are version-stamped and logged
- migration tooling operates through the DataBank layer, not raw SQL from
  arbitrary modules
- migration scripts must be idempotent
- migration tools must declare the schema version they target
- silent background rewriting is forbidden
- treat backup as the operator's responsibility before structural changes
