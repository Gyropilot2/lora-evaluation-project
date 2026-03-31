# Lora Evaluation Project - Code Organization and DataBank Law
Generated: 2026-02-17
See also: `01_GLOSSARY.md` for canonical terms/enums,
`03_SYSTEM_BLUEPRINT.md` for module responsibilities, and
`08_ASSET_AND_VALUEREF_POLICY.md` for detailed storage law.

This file answers two practical questions:

1. Where does code belong in this repo?
2. What are the implementation laws around DataBank and persistence?

Detailed asset format and ValueRef policy live in `08`.

---

# 1. What This File Owns

`03_SYSTEM_BLUEPRINT.md` defines the system architecture.
This file defines the practical code-organization and persistence rules that
keep that architecture legible in the repo.

Use this file for:
- folder ownership and naming expectations
- import and boundary discipline
- DataBank table shape and truth-vs-index rules
- path/config rules
- asset ownership boundaries and portability constraints
- execution rules and source-of-truth split

---

# 2. Repo Surface Ownership

Top-level areas and what they own:

- `core/`
  Shared primitives usable everywhere: diagnostics, hashing, paths, json
  codec, time IDs.

- `contracts/`
  Shared contracts and registries: Evidence schema, metric registry,
  validation-error helpers.

- `extractor/`
  Entry-point extraction logic. The only production subsystem allowed to touch
  live ComfyUI-side objects before converting them into portable primitives.

- `bouncer/`
  Evidence validation, deduplication, invalid-wrapper handling, extras split.

- `databank/`
  Persistence subsystem. Owns SQL, backend implementation, asset staging, and
  DB health helpers.
  - `databank/operations/` - one-off maintenance and migration scripts that act
    on production data via Treasurer / patching primitives. They are not
    production imports. Each script should declare its cleanup condition and
    whether it is safe to re-run.

- `command_center/`
  Thin control facade over the subsystem interfaces it depends on.

- `lab/`
  Off-grid discovery tooling. Not production ingestion. Never writes production
  DB state.

- `operator_app/`
  Local review workspace. Frontend/backend surface for interactive review
  and control.

- `comfyui/`
  Node registration glue only. Lab probe nodes (`CATEGORY = "lora_eval/lab"`)
  belong here only when they are part of the active lab-facing node surface.

- `.dev/`
  Developer tooling and architecture checks only: `dev_healthcheck.py`
  (architectural linter) and `cli.py` (dev CLI harness). **Not** the home for
  migration scripts, data-acting utilities, or anything that touches production
  DB state. Those belong in `databank/operations/`.

- `.docs/`
  Canon law, blueprint, methodology, and stable design intent.

---

# 3. Boundary Laws in Practice

These are implementation-facing restatements of the architectural boundaries.

## 3.1 SQL isolation

- Only `databank/` executes SQL.
- No DB driver imports outside `databank/`.
- No raw SQL outside `databank/`.
- The public persistence door is `databank/treasurer.py`.

## 3.2 ComfyUI isolation

- No ComfyUI imports outside `extractor/` and `comfyui/` in production code.
- `lab/` may depend on a ComfyUI-capable environment because it is off-grid
  discovery tooling, but that dependency must not leak into production module
  boundaries.
- Core modules must remain importable in plain Python with no ComfyUI present.

## 3.3 Path isolation

- No hard-coded path strings outside `core/paths.py`.
- Filesystem locations flow through `config/paths.json` and `core/paths.py`.

## 3.4 Support-surface discipline

- `lab/` and `operator_app/` do not get to bypass the core
  architectural doors just because they are support surfaces.
- If a support surface needs DB data, it goes through Treasurer /
  Command Center / review services rather than inventing its own raw
  persistence path.

---

# 4. DataBank Model

The persistence model is hybrid:

- one authoritative object-level record per row
- plus a small set of promoted columns for fast filtering and indexing

Promoted columns are lookup handles, not the source of semantic truth.
The authoritative payload remains the stored record blobs.

## 4.1 Tables

The DB has four tables:

- `Methods`
- `Evals`
- `Samples`
- `LoRAs`

Hierarchy:
- one Method -> many Evals
- one Eval -> many Samples
- LoRAs is a flat catalog keyed by `lora_hash`

## 4.2 Truth vs promoted columns

Promote only stable, query-friendly scalars that are genuinely useful for
filtering, joining, health inspection, or operator lookup.

Examples of promoted-column categories:
- identity hashes
- foreign-key linkage
- timestamps
- `seed`
- `lora_strength`
- `ingest_status`
- `is_dirty`

Do not confuse promoted convenience with truth ownership:
- the object-level JSON blobs remain authoritative
- promoted columns exist so common queries do not require scanning blobs

## 4.3 Dirty, errors, and invalid data

- Invalid non-vital fields are stored explicitly via status wrappers.
- Error details are stored separately rather than being silently swallowed.
- `is_dirty` is a persistent flag, not a transient UI mood.
- Dirtiness is not auto-cleared by the system.

## 4.4 Extras

- Unknown keys do not silently disappear.
- They remain attached to the originating record via `extras_json`.
- Promotion from extras into canon schema is an intentional later decision, not
  an implicit side effect of repeated sightings.

## 4.5 DataBank health

DB quality reporting belongs to `command_center/health.py`.
It may summarize counts, dirty rates, extras frequency, and related health
signals, but it still operates through Treasurer-facing surfaces rather than
inventing a second persistence interface.

---

# 5. Treasurer and Portability Law

The system talks to persistence through the Treasurer surface.
That is what keeps backend replacement localized.

Practical meaning:
- backend-specific SQL details stay inside `databank/`
- the rest of the system asks for methods/evals/samples/health through named
  calls, not backend knowledge

This keeps backend replacement possible without spreading SQL assumptions across
the repo.

---

# 6. SQLite Concurrency Law

SQLite runs in WAL mode.

Design assumptions:
- one logical writer at a time
- many concurrent readers allowed
- transaction handling and PRAGMA ownership stay inside `databank/`
- no other module gets to assume lock semantics or transaction policy

This is not an invitation for the rest of the repo to become "SQLite-aware."
It is a reminder that concurrency policy is centralized in DataBank.

---

# 7. Path Configuration Law

All important filesystem paths are declared once and consumed through one small
helper layer.

Requirements:
- config file: `config/paths.json`
- helper module: `core/paths.py`
- path operations use `pathlib.Path`
- code outside `core/paths.py` does not hard-code directory strings

Recommended key categories:
- `project_root`
- `db_file`
- `assets_root`
- `logs_root`
- `tmp_root`
- any other durable top-level storage locations the system genuinely owns

Environment overrides are allowed, but still route through `core/paths.py`.

---

# 8. Asset Lifecycle Policy

DataBank owns the persistence side of asset handling: references, staging
coordination, and the persistence boundary between records and blobs.

Detailed lifecycle, GC, hashing, and format law live in
`08_ASSET_AND_VALUEREF_POLICY.md`.

---

# 9. Asset Storage Policy

This file no longer owns the detailed storage-format table.

Use `08_ASSET_AND_VALUEREF_POLICY.md` for:
- locked asset formats
- ValueRef contract
- canonical hashing
- safetensors content-hash policy
- detailed storage and GC law

---

# 10. Code Organization Hygiene

Practical repo hygiene rules:

- Split by responsibility boundary, not aesthetics.
- Do not create "misc" or `utils.py` dumping grounds when a real name exists.
- Support surfaces may move faster than production code, but they still respect
  the same boundaries.
- Keep files narrow enough that one primary owner and purpose remain obvious.

---

# 11. Doc Index

The `.docs/` folder is the authoritative source for design law and intended structure.

- `01_GLOSSARY.md` - canonical terms and object meanings
- `02_CONSTITUTION.md` - non-negotiable design laws
- `03_SYSTEM_BLUEPRINT.md` - module responsibilities and architecture
- `04_CODE_ORG_AND_DATABANK.md` (this file) - code organization and persistence law
- `05_VALIDITY_AND_DIAGNOSTICS.md` - invalid, dirty, diagnostics, extras, and schema-evolution law
- `06_EVIDENCE_REFERENCE.md` - raw measurement domains and what Evidence contains
- `07_SYNTHESIS_PROVISIONAL.md` - metric interpretation, HeroMetric aggregation, comparison principles
- `08_ASSET_AND_VALUEREF_POLICY.md` - asset lifecycle, hashing, and storage format law
- `09_OPERATOR_APP_AND_COMMAND_CENTER.md` - Operator App structure and UX law
