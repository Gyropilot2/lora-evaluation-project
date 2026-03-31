# Lora Evaluation Project - System Blueprint

Generated: 2026-03-30
See also: `01_GLOSSARY.md` for canonical terms and record identities,
`05_VALIDITY_AND_DIAGNOSTICS.md` for invalid/partial-record behavior,
`08_ASSET_AND_VALUEREF_POLICY.md` for storage law.

This document defines the live architecture of Lora Evaluation Project.
It describes how the production pipeline is split, what each subsystem owns,
and why those boundaries matter.

For validity policy, diagnostics contract, schema evolution, and storage law,
use the docs that own those subjects directly. This file is the blueprint.

---

# 1. System Topology

Off-grid module:
- Lab

Production pipeline:
Extractor -> Bouncer -> DataBank -> Review Assembly -> Command Center -> Access Layer

Access layer:
- CLI
- Operator App

These modules are separate on purpose. Extraction, persistence, review
assembly, and presentation answer different questions and should not blur into
one another.

# 2. Module Responsibilities

## 2.1 Lab

Lab is the exploratory surface.
It is where the project checks what is extractable, what looks stable, and what
seems worth promoting into production evidence.

Lab produces structured dumps and field catalogs.
It never writes production DB state.

Lab depends on a ComfyUI-capable environment because it observes live runtime
objects directly. That dependency is allowed to stay there because lab is
off-grid discovery tooling, not the production path.

## 2.2 Extractor

Extractor is the production entry point.
It receives live runtime objects directly from
generation: tensors, LoRA model objects, ClipVision outputs, prompts, and
settings.

Internally, Extractor converts ComfyUI-specific inputs into portable
primitives. Everything it emits downstream is clean Python with no ComfyUI
types attached.

Responsibilities:
- receive inputs from the live generation environment
- convert runtime-specific types into portable primitives
- emit one Evidence candidate record for one sample
- perform primitive measurements only

Extractor does not normalize, compare, or score.

## 2.3 Bouncer

Bouncer is the representability gate between observation and storage.
It validates incoming records against Evidence, resolves deduplication and
enrichment, and decides whether the record can enter persistent storage
cleanly.

Its purpose is not to judge the sample. Its purpose is to protect lineage,
identity, and storage truth.

Bouncer owns:
- structural validation
- duplicate handling
- enrichment of missing measurement slots
- extras separation before persistence

For exact invalid-wrapper and vital-field behavior, see
`05_VALIDITY_AND_DIAGNOSTICS.md`.

## 2.4 DataBank

DataBank is the persistence layer.
It stores Methods, Evals, Samples, the LoRA catalog, and the asset references
needed to retrieve non-JSON measurement material later.

DataBank is where the project remembers what happened.
Interpretation lives downstream in Synthesis and the review surfaces.

Architectural rules:
- SQL stays inside `databank/`
- the rest of the system talks to persistence through Treasurer
- backend-specific knowledge stays localized

DataBankHealth is the inspection and reporting surface for stored-record
quality. It belongs to the DataBank side of the boundary even when it is
surfaced through Command Center.

## 2.5 Review Assembly

Review assembly is the only opinionated production layer.
It reads stored Evidence and produces Synthesis: comparisons, aggregates, hero
packages, and review-ready payloads.

It owns:
- pair construction
- metric procedures and aggregation
- review payload assembly
- consumer-facing comparison packages

This is where stored facts become review surfaces.

## 2.6 Command Center

Command Center gathers durable control and reporting actions over the subsystem
interfaces the rest of the project already depends on.

It routes through the target module's own interface instead of reaching through
module walls directly.

Command Center owns:
- durable lookup and status actions
- review-payload regeneration
- workflow inventory and onboarding hooks
- diagnostics and health-facing runtime surfaces

It does not own the science.
It does not own persistence.
It does not become a second backend.

## 2.7 Access Layer

The access layer is how local users reach the system.

**CLI**
- precise and repeatable
- reliable when the app is not open
- routes through established Python surfaces

**Operator App**
- local interactive review and control surface
- opens the stored hierarchy and assembled review packages
- does not redefine the meaning of metrics or stored facts

Both talk through Command Center or review services rather than bypassing
system boundaries.

# 3. Core Flow

At a high level, one run moves through the system like this:

1. a generation run produces one sample
2. Extractor measures that sample and emits Evidence
3. Bouncer validates, enriches, and admits the record
4. DataBank stores the record and its asset references
5. Review assembly compares stored records and builds Synthesis
6. Command Center and the access layer expose that review surface locally

This split is what makes the project reusable. One stored sample can be
reviewed more than one way later without re-measuring or redefining the facts.

# 4. Architecture Invariants

- Only generative inputs determine object identity.
- Measurement instruments do not change sample identity and can be enriched
  post-hoc.
- DataBank remains semantically dumb persistence.
- Review assembly is the only opinionated production layer.
- Access layers format and expose structured outputs; they do not invent new
  science.
- Lab remains off-grid and does not write production DB state.
- Boundaries stay explicit so replacement and correction remain localized.
