# Lora Evaluation Project - Constitution

Generated: 2026-02-17
See also: `01_GLOSSARY.md` for canonical terms/enums.

## Core identity
A measurement instrument.  
Not an aesthetic oracle.  
Not a vibes engine.

## Generative inputs vs measurement instruments

**Generative inputs** causally determine the generated output.
They are the only legal material for method identity (method.hash) and sample identity (sample.hash).
  Examples: base_model, vae, conditioning (positive + negative), steps, cfg, sampler, scheduler,
  denoise, latent dimensions - these form **method.hash**.
  Additionally: lora identity forms **eval.hash**; seed and lora_strength form **sample.hash**.

**LoRA catalog (4th DB table):** The `LoRAs` table is a flat, SQL-queryable catalog keyed by
`lora_hash`. It stores LoRA metadata - rank, alpha, target_blocks, affects_text_encoder,
raw_metadata - extracted from the .safetensors header at Extractor time. The Evals table
references it via `lora_hash` (logical FK). This is metadata, not a generative input; it does
not participate in any hash computation. The three-level hierarchy (Method -> Eval -> Sample)
remains the canonical identity structure.

**Measurement instruments** observe and analyze the output after generation.
They do not affect what was generated. They are enrichable post-hoc.
  Examples: ClipVision models, InsightFace, segmentation masks, auxiliary comparison images

This distinction governs two contracts:
1. Only generative inputs determine object identity. Two samples that differ only in
   measurement instrument configuration are the same sample - they produced the same output.
2. The enrichment pattern: if a lean run (no instruments) and a rich run (instruments
   connected) share the same sample.hash, the Bouncer enriches the existing sample's
   measurement slots rather than discarding the richer record.

## Boundary Door Law (no wall-reaching)
Crossing a module boundary requires a named interface ("a door").  
The point is not ceremony. The point is to keep ownership clear, make
replacement localized, and stop meaning from leaking across layers.

Practical rules:
- Procedures do not traverse Evidence dicts; they use Context getters/loaders only.
- Non-DataBank modules do not execute SQL; they call Treasurer methods only.
- Non-asset modules do not `open()` blobs directly; they call `assets.*` methods only.
- Non-diagnostics modules do not format log lines; they emit structured diagnostic records via Command Center.
- Diagnostic `code` values are stable identifiers (not free text). Treat them as enum-like and keep them centralized.
- The presentation layer does not compute canonical math; it consumes assembled Synthesis and only formats/presents.

Enforcement:
- Violations are treated as architectural bugs.
- Block by dev_healthcheck where feasible; otherwise block in review.
- For known first-party modules, dev_healthcheck is deny-by-default: a
  cross-module import is illegal unless it passes through an explicitly
  allowlisted boundary door.
- The only open `databank/` door for non-`databank` modules is Treasurer.

## Truth rules (non-negotiable)
- Truth-or-crash, implemented as: **store-invalid + explicit errors**; never fake validity.
- No opinions in Extractor or DB.
- Facts before interpretation.
- Transparency over mystique.
- Reproducibility over cleverness.

## Cultural rules
- Explicit > implicit.
- Stability before expansion.
- Small controlled increments.
- Structural integrity over speed.

## Practices (default tradeoffs)
Primary goal: a maintainable system with long-term integrity. Clean, reviewable code beats clever speed tricks.

Defaults:
- Prefer correctness, clarity, and reproducibility over throughput.
- Prefer recomputation over caching unless profiling proves a real bottleneck.
- Any cache must be explicitly labeled "cache", must be safely regenerable, and must never pollute Evidence/DB truth.
- Performance optimizations are allowed only behind stable interfaces (Context/DataBank/Assets) so they do not leak coupling.
- Version identities honestly: procedure changes -> bump procedure/recipe ids; semantic metric changes -> new metric keys.

## Doc correction protocol

Docs are authoritative, but they are still maintained artifacts.
They need corrections when implementation reality exposes a genuine gap,
ambiguity, or missing constraint.

**Allowed corrections:**
- Adding a field, case, or constraint the docs omitted but implementation genuinely requires
- Clarifying language that created two valid interpretations
- Extending a list (e.g. `vital_fields`) when the omission would cause architectural incoherence or make safe implementation impossible
- Documenting a newly discovered edge case as a separate, clearly-labelled section (not rewriting the primary rules)

**Not allowed:**
- Weakening a constraint because it was inconvenient
- Editing retroactively to match a bad decision ("I did Y, now I'll say Y was always right")
- Removing or softening hard stops, boundary laws, or refusal rules

**How to handle a correction:**
- Keep changes minimal and surgical - fix the gap, do not rewrite the section
- Note the change and reason in a comment in the doc or in the commit message
- Flag residual uncertainty clearly so reviewers can assess it

**The test:** could you show this diff and honestly say "the doc had a gap that
made correct implementation impossible or incoherent, so I corrected it"?
If not, the change probably belongs in code or review discussion instead.

## Architectural cheat ledger
Temporary scaffolding is allowed when it accelerates discovery without
pretending to be settled architecture.  
However, **every intentional rule-break must be recorded** so it does not stay
in the repo as unnamed debt.

Preferred mechanism (in-code placeholder warning):
- Use Command Center to emit a structured diagnostic:
  - `code`: `ARCH.CHEAT`
  - `where`: file/class/function
  - `msg`: what was done
  - `ctx`: why, and the cleanup condition ("remove when X exists")

Fallback mechanism (ledger file):
- If an in-code warning is not feasible, record it in a single ledger (e.g., `.docs/CHEAT_LEDGER.md`) with:
  - File/Class/Function
  - What rule was broken
  - Why it was justified
  - Cleanup condition

Rule:
- Unlogged scaffolding is treated as an architectural bug.

## Avoid
- Hidden weights.
- Undocumented normalization.
- Cross-layer shortcuts.
- Global mutable state.
- Premature optimization.
- Import stubs.
