# 09 - Operator App and Command Center

## What The Operator App Is

A local React + FastAPI application for browsing, comparing, and acting on
evaluation data.

---

## Architectural Position

```
databank/ + extractor/ + bouncer/    ->  core system
contracts/ + command_center/         ->  Synthesis layer + control surface
operator_app/                        ->  local interactive review surface
```

Command Center is the control surface the app, CLI, and local scripts use for
summary, lookup, health, workflow, and review-export actions. The Operator App
consumes those Python surfaces; it does not become a second backend with its
own truth rules.

### Synthesis

Synthesis is assembled by the review layer and surfaced in the Operator App.
The assembly chain is:

- `contracts/aggregation.py` - reduction functions and pair math
- `contracts/procedures_registry.py` - per-sample interpretation procedures
  (gating, drop-state, attribution)
- `contracts/recipe.py` - which metrics are featured and in what order
- `command_center/review_builder.py` - assembles Synthesis from Evidence pairs;
  the heavy computation layer
- `command_center/review_payload.py` - shapes the assembled payload for
  consumers (app backend, review JSON export)

The frontend consumes this Synthesis. It does not recompute it in TypeScript.

---

## The App Surfaces

### Top Bar

Persistent across all views. Shows:
- **App name** ("LEP Operator App")
- **Method count** and **sample count** - live DB totals
- **Workspace tabs** - "Workbench" and "Logs / Ops" (see below)
- **Loading bar** - two-phase progress: method loading (first 50%), asset
  enrichment (second 50%)
- **Context badge** - current focus level or active comparison count

---

### Workbench (three-pane layout)

The main review surface. Three persistent panes side by side.

#### Left - Tree Pane

Navigation and selection. Hierarchy: Method -> Eval -> Strength -> Sample.

- **Search box** - filters visible methods and evals by name/hash
- **Method rows** - each shows the method label (base model + prompt hash).
  Click to focus. Expand to reveal evals.
- **Eval rows** - each shows the LoRA name (or "baseline"). Click to focus.
  Expand to reveal strengths.
- **Strength rows** - each shows the strength value and sample count. Click
  to focus. Expand to reveal individual samples.
- **Sample rows** - each shows seed + strength. Click to focus.
- **Disclosure triangles** - expand/collapse each level
- **Comparison checkboxes** - add any node to the active comparison set.
  Same-level-only; the tree enforces the comparison law.
- **Context menu actions** on nodes:
  - "Select all evals for this method"
  - "Select all strengths for this eval"
  - "Select all samples for this eval / this strength"
  - "Select same-seed peers" - selects all samples sharing the focused seed
  - "Select same-strength peers" - selects all samples at the same strength

#### Center - Work Stage

Content changes based on what is focused and whether a comparison set is active.

**Center top bar** (persistent):
- **Scope line** - breadcrumb showing the current focus path (method -> eval ->
  strength -> sample)
- **Title** - name of the focused entity
- **Metric picker dropdown** - selects which metric to highlight across the
  center views; only visible when inspection options exist for the current focus
- **Prompt hint** - first portion of the prompt used by the focused method,
  shown as a readable label

**Compare Tray** - appears when a comparison set is active:
- Chips showing each selected entity by name
- Click a chip to shift focus to that entity
- x on each chip removes it from the set
- "Clear" button clears the whole set

**Main canvas content** - one of:

- **Overview section** - when focus is on a method, eval, or strength. Shows
  summary tables of child entities with their hero metric scores. Clicking a
  row focuses that entity.
- **Image stage** - when focus is on a sample or a comparison set of samples.
  Shows the generated image alongside the baseline image. Hovering a metric
  in the rail previews the corresponding aux asset (depth, normal, edge, mask,
  luminance) over the image.
- **Trend section** - when focus is on an eval or strength. Shows strength-level
  trend charts for each hero metric across all seeds.
- **Metric inspection panel** - when a specific metric is selected via the
  metric picker. Shows a per-sample breakdown of that metric's value across
  the comparison set or current eval.

#### Right - Metric Rail (Inspector)

Always anchored to the focused item, even when the center is showing a
comparison set.

**Focused item identity block:**
- Focus title (entity name)
- Scope line (breadcrumb)
- Copy button - copies the full rail content to clipboard

**Focused Facts** (sample-level only):
- Sample id, Seed, Strength, Eval name - stable identifiers for the focused
  sample

**Hero Metric Groups** - one group per active hero metric (Identity, Pose,
Background, Composition, Global Semantic, Clothing):
- **Score** - the primary synthesized value for this hero
- **Reliability** - support channel (detector confidence, joint retention ratio,
  etc.). Metric-specific unit.
- **Dropped** - whether the metric pipeline produced no score at all ("yes" /
  "no" / blank when undefined)
- **Selection** - which specific metric key won as the hero score source
- **Components** - individual sub-metrics that contribute to the hero package
  (e.g. Face Cosine Distance, SigLIP Face-region Drift, ViT-L/14 Face-region
  Drift under Identity)
- **Peer Checks** - related package variants shown for comparison

Metric labels show a tooltip on hover (sourced from `operator_app/backend/review_meta.py`
`FIELD_META`) explaining what the metric measures.

**Remaining metric rows** - flat dump of all other computed metrics not
promoted into a hero group. Always available below the hero section.

**Hover asset preview** - hovering certain metric rows (depth, normal, edge,
mask, luminance) previews the corresponding asset over the currently focused
image in the center stage.

---

### Logs / Ops Workspace

Activated via the "Logs / Ops" tab. Replaces the Workbench entirely while active.

#### Preset Rail (left column)

Grouped command presets. Each preset is either:
- **Run** - executes immediately and shows formatted output
- **Prefill** - fills the command console with a template the user completes
  before running

Preset groups:
- **Command Center quick run** - Health, Errors, Summary, Review dump, Batch
  replay, Workflows, Onboard, LoRAs, Evals
- **Command Center lookup** - Method, Eval, Sample (prefill lookups), Help

#### Command Console (center)

Text input for manual commands. Supports the same grammar as the CLI. Submit
runs the command against the FastAPI backend and appends the result to the
history panel.

#### History Panel (right / main area)

Chronological log of commands run and their formatted output. Each entry shows:
- Command text
- Timestamp
- Formatted result (tables, key-value blocks, or raw text depending on command)

---

## Command Center Boundaries

### What CC Owns

Command Center exposes the app-safe and CLI-safe control surface. Its current
public actions are:

- `health()` - DB quality scorecard and diagnostics summary
- `errors(limit)` - recent WARN/ERROR/FATAL records from diagnostics log
- `summary(limit)` - counts and recent evals/samples overview
- `list_loras(limit)` - LoRA inventory with DB coverage
- `list_evals(limit, method, lora)` - eval listing with filters
- `get_method(hash)`, `get_eval(hash)`, `get_sample(hash)` - record lookup
- `run_batch(...)` - batch replay orchestration; delegates to `command_center/batch.py`
- `write_review_dump()` - regenerate the review JSON export
- `list_workflows()` - workspace and staging workflow inventory
- `onboard_workflow(filename)` - onboard a raw workflow from staging

This is the surface mirrored by
`operator_app/backend/app/routes/command_center_api.py`.
The FastAPI routes stay thin and delegate back to `command_center` rather than
redefining the same behavior in the app backend.

### What CC Does Not Own

- **Batch run orchestration** - `command_center/batch.py` is an independent
  process that can run for hours. The app may configure and optionally launch
  it; CC does not manage it.
- **One-off maintenance scripts** - live in `databank/operations/`, run
  explicitly.

### CC Growth Rule

Before adding a new function to `command_center/__init__.py`:

1. The action must be something the app, CLI, or local workflow will invoke repeatedly and reliably.
2. The action must route through Treasurer or other established subsystem interfaces.
3. The function must be registered in all three interface layers (Python API,
   CLI, backend route) or explicitly noted as Python-API-only with a reason.

### CC Interface Layers

- **Python API** - `cc.method()` calls in `command_center/__init__.py`
- **CLI** - `python .dev/cli.py <command>`. Thin wrapper.
- **FastAPI routes** - `/api/cc/...`. Same surface exposed to the frontend.

The expectation is not that every layer looks the same.
The expectation is that they mean the same thing.

---

## Review Assembly Layer

- `contracts/review_surface.py` - shared review metadata and hero roster helpers
- `contracts/review_transport.py` - canonical settled review transport shapes
- `contracts/recipe.py` - consumer policy: which metrics are featured and in what order
- `contracts/aggregation.py` - lower-level review computation primitives
- `command_center/review_builder.py` - heavy review-data assembly from project records
- `command_center/review_payload.py` - app-facing review payload shaping

---

## Core UX Model

### Focus vs Selection

**Focus** - clicking a node updates the workspace to show that item. Focus is
the persistent anchor. The right metric rail always shows the focused item,
even while the center displays a comparison set.

**Selection** - builds a comparison set. The Compare Tray shows the current set;
chips can be removed individually or cleared at once. Focus does not dismantle
the current comparison set.

### Comparison Law

Same-level only: method-to-method, eval-to-eval, strength-to-strength, sample-to-sample.
Mixed levels are not allowed and the tree enforces this.

### Right Rail Ownership

The rail shows the focused item's individual Synthesis - hero groups, component
metrics, raw flat dump. It stays anchored to one item while the center stage is
free to compare many at once.
