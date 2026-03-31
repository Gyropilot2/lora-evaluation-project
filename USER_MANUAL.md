# User Manual

This manual is for people who want to run, inspect, and extend the open-source release of the LoRA Evaluation Project on their own machine.

It is written for a power-user or developer audience. You should already be comfortable with terminals, Python environments, ComfyUI, and reading or changing code when needed. This is not a consumer-facing setup where everything important is exposed through a few easy config knobs; meaningful use may require changing procedures, metrics, workflows, and local paths.

## What You Get Out Of The Box

Right after cloning the repo, installing dependencies, and opening the app, you already have:

- the Operator App local review interface
- a SQLite-backed local dataset with sample images and measurements
- batch commands for generating, rerunning, and replaying measurements
- a healthcheck that validates repo boundaries and contract integrity
- lightweight CLI commands for quick inspection

The included sample dataset currently contains 1 method, 5 evals, and 65 samples, so you can get oriented in the review surface before pointing the system at your own workflows and questions.

## Setup And First Launch

### 1. Put the repo in `custom_nodes`

This repo is a ComfyUI custom node package. It should live inside your ComfyUI `custom_nodes` folder as `lora-evaluation-project`.

If you are using Stability Matrix, a typical location is:

```text
StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\lora-evaluation-project
```

Clone it there:

```powershell
cd StabilityMatrix\Data\Packages\ComfyUI\custom_nodes
git clone https://github.com/Gyropilot2/lora-evaluation-project.git
cd lora-evaluation-project
```

### 2. What you need

- Windows with PowerShell and batch-file support
- Python 3.11+ recommended
- Node.js and npm
- a local ComfyUI setup, with this repo cloned into `custom_nodes`
- the models, LoRAs, and workflow assets you want to use
- any custom node packs those workflows depend on
- the `insightface` package plus a compatible face-analysis pack for the full face-aware review path

When multiple InsightFace packs are present, the current preference order is:

- `antelopev2`
- `buffalo_l`
- the first available pack if neither of those exists

The resolver looks in this order:

- a registered `insightface` model path from your ComfyUI environment
- `ComfyUI/models/insightface`

If InsightFace or its model pack is missing, extraction still continues, but `face_analysis` becomes unavailable. That means face-dependent review signals such as identity support, face detection confidence, and some face-gated metrics may stay null, drop out, or surface as unavailable in the app.

### 3. Install dependencies

From the repo root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r operator_app/backend/requirements.txt
cd operator_app/frontend
npm install
cd ../..
```

### 4. Open the app

From the repo root, run:

```powershell
operator_app\start_operator_app.bat
```

That launcher starts the backend, starts the frontend, waits for both to become reachable, and then opens the browser.

If you close the browser tab later, go to the frontend CLI window, type `o`, and press Enter to open it again without rerunning the whole launcher.

### 5. Important paths

Project paths live in `config/paths.json`.

Important defaults:

- database: `data/lora_eval.db`
- assets: `data/assets`
- logs: `data/logs`
- recipes: `data/recipes`
- temporary files: `data/tmp`
- LoRA root: `data/loras`
- workflow workspace: `config/workflows`
- workflow staging: `config/workflows/staging`

You do not need to change these paths to inspect the included sample data. You only need to edit `config/paths.json` when you want the batch commands to use your own LoRA library or a different local layout.

## First Session In Operator App

If everything is wired correctly, the left tree should open on the included sample dataset. That gives you a real method, several evals, and multiple strengths/seeds to inspect immediately.

In the app, the tree follows `method -> eval -> strength -> sample`. In plain terms: a `method` is the shared setup, an `eval` is the baseline or LoRA being tested within that setup, and a `sample` is one concrete run under a chosen seed and strength.

A good first pass is:

1. Open the only included method in the left tree.
2. Click a LoRA eval to focus it, then add it to comparison and do the same with the baseline so the center starts showing a real pair instead of a single record.
3. Try building the comparison set in more than one way: focus one item and toggle it into compare, add peers at the same level, or move down into strengths and samples when you want the comparison to become more specific.
4. Use the metric picker dropdown to choose which metric the center views should emphasize, then switch between image view, overview tables, trends, and metric inspection to see how that one question changes across the same data.
5. Watch the right rail while changing focus. The center can compare several things at once, but the rail stays anchored to the focused item and keeps its score, support facts, dropped state, and component metrics together.
6. Use the Compare Tray when it appears: click chips to refocus, remove individual items, or clear the whole set and start over.
7. Open `Logs / Ops` if you want a command-style local control surface without leaving the app.

The included data is there to orient you inside the app. The project becomes useful when you bring it your own review question, workflows, and LoRAs.

## Core Maintenance And Inspection Commands

For routine inspection, start in `Logs / Ops`. You can already run these there:

```text
health
summary 5
list-evals 10
list-loras 10
write-review-dump
```

If you would rather run them outside the app, use:

```powershell
python .dev/cli.py health
python .dev/cli.py summary --limit 5
python .dev/cli.py list-evals --limit 10
python .dev/cli.py list-loras --limit 10
```

The one command in this section that is still CLI-only is the repo healthcheck.

### CLI-only repo healthcheck

Run this after environment setup and again after architecture-sensitive changes:

```powershell
python .dev/dev_healthcheck.py
```

`dev_healthcheck.py` is the repo-side architectural linter. It runs checks for things like boundary doors, forbidden imports, hard-coded paths, diagnostics discipline, and contract or review-surface integrity. It is mainly for developers changing code in the repo, especially around architecture, contracts, diagnostics, or review assembly.

### Run the review export from a terminal

In `Logs / Ops`, the matching app command is `write-review-dump`. Outside the app, use:

```powershell
python -m command_center.review_payload
```

By default it writes to:

```text
data/exports/lora_review.json
```

Use this when you want a fresh machine-readable snapshot of the current review surface.

## Running New Samples And Replaying Measurements

This workflow path is more opinionated than it looks. In this project, the same workflow usually exists in three forms:

- a raw ComfyUI API export in `config/workflows/staging/`
- an onboarded workspace template in `config/workflows/`
- a stored `workflow_ref` attached to the Method created by a batch run

That split matters. The staging file is just a drop zone. The onboarded template is the reusable file that `batch run` reads. The stored `workflow_ref` is the copy that stays attached to a Method in the DB so that specific Method can be rerun later.

### 1. Start from the bundled example if you want a reference

One staged example ships with the repo already:

```text
config/workflows/staging/Flux2KleinBase_Lora_Extractor.json
```

Use it as a reference when building your own compatible workflow. Do not assume it will run unchanged on your machine.

It is there mainly to show the expected wiring, node placement, and sweep fields. It uses this project's nodes plus other custom nodes and preprocessors, so ComfyUI still needs every node pack used by that graph to exist on your machine. If ComfyUI opens it with missing-node errors, install those packs first.

### 2. Make or adapt a compatible workflow

A workflow can be valid in ComfyUI and still not be compatible with this project's batch flow. For the current onboarding and rerun path, the eval loop has to be explicit:

```text
base model -> eval LoRA loader -> LoraEvalSampleGuard -> sampler path -> image -> LoraEvalExtractor
                      ^                        ^
             LoRA strength input         seed input
```

For a workflow to onboard cleanly, keep these rules in mind:

- Export it as a **ComfyUI API-format JSON**. In ComfyUI, use **Save (API Format)**, not the normal workflow save.
- Include exactly one **`LoraEvalSampleGuard`** node and one **`LoraEvalExtractor`** node.
- Route the tested LoRA through one clear eval LoRA loader. The current auto-discovery expects **`LoraLoaderModelOnly`**.
- Feed the **seed** into `LoraEvalSampleGuard` from a linked **`PrimitiveInt`** node.
- Feed the **LoRA strength** into the eval LoRA loader from a linked **`PrimitiveFloat`** node.
- Keep the base-model path for that eval loop unambiguous.
- Wire the Extractor's required inputs from the real run: `model`, `vae`, `positive`, `negative`, `latent_image`, `image`, `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, and `denoise`.
- Masks, auxiliary maps, pose inputs, and CLIP vision inputs are optional, but leaving them disconnected means those evidence surfaces will simply be missing later.

Those class names are part of the current v1 workflow discovery path. Different node types may still work in ComfyUI, but they will not onboard cleanly unless you mark the eval loop yourself.

### 3. Understand what onboarding means

`workflow_onboard` does not generate samples. It converts a raw ComfyUI API export into a reusable project template.

In practice that means:

- it reads the raw JSON from `config/workflows/staging/`
- it makes sure the LoRA name, LoRA strength, and seed sweep slots are explicitly marked
- it writes the processed template to `config/workflows/`

After onboarding, the workspace copy is the file that `batch run` uses. The raw staged file can stay where it is for reference, or you can replace it later with a newer export. The system does not keep reading from staging after onboarding.

For a simple single-LoRA workflow, `workflow_onboard` can usually mark the sweep fields for you automatically. If it fails, the tool could not figure out by itself which `lora_name`, strength, and seed fields belong to the project's eval loop. In that case, open the workflow in ComfyUI, place these marker values directly into those fields, then export the API workflow again:

- `__LEP_EVAL_LORA__`
- `__LEP_EVAL_STRENGTH__`
- `__LEP_EVAL_SEED__`

### 4. Onboard the workflow

Place the raw API export in:

```text
config/workflows/staging/
```

Then run:

```powershell
python -m command_center.workflow_onboard config/workflows/staging/Flux2KleinBase_Lora_Extractor.json
```

If you keep the default name, that produces:

```text
config/workflows/Flux2KleinBase_Lora_Extractor.json
```

That onboarded file is now the reusable template for batch generation in this project.

### 5. Run a new batch from the onboarded template

Use `run` when you want the project to sweep LoRAs, strengths, and seeds automatically.

Example:

```powershell
python -m command_center.batch run --workflow Flux2KleinBase_Lora_Extractor --loras-dir "MyLoras" --strengths 0.5 1.0 --seeds 0 1 2
```

Here `--workflow Flux2KleinBase_Lora_Extractor` means:

```text
config/workflows/Flux2KleinBase_Lora_Extractor.json
```

During a real batch run, the batch runner loads that onboarded template, fills in the LoRA name, strength, and seed for each concrete run, submits those materialized workflows to ComfyUI, and stores the tagged workflow declaration in the DB as a `workflow_ref` on the resulting Method. That stored copy is what makes DB-backed `rerun` possible later.

If you edit the workspace template later, that does not rewrite older Methods. They keep the `workflow_ref` that was stored when they were created.

If you want to preview what would be submitted before sending work to ComfyUI, add:

```powershell
--dry-run
```

### 6. Or prove the workflow manually in ComfyUI first

If batch feels like too much at first, you can start by running the workflow directly in ComfyUI.

- Load the workflow in ComfyUI and make sure every node pack it uses is installed locally.
- Set the LoRA name, strength, and seed in the graph itself.
- Queue the prompt normally in ComfyUI.
- As long as `LoraEvalSampleGuard` and `LoraEvalExtractor` are wired correctly, the run will still be measured and stored.

This is a good way to prove the graph works before automating it. The tradeoff is that this manual path does not give you the full workflow lifecycle automatically: you are not using the batch sweep, and you should not assume the resulting Method will have a stored `workflow_ref` that can be rerun cleanly later.

### 7. Rerun a stored Method

Use `rerun` when you already have a Method in the DB and want to replay that Method's stored workflow with a selected set of LoRAs, strengths, or seeds.

To get a `method_hash`, focus the Method in the Operator App and read `Method id` in the right-hand facts panel. If you prefer a terminal, `summary` and `list-evals` in `Logs / Ops` or `.dev/cli.py` also show `method_hash` values.

Example:

```powershell
python -m command_center.batch rerun --method-hash <method_hash> --loras-dir "MyLoras" --strengths 1.0 --seeds 0 1
```

This does not read from the raw staging file. It does not need the workspace template either. It reads the `workflow_ref` stored on that Method in the DB, uses that as the template, and materializes fresh runs from it.

If `rerun` says the Method has no stored workflow, that Method predates workflow-backed reruns or came from a path that did not attach one. In that case it cannot be replayed from the DB alone; go back to a workflow template and run it as a new batch instead.

### 8. Replay or backfill measurements

Use `replay` when you want to add missing measurement paths to existing samples without running the full generation step again.

Example:

```powershell
python -m command_center.batch replay
```

This is the right tool for backfilling measurement surfaces into samples that already exist in the DB.

### 9. After a run

Once new work lands in the DB, the usual next steps are:

```powershell
python -m command_center.review_payload
```

Then open or refresh the Operator App and inspect the new method/evals.

## Data Repairs And Cleanup

These are the two maintenance scripts you are most likely to need.

### Remove a Method tree

Use this to remove a Method together with all of its child Evals and Samples.

Dry-run first:

```powershell
python databank/operations/purge_method.py --list
python databank/operations/purge_method.py --method-hash <hash>
```

Actually delete:

```powershell
python databank/operations/purge_method.py --method-hash <hash> --delete
```

Use it when you want to clear one Method out of the DB before running or reviewing something else.

### Remove unreferenced asset files

`purge_method.py` removes DB records, but it does not delete asset files from `data/assets`. If you also want to remove files that are no longer referenced by the DB, use:

```powershell
python databank/operations/prune_unreferenced_blobs.py
python databank/operations/prune_unreferenced_blobs.py --delete
```

The default run is a report only. Add `--delete` only after checking the dry-run summary.

## Where To Extend Metrics And Workflows

### Metrics and review meaning

If you want to understand which metrics exist, how they are labeled, and how they are surfaced in review:

- `contracts/metrics_registry.py`
- `contracts/procedures_registry.py`

If you want the conceptual distinction between stored facts and interpreted comparison:

- [.docs/05_VALIDITY_AND_DIAGNOSTICS.md](.docs/05_VALIDITY_AND_DIAGNOSTICS.md)
- [.docs/06_EVIDENCE_REFERENCE.md](.docs/06_EVIDENCE_REFERENCE.md)
- [.docs/07_SYNTHESIS_PROVISIONAL.md](.docs/07_SYNTHESIS_PROVISIONAL.md)
- [.docs/08_ASSET_AND_VALUEREF_POLICY.md](.docs/08_ASSET_AND_VALUEREF_POLICY.md)

If you want exploratory reference material that is useful while probing or extending fields:

- [.docs/lab](.docs/lab)

### Workflows and local paths

If you want to add or test new workflows, work through:

- `config/workflows/`
- `config/workflows/staging/`
- `python -m command_center.workflow_onboard ...`

If batch commands cannot find your LoRAs, check `config/paths.json` first. The shipped default is `data/loras`, which is useful for the repo but may not match your own local library.

## Troubleshooting

### ComfyUI does not see the project

- Confirm the repo lives at `ComfyUI\custom_nodes\lora-evaluation-project`.
- If you are using Stability Matrix, that usually means `StabilityMatrix\Data\Packages\ComfyUI\custom_nodes\lora-evaluation-project`.
- If the repo was cloned somewhere else, move it or clone it again into `custom_nodes`, then reinstall dependencies from that repo root.

### The Operator App does not open

- Confirm both Python install commands completed successfully.
- Confirm `npm install` ran in `operator_app/frontend`.
- Try running backend and frontend separately so you can see which side is failing.
- Check that `operator_app/runtime_config.json` still matches the backend/frontend ports you expect.

### The app opens, but the tree is empty

- Confirm `data/lora_eval.db` exists.
- Run `python .dev/cli.py summary --limit 5` and make sure the DB still reports data.

### Batch commands cannot find your LoRAs

- Open `config/paths.json`.
- Check `lora_root`.
- Make sure the folder used in `--loras-dir` actually exists under that root.

### Generation or replay commands fail immediately

- Make sure your local ComfyUI setup is running and reachable.
- Confirm the workflow template you selected is present in `config/workflows/`.
- Try the same command with `--dry-run` first to verify the intended submission.

### The healthcheck fails

- Read the exact violation text from `python .dev/dev_healthcheck.py`.
- Most failures here are boundary or contract issues, not generic test failures.
- Fix the reported violation first, then rerun the healthcheck.
