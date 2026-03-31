# Operator App

The Operator App is the local review interface for the LoRA Evaluation Project.

It is split into:

- `backend/`: FastAPI services
- `frontend/`: React + Vite UI

## Start it

From the repo root:

```powershell
operator_app\start_operator_app.bat
```

Default local URLs:

- frontend: `http://127.0.0.1:4173`
- backend: `http://127.0.0.1:8765`

Runtime host/port settings live in `runtime_config.json`.

## What it is for

The app is the main local review surface for:

- browsing methods, evals, strengths, and samples
- comparing related items
- inspecting metrics and diagnostics
- using the `Logs / Ops` workspace for local operator actions

For full setup and workflow instructions, see the repo-level
[USER_MANUAL.md](../USER_MANUAL.md).
