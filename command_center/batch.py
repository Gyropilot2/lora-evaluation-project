"""
command_center/batch.py — Unified batch orchestrator for LoRA evaluation.

Three subcommands, one engine:

  run     Generate new samples from a workflow file.
  rerun   Re-run a known Method (loads its stored workflow from DB).
  replay  Backfill measurements into existing samples (no KSampler).

The ComfyUI HTTP transport, dedup logic, and result logging are shared
across all three modes.  Completed work is never re-submitted — each mode
skips samples / measurements that already exist in the DB.

USAGE:
  python -m command_center.batch run   --workflow flux_simple
                                       --loras "Flux\\Core Physics.safetensors" "Flux\\BoringLife.safetensors"
                                       --strengths 0.5 1.0
                                       --seeds 0 1 2 3 4

  python -m command_center.batch run   --workflow flux_simple
                                       --loras-dir "Flux Klein 9B Base"
                                       --strengths 1.0
                                       --seeds 0 1 2 3 4

  python -m command_center.batch rerun --method-hash <hash>
                                       --loras "Flux\\Core Physics.safetensors"
                                       --strengths 0.5 1.0
                                       --seeds 0 1 2 3 4

  python -m command_center.batch replay --workflow config/workflow_replay_api.json
                                        [--paths pose_evidence.openpose_body,pose_evidence.dw_body]
                                        [--sample-hash <hash>] [--limit 20]
                                        [--include-dirty] [--force]

All subcommands support --dry-run and --poll SECS.

ADDING A NEW WORKFLOW (run / rerun — one-time):
  1. In ComfyUI: export your workflow via Save (API Format).
  2. Drop the raw JSON in config/workflows/staging/.
  3. Run: python -m command_center.workflow_onboard config/workflows/staging/<file>.json
  4. Use: python batch.py run --workflow <name> ...

AFTER A RUN:
  python -m command_center.review_payload
  (or open the Operator App)
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from command_center import comfyui_client as _http
from command_center.batch_replay import (
    build_replay_plan,
    parse_requested_paths,
    patch_replay_workflow,
)
from command_center.workflow_batch import (
    KnownEvalIndex,
    RunSpec,
    build_known_eval_index,
    build_run_matrix,
    find_existing_sample_hash_for_run,
    find_method_by_workflow,
    load_workflow_from_method,
    load_workflow_template,
    materialize_run_workflow,
    store_workflow_ref,
    workflow_ref_json as serialize_workflow_ref_json,
)
from core.diagnostics import emit
from core.paths import get_path
from databank.treasurer import open_treasurer


# ---------------------------------------------------------------------------
# Shared run-loop (used by both `run` and `rerun`)
# ---------------------------------------------------------------------------

def _run_loop(
    template: dict[str, Any],
    runs: list[RunSpec],
    *,
    dry_run: bool,
    poll: float,
) -> int:
    """Execute the generate-new-samples loop and return exit code.

    Handles DB dedup, workflow materialization, ComfyUI submission, and
    progress reporting.  The template must already be inline-tagged.
    """
    workflow_ref_json: str | None = None
    known_evals: KnownEvalIndex | None = None
    treasurer = None

    try:
        treasurer = open_treasurer(event_emitter=emit, read_only=dry_run)
        existing_method = find_method_by_workflow(treasurer, template)
        if existing_method is not None:
            method_hash = existing_method.get("method_hash", "<missing>")
            sys.stdout.write(f"Known workflow-backed Method: {method_hash}\n")
            if isinstance(method_hash, str) and method_hash != "<missing>":
                known_evals = build_known_eval_index(treasurer, method_hash)
        if not dry_run:
            try:
                workflow_ref = store_workflow_ref(treasurer, template)
                workflow_ref_json = serialize_workflow_ref_json(workflow_ref)
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARN] Workflow declaration storage failed; continuing without workflow_ref: {exc}\n"
                )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[WARN] Workflow declaration lookup failed; continuing without workflow_ref reuse: {exc}\n"
        )

    n_baseline = sum(1 for r in runs if r.is_baseline)
    n_lora     = len(runs) - n_baseline
    sys.stdout.write(
        f"Run plan: {len(runs)} total  "
        f"({n_baseline} baselines + {n_lora} LoRA runs)\n\n"
    )

    if not dry_run and not _http.check_alive():
        if treasurer is not None:
            treasurer.close()
        sys.stderr.write(
            f"[ERROR] Cannot reach ComfyUI at {_http.COMFYUI_HOST}\n"
            "        Make sure ComfyUI is running before starting the batch.\n"
        )
        return 1

    client_id = str(uuid.uuid4())
    errors: list[str] = []
    lora_file_hash_cache: dict[str, str | None] = {}
    externally_skipped = 0

    try:
        for i, run in enumerate(runs, 1):
            if run.is_baseline:
                tag = f"BASELINE{' ' * 32}seed={run.seed}"
            else:
                stem = Path(run.lora_name or "").stem
                tag  = f"{stem:<30}  s={run.strength:<4}  seed={run.seed}"

            if treasurer is not None and known_evals is not None:
                existing_hash = find_existing_sample_hash_for_run(
                    treasurer,
                    known_evals,
                    run,
                    lora_file_hash_cache=lora_file_hash_cache,
                )
                if existing_hash is not None:
                    externally_skipped += 1
                    sys.stdout.write(
                        f"[{i:3d}/{len(runs)}] {tag} ... SKIP (existing {existing_hash[:8]})\n"
                    )
                    continue

            try:
                wf = materialize_run_workflow(
                    template,
                    lora_name=run.lora_name,
                    strength=run.strength,
                    seed=run.seed,
                    workflow_ref_json=workflow_ref_json,
                )
            except Exception as exc:  # noqa: BLE001
                sys.stdout.write(f"[{i:3d}/{len(runs)}] {tag} ... FAILED: {exc}\n")
                errors.append(f"[{i}] {tag}: {exc}")
                continue

            if dry_run:
                sys.stdout.write(f"  [{i:3d}/{len(runs)}] {tag}\n")
                continue

            sys.stdout.write(f"[{i:3d}/{len(runs)}] {tag} ... ")
            sys.stdout.flush()

            try:
                prompt_id = _http.queue_prompt(wf, client_id)
                entry = _http.wait_for_completion(prompt_id, poll_secs=poll)
                ok, status_text = _http.summarize_status(entry)
                if ok:
                    sys.stdout.write(f"OK ({prompt_id[:8]})\n")
                else:
                    sys.stdout.write(f"FAILED: {status_text}\n")
                    errors.append(f"[{i}] {tag}: {status_text}")
            except Exception as exc:  # noqa: BLE001
                sys.stdout.write(f"FAILED: {exc}\n")
                errors.append(f"[{i}] {tag}: {exc}")
    finally:
        if treasurer is not None:
            treasurer.close()

    sys.stdout.write("\n")
    if externally_skipped:
        sys.stdout.write(f"[INFO] {externally_skipped} run(s) skipped via pre-dedup.\n")
    if dry_run:
        sys.stdout.write("(dry-run — nothing sent to ComfyUI)\n")
        return 0 if not errors else 1
    if errors:
        sys.stdout.write(f"[WARN] {len(errors)} run(s) failed:\n")
        for e in errors:
            sys.stdout.write(f"  {e}\n")
        sys.stdout.write(
            "Re-run to retry — completed samples are skipped automatically.\n"
        )
        return 1

    sys.stdout.write(
        "All runs complete.\n"
        "Review results:\n"
        "  python -m command_center.review_payload\n"
    )
    return 0


# ---------------------------------------------------------------------------
# `run` subcommand
# ---------------------------------------------------------------------------

def _resolve_workflow_path(arg: str) -> tuple[Path, str]:
    """Resolve --workflow arg to a Path and a display source label.

    Resolution order:
    1. Bare name (e.g. "flux_simple") → workflows_root/flux_simple.json
    2. Direct path (absolute or relative) → as-is

    Returns (resolved_path, source_label) where source_label is "workspace" or "file".
    Raises FileNotFoundError if neither candidate exists.
    """
    workspace_candidate = get_path("workflows_root") / f"{arg}.json"
    if workspace_candidate.exists():
        return workspace_candidate, "workspace"

    direct_candidate = Path(arg)
    if direct_candidate.exists():
        return direct_candidate, "file"

    raise FileNotFoundError(
        f"Workflow not found.\n"
        f"  Tried workspace: {workspace_candidate}\n"
        f"  Tried direct:    {direct_candidate}\n"
        f"Onboard a workflow first:  python -m command_center.workflow_onboard <raw_file>"
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Generate new samples from a workflow file."""
    try:
        workflow_path, source = _resolve_workflow_path(args.workflow)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[ERROR] {exc}\n")
        return 1

    sys.stdout.write(f"Loading from {source}: {workflow_path}\n")

    try:
        template = load_workflow_template(workflow_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[ERROR] Workflow template load/tag failed: {exc}\n"
            "        Use an inline-tagged workflow or a simple export that v1 auto-discovery can understand.\n"
        )
        return 1

    if args.loras_dir:
        try:
            loras = _loras_from_dir(args.loras_dir)
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"[ERROR] --loras-dir: {exc}\n")
            return 1
    else:
        loras = list(args.loras or [])
    strengths: list[float] = list(args.strengths or [])
    seeds: list[int] = [int(s) for s in (args.seeds or [])]

    if not loras:
        sys.stderr.write(
            "[ERROR] No LoRAs specified.  Use --loras <name> [<name> ...] "
            "or --loras-dir <directory>.\n"
        )
        return 1
    if not strengths:
        sys.stderr.write(
            "[ERROR] No strengths specified.  Use --strengths <value> [<value> ...].\n"
        )
        return 1
    if not seeds:
        sys.stderr.write(
            "[ERROR] No seeds specified.  Use --seeds <int> [<int> ...].\n"
        )
        return 1

    runs = build_run_matrix(loras, strengths, seeds)
    sys.stdout.write(
        f"  LoRAs:     {loras}\n"
        f"  Strengths: {strengths}\n"
        f"  Seeds:     {seeds}\n\n"
    )
    return _run_loop(template, runs, dry_run=args.dry_run, poll=args.poll)


# ---------------------------------------------------------------------------
# `rerun` subcommand
# ---------------------------------------------------------------------------

def cmd_rerun(args: argparse.Namespace) -> int:
    """Re-run a known Method using its stored workflow from DB."""
    method_hash: str = args.method_hash

    if args.loras_dir:
        try:
            loras = _loras_from_dir(args.loras_dir)
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"[ERROR] --loras-dir: {exc}\n")
            return 1
    else:
        loras = list(args.loras or [])
    strengths: list[float] = list(args.strengths or [])
    seeds: list[int] = [int(s) for s in (args.seeds or [])]

    if not loras:
        sys.stderr.write(
            "[ERROR] No LoRAs specified.  Use --loras <name> [<name> ...] "
            "or --loras-dir <directory>.\n"
        )
        return 1
    if not strengths:
        sys.stderr.write(
            "[ERROR] No strengths specified.  Use --strengths <value> [<value> ...].\n"
        )
        return 1
    if not seeds:
        sys.stderr.write(
            "[ERROR] No seeds specified.  Use --seeds <int> (repeatable).\n"
        )
        return 1

    # Load the stored workflow from DB
    # ARCH.CHEAT: loading from DB for run logistics is the wrong permanent direction.
    # The workspace should be the run source; the DB copy is audit trail only.
    emit(
        "WARN", "ARCH.CHEAT", "batch.cmd_rerun",
        "loading workflow from DB for run logistics",
        why="DB-stored workflow is audit trail; workspace should be the run source",
        cleanup="replace DB retrieval with workspace name lookup once Methods have "
                "corresponding workspace files (4.13.c settled the lookup path)",
    )
    try:
        treasurer = open_treasurer(event_emitter=emit, read_only=True)
        try:
            template = load_workflow_from_method(treasurer, method_hash)
        finally:
            treasurer.close()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[ERROR] Could not open DB to load workflow: {exc}\n")
        return 1

    if template is None:
        sys.stderr.write(
            f"[ERROR] No stored workflow found for Method {method_hash}.\n"
            "        This Method predates workflow-backed reruns.\n"
            "        Use `batch.py run` to generate new samples from a workflow file.\n"
        )
        return 1

    sys.stdout.write(f"Rerun Method: {method_hash}\n")
    runs = build_run_matrix(loras, strengths, seeds)
    sys.stdout.write(
        f"  LoRAs:     {loras}\n"
        f"  Strengths: {strengths}\n"
        f"  Seeds:     {seeds}\n\n"
    )
    return _run_loop(template, runs, dry_run=args.dry_run, poll=args.poll)


# ---------------------------------------------------------------------------
# `replay` subcommand
# ---------------------------------------------------------------------------

def cmd_replay(args: argparse.Namespace) -> int:
    """Backfill measurements into existing samples (no KSampler)."""
    from command_center.batch_replay import N_REPLAY_ENRICHER, N_REPLAY_GUARD, N_REPLAY_LOAD

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        sys.stderr.write(f"[ERROR] Replay workflow file not found: {workflow_path}\n")
        return 1

    # Validate node IDs before touching the DB
    template: dict[str, Any] = json.loads(workflow_path.read_text(encoding="utf-8"))
    missing_nodes = [nid for nid in (N_REPLAY_LOAD, N_REPLAY_GUARD, N_REPLAY_ENRICHER) if nid not in template]
    if missing_nodes:
        sys.stderr.write(
            f"[ERROR] Expected replay node IDs not found in workflow: {missing_nodes}\n"
            "        Use config/workflow_replay_api.json or update the replay node constants.\n"
        )
        return 1

    requested_paths = parse_requested_paths(args.paths)
    treasurer = open_treasurer(read_only=True)
    try:
        plan, stats = build_replay_plan(
            treasurer,
            requested_paths=requested_paths,
            sample_hashes=args.sample_hash or None,
            limit=args.limit,
            include_dirty=bool(args.include_dirty),
            force=bool(args.force),
        )
    finally:
        treasurer.close()

    sys.stdout.write(
        "Replay plan:\n"
        f"  workflow:      {workflow_path}\n"
        f"  requested:     {requested_paths}\n"
        f"  scanned:       {stats['scanned']}\n"
        f"  selected:      {stats['selected']}\n"
        f"  skipped_dirty: {stats['skipped_dirty']}\n"
        f"  skipped_done:  {stats['skipped_complete']}\n"
        f"  miss_openpose: {stats['missing_openpose']}\n"
        f"  miss_dw:       {stats['missing_dw']}\n\n"
    )

    if not plan:
        sys.stdout.write("Nothing to replay.\n")
        return 0

    if args.dry_run:
        for idx, row in enumerate(plan, 1):
            sys.stdout.write(
                f"  [{idx:3d}/{len(plan)}] {row['sample_hash']}  "
                f"missing={','.join(row['missing_paths'])}  "
                f"dirty={row['is_dirty']}\n"
            )
        sys.stdout.write("\n(dry-run — nothing sent to ComfyUI)\n")
        return 0

    if not _http.check_alive():
        sys.stderr.write(
            f"[ERROR] Cannot reach ComfyUI at {_http.COMFYUI_HOST}\n"
            "        Make sure ComfyUI is running before starting replay.\n"
        )
        return 1

    client_id = str(uuid.uuid4())
    errors: list[str] = []

    for idx, row in enumerate(plan, 1):
        sample_hash = row["sample_hash"]
        needed = row["missing_paths"]
        tag = f"{sample_hash}  missing={','.join(needed)}"
        sys.stdout.write(f"[{idx:3d}/{len(plan)}] {tag} ... ")
        sys.stdout.flush()
        try:
            wf = patch_replay_workflow(template, sample_hash, needed)
            prompt_id = _http.queue_prompt(wf, client_id)
            entry = _http.wait_for_completion(prompt_id, poll_secs=args.poll)
            ok, status_text = _http.summarize_status(entry)
            if ok:
                sys.stdout.write(f"OK ({prompt_id[:8]})\n")
            else:
                sys.stdout.write(f"FAILED: {status_text}\n")
                errors.append(f"[{idx}] {tag}: {status_text}")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(f"FAILED: {exc}\n")
            errors.append(f"[{idx}] {tag}: {exc}")

    sys.stdout.write("\n")
    if errors:
        sys.stdout.write(f"[WARN] {len(errors)} replay job(s) failed:\n")
        for err in errors:
            sys.stdout.write(f"  {err}\n")
        sys.stdout.write(
            "Re-run to retry.  Already-complete samples are skipped unless --force is used.\n"
        )
        return 1

    sys.stdout.write("All replay jobs complete.\n")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _loras_from_dir(dir_arg: str) -> list[str]:
    """Expand a directory argument into a list of lora_name strings.

    Accepts a path relative to lora_root or an absolute path.
    Returns paths relative to lora_root (the format ComfyUI and the batch
    engine both expect), sorted alphabetically.

    Raises FileNotFoundError if the directory cannot be found.
    Raises ValueError if the directory contains no .safetensors files.
    """
    lora_root = get_path("lora_root")
    candidate = (lora_root / dir_arg).resolve()
    if not candidate.is_dir():
        candidate = Path(dir_arg).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"LoRA directory not found.\n"
            f"  Tried relative to lora_root: {lora_root / dir_arg}\n"
            f"  Tried direct path:           {Path(dir_arg).resolve()}\n"
        )
    files = sorted(candidate.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files found in {candidate}")
    return [str(f.relative_to(lora_root)) for f in files]


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add the LoRA/strength/seed sweep args shared by `run` and `rerun`."""
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--loras", nargs="+", metavar="NAME",
        help="One or more LoRA filenames (relative to lora_root, space-separated).",
    )
    lora_group.add_argument(
        "--loras-dir", metavar="DIR",
        help=(
            "Directory of LoRA files — relative to lora_root or absolute. "
            "All .safetensors files found directly in that folder are added. "
            "Cannot be combined with --loras."
        ),
    )
    parser.add_argument(
        "--strengths", nargs="+", type=float, metavar="FLOAT",
        help="One or more LoRA strength values (space-separated).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, metavar="INT",
        help="One or more seed values (space-separated).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without sending anything to ComfyUI.",
    )
    parser.add_argument(
        "--poll", type=float, default=3.0, metavar="SECS",
        help="Polling interval in seconds while waiting for each run (default: 3).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m command_center.batch",
        description="Unified batch orchestrator for LoRA evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # -- run --
    p_run = sub.add_parser(
        "run",
        help="Generate new samples from a workflow file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "--workflow", required=True, metavar="NAME_OR_PATH",
        help=(
            "Workflow name (e.g. 'flux_simple' resolves to config/workflows/flux_simple.json) "
            "or direct path to a workflow JSON file."
        ),
    )
    _add_run_args(p_run)

    # -- rerun --
    p_rerun = sub.add_parser(
        "rerun",
        help="Re-run a known Method using its stored workflow from DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_rerun.add_argument(
        "--method-hash", required=True, metavar="HASH",
        help="method_hash of the Method to rerun.",
    )
    _add_run_args(p_rerun)

    # -- replay --
    p_replay = sub.add_parser(
        "replay",
        help="Backfill measurements into existing samples (no KSampler)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_replay.add_argument(
        "--workflow",
        default="config/workflow_replay_api.json",
        metavar="PATH",
        help="Path to replay workflow JSON (default: config/workflow_replay_api.json).",
    )
    p_replay.add_argument(
        "--paths",
        default="pose_evidence.openpose_body,pose_evidence.dw_body",
        metavar="PATHS",
        help="Comma-separated requested replay paths.",
    )
    p_replay.add_argument(
        "--sample-hash", action="append", default=[], metavar="HASH",
        help="Replay a specific sample_hash. Repeatable.",
    )
    p_replay.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Limit selected samples after filtering.",
    )
    p_replay.add_argument(
        "--include-dirty", action="store_true",
        help="Include dirty samples in the replay plan (default: skip).",
    )
    p_replay.add_argument(
        "--force", action="store_true",
        help="Queue the requested paths even if the sample already has them.",
    )
    p_replay.add_argument(
        "--dry-run", action="store_true",
        help="Print the replay plan without sending anything to ComfyUI.",
    )
    p_replay.add_argument(
        "--poll", type=float, default=3.0, metavar="SECS",
        help="Polling interval while waiting for each replay job (default: 3).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "run":
        return cmd_run(args)
    if args.subcommand == "rerun":
        return cmd_rerun(args)
    if args.subcommand == "replay":
        return cmd_replay(args)

    parser.error(f"unknown subcommand: {args.subcommand!r}")
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
