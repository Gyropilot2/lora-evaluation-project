"""command_center/workflow_onboard.py — one-time workflow onboarding tool.

Converts a raw ComfyUI API export into a ready-to-run sentinel-tagged template
in the workspace directory (config/workflows/).

The onboarding step is explicitly invoked once per new workflow. After it runs,
the processed template in the workspace is the run source for batch.py. The raw
export can be discarded or kept in staging — it is never used again by the system.

CLI:
    python -m command_center.workflow_onboard <raw_file> [--name <name>]

Public API:
    onboard(raw_path, name=None) -> Path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from command_center.workflow_batch import tag_workflow_template
from core.diagnostics import emit
from core.paths import get_path


def onboard(raw_path: Path, name: str | None = None) -> Path:
    """Convert a raw ComfyUI export to a sentinel-tagged workspace template.

    Reads the raw workflow, runs auto-topology inference via
    ``tag_workflow_template()``, and writes the processed template to the
    workspace directory under the given name (or the raw file's stem).

    Args:
        raw_path: Path to the raw ComfyUI API workflow JSON.
        name: Output filename stem (no ``.json``). Defaults to ``raw_path.stem``.

    Returns:
        The Path written inside the workspace directory.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist.
        ValueError: If topology inference fails (ambiguous or missing required
            nodes). The caller must then create a companion sidecar manually
            and pre-tag the workflow before retrying.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"workflow file not found: {raw_path}")

    raw: dict[str, Any] = json.loads(raw_path.read_text(encoding="utf-8"))

    sys.stdout.write(f"Onboarding: {raw_path}\n")
    sys.stdout.write(f"  Node count in raw export: {len(raw)}\n")

    try:
        tagged = tag_workflow_template(raw)
    except ValueError as exc:
        emit(
            "WARN", "ARCH.CHEAT", "workflow_onboard.onboard",
            "topology inference failed during workflow onboarding — operator must pre-tag eval loop",
            why="auto-inference only handles simple single-LoRA-loader topology; "
                "complex workflows must be pre-tagged inline by the operator",
            cleanup="resolve once a manual pre-tagging guide or sidecar approach "
                    "exists for complex multi-LoRA topologies",
            raw_path=str(raw_path),
            error=str(exc),
        )
        raise ValueError(
            f"Cannot auto-onboard workflow — topology inference failed.\n"
            f"Reason: {exc}\n\n"
            f"To fix: open the workflow in ComfyUI, set the eval slot values to the\n"
            f"  LEP sentinel strings, then re-export as API format and retry.\n"
            f"  Sentinels to inject:\n"
            f"    lora_name  field -> __LEP_EVAL_LORA__\n"
            f"    strength   field -> __LEP_EVAL_STRENGTH__\n"
            f"    seed       field -> __LEP_EVAL_SEED__"
        ) from exc

    workspace = get_path("workflows_root")
    workspace.mkdir(parents=True, exist_ok=True)

    stem = name if name else raw_path.stem
    out_path = workspace / f"{stem}.json"

    out_path.write_text(
        json.dumps(tagged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    emit(
        "INFO", "WORKFLOW.ONBOARD.OK", "workflow_onboard.onboard",
        "workflow onboarded successfully",
        raw_path=str(raw_path),
        workspace_path=str(out_path),
        workflow_name=stem,
    )

    sys.stdout.write(f"  Tagged template written: {out_path}\n")
    sys.stdout.write(f"  Ready. Use: python batch.py run --workflow {stem} ...\n")

    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for workflow onboarding.

    Usage:
        python -m command_center.workflow_onboard <raw_file> [--name <name>]
    """
    parser = argparse.ArgumentParser(
        prog="python -m command_center.workflow_onboard",
        description=(
            "Onboard a raw ComfyUI API workflow export into the LEP workspace.\n"
            "The processed template is written to config/workflows/ and is ready\n"
            "for use with: python batch.py run --workflow <name>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "raw_file",
        help="Path to the raw ComfyUI API workflow JSON export.",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help=(
            "Workspace name for the template (without .json). "
            "Defaults to the input filename stem."
        ),
    )
    args = parser.parse_args(argv)

    try:
        onboard(Path(args.raw_file), name=args.name)
        return 0
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"\nOnboarding failed:\n{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
