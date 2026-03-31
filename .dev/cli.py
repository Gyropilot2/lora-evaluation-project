"""Developer/operator CLI routed through command_center."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import command_center as cc


def _emit_json(payload: object, output_path: str | None) -> None:
    """Serialise payload as pretty JSON to stdout or a file."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output_path:
        p = Path(output_path)

        # If PATH is a directory, write a default scorecard filename inside it.
        if p.exists() and p.is_dir():
            p = p / "scorecard.json"

        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(text + "\n", encoding="utf-8")
        except PermissionError as exc:
            raise SystemExit(
                f"[ERROR] Cannot write JSON output to '{p}'. "
                "Pass a writable file path, e.g. rnd/reports/scorecard_YYYYMMDD_HHMMSS.json "
                f"(details: {exc})"
            )
        print(f"Written to {p}", file=sys.stderr)
    else:
        print(text)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="LEP operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_summary = sub.add_parser("summary", help="Show high-level ingest summary")
    p_summary.add_argument("--limit", type=int, default=10)

    p_loras = sub.add_parser("list-loras", help="Show LoRA inventory")
    p_loras.add_argument("--limit", type=int, default=100)

    p_sample = sub.add_parser("get-sample", help="Get one sample by sample_hash")
    p_sample.add_argument("sample_hash")

    p_method = sub.add_parser("get-method", help="Get one method by method_hash")
    p_method.add_argument("method_hash")

    p_eval = sub.add_parser("get-eval", help="Get one eval by eval_hash")
    p_eval.add_argument("eval_hash")

    p_evals = sub.add_parser("list-evals", help="List eval records")
    p_evals.add_argument("--limit", type=int, default=10)
    p_evals.add_argument("--method-hash")
    p_evals.add_argument("--lora-hash")

    p_health = sub.add_parser("health", help="Show datastore + diagnostics health")
    p_health.add_argument(
        "--scorecard-only",
        action="store_true",
        help="Print only the scorecard JSON (omits legacy db/diagnostics keys)",
    )
    p_health.add_argument(
        "--output",
        metavar="PATH",
        help="Write the JSON output to PATH instead of stdout",
    )

    p_errors = sub.add_parser("errors", help="Show recent warning/error diagnostics")
    p_errors.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command == "summary":
        _print_json(cc.summary(limit=args.limit))
        return 0
    if args.command == "list-loras":
        _print_json(cc.list_loras(limit=args.limit))
        return 0
    if args.command == "get-sample":
        _print_json(cc.get_sample(args.sample_hash))
        return 0
    if args.command == "get-method":
        _print_json(cc.get_method(args.method_hash))
        return 0
    if args.command == "get-eval":
        _print_json(cc.get_eval(args.eval_hash))
        return 0
    if args.command == "list-evals":
        _print_json(
            cc.list_evals(
                limit=args.limit,
                method_hash=args.method_hash,
                lora_hash=args.lora_hash,
            )
        )
        return 0
    if args.command == "health":
        result = cc.health()
        payload = result["scorecard"] if args.scorecard_only else result
        _emit_json(payload, getattr(args, "output", None))
        return 0
    if args.command == "errors":
        _print_json(cc.errors(limit=args.limit))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
