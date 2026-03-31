"""lab/field_catalog.py — reads dump directory and produces a field catalog.

Consumes the JSON dump files written by dump_writer.py and produces a
structured catalog describing:
  - which fields were observed
  - their types and value ranges
  - how frequently each field appeared across dumps
  - anomalies (fields missing from some but not all dumps)

No ComfyUI dependency. No production DB access. Pure analysis tool.

Usage:
    from lab.field_catalog import build_catalog
    catalog = build_catalog("data/lab_dumps")
    # or from CLI: python lab/field_catalog.py data/lab_dumps
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _settings_fingerprint(dump: dict[str, Any]) -> str:
    """Compute a deterministic pairing key from a dump's settings + conditioning + base model.

    Two dumps with the same fingerprint represent the same experiment — same seed,
    sampler settings, conditioning, and base model — differing only in whether a LoRA
    was applied. No user-provided tag is needed; pairing is purely content-based.

    Probe version compatibility:
      v0.6.0+  — uses conditioning.positive.hash (BLAKE3 of conditioner tensors).
                  Two different prompt strings that produce the same conditioning
                  are treated as the same experiment (correct behaviour).
      < v0.6.0 — falls back to prompt.hash or prompt.text for old dumps.
    """
    settings = dump.get("settings") or {}
    pipeline = dump.get("model_pipeline") or {}

    seed = settings.get("seed", "?")
    steps = settings.get("steps", "?")
    cfg = settings.get("cfg", "?")
    sampler = settings.get("sampler_name", "?")
    scheduler = settings.get("scheduler", "?")
    denoise = settings.get("denoise", "?")

    # Base model config class extracted from _try wrapper structure
    config_class = "?"
    bm = pipeline.get("base_model")
    if isinstance(bm, dict) and bm.get("status") == "ok":
        config_class = bm["value"].get("config_class", "?")

    # Prompt/conditioning component — version-aware extraction
    prompt_component = "?"

    # v0.6.0+: conditioning.positive.hash is the ground truth
    cond = dump.get("conditioning") or {}
    pos_cond = cond.get("positive") or {}
    if isinstance(pos_cond, dict) and isinstance(pos_cond.get("hash"), str):
        prompt_component = f"cond_hash={pos_cond['hash'][:32]}"

    # Fallback for pre-v0.6.0 dumps: prompt.hash or prompt.text
    if prompt_component == "?":
        prompt_info = dump.get("prompt") or {}
        ph = prompt_info.get("hash")
        if isinstance(ph, dict) and ph.get("status") == "ok":
            prompt_component = f"prompt_hash={ph['value'][:32]}"
        elif isinstance(ph, str):
            prompt_component = f"prompt_hash={ph[:32]}"
        else:
            prompt_component = str(prompt_info.get("text", ""))

    return (
        f"seed={seed}|steps={steps}|cfg={cfg}"
        f"|sampler={sampler}|scheduler={scheduler}|denoise={denoise}"
        f"|model={config_class}|prompt={prompt_component}"
    )


def _fingerprint_summary(fp: str) -> str:
    """Return a short display label for a fingerprint (for display only)."""
    # Extract the non-prompt portion and show a prompt excerpt
    try:
        main, prompt_part = fp.split("|prompt=", 1)
        excerpt = (prompt_part[:40] + "...") if len(prompt_part) > 40 else prompt_part
        return f"{main}|prompt={excerpt!r}"
    except ValueError:
        return fp[:80]


def build_paired_catalog(dump_dir: Path | str) -> dict[str, Any]:
    """Group dumps by settings fingerprint and compute field-level deltas.

    Pairing is content-based — no user-provided tag is needed. Two dumps are
    candidates for pairing when they share the same seed, sampler settings, prompt,
    and base model (i.e. the same experiment), one with a LoRA applied and one
    without (is_baseline=True).

    This mirrors the V2+ baseline lookup: query for a run where all
    non-LoRA fields match and lora_hash IS NULL. No session_tag needed or used.

    Returns a dict with keys:
        experiment_count        — number of matched fingerprint groups
        unmatched_fingerprints  — fingerprints that lacked a baseline or a LoRA run
        experiments             — {fingerprint: ExperimentDiff}
        errors                  — list of files that could not be parsed

    ExperimentDiff has:
        fingerprint_summary     — display label (truncated)
        baseline_run_id         — run_id of the most recent baseline in this group
        baseline_count          — total baselines found (>1 means repeated baseline runs)
        lora_run_count
        lora_runs               — list of {run_id, lora_name, deltas}
        deltas are {dotted_path: {baseline, lora, delta}} for numeric leaves only
    """
    dump_dir = Path(dump_dir)
    dumps: list[dict[str, Any]] = []
    errors: list[str] = []

    for p in sorted(dump_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            dumps.append(data)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p.name}: {exc}")

    # Group by settings fingerprint — all dumps participate (no tag required)
    groups: dict[str, list[dict[str, Any]]] = {}
    for dump in dumps:
        fp = _settings_fingerprint(dump)
        groups.setdefault(fp, []).append(dump)

    experiments: dict[str, Any] = {}
    unmatched: list[str] = []

    for fp, group_dumps in sorted(groups.items()):
        baselines = [d for d in group_dumps if d.get("is_baseline") is True]
        lora_runs = [d for d in group_dumps if not d.get("is_baseline")]

        if not baselines or not lora_runs:
            unmatched.append(fp)
            continue

        # Use the most recent baseline (last in file-sorted order) as the reference.
        # Multiple baselines are noted but not an error — user may have re-run the baseline.
        baseline = baselines[-1]
        baseline_flat = _flatten(baseline)
        baseline_run_id = baseline.get("_dump_meta", {}).get("run_id", "?")

        lora_diffs: list[dict[str, Any]] = []
        for lrun in lora_runs:
            lora_flat = _flatten(lrun)
            lrun_id = lrun.get("_dump_meta", {}).get("run_id", "?")
            lrun_name = lrun.get("lora_name") or ""

            deltas: dict[str, Any] = {}
            all_keys = set(baseline_flat) | set(lora_flat)
            for key in sorted(all_keys):
                bval = baseline_flat.get(key)
                lval = lora_flat.get(key)
                # Only compute numeric deltas
                if (
                    isinstance(bval, (int, float)) and not isinstance(bval, bool)
                    and isinstance(lval, (int, float)) and not isinstance(lval, bool)
                ):
                    deltas[key] = {
                        "baseline": bval,
                        "lora": lval,
                        "delta": lval - bval,
                    }

            lora_diffs.append({
                "run_id": lrun_id,
                "lora_name": lrun_name,
                "deltas": deltas,
            })

        experiments[fp] = {
            "fingerprint_summary": _fingerprint_summary(fp),
            "baseline_run_id": baseline_run_id,
            "baseline_count": len(baselines),
            "lora_run_count": len(lora_runs),
            "lora_runs": lora_diffs,
        }

    return {
        "experiment_count": len(experiments),
        "unmatched_fingerprints": unmatched,
        "experiments": experiments,
        "errors": errors,
    }


def print_paired_catalog(paired: dict[str, Any]) -> None:
    """Print a readable summary of a paired (delta) catalog.

    Allowed to use print() — this is a lab analysis tool.
    """
    print(f"\n=== Paired Catalog ({paired['experiment_count']} matched experiments) ===\n")
    print("Pairing is content-based: same seed + settings + prompt + base model.")
    print("No session tag required — runs are matched automatically.\n")

    if paired["errors"]:
        print("PARSE ERRORS:")
        for e in paired["errors"]:
            print(f"  {e}")
        print()

    if paired["unmatched_fingerprints"]:
        print(f"UNMATCHED GROUPS ({len(paired['unmatched_fingerprints'])}) — missing baseline or LoRA run:")
        for fp in paired["unmatched_fingerprints"]:
            print(f"  {_fingerprint_summary(fp)}")
        print()

    for _fp, exp in paired["experiments"].items():
        print(f"Experiment: {exp['fingerprint_summary']}")
        print(f"  Baseline run_id : {exp['baseline_run_id']}")
        if exp["baseline_count"] > 1:
            print(f"  NOTE: {exp['baseline_count']} baseline dumps found; using most recent")
        print(f"  LoRA runs       : {exp['lora_run_count']}")
        for lr in exp["lora_runs"]:
            lname = lr["lora_name"] or "(name not recorded)"
            print(f"\n  -- LoRA run: {lr['run_id']}  [{lname}]")
            deltas = lr["deltas"]
            if not deltas:
                print("     (no numeric fields in common for delta)")
            else:
                for field, d in list(deltas.items())[:30]:  # cap display at 30 fields
                    print(f"     {field}")
                    print(f"       baseline={d['baseline']:.6g}  lora={d['lora']:.6g}  delta={d['delta']:+.6g}")
                if len(deltas) > 30:
                    print(f"     ... ({len(deltas) - 30} more fields)")
        print()


def build_catalog(dump_dir: Path | str) -> dict[str, Any]:
    """Read all JSON dumps in dump_dir and return a field catalog.

    Args:
        dump_dir: Path to the directory of .json dump files.

    Returns:
        A dict with keys:
          dump_count      — number of dumps read
          dump_files      — list of filenames
          fields          — {dotted_path: FieldStat} for every field seen
          anomaly_fields  — dotted paths absent from at least one dump
          errors          — list of files that could not be parsed
    """
    dump_dir = Path(dump_dir)
    dumps: list[dict[str, Any]] = []
    errors: list[str] = []
    dump_files: list[str] = []

    for p in sorted(dump_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            dumps.append(data)
            dump_files.append(p.name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p.name}: {exc}")

    dump_count = len(dumps)
    if dump_count == 0:
        return {
            "dump_count": 0,
            "dump_files": [],
            "fields": {},
            "anomaly_fields": [],
            "errors": errors,
        }

    # Flatten each dump into dotted-path → value entries.
    # "_dump_meta" is internal bookkeeping — catalog it but tag it.
    field_occurrences: dict[str, list[Any]] = {}
    for dump in dumps:
        flat = _flatten(dump)
        for dotted_path, val in flat.items():
            field_occurrences.setdefault(dotted_path, [])
            field_occurrences[dotted_path].append(val)

    fields: dict[str, Any] = {}
    for dotted_path, values in sorted(field_occurrences.items()):
        present_count = len(values)
        frequency = present_count / dump_count

        types_seen = sorted({type(v).__name__ for v in values})

        # Collect up to 5 sample values (deduplicated where possible).
        seen_reprs: list[str] = []
        sample_values: list[Any] = []
        for v in values:
            r = repr(v)
            if r not in seen_reprs:
                seen_reprs.append(r)
                sample_values.append(v)
            if len(sample_values) >= 5:
                break

        # Numeric range (only when all values are numeric).
        numeric_range: dict[str, float] | None = None
        numeric_vals: list[float] = []
        for v in values:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v))
        if numeric_vals and len(numeric_vals) == present_count:
            numeric_range = {
                "min": min(numeric_vals),
                "max": max(numeric_vals),
                "mean": statistics.mean(numeric_vals),
            }
            if len(numeric_vals) > 1:
                numeric_range["stdev"] = statistics.stdev(numeric_vals)

        anomaly = frequency < 1.0

        fields[dotted_path] = {
            "present_count": present_count,
            "frequency": round(frequency, 4),
            "types_seen": types_seen,
            "sample_values": sample_values,
            "numeric_range": numeric_range,
            "anomaly": anomaly,
        }

    anomaly_fields = sorted(k for k, v in fields.items() if v["anomaly"])

    return {
        "dump_count": dump_count,
        "dump_files": dump_files,
        "fields": fields,
        "anomaly_fields": anomaly_fields,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def print_catalog(catalog: dict[str, Any]) -> None:
    """Print a readable summary of a field catalog to stdout.

    Allowed to use print() — this is a lab analysis tool.
    """
    print(f"\n=== Field Catalog ({catalog['dump_count']} dumps) ===\n")

    if catalog["errors"]:
        print("PARSE ERRORS:")
        for e in catalog["errors"]:
            print(f"  {e}")
        print()

    fields = catalog["fields"]
    if not fields:
        print("  (no fields found)")
        return

    for path, stat in fields.items():
        freq_pct = stat["frequency"] * 100
        anomaly_tag = "  [ANOMALY]" if stat["anomaly"] else ""
        print(f"{path}{anomaly_tag}")
        print(f"  present: {stat['present_count']}/{catalog['dump_count']} ({freq_pct:.0f}%)")
        print(f"  types:   {', '.join(stat['types_seen'])}")
        if stat["numeric_range"]:
            r = stat["numeric_range"]
            print(f"  range:   min={r['min']:.4g}  max={r['max']:.4g}  mean={r['mean']:.4g}")
        if stat["sample_values"]:
            sample_str = ", ".join(repr(v) for v in stat["sample_values"][:3])
            print(f"  samples: {sample_str}")
        print()

    if catalog["anomaly_fields"]:
        print(f"ANOMALY FIELDS ({len(catalog['anomaly_fields'])}):")
        for f in catalog["anomaly_fields"]:
            print(f"  {f}")
        print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict into dotted-path → leaf-value entries.

    Lists are indexed: "settings.run_ids.0", "settings.run_ids.1", etc.
    Non-dict, non-list values are leaf entries.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_key = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, child_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child_key = f"{prefix}.{i}" if prefix else str(i)
            out.update(_flatten(item, child_key))
    else:
        # Leaf value.
        out[prefix] = obj
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lab/field_catalog.py <dump_dir> [--paired]")
        sys.exit(1)

    dump_path = sys.argv[1]
    if "--paired" in sys.argv:
        paired = build_paired_catalog(dump_path)

        # Write paired results to a JSON file alongside the dumps
        out_path = Path(dump_path) / "_paired_results.json"
        try:
            out_path.write_text(
                json.dumps(paired, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Paired results written to: {out_path}")
        except OSError as exc:
            print(f"WARNING: could not write paired results file: {exc}")

        # Also print terminal summary
        print_paired_catalog(paired)
    else:
        catalog = build_catalog(dump_path)
        print_catalog(catalog)
