"""Workflow declaration owner for batch-run orchestration.

Layer 1 law:
- workflow JSON is the declaration artifact
- Method remains the confirmed semantic comparability truth
- loop slots are declared inline in the workflow for LoRA / strength / seed

This module owns the v1 workflow declaration mechanics used by `command_center.batch`:
- loading exported API workflow JSON
- tagging the eval-loop slots inline when the workflow is still raw
- rehydrating concrete baseline / LoRA runs from a tagged template

Hardcoded node IDs from one workflow export are not architecture law.
For untagged workflows, v1 uses the current simple topology only as transitional
scaffolding. If the workflow shape is more complex or ambiguous, the operator
must pre-tag the eval loop inline instead of expecting blind inference.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.hashing import hash_bytes, sample_hash as compute_sample_hash
from core.json_codec import canonical_json
from core.paths import get_path
from databank.treasurer import Treasurer


BASELINE_LORA_NAME = "(baseline)"
WORKFLOW_ASSET_TYPE = "workflow_json"
WORKFLOW_FORMAT = "json"

WORKFLOW_LORA_SENTINEL = "__LEP_EVAL_LORA__"
WORKFLOW_STRENGTH_SENTINEL = "__LEP_EVAL_STRENGTH__"
WORKFLOW_SEED_SENTINEL = "__LEP_EVAL_SEED__"

_SENTINELS = (
    WORKFLOW_LORA_SENTINEL,
    WORKFLOW_STRENGTH_SENTINEL,
    WORKFLOW_SEED_SENTINEL,
)

_SAMPLE_GUARD_CLASS = "LoraEvalSampleGuard"
_EXTRACTOR_CLASS = "LoraEvalExtractor"
_LORA_LOADER_CLASS = "LoraLoaderModelOnly"
_SEED_PRIMITIVE_CLASS = "PrimitiveInt"
_STRENGTH_PRIMITIVE_CLASS = "PrimitiveFloat"


@dataclass(frozen=True)
class WorkflowHandles:
    sample_guard_id: str
    extractor_id: str
    lora_loader_id: str
    base_model_id: str
    seed_primitive_id: str
    strength_primitive_id: str


@dataclass(frozen=True)
class RunSpec:
    lora_name: str | None
    strength: float
    seed: int

    @property
    def is_baseline(self) -> bool:
        return self.lora_name is None


@dataclass(frozen=True)
class KnownEvalIndex:
    method_hash: str
    baseline_eval_hash: str | None
    file_hash_to_eval_hash: dict[str, str]


def build_run_matrix(
    loras: list[str],
    strengths: list[float],
    seeds: list[int],
) -> list[RunSpec]:
    """Return the ordered run matrix: baselines first, then LoRA sweeps."""
    runs: list[RunSpec] = []
    for seed in seeds:
        runs.append(RunSpec(lora_name=None, strength=0.0, seed=seed))
    for lora_name in loras:
        for strength in strengths:
            for seed in seeds:
                runs.append(RunSpec(lora_name=lora_name, strength=float(strength), seed=seed))
    return runs


def load_workflow_template(path: str | Path) -> dict[str, Any]:
    """Load workflow JSON and ensure the eval loop is inline-tagged."""
    workflow_path = Path(path)
    raw: dict[str, Any] = json.loads(workflow_path.read_text(encoding="utf-8"))
    return tag_workflow_template(raw)


def tag_workflow_template(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return a tagged workflow template with inline eval-loop sentinels.

    If the workflow is already tagged, it is returned unchanged (deep-copied).

    For raw exported workflows, v1 auto-tags only the current simple topology:
    one SampleGuard, one Extractor, one eval LoRA loader, and primitive nodes
    for seed/strength linked into those surfaces. More complex workflows must
    be pre-tagged inline by the operator rather than guessed here.
    """
    tagged = copy.deepcopy(workflow)
    if workflow_has_inline_slots(tagged):
        return tagged

    handles = discover_workflow_handles(tagged)

    tagged[handles.seed_primitive_id]["inputs"]["value"] = WORKFLOW_SEED_SENTINEL
    tagged[handles.strength_primitive_id]["inputs"]["value"] = WORKFLOW_STRENGTH_SENTINEL
    tagged[handles.lora_loader_id]["inputs"]["lora_name"] = WORKFLOW_LORA_SENTINEL

    for node_id in (handles.sample_guard_id, handles.extractor_id):
        inputs = tagged[node_id].setdefault("inputs", {})
        if "lora_name" in inputs and not isinstance(inputs["lora_name"], list):
            inputs["lora_name"] = WORKFLOW_LORA_SENTINEL
        if "lora_strength" in inputs and not isinstance(inputs["lora_strength"], list):
            inputs["lora_strength"] = WORKFLOW_STRENGTH_SENTINEL

    if not workflow_has_inline_slots(tagged):
        raise ValueError("failed to tag workflow template with all required eval-loop sentinels")

    return tagged


def workflow_has_inline_slots(workflow: dict[str, Any]) -> bool:
    """Return True when all required inline eval-loop sentinels are present."""
    found = {sentinel: False for sentinel in _SENTINELS}

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)
            return
        if isinstance(value, list):
            for child in value:
                _walk(child)
            return
        if isinstance(value, str) and value in found:
            found[value] = True

    _walk(workflow)
    return all(found.values())


def workflow_digest(workflow: dict[str, Any]) -> str:
    """Return the BLAKE3 digest of canonical tagged workflow JSON."""
    return hash_bytes(canonical_workflow_json(workflow).encode("utf-8"))


def workflow_bytes(workflow: dict[str, Any]) -> bytes:
    """Return canonical UTF-8 bytes for a workflow declaration artifact."""
    return canonical_workflow_json(workflow).encode("utf-8")


def canonical_workflow_json(workflow: dict[str, Any]) -> str:
    """Return canonical JSON for a workflow declaration artifact."""
    return canonical_json(workflow)


def store_workflow_ref(treasurer: Treasurer, tagged_template: dict[str, Any]) -> dict[str, Any]:
    """Persist one tagged workflow declaration and return its committed ValueRef."""
    return treasurer.store_asset_blob(
        workflow_bytes(tagged_template),
        asset_type=WORKFLOW_ASSET_TYPE,
        fmt=WORKFLOW_FORMAT,
    )


def workflow_ref_json(workflow_ref: dict[str, Any]) -> str:
    """Serialize a workflow_ref dict for the explicit Extractor ingest seam."""
    return json.dumps(workflow_ref, ensure_ascii=False)


def load_workflow_from_method(treasurer: Treasurer, method_hash: str) -> dict[str, Any] | None:
    """Load the stored workflow template for a known Method by its hash.

    Returns the tagged workflow template dict when the Method record exists,
    carries a workflow_ref, and the asset blob is readable and valid JSON.
    Returns None in all failure cases — the caller decides whether that is
    fatal (rerun requires a workflow; run can fall back to a file path).
    """
    method = treasurer.get_method(method_hash)
    if method is None:
        return None
    workflow_ref = method.get("workflow_ref")
    if not isinstance(workflow_ref, dict):
        return None
    try:
        blob = treasurer.load_asset_blob(workflow_ref)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(blob, dict):
        # load_asset_blob returned an Invalid wrapper — asset unavailable
        return None
    try:
        template = json.loads(blob.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(template, dict):
        return None
    return template


def find_method_by_workflow(treasurer: Treasurer, tagged_template: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a known Method by the digest of its stored workflow declaration."""
    return treasurer.find_method_by_workflow_digest(workflow_digest(tagged_template))


def build_known_eval_index(treasurer: Treasurer, method_hash: str) -> KnownEvalIndex:
    """Build a conservative eval lookup index for one already-known Method.

    Only raw file hashes that map to exactly one existing eval are included.
    Ambiguous or missing file-hash cases are intentionally dropped so v1
    external pre-dedup never guesses fresh eval identity outside ComfyUI.
    """
    evals = treasurer.query_evals({"method_hash": method_hash})
    baseline_eval_hash: str | None = None
    file_hash_buckets: dict[str, list[str]] = {}

    for eval_record in evals:
        eval_hash = eval_record.get("eval_hash")
        lora_obj = eval_record.get("lora") if isinstance(eval_record.get("lora"), dict) else {}
        lora_hash = lora_obj.get("hash")
        file_hash = lora_obj.get("file_hash")

        if lora_hash is None:
            if isinstance(eval_hash, str):
                baseline_eval_hash = eval_hash
            continue

        if isinstance(eval_hash, str) and isinstance(file_hash, str):
            file_hash_buckets.setdefault(file_hash, []).append(eval_hash)

    file_hash_to_eval_hash = {
        file_hash: hashes[0]
        for file_hash, hashes in file_hash_buckets.items()
        if len(hashes) == 1
    }
    return KnownEvalIndex(
        method_hash=method_hash,
        baseline_eval_hash=baseline_eval_hash,
        file_hash_to_eval_hash=file_hash_to_eval_hash,
    )


def find_existing_sample_hash_for_run(
    treasurer: Treasurer,
    known_evals: KnownEvalIndex,
    run: RunSpec,
    lora_file_hash_cache: dict[str, str | None] | None = None,
) -> str | None:
    """Return an already-existing exact sample_hash for this run, if provable.

    Returns None when eval identity is not already known honestly outside
    ComfyUI. That includes unresolved LoRA paths, missing file-hash matches,
    and ambiguous file-hash -> eval mappings.
    """
    eval_hash: str | None
    if run.is_baseline:
        eval_hash = known_evals.baseline_eval_hash
    else:
        if run.lora_name is None:
            return None
        file_hash: str | None
        if lora_file_hash_cache is not None and run.lora_name in lora_file_hash_cache:
            file_hash = lora_file_hash_cache[run.lora_name]
        else:
            lora_path = resolve_lora_path(run.lora_name)
            file_hash = None if lora_path is None else hash_bytes(lora_path.read_bytes())
            if lora_file_hash_cache is not None:
                lora_file_hash_cache[run.lora_name] = file_hash
        eval_hash = known_evals.file_hash_to_eval_hash.get(file_hash)

    if not isinstance(eval_hash, str):
        return None

    sample_hash = compute_sample_hash(
        eval_hash,
        int(run.seed),
        None if run.is_baseline else float(run.strength),
    )
    existing = treasurer.get_sample(sample_hash)
    if existing is None:
        return None
    return sample_hash


def materialize_run_workflow(
    tagged_template: dict[str, Any],
    *,
    lora_name: str | None,
    strength: float,
    seed: int,
    workflow_ref_json: str | None = None,
) -> dict[str, Any]:
    """Return a runnable workflow for either a baseline or LoRA run."""
    handles = discover_workflow_handles(tagged_template)
    hydrated = _hydrate_workflow_slots(
        tagged_template,
        lora_name=BASELINE_LORA_NAME if lora_name is None else lora_name,
        strength=0.0 if lora_name is None else float(strength),
        seed=int(seed),
    )
    if lora_name is None:
        _patch_baseline_topology(hydrated, handles)
    else:
        _patch_lora_topology(hydrated, handles)
    _inject_workflow_ref_json(hydrated, handles.extractor_id, workflow_ref_json)
    return hydrated


def resolve_lora_path(lora_name: str) -> Path | None:
    """Resolve a batch-run LoRA name against the configured non-ComfyUI LoRA root."""
    lora_root = get_path("lora_root")
    candidate = (lora_root / Path(lora_name)).resolve()
    return candidate if candidate.is_file() else None


def discover_workflow_handles(workflow: dict[str, Any]) -> WorkflowHandles:
    """Discover the current eval-loop patch points from workflow structure.

    This is intentionally strict. If a raw workflow is more complex than the
    current simple shape, v1 does not guess; the operator must pre-tag the eval
    slot inline so later rehydration is explicit.
    """
    sample_guard_id = _find_unique_node_id(workflow, _SAMPLE_GUARD_CLASS)
    extractor_id = _find_unique_node_id(workflow, _EXTRACTOR_CLASS)
    lora_loader_id = _find_eval_lora_loader_id(workflow)
    seed_primitive_id = _resolve_linked_node_id(
        workflow,
        sample_guard_id,
        "seed",
        expected_class_type=_SEED_PRIMITIVE_CLASS,
    )
    strength_primitive_id = _resolve_linked_node_id(
        workflow,
        lora_loader_id,
        "strength_model",
        expected_class_type=_STRENGTH_PRIMITIVE_CLASS,
    )
    base_model_id = _discover_base_model_id(workflow, sample_guard_id, lora_loader_id)
    return WorkflowHandles(
        sample_guard_id=sample_guard_id,
        extractor_id=extractor_id,
        lora_loader_id=lora_loader_id,
        base_model_id=base_model_id,
        seed_primitive_id=seed_primitive_id,
        strength_primitive_id=strength_primitive_id,
    )


def _find_unique_node_id(workflow: dict[str, Any], class_type: str) -> str:
    matches = [node_id for node_id, node in workflow.items() if _class_type(node) == class_type]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"workflow is missing required node class {class_type!r}")
    raise ValueError(
        f"workflow has multiple {class_type!r} nodes; pre-tag the eval loop inline instead of relying on v1 auto-discovery"
    )


def _find_eval_lora_loader_id(workflow: dict[str, Any]) -> str:
    matches = [node_id for node_id, node in workflow.items() if _class_type(node) == _LORA_LOADER_CLASS]
    if len(matches) == 1:
        return matches[0]

    sentinel_matches = []
    for node_id in matches:
        lora_name = workflow.get(node_id, {}).get("inputs", {}).get("lora_name")
        if lora_name == WORKFLOW_LORA_SENTINEL:
            sentinel_matches.append(node_id)

    if len(sentinel_matches) == 1:
        return sentinel_matches[0]
    if not matches:
        raise ValueError("workflow is missing required eval LoRA loader node")
    raise ValueError(
        "workflow has multiple LoRA loader nodes; declare the eval loader inline with the workflow sentinels before using batch_run"
    )


def _discover_base_model_id(workflow: dict[str, Any], sample_guard_id: str, lora_loader_id: str) -> str:
    sample_guard_model = _read_linked_node_id(workflow, sample_guard_id, "model")
    if sample_guard_model and _class_type(workflow.get(sample_guard_model)) != _LORA_LOADER_CLASS:
        return sample_guard_model

    loader_model = _read_linked_node_id(workflow, lora_loader_id, "model")
    if loader_model:
        return loader_model

    raise ValueError("could not resolve the base model source for the eval loop")


def _resolve_linked_node_id(
    workflow: dict[str, Any],
    node_id: str,
    input_name: str,
    *,
    expected_class_type: str,
) -> str:
    linked_node_id = _read_linked_node_id(workflow, node_id, input_name)
    if linked_node_id is None:
        raise ValueError(
            f"workflow input {node_id}.{input_name} is not linked; v1 auto-discovery expects a linked {expected_class_type}"
        )
    linked_class = _class_type(workflow.get(linked_node_id))
    if linked_class != expected_class_type:
        raise ValueError(
            f"workflow input {node_id}.{input_name} links to {linked_class!r}, expected {expected_class_type!r}"
        )
    return linked_node_id


def _read_linked_node_id(workflow: dict[str, Any], node_id: str, input_name: str) -> str | None:
    inputs = workflow.get(node_id, {}).get("inputs", {})
    value = inputs.get(input_name)
    if isinstance(value, list) and value:
        linked_node_id = value[0]
        if isinstance(linked_node_id, str) and linked_node_id in workflow:
            return linked_node_id
    return None


def _class_type(node: Any) -> str | None:
    if isinstance(node, dict):
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) else None
    return None


def _hydrate_workflow_slots(
    tagged_template: dict[str, Any],
    *,
    lora_name: str,
    strength: float,
    seed: int,
) -> dict[str, Any]:
    replacements = {
        WORKFLOW_LORA_SENTINEL: lora_name,
        WORKFLOW_STRENGTH_SENTINEL: float(strength),
        WORKFLOW_SEED_SENTINEL: int(seed),
    }
    return _replace_sentinels(copy.deepcopy(tagged_template), replacements)


def _replace_sentinels(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_sentinels(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_sentinels(child, replacements) for child in value]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def _patch_lora_topology(workflow: dict[str, Any], handles: WorkflowHandles) -> None:
    workflow[handles.lora_loader_id].setdefault("inputs", {})["model"] = [handles.base_model_id, 0]
    workflow[handles.sample_guard_id].setdefault("inputs", {})["model"] = [handles.lora_loader_id, 0]


def _patch_baseline_topology(workflow: dict[str, Any], handles: WorkflowHandles) -> None:
    workflow.pop(handles.lora_loader_id, None)
    workflow[handles.strength_primitive_id].setdefault("inputs", {})["value"] = 0.0
    workflow[handles.sample_guard_id].setdefault("inputs", {})["model"] = [handles.base_model_id, 0]


def _inject_workflow_ref_json(
    workflow: dict[str, Any],
    extractor_id: str,
    workflow_ref_json: str | None,
) -> None:
    inputs = workflow[extractor_id].setdefault("inputs", {})
    if workflow_ref_json is None:
        inputs.pop("workflow_ref_json", None)
    else:
        inputs["workflow_ref_json"] = workflow_ref_json
