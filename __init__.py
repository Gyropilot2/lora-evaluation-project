# lora_evaluation_project — ComfyUI custom node package entry point.
#
# ComfyUI discovers nodes by importing this file and reading
# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS from it.
#
# All node registration logic lives in comfyui/nodes.py — this file
# is the single re-export surface required by the ComfyUI loader.
#
# sys.path bootstrap — standard ComfyUI custom node pattern:
# ComfyUI adds custom_nodes/ to sys.path, but not the package directory
# itself. We insert it here so our own submodules (comfyui/, lab/, core/,
# bouncer/, etc.) are all importable with absolute paths.

import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

def _load_node_mappings() -> tuple[dict, dict]:
    """Load node mapping exports lazily.

    This keeps plain-Python tooling (tests, healthchecks, scripts) from importing
    the ComfyUI/Lab dependency graph just by importing the package root.
    ComfyUI itself still reads these attributes normally.
    """
    from comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    NODE_DISPLAY_NAME_MAPPINGS["LoraEvalPoseSceneProbe"] = "LoRA Eval - Pose Scene Lab"
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def __getattr__(name: str):
    if name in {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"}:
        node_mappings, display_mappings = _load_node_mappings()
        if name == "NODE_CLASS_MAPPINGS":
            return node_mappings
        return display_mappings
    raise AttributeError(name)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
