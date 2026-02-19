"""Core primitives package — first-class engine infrastructure.

Provides Pydantic-contracted primitives that define the engine's I/O
contracts. Each primitive has Input/Output models and an execute() function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

_PRIMITIVE_REGISTRY: dict[str, str] = {
    "user_interaction/get_user_input": (
        "llm_orc.primitives.user_interaction.get_user_input"
    ),
    "user_interaction/confirm_action": (
        "llm_orc.primitives.user_interaction.confirm_action"
    ),
    "data_transform/json_extract": ("llm_orc.primitives.data_transform.json_extract"),
    "file_ops/read_file": ("llm_orc.primitives.file_ops.read_file"),
    "file_ops/write_file": ("llm_orc.primitives.file_ops.write_file"),
    "control_flow/replicate_n_times": (
        "llm_orc.primitives.control_flow.replicate_n_times"
    ),
}

# Map script_ref patterns to registry keys
# Hyphen variants are not needed: the resolver normalizes hyphens to
# underscores before reaching this lookup.
_SCRIPT_REF_PATTERNS: dict[str, str] = {
    "primitives/user_interaction/get_user_input": "user_interaction/get_user_input",
    "primitives/user_interaction/confirm_action": "user_interaction/confirm_action",
    "primitives/data_transform/json_extract": "data_transform/json_extract",
    "primitives/file_ops/read_file": "file_ops/read_file",
    "primitives/file_ops/write_file": "file_ops/write_file",
    "primitives/control_flow/replicate_n_times": "control_flow/replicate_n_times",
}


def _normalize_script_ref(script_ref: str) -> str | None:
    """Normalize a script reference to a registry key."""
    # Strip .py extension
    ref = script_ref.removesuffix(".py")

    # Direct lookup
    if ref in _SCRIPT_REF_PATTERNS:
        return _SCRIPT_REF_PATTERNS[ref]

    # Try hyphen-to-underscore
    normalized = ref.replace("-", "_")
    if normalized in _SCRIPT_REF_PATTERNS:
        return _SCRIPT_REF_PATTERNS[normalized]

    return None


def get_output_schema(script_ref: str) -> type[BaseModel] | None:
    """Get the Pydantic output schema for a known primitive.

    Returns None if the script_ref doesn't match a registered primitive.
    """
    import importlib

    registry_key = _normalize_script_ref(script_ref)
    if registry_key is None:
        return None

    module_path = _PRIMITIVE_REGISTRY.get(registry_key)
    if module_path is None:
        return None

    try:
        module = importlib.import_module(module_path)
        # Convention: each module has a class ending in "Output"
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and name.endswith("Output")
                and name != "BaseModel"
            ):
                return obj
    except ImportError:
        pass

    return None
