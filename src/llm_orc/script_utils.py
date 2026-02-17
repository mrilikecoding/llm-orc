"""Shared utilities for llm-orc script agents.

Provides envelope unwrapping for scripts that receive input via stdin.
Scripts can import directly::

    from llm_orc.script_utils import unwrap_input

    raw = sys.stdin.read()
    data, params = unwrap_input(raw)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def unwrap_input(
    raw_json: str,
    debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap llm-orc envelope to get actual data and parameters.

    Handles three input formats:
    1. ScriptAgentInput: ``{"agent_name": ..., "input_data": "<json>", ...}``
    2. Legacy wrapper:   ``{"input": "<json or dict>", "parameters": {...}}``
    3. Direct:           ``{"nodes": [...], "edges": [...]}``

    Parameters from the ``AGENT_PARAMETERS`` env var are used as a
    fallback when the envelope doesn't carry them (e.g. format 1).

    Args:
        raw_json: Raw JSON string from stdin.
        debug: If truthy, log envelope diagnostics to stderr.

    Returns:
        Tuple of (data_dict, parameters_dict).
    """
    envelope = json.loads(raw_json) if raw_json.strip() else {}

    if debug:
        _debug_envelope(envelope)

    data, params = _unwrap_envelope(envelope)

    # Fallback: read parameters from env var if envelope didn't have them
    if not params:
        params = _params_from_env()

    if debug:
        _debug_result(data, params)

    return data, params


def _unwrap_envelope(
    envelope: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Core unwrapping logic without debug logging."""
    # Format 1: ScriptAgentInput envelope
    input_data = envelope.get("input_data", "")
    if isinstance(input_data, str) and input_data.strip():
        inner = _try_parse_json(input_data)
        if inner is not None:
            return inner, envelope.get("parameters", {}) or {}

    # Format 2: Legacy wrapper {"input": ..., "parameters": ...}
    if "input" in envelope and "parameters" in envelope:
        return _unwrap_legacy(envelope)

    # Format 3: Direct invocation -- envelope IS the data
    return envelope, {}


def _unwrap_legacy(
    envelope: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unwrap legacy ``{"input": ..., "parameters": ...}`` format."""
    inner = envelope["input"]
    params: dict[str, Any] = envelope.get("parameters", {}) or {}

    if isinstance(inner, str) and inner.strip():
        parsed = _try_parse_json(inner)
        if parsed is not None:
            return parsed, params
        return envelope, params

    if isinstance(inner, dict):
        return inner, params

    return envelope, params


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse a JSON string; return None on failure."""
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except (json.JSONDecodeError, TypeError):
        return None


def _params_from_env() -> dict[str, Any]:
    """Read AGENT_PARAMETERS from environment variable."""
    raw = os.environ.get("AGENT_PARAMETERS", "")
    if raw:
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed
    return {}


def _debug_envelope(envelope: dict[str, Any]) -> None:
    """Log envelope shape to stderr for diagnostics."""
    info = {
        "envelope_keys": sorted(envelope.keys()) if envelope else [],
        "envelope_size": len(json.dumps(envelope)),
    }
    # Check key field types/sizes
    for key in ("input_data", "input"):
        val = envelope.get(key)
        if val is not None:
            info[f"{key}_type"] = type(val).__name__
            if isinstance(val, str):
                info[f"{key}_len"] = len(val)
            elif isinstance(val, dict):
                info[f"{key}_keys"] = sorted(val.keys())
    print(
        f"[llm-orc debug] envelope: {json.dumps(info)}",
        file=sys.stderr,
    )


def _debug_result(data: dict[str, Any], params: dict[str, Any]) -> None:
    """Log unwrapped result shape to stderr for diagnostics."""
    info = {
        "data_keys": sorted(data.keys()) if data else [],
        "data_size": len(json.dumps(data)),
        "params_keys": sorted(params.keys()) if params else [],
    }
    print(
        f"[llm-orc debug] unwrapped: {json.dumps(info)}",
        file=sys.stderr,
    )
