"""Shared utility functions for execution components."""

from typing import Any


def dep_name(dep: str | dict[str, Any]) -> str:
    """Extract the agent name from a dependency entry.

    Dependency entries are either a plain string or a dict with a single
    ``"agent_name"`` key, e.g. ``{"agent_name": "b"}`` for conditional deps.
    """
    if isinstance(dep, dict):
        return str(dep["agent_name"])
    return dep
