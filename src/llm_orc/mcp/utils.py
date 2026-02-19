"""Shared utilities for MCP server modules."""

from typing import Any


def get_agent_attr(agent: Any, attr: str, default: Any = None) -> Any:
    """Get agent attribute handling both dict and object forms."""
    if isinstance(agent, dict):
        return agent.get(attr, default)
    return getattr(agent, attr, default)
