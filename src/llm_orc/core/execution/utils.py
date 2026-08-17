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


def resolve_agent_timeout(
    agent_config: dict[str, Any], performance_config: dict[str, Any]
) -> int:
    """Seconds an agent may run: its own ``timeout_seconds`` when set,
    else the operator's ``performance.execution.default_timeout``, else
    60.

    One rule, one home. The dispatcher applies it as an outer bound and
    the script-agent runner applies it as the subprocess bound; two
    answers to "how long may this agent take" is how a script agent came
    to have no bound at all (#157). ``None`` means unset and defers —
    never a value in its own right.
    """
    timeout = agent_config.get("timeout_seconds")
    if timeout is not None:
        return int(timeout)
    default = performance_config.get("execution", {}).get("default_timeout", 60)
    return int(default)
