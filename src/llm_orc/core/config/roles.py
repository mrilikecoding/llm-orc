"""Flexible role system for LLM agents."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RoleDefinition:
    """Defines a role for an LLM agent."""

    name: str
    prompt: str
    context: dict[str, Any] | None = None
