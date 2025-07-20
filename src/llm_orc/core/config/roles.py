"""Flexible role system for LLM agents."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RoleDefinition:
    """Defines a role for an LLM agent."""

    name: str
    prompt: str
    context: dict[str, Any] | None = None


class RoleManager:
    """Manages role definitions and retrieval."""

    def __init__(self) -> None:
        self.roles: dict[str, RoleDefinition] = {}

    def register_role(self, role: RoleDefinition) -> None:
        """Register a new role."""
        self.roles[role.name] = role

    def get_role(self, name: str) -> RoleDefinition:
        """Retrieve a role by name."""
        if name not in self.roles:
            raise KeyError(f"Role '{name}' not found")
        return self.roles[name]
