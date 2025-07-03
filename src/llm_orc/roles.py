"""Flexible role system for LLM agents."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RoleDefinition:
    """Defines a role for an LLM agent."""
    
    name: str
    prompt: str
    context: Optional[Dict[str, Any]] = None


class RoleManager:
    """Manages role definitions and retrieval."""
    
    def __init__(self):
        self.roles: Dict[str, RoleDefinition] = {}
    
    def register_role(self, role: RoleDefinition) -> None:
        """Register a new role."""
        self.roles[role.name] = role
    
    def get_role(self, name: str) -> RoleDefinition:
        """Retrieve a role by name."""
        if name not in self.roles:
            raise KeyError(f"Role '{name}' not found")
        return self.roles[name]