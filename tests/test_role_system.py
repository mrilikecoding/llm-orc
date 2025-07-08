"""Test suite for the flexible role system."""

import pytest

from llm_orc.roles import RoleDefinition, RoleManager


class TestRoleDefinition:
    """Test role definition creation and validation."""

    def test_create_basic_role(self) -> None:
        """Should create a basic role with name and prompt."""
        role = RoleDefinition(
            name="engineer",
            prompt="You are a senior software engineer focused on clean code and TDD.",
        )
        assert role.name == "engineer"
        assert "senior software engineer" in role.prompt

    def test_create_role_with_context(self) -> None:
        """Should create role with additional context and capabilities."""
        role = RoleDefinition(
            name="shakespeare",
            prompt="You are William Shakespeare, the renowned playwright.",
            context={
                "era": "Elizabethan",
                "specialties": ["poetry", "drama", "language"],
                "personality": "eloquent, witty, passionate",
            },
        )
        assert role.name == "shakespeare"
        assert role.context is not None
        assert role.context["era"] == "Elizabethan"
        assert "poetry" in role.context["specialties"]


class TestRoleManager:
    """Test role management and retrieval."""

    def test_register_role(self) -> None:
        """Should register a new role."""
        manager = RoleManager()
        role = RoleDefinition(
            name="designer",
            prompt="You are a creative designer focused on user experience.",
        )
        manager.register_role(role)
        assert "designer" in manager.roles

    def test_get_role(self) -> None:
        """Should retrieve registered role."""
        manager = RoleManager()
        role = RoleDefinition(
            name="artist", prompt="You are a creative artist exploring digital mediums."
        )
        manager.register_role(role)
        retrieved = manager.get_role("artist")
        assert retrieved.name == "artist"
        assert "creative artist" in retrieved.prompt

    def test_get_nonexistent_role_raises_error(self) -> None:
        """Should raise error for non-existent role."""
        manager = RoleManager()
        with pytest.raises(KeyError):
            manager.get_role("nonexistent")
