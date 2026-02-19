"""Unit tests for role management system."""

from llm_orc.core.config.roles import RoleDefinition


class TestRoleDefinition:
    """Test suite for RoleDefinition dataclass."""

    def test_role_definition_creation_minimal(self) -> None:
        """Test creating a role definition with minimal required fields."""
        role = RoleDefinition(name="test_role", prompt="Test prompt")

        assert role.name == "test_role"
        assert role.prompt == "Test prompt"
        assert role.context is None

    def test_role_definition_creation_with_context(self) -> None:
        """Test creating a role definition with context."""
        context = {"temperature": 0.7, "max_tokens": 100}
        role = RoleDefinition(
            name="contextual_role", prompt="Contextual prompt", context=context
        )

        assert role.name == "contextual_role"
        assert role.prompt == "Contextual prompt"
        assert role.context == context

    def test_role_definition_immutability_features(self) -> None:
        """Test dataclass features and field access."""
        role = RoleDefinition(name="mutable_role", prompt="Original prompt")

        # Test field modification
        role.name = "modified_role"
        role.prompt = "Modified prompt"
        role.context = {"new": "context"}

        assert role.name == "modified_role"
        assert role.prompt == "Modified prompt"
        assert role.context == {"new": "context"}

    def test_role_definition_equality(self) -> None:
        """Test role definition equality comparison."""
        role1 = RoleDefinition(name="equal_role", prompt="Same prompt")
        role2 = RoleDefinition(name="equal_role", prompt="Same prompt")
        role3 = RoleDefinition(name="different_role", prompt="Same prompt")

        assert role1 == role2
        assert role1 != role3

    def test_role_definition_with_empty_values(self) -> None:
        """Test role definition with empty string values."""
        role = RoleDefinition(name="", prompt="")

        assert role.name == ""
        assert role.prompt == ""
        assert role.context is None

    def test_role_definition_with_complex_context(self) -> None:
        """Test role definition with complex context data."""
        complex_context = {
            "model_params": {"temperature": 0.8, "max_tokens": 150},
            "instructions": ["Be helpful", "Be accurate"],
            "metadata": {"version": "1.0", "author": "test"},
            "nested": {"deep": {"value": 42}},
        }

        role = RoleDefinition(
            name="complex_role", prompt="Complex prompt", context=complex_context
        )

        assert role.context == complex_context
        assert role.context is not None
        assert role.context["model_params"]["temperature"] == 0.8
        assert role.context["instructions"] == ["Be helpful", "Be accurate"]
        assert role.context["nested"]["deep"]["value"] == 42

    def test_role_definition_repr(self) -> None:
        """Test string representation of role definition."""
        role = RoleDefinition(name="repr_role", prompt="Repr prompt")

        repr_str = repr(role)
        assert "RoleDefinition" in repr_str
        assert "repr_role" in repr_str
        assert "Repr prompt" in repr_str
