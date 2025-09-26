"""Unit tests for role management system."""

import pytest

from llm_orc.core.config.roles import RoleDefinition, RoleManager


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


class TestRoleManager:
    """Test suite for RoleManager class."""

    def test_role_manager_initialization(self) -> None:
        """Test role manager initialization."""
        manager = RoleManager()

        assert isinstance(manager.roles, dict)
        assert len(manager.roles) == 0

    def test_register_single_role(self) -> None:
        """Test registering a single role."""
        manager = RoleManager()
        role = RoleDefinition(name="single_role", prompt="Single prompt")

        manager.register_role(role)

        assert len(manager.roles) == 1
        assert "single_role" in manager.roles
        assert manager.roles["single_role"] == role

    def test_register_multiple_roles(self) -> None:
        """Test registering multiple roles."""
        manager = RoleManager()

        role1 = RoleDefinition(name="role1", prompt="Prompt 1")
        role2 = RoleDefinition(name="role2", prompt="Prompt 2")
        role3 = RoleDefinition(name="role3", prompt="Prompt 3")

        manager.register_role(role1)
        manager.register_role(role2)
        manager.register_role(role3)

        assert len(manager.roles) == 3
        assert manager.roles["role1"] == role1
        assert manager.roles["role2"] == role2
        assert manager.roles["role3"] == role3

    def test_register_role_overwrite(self) -> None:
        """Test overwriting an existing role."""
        manager = RoleManager()

        original_role = RoleDefinition(name="overwrite_role", prompt="Original")
        new_role = RoleDefinition(name="overwrite_role", prompt="New")

        manager.register_role(original_role)
        assert manager.roles["overwrite_role"].prompt == "Original"

        manager.register_role(new_role)
        assert manager.roles["overwrite_role"].prompt == "New"
        assert len(manager.roles) == 1

    def test_get_existing_role(self) -> None:
        """Test retrieving an existing role."""
        manager = RoleManager()
        role = RoleDefinition(name="existing_role", prompt="Existing prompt")

        manager.register_role(role)
        retrieved_role = manager.get_role("existing_role")

        assert retrieved_role == role
        assert retrieved_role.name == "existing_role"
        assert retrieved_role.prompt == "Existing prompt"

    def test_get_nonexistent_role(self) -> None:
        """Test retrieving a non-existent role raises KeyError."""
        manager = RoleManager()

        with pytest.raises(KeyError, match="Role 'nonexistent' not found"):
            manager.get_role("nonexistent")

    def test_get_role_after_registration(self) -> None:
        """Test retrieving role immediately after registration."""
        manager = RoleManager()
        role = RoleDefinition(
            name="immediate_role",
            prompt="Immediate prompt",
            context={"immediate": True},
        )

        manager.register_role(role)
        retrieved = manager.get_role("immediate_role")

        assert retrieved is role  # Same instance
        assert retrieved.context is not None
        assert retrieved.context["immediate"] is True

    def test_role_manager_isolation(self) -> None:
        """Test that different role managers are isolated."""
        manager1 = RoleManager()
        manager2 = RoleManager()

        role1 = RoleDefinition(name="manager1_role", prompt="Manager 1")
        role2 = RoleDefinition(name="manager2_role", prompt="Manager 2")

        manager1.register_role(role1)
        manager2.register_role(role2)

        assert len(manager1.roles) == 1
        assert len(manager2.roles) == 1
        assert "manager1_role" in manager1.roles
        assert "manager2_role" in manager2.roles
        assert "manager2_role" not in manager1.roles
        assert "manager1_role" not in manager2.roles

    def test_role_manager_with_complex_roles(self) -> None:
        """Test role manager with roles containing complex context."""
        manager = RoleManager()

        analyst_role = RoleDefinition(
            name="data_analyst",
            prompt="You are a skilled data analyst. Analyze the provided data.",
            context={
                "expertise": ["statistics", "visualization", "machine_learning"],
                "tools": {"preferred": "python", "libraries": ["pandas", "numpy"]},
                "output_format": "markdown",
            },
        )

        writer_role = RoleDefinition(
            name="technical_writer",
            prompt="You are a technical writer. Create clear documentation.",
            context={
                "style": "concise",
                "audience": "developers",
                "formats": ["markdown", "rst"],
            },
        )

        manager.register_role(analyst_role)
        manager.register_role(writer_role)

        retrieved_analyst = manager.get_role("data_analyst")
        retrieved_writer = manager.get_role("technical_writer")

        assert retrieved_analyst.context is not None
        assert retrieved_analyst.context["expertise"] == [
            "statistics",
            "visualization",
            "machine_learning",
        ]
        assert retrieved_writer.context is not None
        assert retrieved_writer.context["audience"] == "developers"

    def test_role_name_case_sensitivity(self) -> None:
        """Test that role names are case-sensitive."""
        manager = RoleManager()

        role_lower = RoleDefinition(name="lowercase", prompt="Lower case role")
        role_upper = RoleDefinition(name="LOWERCASE", prompt="Upper case role")

        manager.register_role(role_lower)
        manager.register_role(role_upper)

        assert len(manager.roles) == 2
        assert manager.get_role("lowercase").prompt == "Lower case role"
        assert manager.get_role("LOWERCASE").prompt == "Upper case role"

        with pytest.raises(KeyError):
            manager.get_role("Lowercase")

    def test_role_name_special_characters(self) -> None:
        """Test role names with special characters."""
        manager = RoleManager()

        special_role = RoleDefinition(
            name="role@domain.com", prompt="Role with email-like name"
        )
        path_role = RoleDefinition(name="role/path/name", prompt="Role with path")
        symbol_role = RoleDefinition(name="role-with_symbols", prompt="Role symbols")

        manager.register_role(special_role)
        manager.register_role(path_role)
        manager.register_role(symbol_role)

        assert manager.get_role("role@domain.com").prompt == "Role with email-like name"
        assert manager.get_role("role/path/name").prompt == "Role with path"
        assert manager.get_role("role-with_symbols").prompt == "Role symbols"

    def test_role_manager_workflow(self) -> None:
        """Test a complete workflow with role manager."""
        manager = RoleManager()

        # Register roles for a conversational system
        system_role = RoleDefinition(
            name="system",
            prompt="You are a helpful AI assistant.",
            context={"behavior": "professional", "tone": "friendly"},
        )

        expert_role = RoleDefinition(
            name="domain_expert",
            prompt="You are an expert in the specified domain.",
            context={"adaptable": True, "domains": []},
        )

        manager.register_role(system_role)
        manager.register_role(expert_role)

        # Simulate usage
        active_role = manager.get_role("system")
        assert active_role.context is not None
        assert active_role.context["behavior"] == "professional"

        # Switch to expert role
        expert = manager.get_role("domain_expert")
        assert expert.context is not None
        expert.context["domains"] = ["machine_learning", "data_science"]

        # Verify state
        updated_expert = manager.get_role("domain_expert")
        assert updated_expert.context is not None
        assert "machine_learning" in updated_expert.context["domains"]

    def test_empty_role_names_and_prompts(self) -> None:
        """Test handling of empty role names and prompts."""
        manager = RoleManager()

        empty_name_role = RoleDefinition(name="", prompt="Valid prompt")
        empty_prompt_role = RoleDefinition(name="valid_name", prompt="")

        manager.register_role(empty_name_role)
        manager.register_role(empty_prompt_role)

        assert manager.get_role("").prompt == "Valid prompt"
        assert manager.get_role("valid_name").prompt == ""
