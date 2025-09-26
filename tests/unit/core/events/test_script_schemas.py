"""Tests for script validation schemas using TDD methodology."""

import pytest
from pydantic import ValidationError

from llm_orc.core.events.script_schemas import GetUserInputScript


class TestGetUserInputScript:
    """Test suite for GetUserInputScript Pydantic schema."""

    def test_get_user_input_script_creation_with_required_fields(self) -> None:
        """Test that GetUserInputScript can be created with all required fields."""
        # Given
        prompt = "Enter your name:"
        script_name = "get_user_input"

        # When
        script = GetUserInputScript(
            prompt=prompt,
            script_name=script_name,
        )

        # Then
        assert script.prompt == prompt
        assert script.script_name == script_name
        assert script.input_type == "string"  # default value
        assert script.default_value is None  # default value

    def test_get_user_input_script_optional_fields(self) -> None:
        """Test GetUserInputScript with optional fields."""
        # Given
        prompt = "Enter value:"
        script_name = "get_user_input"
        default_value = "default"
        input_type = "number"

        # When
        script = GetUserInputScript(
            prompt=prompt,
            script_name=script_name,
            default_value=default_value,
            input_type=input_type,
        )

        # Then
        assert script.prompt == prompt
        assert script.script_name == script_name
        assert script.default_value == default_value
        assert script.input_type == input_type

    def test_get_user_input_script_validation_requires_prompt(self) -> None:
        """Test that GetUserInputScript validation fails without prompt."""
        # When/Then
        with pytest.raises(ValidationError):
            GetUserInputScript(script_name="get_user_input")  # type: ignore[call-arg]

    def test_get_user_input_script_validation_requires_script_name(self) -> None:
        """Test that GetUserInputScript validation fails without script_name."""
        # When/Then
        with pytest.raises(ValidationError):
            GetUserInputScript(prompt="Enter value:")  # type: ignore[call-arg]

    def test_get_user_input_script_to_dict_serialization(self) -> None:
        """Test that GetUserInputScript can be serialized to dict."""
        # Given
        script = GetUserInputScript(
            prompt="Enter name:",
            script_name="get_user_input",
            default_value="John",
        )

        # When
        result = script.model_dump()

        # Then
        assert isinstance(result, dict)
        assert result["prompt"] == "Enter name:"
        assert result["script_name"] == "get_user_input"
        assert result["default_value"] == "John"
        assert result["input_type"] == "string"
