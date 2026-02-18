"""Tests for get_user_input primitive."""

import json

import pytest
from pydantic import ValidationError

from llm_orc.primitives.user_interaction.get_user_input import (
    GetUserInputInput,
    GetUserInputOutput,
    execute,
)


class TestGetUserInputInput:
    """Test input model validation."""

    def test_valid_input_with_defaults(self) -> None:
        model = GetUserInputInput(prompt="Enter name:")
        assert model.prompt == "Enter name:"
        assert model.multiline is False

    def test_valid_input_multiline(self) -> None:
        model = GetUserInputInput(prompt="Enter text:", multiline=True)
        assert model.multiline is True

    def test_prompt_required(self) -> None:
        with pytest.raises(ValidationError):
            GetUserInputInput()  # type: ignore[call-arg]

    def test_pre_collected_input(self) -> None:
        model = GetUserInputInput(prompt="Enter:", pre_collected="already have it")
        assert model.pre_collected == "already have it"


class TestGetUserInputOutput:
    """Test output model and field name compatibility."""

    def test_success_output_fields(self) -> None:
        output = GetUserInputOutput(
            success=True, input="hello", multiline=False, length=5
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["input"] == "hello"
        assert data["multiline"] is False
        assert data["length"] == 5

    def test_error_output_fields(self) -> None:
        output = GetUserInputOutput(
            success=False, error="User cancelled input", input=""
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["error"] == "User cancelled input"
        assert data["input"] == ""


class TestGetUserInputExecute:
    """Test execute function."""

    def test_pre_collected_skips_interactive(self) -> None:
        result = execute(GetUserInputInput(prompt="Enter:", pre_collected="pre-filled"))
        assert result.success is True
        assert result.input == "pre-filled"
        assert result.length == 10

    def test_pre_collected_multiline(self) -> None:
        result = execute(
            GetUserInputInput(
                prompt="Enter:",
                multiline=True,
                pre_collected="line1\nline2",
            )
        )
        assert result.success is True
        assert result.input == "line1\nline2"
        assert result.multiline is True
        assert result.length == 11

    def test_empty_pre_collected_returns_empty(self) -> None:
        result = execute(GetUserInputInput(prompt="Enter:", pre_collected=""))
        assert result.success is True
        assert result.input == ""
        assert result.length == 0
