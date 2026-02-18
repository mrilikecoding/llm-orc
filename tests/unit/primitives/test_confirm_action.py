"""Tests for confirm_action primitive."""

import json

import pytest
from pydantic import ValidationError

from llm_orc.primitives.user_interaction.confirm_action import (
    ConfirmActionInput,
    ConfirmActionOutput,
    execute,
)


class TestConfirmActionInput:
    """Test input model validation."""

    def test_valid_input_with_defaults(self) -> None:
        model = ConfirmActionInput(prompt="Continue?")
        assert model.prompt == "Continue?"
        assert model.default == "n"

    def test_valid_input_custom_default(self) -> None:
        model = ConfirmActionInput(prompt="Proceed?", default="y")
        assert model.default == "y"

    def test_prompt_required(self) -> None:
        with pytest.raises(ValidationError):
            ConfirmActionInput()  # type: ignore[call-arg]

    def test_pre_collected_input(self) -> None:
        model = ConfirmActionInput(prompt="Continue?", pre_collected="y")
        assert model.pre_collected == "y"


class TestConfirmActionOutput:
    """Test output model and field name compatibility."""

    def test_confirmed_output_fields(self) -> None:
        output = ConfirmActionOutput(
            success=True,
            confirmed=True,
            input="y",
            prompt="Continue?",
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["confirmed"] is True
        assert data["input"] == "y"
        assert data["prompt"] == "Continue?"

    def test_declined_output_fields(self) -> None:
        output = ConfirmActionOutput(
            success=True,
            confirmed=False,
            input="n",
            prompt="Continue?",
        )
        data = json.loads(output.model_dump_json())
        assert data["confirmed"] is False

    def test_error_output_fields(self) -> None:
        output = ConfirmActionOutput(
            success=False,
            confirmed=False,
            error="User cancelled confirmation",
            input="",
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["error"] == "User cancelled confirmation"


class TestConfirmActionExecute:
    """Test execute function."""

    def test_pre_collected_yes(self) -> None:
        result = execute(ConfirmActionInput(prompt="Continue?", pre_collected="y"))
        assert result.success is True
        assert result.confirmed is True
        assert result.input == "y"

    def test_pre_collected_yes_full(self) -> None:
        result = execute(ConfirmActionInput(prompt="Continue?", pre_collected="yes"))
        assert result.confirmed is True

    def test_pre_collected_no(self) -> None:
        result = execute(ConfirmActionInput(prompt="Continue?", pre_collected="n"))
        assert result.confirmed is False
        assert result.input == "n"

    def test_pre_collected_empty_uses_default_n(self) -> None:
        result = execute(
            ConfirmActionInput(prompt="Continue?", default="n", pre_collected="")
        )
        assert result.confirmed is False

    def test_pre_collected_empty_uses_default_y(self) -> None:
        result = execute(
            ConfirmActionInput(prompt="Continue?", default="y", pre_collected="")
        )
        assert result.confirmed is True
