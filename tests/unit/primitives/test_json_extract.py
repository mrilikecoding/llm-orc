"""Tests for json_extract primitive."""

import json

import pytest
from pydantic import ValidationError

from llm_orc.primitives.data_transform.json_extract import (
    JsonExtractInput,
    JsonExtractOutput,
    execute,
)


class TestJsonExtractInput:
    """Test input model validation."""

    def test_valid_input(self) -> None:
        model = JsonExtractInput(json_data='{"name": "Alice"}', fields=["name"])
        assert model.json_data == '{"name": "Alice"}'
        assert model.fields == ["name"]

    def test_json_data_required(self) -> None:
        with pytest.raises(ValidationError):
            JsonExtractInput(fields=["name"])  # type: ignore[call-arg]

    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            JsonExtractInput(json_data="{}")  # type: ignore[call-arg]

    def test_dict_json_data(self) -> None:
        model = JsonExtractInput(
            json_data={"name": "Alice"},
            fields=["name"],
        )
        assert model.json_data == {"name": "Alice"}


class TestJsonExtractOutput:
    """Test output model and field name compatibility."""

    def test_success_output_fields(self) -> None:
        output = JsonExtractOutput(
            success=True,
            extracted={"name": "Alice"},
            missing_fields=[],
            total_fields=1,
            extracted_count=1,
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["extracted"] == {"name": "Alice"}
        assert data["missing_fields"] == []
        assert data["total_fields"] == 1
        assert data["extracted_count"] == 1

    def test_error_output_fields(self) -> None:
        output = JsonExtractOutput(
            success=False,
            error="Invalid JSON: ...",
            extracted={},
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["extracted"] == {}


class TestJsonExtractExecute:
    """Test execute function."""

    def test_extract_single_field(self) -> None:
        result = execute(
            JsonExtractInput(
                json_data='{"name": "Alice", "age": 30}',
                fields=["name"],
            )
        )
        assert result.success is True
        assert result.extracted == {"name": "Alice"}
        assert result.extracted_count == 1

    def test_extract_multiple_fields(self) -> None:
        result = execute(
            JsonExtractInput(
                json_data='{"name": "Alice", "age": 30, "city": "NYC"}',
                fields=["name", "age"],
            )
        )
        assert result.success is True
        assert result.extracted == {"name": "Alice", "age": 30}
        assert result.extracted_count == 2
        assert result.total_fields == 2

    def test_missing_fields_tracked(self) -> None:
        result = execute(
            JsonExtractInput(
                json_data='{"name": "Bob"}',
                fields=["name", "age"],
            )
        )
        assert result.success is True
        assert result.extracted == {"name": "Bob"}
        assert result.missing_fields == ["age"]
        assert result.extracted_count == 1
        assert result.total_fields == 2

    def test_invalid_json_string(self) -> None:
        result = execute(
            JsonExtractInput(
                json_data="not valid json",
                fields=["name"],
            )
        )
        assert result.success is False
        assert "Invalid JSON" in (result.error or "")
        assert result.extracted == {}

    def test_dict_input_directly(self) -> None:
        result = execute(
            JsonExtractInput(
                json_data={"x": 1, "y": 2},
                fields=["x"],
            )
        )
        assert result.success is True
        assert result.extracted == {"x": 1}
