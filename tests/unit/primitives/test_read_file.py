"""Tests for read_file primitive."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_orc.primitives.file_ops.read_file import (
    ReadFileInput,
    ReadFileOutput,
    execute,
)


class TestReadFileInput:
    """Test input model validation."""

    def test_valid_input_with_defaults(self) -> None:
        model = ReadFileInput(path="input.txt")
        assert model.path == "input.txt"
        assert model.encoding == "utf-8"

    def test_custom_encoding(self) -> None:
        model = ReadFileInput(path="data.csv", encoding="latin-1")
        assert model.encoding == "latin-1"

    def test_path_required(self) -> None:
        with pytest.raises(ValidationError):
            ReadFileInput()  # type: ignore[call-arg]


class TestReadFileOutput:
    """Test output model and field name compatibility."""

    def test_success_output_fields(self) -> None:
        output = ReadFileOutput(
            success=True,
            content="hello world",
            path="test.txt",
            size=11,
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["content"] == "hello world"
        assert data["path"] == "test.txt"
        assert data["size"] == 11

    def test_error_output_fields(self) -> None:
        output = ReadFileOutput(
            success=False,
            error="File not found",
            path="missing.txt",
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["path"] == "missing.txt"


class TestReadFileExecute:
    """Test execute function."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = execute(ReadFileInput(path=str(test_file)))
        assert result.success is True
        assert result.content == "hello world"
        assert result.path == str(test_file)
        assert result.size == 11

    def test_read_nonexistent_file(self) -> None:
        result = execute(ReadFileInput(path="/nonexistent/path.txt"))
        assert result.success is False
        assert result.error is not None
        assert result.path == "/nonexistent/path.txt"

    def test_read_with_encoding(self, tmp_path: Path) -> None:
        test_file = tmp_path / "encoded.txt"
        test_file.write_text("caf\u00e9", encoding="utf-8")

        result = execute(ReadFileInput(path=str(test_file), encoding="utf-8"))
        assert result.success is True
        assert result.content == "caf\u00e9"
