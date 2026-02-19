"""Tests for write_file primitive."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_orc.primitives.file_ops.write_file import (
    WriteFileInput,
    WriteFileOutput,
    execute,
)


class TestWriteFileInput:
    """Test input model validation."""

    def test_valid_input_with_defaults(self) -> None:
        model = WriteFileInput(path="output.txt", content="hello")
        assert model.path == "output.txt"
        assert model.content == "hello"
        assert model.encoding == "utf-8"

    def test_path_required(self) -> None:
        with pytest.raises(ValidationError):
            WriteFileInput(content="hello")  # type: ignore[call-arg]

    def test_content_required(self) -> None:
        with pytest.raises(ValidationError):
            WriteFileInput(path="out.txt")  # type: ignore[call-arg]


class TestWriteFileOutput:
    """Test output model and field name compatibility."""

    def test_success_output_fields(self) -> None:
        output = WriteFileOutput(
            success=True,
            path="output.txt",
            size=5,
            bytes_written=5,
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is True
        assert data["path"] == "output.txt"
        assert data["size"] == 5
        assert data["bytes_written"] == 5

    def test_error_output_fields(self) -> None:
        output = WriteFileOutput(
            success=False,
            error="Permission denied",
            path="readonly.txt",
        )
        data = json.loads(output.model_dump_json())
        assert data["success"] is False
        assert data["error"] == "Permission denied"


class TestWriteFileExecute:
    """Test execute function."""

    def test_write_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        result = execute(WriteFileInput(path=str(target), content="hello world"))
        assert result.success is True
        assert result.path == str(target)
        assert result.size == 11
        assert result.bytes_written == 11
        assert target.read_text() == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "dir" / "output.txt"
        result = execute(WriteFileInput(path=str(target), content="nested"))
        assert result.success is True
        assert target.read_text() == "nested"

    def test_write_error_on_invalid_path(self) -> None:
        result = execute(WriteFileInput(path="/proc/nonexistent/file.txt", content="x"))
        assert result.success is False
        assert result.error is not None

    def test_bytes_written_matches_encoding(self, tmp_path: Path) -> None:
        target = tmp_path / "encoded.txt"
        content = "caf\u00e9"
        result = execute(WriteFileInput(path=str(target), content=content))
        assert result.success is True
        assert result.size == 4
        assert result.bytes_written == len(content.encode("utf-8"))
