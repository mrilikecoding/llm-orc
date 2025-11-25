"""Unit tests for FileWrite Script reference implementation - ADR-003."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from llm_orc.reference.file_write_script import (
    FileWriteInput,
    FileWriteOutput,
    FileWriteScript,
)


class TestFileWriteScript:
    """Test the FileWriteScript reference implementation."""

    @pytest.fixture
    def script(self) -> FileWriteScript:
        """Provide a FileWriteScript instance."""
        return FileWriteScript()

    @pytest.fixture
    def temp_dir(self) -> Any:
        """Provide a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_script_contract_compliance(self, script: FileWriteScript) -> None:
        """Test that FileWriteScript implements ScriptContract correctly."""
        # Test metadata
        metadata = script.metadata
        assert metadata.name == "file_write"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Write JSON data to file"
        assert "file_operations" in str(metadata.capabilities)

        # Test schemas
        assert script.input_schema() == FileWriteInput
        assert script.output_schema() == FileWriteOutput

        # Test test cases
        test_cases = script.get_test_cases()
        assert len(test_cases) >= 2  # Should have at least success and error cases

    async def test_successful_file_write(
        self, script: FileWriteScript, temp_dir: Path
    ) -> None:
        """Test successful file writing."""
        test_file = temp_dir / "test_output.json"

        input_data = FileWriteInput(
            file_path=str(test_file),
            content={"name": "Alice", "age": 30, "city": "New York"},
            format_json=True,
        )

        result = await script.execute(input_data)

        assert isinstance(result, FileWriteOutput)
        assert result.success is True
        assert result.file_path == str(test_file)
        assert result.bytes_written > 0
        assert result.error is None

        # Verify file was actually written
        assert test_file.exists()
        content = test_file.read_text()
        assert "Alice" in content
        assert "30" in content
        # Should be formatted JSON (with indentation)
        assert "\n" in content

    async def test_unformatted_json_write(
        self, script: FileWriteScript, temp_dir: Path
    ) -> None:
        """Test writing compact JSON without formatting."""
        test_file = temp_dir / "compact.json"

        input_data = FileWriteInput(
            file_path=str(test_file),
            content={"test": "data", "compact": True},
            format_json=False,
        )

        result = await script.execute(input_data)

        assert isinstance(result, FileWriteOutput)
        assert result.success is True
        assert result.file_path == str(test_file)
        assert result.bytes_written > 0

        # Verify compact JSON (no extra whitespace)
        content = test_file.read_text()
        assert content == '{"test": "data", "compact": true}'

    async def test_directory_creation(
        self, script: FileWriteScript, temp_dir: Path
    ) -> None:
        """Test that parent directories are created if needed."""
        nested_file = temp_dir / "nested" / "deep" / "test.json"

        input_data = FileWriteInput(
            file_path=str(nested_file),
            content={"nested": "structure"},
            format_json=True,
        )

        result = await script.execute(input_data)

        assert isinstance(result, FileWriteOutput)
        assert result.success is True
        assert nested_file.exists()
        assert nested_file.parent.exists()

    async def test_invalid_path_error(self, script: FileWriteScript) -> None:
        """Test error handling for invalid paths."""
        # Try to write to a path that should fail
        input_data = FileWriteInput(
            file_path="/root/should/not/be/writable/test.json",  # Likely to fail
            content={"test": "data"},
        )

        result = await script.execute(input_data)

        assert isinstance(result, FileWriteOutput)
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()

    async def test_schema_validation(self, script: FileWriteScript) -> None:
        """Test input schema validation."""
        temp_dir = tempfile.gettempdir()

        # Valid input
        test_path = f"{temp_dir}/test.json"
        valid_input = FileWriteInput(file_path=test_path, content={"valid": "data"})
        assert valid_input.file_path == test_path
        assert valid_input.content == {"valid": "data"}
        assert valid_input.format_json is True  # Default

        # Test with format_json=False
        compact_path = f"{temp_dir}/compact.json"
        compact_input = FileWriteInput(
            file_path=compact_path, content={"compact": True}, format_json=False
        )
        assert compact_input.format_json is False

    def test_composition_compatibility_with_json_extract(
        self, script: FileWriteScript
    ) -> None:
        """Test that FileWriteScript can receive output from JsonExtractScript."""
        from llm_orc.reference.json_extract_script import JsonExtractScript

        json_extract = JsonExtractScript()

        # Get output schema from JsonExtract
        extract_output_schema = json_extract.output_schema()
        write_input_schema = script.input_schema()

        # Verify schemas are compatible for composition
        # JsonExtract outputs: success, extracted_data, error
        # FileWrite expects: file_path, content, format_json

        # The extracted_data field should be compatible with content field
        extract_fields = extract_output_schema.model_fields
        write_fields = write_input_schema.model_fields

        assert "extracted_data" in extract_fields
        assert "content" in write_fields

        # Both should handle dict[str, Any] type content
        # This tests our composition validation logic
