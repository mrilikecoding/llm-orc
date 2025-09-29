"""File Write Script - Reference implementation for ScriptContract composition."""

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)


class FileWriteInput(BaseModel):
    """Input schema for file writing operations."""

    file_path: str = Field(..., description="Path where file should be written")
    content: dict[str, Any] = Field(..., description="JSON data to write to file")
    format_json: bool = Field(default=True, description="Format JSON with indentation")


class FileWriteOutput(BaseModel):
    """Output schema for file writing operations."""

    success: bool
    file_path: str | None = None
    bytes_written: int = 0
    error: str | None = None


class FileWriteScript(ScriptContract):
    """Reference implementation for file writing that composes with JsonExtract."""

    @property
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        return ScriptMetadata(
            name="file_write",
            version="1.0.0",
            description="Write JSON data to file",
            author="llm-orchestra",
            category="file_operations",
            capabilities=[ScriptCapability.FILE_OPERATIONS],
            tags=["file", "write", "json", "composition"],
            examples=[
                {
                    "name": "write_json_data",
                    "input": {
                        "file_path": f"{tempfile.gettempdir()}/output.json",
                        "content": {"name": "Alice", "age": 30},
                        "format_json": True,
                    },
                    "output": {
                        "success": True,
                        "file_path": f"{tempfile.gettempdir()}/output.json",
                        "bytes_written": 42,
                    },
                }
            ],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        """Input schema for validation."""
        return FileWriteInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        """Output schema for validation."""
        return FileWriteOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute file writing."""
        # Validate and cast input to expected type
        if not isinstance(input_data, FileWriteInput):
            validated_input = FileWriteInput(**input_data.model_dump())
        else:
            validated_input = input_data

        try:
            # Create parent directories if needed
            file_path = Path(validated_input.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON content
            if validated_input.format_json:
                content_str = json.dumps(validated_input.content, indent=2)
            else:
                content_str = json.dumps(validated_input.content)

            with file_path.open("w") as f:
                f.write(content_str)

            return FileWriteOutput(
                success=True,
                file_path=str(file_path),
                bytes_written=len(content_str.encode()),
            )

        except Exception as e:
            return FileWriteOutput(
                success=False,
                error=f"File write failed: {e}",
            )

    def get_test_cases(self) -> list[TestCase]:
        """Return test cases for contract validation.

        Uses tempfile.gettempdir() to avoid hardcoded /tmp paths (CWE-377).
        This prevents predictable file paths that could lead to:
        - Race conditions
        - Symlink attacks
        - Conflicts with other processes
        """
        temp_dir = tempfile.gettempdir()
        return [
            TestCase(
                name="successful_write",
                description="Write JSON data to file successfully",
                input_data={
                    "file_path": f"{temp_dir}/test_output.json",
                    "content": {"name": "Alice", "age": 30},
                    "format_json": True,
                },
                expected_output={
                    "success": True,
                    "file_path": f"{temp_dir}/test_output.json",
                },
            ),
            TestCase(
                name="unformatted_json",
                description="Write compact JSON without formatting",
                input_data={
                    "file_path": f"{temp_dir}/compact.json",
                    "content": {"test": "data", "compact": True},
                    "format_json": False,
                },
                expected_output={
                    "success": True,
                    "file_path": f"{temp_dir}/compact.json",
                },
            ),
            TestCase(
                name="invalid_path_error",
                description="Handle invalid file path gracefully",
                input_data={
                    "file_path": "/invalid/nonexistent/deeply/nested/path/test.json",
                    "content": {"test": "data"},
                },
                expected_output={"success": False},
                should_succeed=False,
            ),
        ]
