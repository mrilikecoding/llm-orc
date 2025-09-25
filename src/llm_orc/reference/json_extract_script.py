"""JSON Extract Script - Reference implementation for ScriptContract ADR-003."""

import json
from typing import Any

from pydantic import BaseModel, Field

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)


class JsonExtractInput(BaseModel):
    """Input schema for JSON field extraction."""

    json_data: str = Field(..., description="JSON string to parse and extract from")
    fields: list[str] = Field(
        ..., description="List of field names to extract from JSON"
    )


class JsonExtractOutput(BaseModel):
    """Output schema for JSON field extraction."""

    success: bool
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JsonExtractScript(ScriptContract):
    """Reference implementation for JSON field extraction."""

    @property
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        return ScriptMetadata(
            name="json_extract",
            version="1.0.0",
            description="Extract specified fields from JSON data",
            author="llm-orchestra",
            category="data_transformation",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION],
            tags=["json", "extract", "reference", "data"],
            examples=[
                {
                    "name": "basic_extraction",
                    "input": {
                        "json_data": '{"name": "Alice", "age": 30}',
                        "fields": ["name"],
                    },
                    "output": {"success": True, "extracted_data": {"name": "Alice"}},
                }
            ],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        """Input schema for validation."""
        return JsonExtractInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        """Output schema for validation."""
        return JsonExtractOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute JSON field extraction."""
        # Validate and cast input to expected type
        if not isinstance(input_data, JsonExtractInput):
            validated_input = JsonExtractInput(**input_data.model_dump())
        else:
            validated_input = input_data

        try:
            # Parse JSON
            data = json.loads(validated_input.json_data)

            # Extract requested fields
            extracted = {}
            for field in validated_input.fields:
                if field in data:
                    extracted[field] = data[field]

            return JsonExtractOutput(success=True, extracted_data=extracted)

        except json.JSONDecodeError as e:
            return JsonExtractOutput(success=False, error=f"Invalid JSON: {e}")
        except Exception as e:
            return JsonExtractOutput(success=False, error=f"Extraction failed: {e}")

    def get_test_cases(self) -> list[TestCase]:
        """Return test cases for contract validation."""
        return [
            TestCase(
                name="successful_extraction",
                description="Extract fields from valid JSON",
                input_data={
                    "json_data": '{"name": "Alice", "age": 30, "city": "New York"}',
                    "fields": ["name", "age"],
                },
                expected_output={
                    "success": True,
                    "extracted_data": {"name": "Alice", "age": 30},
                },
            ),
            TestCase(
                name="invalid_json",
                description="Handle invalid JSON gracefully",
                input_data={
                    "json_data": "invalid json",
                    "fields": ["name"],
                },
                expected_output={"success": False},
                should_succeed=False,
            ),
            TestCase(
                name="missing_fields",
                description="Extract available fields only",
                input_data={
                    "json_data": '{"name": "Bob"}',
                    "fields": ["name", "age"],
                },
                expected_output={
                    "success": True,
                    "extracted_data": {"name": "Bob"},
                },
            ),
        ]
