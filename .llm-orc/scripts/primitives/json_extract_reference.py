#!/usr/bin/env python3
"""JSON Extract Reference Script - Demonstrates ScriptContract ADR-003 patterns."""

import asyncio
import json
import sys
from typing import Any

from pydantic import BaseModel, Field

# For standalone usage, if llm_orc is not installed
try:
    from llm_orc.contracts.script_contract import (
        ScriptCapability,
        ScriptContract,
        ScriptMetadata,
        TestCase,
    )
except ImportError:
    # Fallback for standalone execution
    print("Warning: llm_orc not found. This is a reference implementation.", file=sys.stderr)
    sys.exit(1)


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


class JsonExtractReferenceScript(ScriptContract):
    """Reference implementation for JSON field extraction following ADR-003."""

    @property
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        return ScriptMetadata(
            name="json_extract_reference",
            version="1.0.0",
            description="Reference: Extract specified fields from JSON data",
            author="llm-orchestra-team",
            category="data_transformation",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION],
            tags=["json", "extract", "reference", "primitive", "adr-003"],
            examples=[
                {
                    "name": "basic_field_extraction",
                    "description": "Extract name and age from JSON object",
                    "input": {
                        "json_data": '{"name": "Alice", "age": 30, "city": "NYC"}',
                        "fields": ["name", "age"]
                    },
                    "output": {
                        "success": True,
                        "extracted_data": {"name": "Alice", "age": 30}
                    },
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
        """Execute JSON field extraction with proper error handling."""
        # Validate and cast input to expected type
        if not isinstance(input_data, JsonExtractInput):
            validated_input = JsonExtractInput(**input_data.model_dump())
        else:
            validated_input = input_data

        try:
            # Parse JSON
            data = json.loads(validated_input.json_data)

            # Extract requested fields (only existing ones)
            extracted = {}
            for field in validated_input.fields:
                if field in data:
                    extracted[field] = data[field]

            return JsonExtractOutput(success=True, extracted_data=extracted)

        except json.JSONDecodeError as e:
            return JsonExtractOutput(
                success=False, error=f"Invalid JSON format: {e}"
            )
        except Exception as e:
            return JsonExtractOutput(
                success=False, error=f"Extraction failed: {e}"
            )

    def get_test_cases(self) -> list[TestCase]:
        """Return comprehensive test cases for contract validation."""
        return [
            TestCase(
                name="successful_field_extraction",
                description="Extract multiple fields from valid JSON",
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
                name="invalid_json_handling",
                description="Gracefully handle malformed JSON",
                input_data={
                    "json_data": "{invalid json}",
                    "fields": ["name"],
                },
                expected_output={"success": False},
                should_succeed=False,
            ),
            TestCase(
                name="partial_field_extraction",
                description="Extract only available fields",
                input_data={
                    "json_data": '{"name": "Bob", "email": "bob@example.com"}',
                    "fields": ["name", "age", "email"],  # age missing
                },
                expected_output={
                    "success": True,
                    "extracted_data": {"name": "Bob", "email": "bob@example.com"},
                },
            ),
            TestCase(
                name="empty_json_object",
                description="Handle empty JSON object",
                input_data={
                    "json_data": '{}',
                    "fields": ["name"],
                },
                expected_output={
                    "success": True,
                    "extracted_data": {},
                },
            ),
        ]


# Standalone execution for testing
async def main() -> None:
    """Main function for standalone script execution."""
    if len(sys.argv) < 3:
        print(
            "Usage: python json_extract_reference.py '<json_string>' 'field1,field2,field3'"
        )
        print("Example: python json_extract_reference.py '{\"name\":\"Alice\",\"age\":30}' 'name,age'")
        sys.exit(1)

    json_data = sys.argv[1]
    fields = sys.argv[2].split(",")

    script = JsonExtractReferenceScript()
    input_data = JsonExtractInput(json_data=json_data, fields=fields)

    result = await script.execute(input_data)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())