"""Tests for JsonExtractScript reference implementation - ADR-003."""

from typing import TYPE_CHECKING, cast

import pytest

from llm_orc.contracts.contract_validator import ContractValidator

if TYPE_CHECKING:
    pass


class TestJsonExtractScriptReference:
    """Test cases for JsonExtractScript reference implementation."""

    def test_json_extract_script_imports_successfully(self) -> None:
        """Test that JsonExtractScript can be imported after creation."""
        # This will initially fail, then pass after implementation
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            assert JsonExtractScript is not None
        except (ModuleNotFoundError, ImportError):
            pytest.fail("JsonExtractScript module not found - needs implementation")

    @pytest.mark.asyncio
    async def test_json_extract_basic_functionality(self) -> None:
        """Test basic JSON field extraction functionality."""
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            script = JsonExtractScript()
            input_data = script.input_schema()(
                json_data='{"name": "Alice", "age": 30}', fields=["name", "age"]
            )

            result = await script.execute(input_data)
            assert isinstance(result, script.output_schema())

            # Cast for type checking
            from llm_orc.reference.json_extract_script import JsonExtractOutput

            output_result = cast(JsonExtractOutput, result)
            assert output_result.success is True
            assert "name" in output_result.extracted_data
            assert "age" in output_result.extracted_data
            assert output_result.extracted_data["name"] == "Alice"
            assert output_result.extracted_data["age"] == 30

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")

    @pytest.mark.asyncio
    async def test_json_extract_input_validation(self) -> None:
        """Test input schema validation."""
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            script = JsonExtractScript()

            # Test that schema accepts valid input
            from llm_orc.reference.json_extract_script import (
                JsonExtractInput,
                JsonExtractOutput,
            )

            input_data = script.input_schema()(
                json_data='{"name": "Alice"}', fields=["name"]
            )
            typed_input = cast(JsonExtractInput, input_data)
            assert typed_input.json_data == '{"name": "Alice"}'
            assert typed_input.fields == ["name"]

            # Test that invalid JSON is handled gracefully in execution
            input_data = script.input_schema()(
                json_data="invalid json", fields=["name"]
            )
            result = await script.execute(input_data)
            typed_result = cast(JsonExtractOutput, result)
            assert typed_result.success is False
            assert typed_result.error is not None

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")

    @pytest.mark.asyncio
    async def test_json_extract_error_handling(self) -> None:
        """Test error handling for invalid JSON."""
        try:
            from llm_orc.reference.json_extract_script import (
                JsonExtractOutput,
                JsonExtractScript,
            )

            script = JsonExtractScript()
            input_data = script.input_schema()(
                json_data="invalid json", fields=["name"]
            )

            result = await script.execute(input_data)
            typed_result = cast(JsonExtractOutput, result)
            assert typed_result.success is False
            assert typed_result.error is not None

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")

    def test_json_extract_contract_validation(self) -> None:
        """Test that JsonExtractScript passes contract validation."""
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            validator = ContractValidator()
            result = validator.validate_all_scripts([JsonExtractScript])

            if not result:
                pytest.fail(
                    f"Contract validation failed: {validator.validation_errors}"
                )

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")

    def test_json_extract_metadata_completeness(self) -> None:
        """Test that metadata is complete and correct."""
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            script = JsonExtractScript()
            metadata = script.metadata

            assert metadata.name == "json_extract"
            assert metadata.version is not None
            assert metadata.description is not None
            assert metadata.author is not None
            assert metadata.category == "data_transformation"
            assert len(metadata.capabilities) > 0

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")

    def test_json_extract_test_cases_provided(self) -> None:
        """Test that script provides comprehensive test cases."""
        try:
            from llm_orc.reference.json_extract_script import JsonExtractScript

            script = JsonExtractScript()
            test_cases = script.get_test_cases()

            assert len(test_cases) >= 2  # Success and failure cases
            success_cases = [tc for tc in test_cases if tc.should_succeed]
            failure_cases = [tc for tc in test_cases if not tc.should_succeed]

            assert len(success_cases) >= 1
            assert len(failure_cases) >= 1

        except (ModuleNotFoundError, ImportError):
            pytest.skip("Implementation not created yet")
