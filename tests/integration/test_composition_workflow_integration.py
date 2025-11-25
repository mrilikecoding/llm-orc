"""Integration tests for end-to-end script composition workflows - ADR-003."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from llm_orc.contracts.contract_validator import ContractValidator
from llm_orc.reference.file_write_script import FileWriteOutput, FileWriteScript
from llm_orc.reference.json_extract_script import JsonExtractOutput, JsonExtractScript


class TestCompositionWorkflowIntegration:
    """Test end-to-end script composition workflows."""

    @pytest.fixture
    def temp_dir(self) -> Any:
        """Provide a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def validator(self) -> ContractValidator:
        """Provide a ContractValidator instance."""
        return ContractValidator()

    async def test_json_extract_to_file_write_workflow(self, temp_dir: Path) -> None:
        """Test complete JsonExtract → FileWrite workflow."""
        # Step 1: Create script instances
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # Step 2: Execute JsonExtract to get data
        json_data = '{"name": "Alice", "age": 30, "city": "New York", "active": true}'
        extract_input = json_extract.input_schema()(
            json_data=json_data,
            fields=["name", "age", "city"],
        )

        extract_result = await json_extract.execute(extract_input)
        assert isinstance(extract_result, JsonExtractOutput)
        assert extract_result.success is True
        assert extract_result.extracted_data == {
            "name": "Alice",
            "age": 30,
            "city": "New York",
        }

        # Step 3: Use extracted data as input to FileWrite
        output_file = temp_dir / "extracted_output.json"
        write_input = file_write.input_schema()(
            file_path=str(output_file),
            content=extract_result.extracted_data,  # Composition happens here
            format_json=True,
        )

        write_result = await file_write.execute(write_input)
        assert isinstance(write_result, FileWriteOutput)
        assert write_result.success is True
        assert write_result.file_path == str(output_file)
        assert write_result.bytes_written > 0

        # Step 4: Verify the file was written correctly
        assert output_file.exists()
        content = output_file.read_text()
        assert "Alice" in content
        assert "30" in content
        assert "New York" in content
        # Should not contain "active" since it wasn't extracted
        assert "active" not in content

    def test_workflow_composition_validation(
        self, validator: ContractValidator
    ) -> None:
        """Test that the workflow composition is validated correctly."""
        scripts = [JsonExtractScript, FileWriteScript]

        # Validate the composition
        result = validator.validate_composition_workflow(scripts)

        assert result["valid"] is True
        assert result["workflow_steps"] == 2
        assert len(result["composition_checks"]) == 1

        # Check the specific composition validation
        composition_check = result["composition_checks"][0]
        assert composition_check["source"] == "json_extract"
        assert composition_check["target"] == "file_write"
        assert composition_check["result"]["compatible"] is True

        # Verify field mapping
        field_mappings = composition_check["result"]["field_mappings"]
        assert "extracted_data" in field_mappings
        assert field_mappings["extracted_data"] == "content"

    def test_contract_validation_for_both_scripts(
        self, validator: ContractValidator
    ) -> None:
        """Test that both scripts pass individual contract validation."""
        scripts = [JsonExtractScript, FileWriteScript]

        # Validate each script individually
        json_extract_valid = validator.validate_all_scripts([JsonExtractScript])
        file_write_valid = validator.validate_all_scripts([FileWriteScript])

        assert json_extract_valid is True
        assert file_write_valid is True

        # Validate both together
        both_valid = validator.validate_all_scripts(scripts)
        assert both_valid is True
        assert len(validator.validation_errors) == 0

    async def test_error_propagation_in_workflow(self, temp_dir: Path) -> None:
        """Test error handling and propagation in composition workflow."""
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # Step 1: Create an error scenario in JsonExtract
        extract_input = json_extract.input_schema()(
            json_data="invalid json string",  # This will cause JSON parsing error
            fields=["name"],
        )

        extract_result = await json_extract.execute(extract_input)
        assert isinstance(extract_result, JsonExtractOutput)
        assert extract_result.success is False
        assert extract_result.error is not None

        # Step 2: In a real workflow, error handling would prevent FileWrite execution
        # But if we were to continue, FileWrite would get empty data
        write_input = file_write.input_schema()(
            file_path=str(temp_dir / "error_output.json"),
            content=extract_result.extracted_data,  # Will be empty dict
            format_json=True,
        )

        write_result = await file_write.execute(write_input)
        # FileWrite should succeed even with empty data
        assert isinstance(write_result, FileWriteOutput)
        assert write_result.success is True

        # The file should contain empty JSON object
        output_file = Path(write_result.file_path) if write_result.file_path else None
        assert output_file is not None
        assert output_file.exists()
        content = output_file.read_text().strip()
        assert content == "{}"

    def test_field_mapping_validation(self, validator: ContractValidator) -> None:
        """Test detailed field mapping validation between schemas."""
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # Test the schema composition
        result = validator._test_schema_composition(json_extract, file_write)

        assert result["compatible"] is True
        assert result["type_compatible"] is True
        assert len(result["errors"]) == 0

        # Verify specific field mapping
        field_mappings = result["field_mappings"]
        assert "extracted_data" in field_mappings
        assert field_mappings["extracted_data"] == "content"

    def test_complex_workflow_with_multiple_steps(
        self, validator: ContractValidator
    ) -> None:
        """Test validation of longer composition chains."""
        # For now, test with duplicated steps to simulate a longer workflow
        scripts = [JsonExtractScript, FileWriteScript, JsonExtractScript]

        result = validator.validate_composition_workflow(scripts)

        # Should validate all pairs: JsonExtract→FileWrite, FileWrite→JsonExtract
        assert result["workflow_steps"] == 3
        assert len(result["composition_checks"]) == 2

        # First composition should be valid
        first_check = result["composition_checks"][0]
        assert first_check["source"] == "json_extract"
        assert first_check["target"] == "file_write"
        assert first_check["result"]["compatible"] is True

    async def test_end_to_end_data_transformation_pipeline(
        self, temp_dir: Path
    ) -> None:
        """Test a complete data transformation pipeline."""
        # Simulate a real-world scenario: extract specific fields from JSON and save

        # Input data
        source_json = """
        {
            "user_profile": {
                "personal_info": {
                    "name": "John Doe",
                    "age": 28,
                    "email": "john.doe@example.com"
                },
                "preferences": {
                    "theme": "dark",
                    "language": "en",
                    "notifications": true
                },
                "metadata": {
                    "created_at": "2023-01-15",
                    "last_login": "2024-01-20",
                    "session_count": 42
                }
            }
        }
        """

        # Step 1: Extract only personal info
        json_extract = JsonExtractScript()
        extract_input = json_extract.input_schema()(
            json_data=source_json,
            fields=["user_profile"],  # Extract the nested object
        )

        extract_result = await json_extract.execute(extract_input)
        assert isinstance(extract_result, JsonExtractOutput)
        assert extract_result.success is True
        assert "user_profile" in extract_result.extracted_data

        # Step 2: Save extracted data to file
        file_write = FileWriteScript()
        output_file = temp_dir / "user_profile.json"
        write_input = file_write.input_schema()(
            file_path=str(output_file),
            content=extract_result.extracted_data,
            format_json=True,
        )

        write_result = await file_write.execute(write_input)
        assert isinstance(write_result, FileWriteOutput)
        assert write_result.success is True

        # Step 3: Verify the pipeline worked correctly
        assert output_file.exists()
        saved_content = output_file.read_text()
        assert "John Doe" in saved_content
        assert "john.doe@example.com" in saved_content
        assert "dark" in saved_content  # From preferences
        assert "2023-01-15" in saved_content  # From metadata

        bytes_written = write_result.bytes_written
        print(
            f"✓ Successfully processed data pipeline with {bytes_written} bytes written"
        )

    def test_schema_evolution_compatibility(self, validator: ContractValidator) -> None:
        """Test that schema composition handles evolution gracefully."""
        # This tests the robustness of our composition validation
        # when schemas might evolve over time

        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # Test current compatibility
        current_result = validator._test_schema_composition(json_extract, file_write)
        assert current_result["compatible"] is True

        # Verify that field mappings are flexible
        # The mapping should work even if field names are slightly different
        # (our implementation specifically handles the extracted_data → content mapping)
        assert "extracted_data" in current_result["field_mappings"]
        assert current_result["field_mappings"]["extracted_data"] == "content"
