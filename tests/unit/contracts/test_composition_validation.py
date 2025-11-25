"""Unit tests for composition validation functionality - TDD Red Phase."""

from pydantic import BaseModel

from llm_orc.contracts.contract_validator import ContractValidator
from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)
from llm_orc.reference.file_write_script import FileWriteOutput, FileWriteScript
from llm_orc.reference.json_extract_script import JsonExtractScript


class IncompatibleInput(BaseModel):
    """Input schema that's incompatible for composition."""

    required_field: int
    different_type: str


class TestIncompatibleScript(ScriptContract):
    """Test script that's incompatible for composition."""

    @property
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        return ScriptMetadata(
            name="incompatible",
            version="1.0.0",
            description="Incompatible script for testing",
            author="test",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        """Input schema for validation."""
        return IncompatibleInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        """Output schema for validation."""
        return FileWriteOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute the incompatible script."""
        return FileWriteOutput(success=False)

    def get_test_cases(self) -> list[TestCase]:
        """Return test cases."""
        return []


class TestCompositionValidation:
    """Test composition validation functionality."""

    def test_composition_compatibility_method_exists(self) -> None:
        """Test that ContractValidator has composition compatibility method."""
        validator = ContractValidator()

        # RED: This should fail because method doesn't exist yet
        assert hasattr(validator, "_test_composition_compatibility")

    def test_schema_composition_method_exists(self) -> None:
        """Test that ContractValidator has schema composition method."""
        validator = ContractValidator()

        # RED: This should fail because method doesn't exist yet
        assert hasattr(validator, "_test_schema_composition")

    def test_json_extract_to_file_write_composition(self) -> None:
        """Test that JsonExtract can compose with FileWrite."""
        # Create script instances
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # RED: This should fail because composition validation doesn't exist
        validator = ContractValidator()
        result = validator._test_schema_composition(json_extract, file_write)

        assert result["compatible"] is True
        assert "field_mappings" in result
        assert len(result["errors"]) == 0

    def test_incompatible_schema_composition_detected(self) -> None:
        """Test that incompatible schemas are properly detected."""
        json_extract = JsonExtractScript()
        incompatible = TestIncompatibleScript()

        # RED: This should fail because composition validation doesn't exist
        validator = ContractValidator()
        result = validator._test_schema_composition(json_extract, incompatible)

        assert result["compatible"] is False
        assert len(result["errors"]) > 0

    def test_field_mapping_between_different_structures(self) -> None:
        """Test field mapping for different but compatible schema structures."""
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # RED: This should fail because composition validation doesn't exist
        validator = ContractValidator()
        result = validator._test_schema_composition(json_extract, file_write)

        # Should be able to map extracted_data -> content
        mappings = result["field_mappings"]
        assert "extracted_data" in mappings
        assert mappings["extracted_data"] == "content"

    def test_type_compatibility_validation(self) -> None:
        """Test that type compatibility is validated for schema composition."""
        json_extract = JsonExtractScript()
        file_write = FileWriteScript()

        # RED: This should fail because composition validation doesn't exist
        validator = ContractValidator()
        result = validator._test_schema_composition(json_extract, file_write)

        # dict[str, Any] should be compatible with dict[str, Any]
        assert result["type_compatible"] is True

    def test_end_to_end_composition_workflow(self) -> None:
        """Test end-to-end composition workflow validation."""
        scripts = [JsonExtractScript, FileWriteScript]

        # RED: This should fail because composition workflow validation doesn't exist
        validator = ContractValidator()
        result = validator.validate_composition_workflow(scripts)

        assert result["valid"] is True
        assert result["workflow_steps"] == 2
        assert (
            len(result["composition_checks"]) == 1
        )  # One composition check between scripts
