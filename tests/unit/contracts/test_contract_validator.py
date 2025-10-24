"""Unit tests for ContractValidator."""

import tempfile
from pathlib import Path

from pydantic import BaseModel

from llm_orc.contracts.contract_validator import ContractValidator
from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)


class MockInput(BaseModel):
    """Mock input schema for testing."""

    value: str


class MockOutput(BaseModel):
    """Mock output schema for testing."""

    result: str


class ValidMockScript(ScriptContract):
    """Valid mock script for testing."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="valid_mock_script",
            version="1.0.0",
            description="A valid mock script for testing",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        typed_input = MockInput(**input_data.model_dump())
        return MockOutput(result=f"processed_{typed_input.value}")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="basic_test",
                description="Basic test case",
                input_data={"value": "test"},
                expected_output={"result": "processed_test"},
            )
        ]


class InvalidMetadataScript(ScriptContract):
    """Script with invalid metadata."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="",  # Missing name
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return MockOutput(result="test")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="test",
                description="test",
                input_data={"value": "test"},
                expected_output={},
            )
        ]


class NoCapabilitiesScript(ScriptContract):
    """Script with no capabilities."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="no_capabilities",
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[],  # No capabilities
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return MockOutput(result="test")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="test",
                description="test",
                input_data={"value": "test"},
                expected_output={},
            )
        ]


class InvalidInputSchemaScript(ScriptContract):
    """Script with invalid input schema."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="invalid_schema",
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return str  # type: ignore[return-value]

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return MockOutput(result="test")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="test",
                description="test",
                input_data={"value": "test"},
                expected_output={},
            )
        ]


class NoTestCasesScript(ScriptContract):
    """Script with no test cases."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="no_tests",
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return MockOutput(result="test")

    def get_test_cases(self) -> list[TestCase]:
        return []  # No test cases


class FailingTestScript(ScriptContract):
    """Script with failing test case."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="failing_test",
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        raise RuntimeError("Test failure")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="failing_test",
                description="Test that fails",
                input_data={"value": "test"},
                expected_output={"result": "expected"},
                should_succeed=True,
            )
        ]


class WrongOutputScript(ScriptContract):
    """Script that returns wrong output type."""

    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="wrong_output",
            version="1.0.0",
            description="Test",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return MockInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return MockOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        # Intentionally return wrong type to test error handling
        return MockInput(value="wrong_type")

    def get_test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                name="wrong_output_test",
                description="Test with wrong output",
                input_data={"value": "test"},
                expected_output={"result": "expected"},
                should_succeed=True,
            )
        ]


class TestContractValidator:
    """Test the ContractValidator class."""

    def test_contract_validator_can_be_instantiated(self) -> None:
        """Test that ContractValidator can be instantiated."""
        validator = ContractValidator()
        assert validator is not None

    def test_validate_all_scripts_returns_boolean(self) -> None:
        """Test that validate_all_scripts returns a boolean result."""
        validator = ContractValidator()
        scripts = [ValidMockScript]

        result = validator.validate_all_scripts(scripts)
        assert isinstance(result, bool)

    def test_validate_all_scripts_with_valid_script_returns_true(self) -> None:
        """Test that valid script passes validation."""
        validator = ContractValidator()
        scripts = [ValidMockScript]

        result = validator.validate_all_scripts(scripts)
        assert result is True
        assert len(validator.validation_errors) == 0

    def test_validate_all_scripts_with_exception_returns_false(self) -> None:
        """Test validation handles exceptions (lines 44-45)."""
        validator = ContractValidator()
        scripts = [InvalidMetadataScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert len(validator.validation_errors) > 0
        assert "InvalidMetadataScript" in validator.validation_errors[0]

    def test_validate_metadata_missing_field(self) -> None:
        """Test metadata validation detects missing fields (line 267)."""
        validator = ContractValidator()
        scripts = [InvalidMetadataScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert any(
            "Missing required metadata field" in err
            for err in validator.validation_errors
        )

    def test_validate_metadata_no_capabilities(self) -> None:
        """Test metadata validation detects no capabilities (line 270)."""
        validator = ContractValidator()
        scripts = [NoCapabilitiesScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert any(
            "at least one capability" in err for err in validator.validation_errors
        )

    def test_validate_schemas_invalid_input_schema(self) -> None:
        """Test schema validation detects invalid input schema (line 286)."""
        validator = ContractValidator()
        scripts = [InvalidInputSchemaScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert any("Pydantic BaseModel" in err for err in validator.validation_errors)

    def test_run_test_cases_no_test_cases(self) -> None:
        """Test execution requires at least one test case (line 310)."""
        validator = ContractValidator()
        scripts = [NoTestCasesScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert any(
            "at least one test case" in err for err in validator.validation_errors
        )

    def test_run_test_cases_test_failure(self) -> None:
        """Test execution handles test case failures (lines 330-332)."""
        validator = ContractValidator()
        scripts = [FailingTestScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert len(validator.validation_errors) > 0

    def test_run_test_cases_wrong_output_type(self) -> None:
        """Test execution detects wrong output type (line 322)."""
        validator = ContractValidator()
        scripts = [WrongOutputScript]

        result = validator.validate_all_scripts(scripts)

        assert result is False
        assert len(validator.validation_errors) > 0

    def test_discover_and_validate_no_scripts(self) -> None:
        """Test discovery with no scripts returns True (lines 58-59)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ContractValidator(script_directory=temp_dir)

            result = validator.discover_and_validate_all_scripts()

            assert result is True
            assert len(validator.validation_errors) == 0

    def test_should_skip_file_dunder_files(self) -> None:
        """Test that __init__.py files are skipped (line 154)."""
        validator = ContractValidator()
        processed: set[str] = set()

        should_skip = validator._should_skip_file(Path("__init__.py"), processed)

        assert should_skip is True

    def test_should_skip_file_already_processed(self) -> None:
        """Test that already processed files are skipped (lines 156-157)."""
        validator = ContractValidator()
        file_path = Path("test.py")
        processed: set[str] = {str(file_path)}

        should_skip = validator._should_skip_file(file_path, processed)

        assert should_skip is True

    def test_create_script_info_with_import_error(self) -> None:
        """Test script info creation with import error (lines 181-183)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ContractValidator(script_directory=temp_dir)

            # Create a Python file with syntax error
            bad_file = Path(temp_dir) / "bad_script.py"
            bad_file.write_text("this is not valid python {{{")

            script_info = validator._create_script_info(bad_file, "test")

            assert script_info["import_error"] is not None
            assert len(script_info["contract_classes"]) == 0
