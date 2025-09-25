"""Unit tests for ContractValidator."""

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
