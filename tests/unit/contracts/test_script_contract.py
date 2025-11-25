"""Unit tests for ScriptContract base class and contract system."""

import pytest
from pydantic import BaseModel

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptDependency,
    ScriptMetadata,
    TestCase,
)


class TestScriptContract:
    """Test the core ScriptContract abstract interface."""

    def test_script_contract_is_abstract_base_class(self) -> None:
        """Test that ScriptContract cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ScriptContract()  # type: ignore

    def test_script_contract_defines_required_abstract_methods(self) -> None:
        """Test that ScriptContract defines all required abstract methods."""
        abstract_methods = ScriptContract.__abstractmethods__

        expected_methods = {
            "metadata",
            "input_schema",
            "output_schema",
            "execute",
            "get_test_cases",
        }

        assert abstract_methods == expected_methods

    def test_script_contract_subclass_must_implement_all_methods(self) -> None:
        """Test that incomplete ScriptContract subclass raises TypeError."""

        class IncompleteScript(ScriptContract):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteScript()  # type: ignore

    def test_complete_script_contract_implementation(self) -> None:
        """Test that complete ScriptContract implementation works correctly."""

        class TestInput(BaseModel):
            value: str

        class TestOutput(BaseModel):
            result: str

        class CompleteScript(ScriptContract):
            @property
            def metadata(self) -> ScriptMetadata:
                return ScriptMetadata(
                    name="test_script",
                    version="1.0.0",
                    description="Test script",
                    author="test_author",
                    category="test",
                    capabilities=[ScriptCapability.COMPUTATION],
                )

            @classmethod
            def input_schema(cls) -> type[BaseModel]:
                return TestInput

            @classmethod
            def output_schema(cls) -> type[BaseModel]:
                return TestOutput

            async def execute(self, input_data: BaseModel) -> BaseModel:
                assert isinstance(input_data, TestInput)
                return TestOutput(result=f"processed_{input_data.value}")

            def get_test_cases(self) -> list[TestCase]:
                return [
                    TestCase(
                        name="basic_test",
                        description="Basic test case",
                        input_data={"value": "test"},
                        expected_output={"result": "processed_test"},
                    )
                ]

        # Should instantiate successfully
        script = CompleteScript()
        assert script is not None
        assert script.metadata.name == "test_script"
        assert script.input_schema() == TestInput
        assert script.output_schema() == TestOutput
        assert len(script.get_test_cases()) == 1


class TestScriptCapability:
    """Test ScriptCapability enum."""

    def test_script_capability_values(self) -> None:
        """Test that ScriptCapability has expected values."""
        expected_capabilities = {
            "user_interaction",
            "data_transformation",
            "file_operations",
            "api_integration",
            "computation",
            "control_flow",
            "external_execution",
        }
        actual_capabilities = {cap.value for cap in ScriptCapability}
        assert actual_capabilities == expected_capabilities

    def test_script_capability_enum_behavior(self) -> None:
        """Test ScriptCapability enum behavior."""
        cap = ScriptCapability.COMPUTATION
        assert cap == ScriptCapability.COMPUTATION
        assert cap.value == "computation"


class TestScriptDependency:
    """Test ScriptDependency model."""

    def test_script_dependency_basic(self) -> None:
        """Test basic ScriptDependency creation."""
        dep = ScriptDependency(name="requests")
        assert dep.name == "requests"
        assert dep.version is None
        assert dep.optional is False
        assert dep.pip_package is None
        assert dep.system_command is None

    def test_script_dependency_full(self) -> None:
        """Test ScriptDependency with all fields."""
        dep = ScriptDependency(
            name="pandas",
            version="2.0.0",
            optional=True,
            pip_package="pandas==2.0.0",
            system_command="pip install pandas==2.0.0",
        )
        assert dep.name == "pandas"
        assert dep.version == "2.0.0"
        assert dep.optional is True
        assert dep.pip_package == "pandas==2.0.0"
        assert dep.system_command == "pip install pandas==2.0.0"


class TestTestCase:
    """Test TestCase model."""

    def test_test_case_basic(self) -> None:
        """Test basic TestCase creation."""
        test_case = TestCase(
            name="test_basic",
            description="Basic test",
            input_data={"key": "value"},
            expected_output={"result": "success"},
        )
        assert test_case.name == "test_basic"
        assert test_case.description == "Basic test"
        assert test_case.input_data == {"key": "value"}
        assert test_case.expected_output == {"result": "success"}
        assert test_case.should_succeed is True
        assert test_case.setup_commands == []
        assert test_case.cleanup_commands == []

    def test_test_case_full(self) -> None:
        """Test TestCase with all fields."""
        test_case = TestCase(
            name="test_full",
            description="Full test",
            input_data={"data": "test"},
            expected_output={"output": "result"},
            should_succeed=False,
            setup_commands=["setup_cmd"],
            cleanup_commands=["cleanup_cmd"],
        )
        assert test_case.name == "test_full"
        assert test_case.should_succeed is False
        assert test_case.setup_commands == ["setup_cmd"]
        assert test_case.cleanup_commands == ["cleanup_cmd"]


class TestScriptMetadata:
    """Test ScriptMetadata model."""

    def test_script_metadata_basic(self) -> None:
        """Test basic ScriptMetadata creation."""
        metadata = ScriptMetadata(
            name="test_script",
            version="1.0.0",
            description="Test script",
            author="test_author",
            category="test",
            capabilities=[ScriptCapability.COMPUTATION],
        )
        assert metadata.name == "test_script"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test script"
        assert metadata.author == "test_author"
        assert metadata.category == "test"
        assert metadata.capabilities == [ScriptCapability.COMPUTATION]
        assert metadata.dependencies == []
        assert metadata.tags == []
        assert metadata.examples == []
        assert metadata.test_cases == []

    def test_script_metadata_full(self) -> None:
        """Test ScriptMetadata with all fields."""
        dependency = ScriptDependency(name="requests")
        metadata = ScriptMetadata(
            name="full_script",
            version="2.0.0",
            description="Full script",
            author="full_author",
            category="utility",
            capabilities=[
                ScriptCapability.API_INTEGRATION,
                ScriptCapability.FILE_OPERATIONS,
            ],
            dependencies=[dependency],
            tags=["api", "utility"],
            examples=[{"input": "test", "output": "result"}],
            test_cases=[{"name": "test1", "input": {"key": "value"}}],
        )
        assert len(metadata.capabilities) == 2
        assert ScriptCapability.API_INTEGRATION in metadata.capabilities
        assert ScriptCapability.FILE_OPERATIONS in metadata.capabilities
        assert len(metadata.dependencies) == 1
        assert metadata.dependencies[0].name == "requests"
        assert metadata.tags == ["api", "utility"]
        assert len(metadata.examples) == 1
        assert len(metadata.test_cases) == 1
