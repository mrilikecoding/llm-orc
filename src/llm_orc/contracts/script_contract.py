"""Core script contract interface and models for ADR-003 implementation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScriptCapability(str, Enum):
    """Enumeration of script capabilities."""

    USER_INTERACTION = "user_interaction"
    DATA_TRANSFORMATION = "data_transformation"
    FILE_OPERATIONS = "file_operations"
    API_INTEGRATION = "api_integration"
    COMPUTATION = "computation"
    CONTROL_FLOW = "control_flow"
    EXTERNAL_EXECUTION = "external_execution"


class ScriptDependency(BaseModel):
    """Declaration of script dependencies."""

    name: str
    version: str | None = None
    optional: bool = False
    pip_package: str | None = None
    system_command: str | None = None


class TestCase(BaseModel):
    """Test case for script validation."""

    name: str
    description: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    should_succeed: bool = True
    setup_commands: list[str] = Field(default_factory=list)
    cleanup_commands: list[str] = Field(default_factory=list)


class ScriptMetadata(BaseModel):
    """Comprehensive metadata for script contract."""

    name: str
    version: str
    description: str
    author: str
    category: str
    capabilities: list[ScriptCapability]
    dependencies: list[ScriptDependency] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)
    test_cases: list[dict[str, Any]] = Field(default_factory=list)


class ScriptContract(ABC):
    """Universal contract that all scripts must implement."""

    @property
    @abstractmethod
    def metadata(self) -> ScriptMetadata:
        """Script metadata and capabilities."""
        pass

    @classmethod
    @abstractmethod
    def input_schema(cls) -> type[BaseModel]:
        """Input schema for validation."""
        pass

    @classmethod
    @abstractmethod
    def output_schema(cls) -> type[BaseModel]:
        """Output schema for validation."""
        pass

    @abstractmethod
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute the script with validated input."""
        pass

    @abstractmethod
    def get_test_cases(self) -> list[TestCase]:
        """Return test cases for contract validation."""
        pass
