"""Test script discovery functionality for ContractValidator."""

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from llm_orc.contracts.contract_validator import ContractValidator


class MockScriptInput(BaseModel):
    """Mock input schema for testing."""

    data: str = Field(..., description="Test data input")


class MockScriptOutput(BaseModel):
    """Mock output schema for testing."""

    success: bool
    result: str = Field(default="")
    error: str | None = None


class TestScriptDiscovery:
    """Test cases for script discovery functionality."""

    def test_discover_scripts_returns_empty_list_when_no_scripts_found(self) -> None:
        """Test discovery returns empty list when no scripts exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ContractValidator(temp_dir)
            discovered = validator._discover_scripts()
            assert discovered == []

    def test_discover_scripts_finds_python_files_with_script_contract(self) -> None:
        """Test discovery finds Python files implementing ScriptContract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock script file
            script_file = Path(temp_dir) / "test_script.py"
            script_content = '''
"""Mock script for testing."""

from typing import Any
from pydantic import BaseModel, Field
from llm_orc.contracts.script_contract import (
    ScriptCapability, ScriptContract, ScriptMetadata, TestCase
)

class TestInput(BaseModel):
    data: str = Field(..., description="Test data")

class TestOutput(BaseModel):
    success: bool
    result: str = ""

class TestScript(ScriptContract):
    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="test_script",
            version="1.0.0",
            description="Test script",
            author="test",
            category="test",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION]
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return TestInput

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return TestOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return TestOutput(success=True, result="test")

    def get_test_cases(self) -> list[TestCase]:
        return []
'''
            script_file.write_text(script_content)

            validator = ContractValidator(temp_dir)
            discovered = validator._discover_scripts()

            assert len(discovered) == 1
            assert discovered[0]["name"] == "test_script.py"
            assert discovered[0]["path"] == str(script_file)
            assert "TestScript" in discovered[0]["contract_classes"]

    def test_discover_scripts_supports_directory_categories(self) -> None:
        """Test discovery supports different script categories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure matching ADR-003
            base_path = Path(temp_dir)
            primitives_dir = base_path / ".llm-orc" / "scripts" / "primitives"
            examples_dir = base_path / ".llm-orc" / "scripts" / "examples"
            community_dir = base_path / ".llm-orc" / "scripts" / "community"

            primitives_dir.mkdir(parents=True)
            examples_dir.mkdir(parents=True)
            community_dir.mkdir(parents=True)

            # Create mock scripts in each directory
            for category_dir, category_name in [
                (primitives_dir, "primitive"),
                (examples_dir, "example"),
                (community_dir, "community"),
            ]:
                script_file = category_dir / f"{category_name}_script.py"
                script_content = f'''
"""Mock {category_name} script."""

from pydantic import BaseModel

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)

class {category_name.title()}Script(ScriptContract):
    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="{category_name}_script",
            version="1.0.0",
            description="Mock {category_name} script",
            author="test",
            category="{category_name}",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION]
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return BaseModel

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return BaseModel

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return BaseModel()

    def get_test_cases(self) -> list[TestCase]:
        return []
'''
                script_file.write_text(script_content)

            validator = ContractValidator(str(base_path))
            discovered = validator._discover_scripts()

            assert len(discovered) == 3
            categories = {script["category"] for script in discovered}
            assert categories == {"primitives", "examples", "community"}

    def test_discover_scripts_handles_import_errors_gracefully(self) -> None:
        """Test discovery handles files that can't be imported."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python file with syntax errors
            bad_file = Path(temp_dir) / "bad_script.py"
            bad_file.write_text("this is not valid python syntax !!!!")

            # Create a valid script file
            good_file = Path(temp_dir) / "good_script.py"
            good_file.write_text('''
"""Valid script."""

from pydantic import BaseModel

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)

class GoodScript(ScriptContract):
    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="good_script",
            version="1.0.0",
            description="Good script",
            author="test",
            category="test",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION]
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return BaseModel

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return BaseModel

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return BaseModel()

    def get_test_cases(self) -> list[TestCase]:
        return []
''')

            validator = ContractValidator(temp_dir)
            discovered = validator._discover_scripts()

            # Should find the good script and skip the bad one
            assert len(discovered) == 1
            assert discovered[0]["name"] == "good_script.py"

    def test_discover_scripts_ignores_non_contract_python_files(self) -> None:
        """Test discovery ignores Python files that don't implement ScriptContract."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a Python file without ScriptContract
            non_contract_file = Path(temp_dir) / "regular_module.py"
            non_contract_file.write_text('''
"""Regular Python module without ScriptContract."""

def some_function():
    return "not a contract"

class RegularClass:
    pass
''')

            validator = ContractValidator(temp_dir)
            discovered = validator._discover_scripts()

            assert len(discovered) == 0

    def test_discover_scripts_returns_structured_data(self) -> None:
        """Test discovery returns properly structured data for each script."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_file = Path(temp_dir) / "structured_test.py"
            script_content = '''
"""Structured test script."""

from pydantic import BaseModel

from llm_orc.contracts.script_contract import (
    ScriptCapability,
    ScriptContract,
    ScriptMetadata,
    TestCase,
)

class StructuredScript(ScriptContract):
    @property
    def metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            name="structured_test",
            version="1.0.0",
            description="Structured test script",
            author="test",
            category="test",
            capabilities=[ScriptCapability.DATA_TRANSFORMATION]
        )

    @classmethod
    def input_schema(cls) -> type[BaseModel]:
        return BaseModel

    @classmethod
    def output_schema(cls) -> type[BaseModel]:
        return BaseModel

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return BaseModel()

    def get_test_cases(self) -> list[TestCase]:
        return []
'''
            script_file.write_text(script_content)

            validator = ContractValidator(temp_dir)
            discovered = validator._discover_scripts()

            assert len(discovered) == 1
            script_info = discovered[0]

            # Verify structure
            assert "name" in script_info
            assert "path" in script_info
            assert "category" in script_info
            assert "contract_classes" in script_info
            assert "import_error" in script_info

            assert script_info["name"] == "structured_test.py"
            assert script_info["path"] == str(script_file)
            assert script_info["category"] == "root"  # Default category
            assert script_info["contract_classes"] == ["StructuredScript"]
            assert script_info["import_error"] is None

    def test_discover_and_validate_all_scripts_returns_true_when_no_scripts(
        self,
    ) -> None:
        """Test integrated discovery and validation with no scripts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ContractValidator(temp_dir)
            result = validator.discover_and_validate_all_scripts()
            assert result is True

    def test_discover_and_validate_all_scripts_finds_and_validates_existing(
        self,
    ) -> None:
        """Test integrated discovery and validation finds existing scripts."""
        # Use the actual project directory
        project_root = Path(__file__).parent.parent
        validator = ContractValidator(str(project_root))

        result = validator.discover_and_validate_all_scripts()
        # Should succeed since it found valid scripts
        assert result is True
