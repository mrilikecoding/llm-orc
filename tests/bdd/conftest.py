"""BDD test configuration for llm-orc script agents."""

import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Import existing fixtures from main test suite - moved to top-level conftest.py


@pytest.fixture
def bdd_context() -> dict[str, Any]:
    """Shared context for BDD scenarios."""
    return {"scripts": {}, "agents": {}, "execution_results": {}, "temp_files": []}


@pytest.fixture
def temp_script_dir() -> Generator[Path, None, None]:
    """Temporary directory for test scripts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_dir = Path(tmpdir) / "scripts" / "primitives"
        script_dir.mkdir(parents=True)
        yield script_dir


@pytest.fixture
def mock_script_agent() -> type:
    """Mock script agent for testing."""

    class MockScriptAgent:
        def __init__(self, script_path: str) -> None:
            self.script_path = script_path

        async def execute(self, input_data: str) -> str:
            return json.dumps(
                {
                    "success": True,
                    "data": "mock_output",
                    "metadata": {"script": self.script_path},
                }
            )

    return MockScriptAgent


@pytest.fixture
def sample_ensemble_config() -> dict[str, Any]:
    """Sample ensemble configuration for testing."""
    return {
        "name": "test-ensemble",
        "agents": [
            {
                "name": "test-script",
                "script": "primitives/test_script.py",
                "parameters": {"test": True},
            }
        ],
    }


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_temp_files(bdd_context: dict[str, Any]) -> Generator[None, None, None]:
    """Automatically cleanup temporary files after each test."""
    yield
    for temp_file in bdd_context.get("temp_files", []):
        if os.path.exists(temp_file):
            os.remove(temp_file)
