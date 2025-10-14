"""BDD test configuration for llm-orc script agents."""

import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.test_primitives import TestPrimitiveFactory

# Import existing fixtures from main test suite - moved to top-level conftest.py


@pytest.fixture
def test_primitives_dir(tmp_path: Path) -> Path:
    """Setup test primitives directory with mock scripts."""
    primitives_dir = tmp_path / "primitives"
    primitives_dir.mkdir(parents=True, exist_ok=True)

    # Create standard test primitives
    TestPrimitiveFactory.create_user_input_script(primitives_dir)
    TestPrimitiveFactory.create_file_read_script(primitives_dir)

    # Create additional primitives needed by Issue #24 tests
    _create_ai_primitives(primitives_dir)
    _create_file_ops_primitives(primitives_dir)
    _create_network_primitives(primitives_dir)

    return primitives_dir


def _create_ai_primitives(primitives_dir: Path) -> None:
    """Create AI category primitives for testing."""
    ai_dir = primitives_dir / "ai"
    ai_dir.mkdir(exist_ok=True)

    # Create generate_story_prompt.py
    story_prompt_script = ai_dir / "generate_story_prompt.py"
    story_prompt_script.write_text("""#!/usr/bin/env python3
\"\"\"Test AI primitive for story prompt generation.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        theme = input_data.get('theme', 'generic')

        # Mock story prompt generation
        result = {
            "success": True,
            "data": f"A {theme} story about...",
            "story_prompt": f"Generate a {theme} narrative",
            "theme_used": theme,
            "supports_agent_requests": True,
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    story_prompt_script.chmod(0o755)


def _create_file_ops_primitives(primitives_dir: Path) -> None:
    """Create file-ops category primitives for testing."""
    file_ops_dir = primitives_dir / "file-ops"
    file_ops_dir.mkdir(exist_ok=True)

    # Create json_extract.py
    json_extract_script = file_ops_dir / "json_extract.py"
    json_extract_script.write_text("""#!/usr/bin/env python3
\"\"\"Test primitive for JSON extraction.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        json_path = input_data.get('json_path', '$.data')
        source_data = input_data.get('source_data', {})

        # Mock JSON extraction
        result = {
            "success": True,
            "data": source_data.get('data', 'extracted_value'),
            "path_used": json_path,
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    json_extract_script.chmod(0o755)

    # Create write_file.py
    write_file_script = file_ops_dir / "write_file.py"
    write_file_script.write_text("""#!/usr/bin/env python3
\"\"\"Test primitive for file writing.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        file_path = input_data.get('file_path')
        content = input_data.get('content', '')

        result = {
            "success": True,
            "data": f"Wrote {len(content)} bytes",
            "bytes_written": len(content),
            "file_path": file_path,
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    write_file_script.chmod(0o755)

    # Create read_protected_file.py (for error handling tests)
    read_protected_script = file_ops_dir / "read_protected_file.py"
    read_protected_script.write_text("""#!/usr/bin/env python3
\"\"\"Test primitive that simulates permission errors.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        file_path = input_data.get('file_path')

        # Simulate permission error
        result = {
            "success": False,
            "error": f"Permission denied: {file_path}",
            "error_type": "PermissionError",
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))
        sys.exit(1)

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    read_protected_script.chmod(0o755)


def _create_network_primitives(primitives_dir: Path) -> None:
    """Create network category primitives for testing."""
    network_dir = primitives_dir / "network"
    network_dir.mkdir(exist_ok=True)

    # Create topology.py
    topology_script = network_dir / "topology.py"
    topology_script.write_text("""#!/usr/bin/env python3
\"\"\"Test network topology primitive.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

        result = {
            "success": True,
            "data": {"nodes": 5, "edges": 8},
            "topology_type": "mock",
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    topology_script.chmod(0o755)

    # Create analyze_topology.py
    analyze_script = network_dir / "analyze_topology.py"
    analyze_script.write_text("""#!/usr/bin/env python3
\"\"\"Test network analysis primitive.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        topology_data = input_data.get('topology_data', {})

        result = {
            "success": True,
            "data": {"centrality_scores": {"node1": 0.8, "node2": 0.6}},
            "analysis_type": "centrality",
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    analyze_script.chmod(0o755)


@pytest.fixture
def bdd_context(test_primitives_dir: Path) -> dict[str, Any]:
    """Shared context for BDD scenarios with test primitives configured."""
    from llm_orc.core.execution.script_resolver import ScriptResolver

    # Create resolver with test primitives directory
    resolver = ScriptResolver(search_paths=[str(test_primitives_dir.parent)])

    return {
        "scripts": {},
        "agents": {},
        "execution_results": {},
        "temp_files": [],
        "test_primitives_dir": test_primitives_dir,
        "script_resolver": resolver,
    }


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
