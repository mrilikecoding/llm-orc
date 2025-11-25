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
def test_primitives_dir(tmp_path: Path) -> Generator[Path, None, None]:
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

    # Set environment variable so all ScriptResolvers can find test primitives
    old_env = os.environ.get("LLM_ORC_TEST_PRIMITIVES_DIR")
    os.environ["LLM_ORC_TEST_PRIMITIVES_DIR"] = str(primitives_dir.parent)

    yield primitives_dir

    # Cleanup: restore environment variable
    if old_env:
        os.environ["LLM_ORC_TEST_PRIMITIVES_DIR"] = old_env
    else:
        os.environ.pop("LLM_ORC_TEST_PRIMITIVES_DIR", None)


def _create_ai_primitives(primitives_dir: Path) -> None:
    """Create AI category primitives for testing."""
    ai_dir = primitives_dir / "ai"
    ai_dir.mkdir(exist_ok=True)

    # Create generate_story_prompt.py
    story_prompt_script = ai_dir / "generate_story_prompt.py"
    story_prompt_script.write_text("""#!/usr/bin/env python3
\"\"\"Test AI primitive for story prompt generation.
Follows ADR-001 PromptGeneratorOutput pattern with top-level fields.
\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        theme = input_data.get('theme', 'generic')
        character_type = input_data.get('character_type', 'protagonist')

        # Generate contextual prompt with rich theming (ADR-001 compliant)
        if theme == "cyberpunk":
            generated_prompt = (
                f"A cyberpunk detective story set in neo-tokyo with neon-lit "
                f"data networks featuring a {character_type} with neural implants"
            )
        else:
            generated_prompt = f"A {theme} story about a {character_type}"

        # ADR-001 PromptGeneratorOutput: top-level fields
        result = {
            "success": True,
            "data": None,  # Optional field per ScriptAgentOutput
            "generated_prompt": generated_prompt,
            "context_metadata": {
                "theme": theme,
                "character_type": character_type,
                "is_test_mode": True
            },
            "agent_requests": [
                {
                    "target_agent_type": "user_input",
                    "parameters": {
                        "prompt": generated_prompt,
                        "multiline": False,
                        "context": {
                            "theme": theme,
                            "character_type": character_type,
                            "generator": "story_prompt_generator"
                        }
                    },
                    "priority": 0
                }
            ]
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

    # Create read_file.py
    read_file_script = file_ops_dir / "read_file.py"
    read_file_script.write_text("""#!/usr/bin/env python3
\"\"\"Test primitive for file reading.\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
        file_path = input_data.get('file_path')

        # Mock file reading with config.json structure
        mock_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            },
            "server": {
                "port": 8080
            }
        }
        result = {
            "success": True,
            "data": {
                "content": json.dumps(mock_config),
                "size": len(json.dumps(mock_config)),
                "path": file_path
            },
            "metadata": {"is_test_mode": True}
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
    read_file_script.chmod(0o755)

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

        # Mock JSON extraction - extract from source_data based on json_path
        # For json_path like "$.database", extract the database section
        extracted_key = json_path.split('.')[-1] if '.' in json_path else json_path
        extracted_value = source_data.get(extracted_key, {})

        result = {
            "success": True,
            "data": {
                "extracted_value": extracted_value,
                "key": extracted_key,
                "path_used": json_path
            },
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

        # Actually write the file for testing
        with open(file_path, 'w') as f:
            f.write(content)

        result = {
            "success": True,
            "data": {
                "bytes_written": len(content),
                "path": file_path,
                "status": "written"
            },
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
            "data": {
                "analysis_results": {
                    "centrality_scores": {"node1": 0.8, "node2": 0.6, "node3": 0.4},
                    "node_rankings": ["node1", "node2", "node3"],
                    "analysis_metadata": {
                        "algorithm": "betweenness_centrality",
                        "timestamp": "2025-10-14T00:00:00Z"
                    }
                }
            },
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
