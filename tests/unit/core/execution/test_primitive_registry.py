"""Tests for primitive registry system (ADR-001)."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from llm_orc.core.execution.primitive_registry import PrimitiveRegistry


class TestPrimitiveRegistry:
    """Test the primitive registry for script agent discovery and validation."""

    def test_primitive_registry_initialization(self) -> None:
        """Test that primitive registry initializes correctly."""
        registry = PrimitiveRegistry()
        assert registry is not None
        assert hasattr(registry, "discover_primitives")
        assert hasattr(registry, "get_primitive_info")
        assert hasattr(registry, "validate_primitive")

    def test_discover_primitives_finds_available_scripts(self) -> None:
        """Test primitive discovery in .llm-orc/scripts/primitives directory."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            # Create test primitive scripts
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            # Create test primitives
            test_primitives = [
                "json_extract.py",
                "json_merge.py",
                "file_read.py",
                "file_write.py",
            ]

            for primitive_name in test_primitives:
                primitive_file = primitives_dir / primitive_name
                primitive_file.write_text(f"""#!/usr/bin/env python3
# Primitive: {primitive_name}
# Input: JSON data
# Output: JSON result
import json
import os

def main():
    input_data = os.environ.get("INPUT_DATA", "{{}}")
    result = {{"success": True, "data": "processed", "error": None,
              "agent_requests": []}}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
""")
                primitive_file.chmod(0o755)

            # Mock the working directory to use our temp directory
            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                primitives = registry.discover_primitives()

            assert len(primitives) == 4
            primitive_names = {p["name"] for p in primitives}
            assert "json_extract.py" in primitive_names
            assert "json_merge.py" in primitive_names
            assert "file_read.py" in primitive_names
            assert "file_write.py" in primitive_names

    def test_get_primitive_info_returns_metadata(self) -> None:
        """Test getting primitive metadata including schema information."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            # Create test primitive with metadata
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            primitive_file = primitives_dir / "json_extract.py"
            primitive_file.write_text("""#!/usr/bin/env python3
# Primitive: JSON extraction utility
# Input: JSON data with extraction path
# Output: Extracted JSON value
# Depends: json, pathlib
import json
import os

def main():
    input_data = os.environ.get("INPUT_DATA", "{}")
    result = {"success": True, "data": "extracted", "error": None, "agent_requests": []}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                primitive_info = registry.get_primitive_info("json_extract.py")

            assert primitive_info["name"] == "json_extract.py"
            assert primitive_info["type"] == "primitive"
            assert "JSON extraction utility" in primitive_info["description"]
            assert "JSON data with extraction path" in primitive_info["input_schema"]
            assert "Extracted JSON value" in primitive_info["output_schema"]
            assert "json" in primitive_info["dependencies"]
            assert "pathlib" in primitive_info["dependencies"]

    def test_validate_primitive_schema_contracts(self) -> None:
        """Test that primitives conform to ScriptAgentInput/Output schemas."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            # Create test primitive that conforms to schema
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            primitive_file = primitives_dir / "json_extract.py"
            primitive_file.write_text("""#!/usr/bin/env python3
import json
import os

def main():
    input_data = os.environ.get("INPUT_DATA", "{}")
    result = {"success": True, "data": "validated", "error": None, "agent_requests": []}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                validation_result = registry.validate_primitive("json_extract.py")

            assert validation_result["valid"] is True
            assert validation_result["schema_compliant"] is True
            assert "output" in validation_result
            assert validation_result["output"]["success"] is True
            assert validation_result["output"]["data"] == "validated"

    def test_primitive_registry_caches_discovery_results(self) -> None:
        """Test that primitive discovery results are cached for performance."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            test_primitive = primitives_dir / "test_cache.py"
            test_primitive.write_text("#!/usr/bin/env python3\nprint('cached')")
            test_primitive.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                # First call should populate cache
                primitives1 = registry.discover_primitives()

                # Second call should use cache
                primitives2 = registry.discover_primitives()

            # Results should be identical (from cache)
            assert primitives1 == primitives2
            assert len(primitives1) == 1
            assert primitives1[0]["name"] == "test_cache.py"

    def test_discover_primitives_skips_init_files(self) -> None:
        """Test that __init__.py files are skipped during discovery."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            category_dir = primitives_dir / "utils"
            category_dir.mkdir(parents=True)

            # Create __init__.py (should be skipped)
            (category_dir / "__init__.py").write_text("# init file")
            # Create regular primitive (should be found)
            test_primitive = category_dir / "test.py"
            test_primitive.write_text("#!/usr/bin/env python3\nprint('test')")
            test_primitive.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                primitives = registry.discover_primitives()

            # Should find only the test.py, not __init__.py
            assert len(primitives) == 1
            assert primitives[0]["name"] == "utils/test.py"

    def test_get_primitive_info_caches_result(self) -> None:
        """Test that get_primitive_info caches results for performance."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            primitive_file = primitives_dir / "cached.py"
            primitive_file.write_text("""#!/usr/bin/env python3
# Primitive: Cached primitive
import json
print(json.dumps({"success": True, "data": "test", "agent_requests": []}))
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                # First call should populate cache
                info1 = registry.get_primitive_info("cached.py")
                # Second call should hit cache
                info2 = registry.get_primitive_info("cached.py")

            assert info1 == info2
            assert info1["name"] == "cached.py"

    def test_get_primitive_info_raises_for_unknown_primitive(self) -> None:
        """Test that get_primitive_info raises FileNotFoundError for unknown primitives."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with pytest.raises(
                    FileNotFoundError, match="Primitive 'nonexistent.py' not found"
                ):
                    registry.get_primitive_info("nonexistent.py")

    def test_validate_primitive_handles_execution_failure(self) -> None:
        """Test validation of primitive that fails during execution."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            # Create primitive that exits with error
            primitive_file = primitives_dir / "failing.py"
            primitive_file.write_text("""#!/usr/bin/env python3
import sys
sys.stderr.write("Execution failed")
sys.exit(1)
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                result = registry.validate_primitive("failing.py")

            assert result["valid"] is False
            assert "Script execution failed" in result["error"]

    def test_validate_primitive_handles_invalid_json(self) -> None:
        """Test validation of primitive that outputs invalid JSON."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            # Create primitive that outputs invalid JSON
            primitive_file = primitives_dir / "bad_json.py"
            primitive_file.write_text("""#!/usr/bin/env python3
print("not valid json {")
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                result = registry.validate_primitive("bad_json.py")

            assert result["valid"] is False
            assert "Invalid JSON output" in result["error"]

    def test_validate_primitive_handles_schema_violation(self) -> None:
        """Test validation of primitive that outputs JSON violating schema."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            # Create primitive that outputs JSON without required fields
            primitive_file = primitives_dir / "bad_schema.py"
            primitive_file.write_text("""#!/usr/bin/env python3
import json
print(json.dumps({"invalid": "schema"}))
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                result = registry.validate_primitive("bad_schema.py")

            assert result["valid"] is False
            assert "Schema validation failed" in result["error"]

    def test_clear_cache_empties_all_caches(self) -> None:
        """Test that clear_cache removes all cached data."""
        registry = PrimitiveRegistry()

        with TemporaryDirectory() as temp_dir:
            primitives_dir = Path(temp_dir) / ".llm-orc" / "scripts" / "primitives"
            primitives_dir.mkdir(parents=True)

            primitive_file = primitives_dir / "test.py"
            primitive_file.write_text("""#!/usr/bin/env python3
# Primitive: Test
import json
print(json.dumps({"success": True, "data": "test", "agent_requests": []}))
""")
            primitive_file.chmod(0o755)

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                # Populate caches
                registry.discover_primitives()
                registry.get_primitive_info("test.py")

                # Verify caches are populated
                assert len(registry._cache) > 0
                assert len(registry._primitive_cache) > 0

                # Clear caches
                registry.clear_cache()

                # Verify caches are empty
                assert len(registry._cache) == 0
                assert len(registry._primitive_cache) == 0
