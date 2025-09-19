"""Primitive registry for discovering and managing script agent primitives (ADR-001)."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput


class PrimitiveRegistry:
    """Registry for discovering and managing primitive script agents."""

    def __init__(self) -> None:
        """Initialize the primitive registry."""
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._primitive_cache: dict[str, dict[str, Any]] = {}

    def discover_primitives(self) -> list[dict[str, Any]]:
        """Discover available primitive scripts in .llm-orc/scripts/primitives.

        Returns:
            List of primitive metadata dictionaries
        """
        cache_key = "primitives"
        if cache_key in self._cache:
            return self._cache[cache_key]

        primitives = []
        cwd = Path.cwd()
        primitives_dir = cwd / ".llm-orc" / "scripts" / "primitives"

        if primitives_dir.exists():
            # Find all Python script files in primitives directory
            for script_file in primitives_dir.glob("*.py"):
                primitive_info = {
                    "name": script_file.name,
                    "path": str(script_file),
                    "relative_path": str(script_file.relative_to(cwd)),
                    "type": "primitive",
                    "executable": script_file.stat().st_mode & 0o111 != 0,
                }
                primitives.append(primitive_info)

        # Cache the results
        self._cache[cache_key] = primitives
        return primitives

    def get_primitive_info(self, primitive_name: str) -> dict[str, Any]:
        """Get detailed information about a specific primitive.

        Args:
            primitive_name: Name of the primitive script

        Returns:
            Primitive metadata dictionary

        Raises:
            FileNotFoundError: If primitive is not found
        """
        if primitive_name in self._primitive_cache:
            return self._primitive_cache[primitive_name]

        primitives = self.discover_primitives()
        for primitive in primitives:
            if primitive["name"] == primitive_name:
                # Extract metadata from script comments
                metadata = self._extract_primitive_metadata(primitive["path"])
                primitive_info = {**primitive, **metadata}
                self._primitive_cache[primitive_name] = primitive_info
                return primitive_info

        raise FileNotFoundError(f"Primitive '{primitive_name}' not found")

    def validate_primitive(self, primitive_name: str) -> dict[str, Any]:
        """Validate that a primitive conforms to ScriptAgentInput/Output schemas.

        Args:
            primitive_name: Name of the primitive to validate

        Returns:
            Validation result dictionary

        Raises:
            FileNotFoundError: If primitive is not found
        """
        primitive_info = self.get_primitive_info(primitive_name)
        script_path = primitive_info["path"]

        # Test the primitive with sample input
        test_input = ScriptAgentInput(
            agent_name="test_validation",
            input_data="test validation input",
            context={"validation": True},
            dependencies={"test": "data"},
        )

        try:
            # Execute primitive with test input
            env = os.environ.copy()
            env["INPUT_DATA"] = test_input.model_dump_json()

            result = subprocess.run(
                [script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )

            if result.returncode != 0:
                return {
                    "valid": False,
                    "error": f"Script execution failed: {result.stderr}",
                    "stdout": result.stdout,
                }

            # Validate output conforms to ScriptAgentOutput schema
            try:
                output_data = json.loads(result.stdout.strip())
                validated_output = ScriptAgentOutput.model_validate(output_data)
                return {
                    "valid": True,
                    "schema_compliant": True,
                    "output": validated_output.model_dump(),
                }
            except json.JSONDecodeError as e:
                return {
                    "valid": False,
                    "error": f"Invalid JSON output: {e}",
                    "stdout": result.stdout,
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Schema validation failed: {e}",
                    "stdout": result.stdout,
                }

        except subprocess.TimeoutExpired:
            return {
                "valid": False,
                "error": "Script execution timed out",
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Execution error: {e}",
            }

    def _extract_primitive_metadata(self, script_path: str) -> dict[str, Any]:
        """Extract metadata from primitive script comments.

        Args:
            script_path: Path to the primitive script

        Returns:
            Metadata dictionary extracted from comments
        """
        metadata = {
            "description": "",
            "input_schema": "",
            "output_schema": "",
            "dependencies": [],
        }

        try:
            with open(script_path, encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith("# Primitive:"):
                    metadata["description"] = line.replace("# Primitive:", "").strip()
                elif line.startswith("# Input:"):
                    metadata["input_schema"] = line.replace("# Input:", "").strip()
                elif line.startswith("# Output:"):
                    metadata["output_schema"] = line.replace("# Output:", "").strip()
                elif line.startswith("# Depends:"):
                    deps = line.replace("# Depends:", "").strip()
                    metadata["dependencies"] = [d.strip() for d in deps.split(",")]

        except Exception:
            # If we can't read the file, return empty metadata
            pass

        return metadata

    def clear_cache(self) -> None:
        """Clear all cached primitive information."""
        self._cache.clear()
        self._primitive_cache.clear()
