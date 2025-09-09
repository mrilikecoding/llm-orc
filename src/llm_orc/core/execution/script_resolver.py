"""Script resolution and discovery for script agents."""

import os
from pathlib import Path
from typing import Any


class ScriptResolver:
    """Resolves script references to executable paths or inline content."""

    def __init__(self) -> None:
        """Initialize the script resolver."""
        self._cache: dict[str, str] = {}

    def resolve_script_path(self, script_ref: str) -> str:
        """Resolve script reference to executable path or inline content.

        Args:
            script_ref: Script reference - can be:
                - Relative path from .llm-orc/scripts/ (e.g., "scripts/test.py")
                - Absolute path (e.g., "/usr/local/bin/analyzer")
                - Inline script content (backward compatibility)

        Returns:
            Resolved script path or inline content

        Raises:
            FileNotFoundError: If script file doesn't exist
        """
        # Check cache first
        if script_ref in self._cache:
            return self._cache[script_ref]

        resolved = self._resolve_uncached(script_ref)
        self._cache[script_ref] = resolved
        return resolved

    def _resolve_uncached(self, script_ref: str) -> str:
        """Resolve script reference without using cache."""
        # Check if it's an absolute path
        if os.path.isabs(script_ref):
            path = Path(script_ref)
            if path.exists():
                return str(path)
            raise FileNotFoundError(f"Script not found: {script_ref}")

        # Check if it looks like a path (contains / or \ or has script extension)
        is_path = (
            "/" in script_ref
            or "\\" in script_ref
            or script_ref.endswith((".py", ".sh", ".bash", ".js", ".rb"))
        )

        if is_path:
            # Try to resolve as relative path from .llm-orc directory
            resolved = self._try_resolve_relative_path(script_ref)
            if resolved:
                return resolved

            # If it looks like a path but wasn't found, raise error
            raise FileNotFoundError(f"Script not found: {script_ref}")

        # Fall back to treating it as inline content (backward compatibility)
        return script_ref

    def _try_resolve_relative_path(self, script_ref: str) -> str | None:
        """Try to resolve script as relative path from .llm-orc directory.

        Args:
            script_ref: Relative script reference

        Returns:
            Resolved path or None if not found
        """
        # Get current working directory
        cwd = Path(os.getcwd())

        # Priority 1: Check .llm-orc/scripts/ directory (hierarchical names)
        llm_orc_scripts_path = cwd / ".llm-orc" / "scripts" / script_ref
        if llm_orc_scripts_path.exists():
            return str(llm_orc_scripts_path)

        # Priority 2: Check .llm-orc/ directory directly
        llm_orc_path = cwd / ".llm-orc" / script_ref
        if llm_orc_path.exists():
            return str(llm_orc_path)

        # Priority 3: Check .llm-orc/ directly (backward compatibility)
        llm_orc_direct = cwd / ".llm-orc" / script_ref.removeprefix("scripts/")
        if llm_orc_direct.exists():
            return str(llm_orc_direct)

        # Priority 4: Check current directory (fallback)
        current_path = cwd / script_ref
        if current_path.exists():
            return str(current_path)

        return None

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._cache.clear()

    def list_available_scripts(self) -> list[dict[str, str | None]]:
        """List all available scripts in .llm-orc/scripts directory and subdirectories.

        Returns:
            List of script dictionaries with name, path, and relative_path
        """
        scripts = []
        cwd = Path(os.getcwd())
        scripts_dir = cwd / ".llm-orc" / "scripts"

        if scripts_dir.exists():
            # Find all script files recursively
            for pattern in ["*.py", "*.sh", "*.bash", "*.js", "*.rb"]:
                for script_file in scripts_dir.rglob(pattern):
                    # Calculate relative path for hierarchical display
                    relative_path = script_file.relative_to(scripts_dir)
                    relative_dir = (
                        str(relative_path.parent)
                        if relative_path.parent != Path(".")
                        else None
                    )
                    display_name = (
                        f"{relative_dir}/{script_file.name}"
                        if relative_dir
                        else script_file.name
                    )

                    scripts.append(
                        {
                            "name": script_file.name,
                            "display_name": display_name,
                            "path": str(script_file),
                            "relative_path": relative_dir,
                        }
                    )

        return sorted(scripts, key=lambda x: x["display_name"] or "")

    def get_script_info(self, script_name: str) -> dict[str, str | list[str]] | None:
        """Get information about a specific script.

        Args:
            script_name: Name of the script

        Returns:
            Script information dictionary or None if not found
        """
        try:
            script_path = self.resolve_script_path(script_name)
            if not os.path.exists(script_path):
                return None

            return {
                "name": script_name,
                "path": script_path,
                "description": f"Script at {script_path}",
                "parameters": [],  # Basic implementation
            }
        except FileNotFoundError:
            return None

    def test_script(
        self, script_name: str, parameters: dict[str, str]
    ) -> dict[str, Any]:
        """Test script execution with given parameters.

        Args:
            script_name: Name of the script to test
            parameters: Dictionary of parameters for the script

        Returns:
            Dictionary with execution results
        """
        import json
        import subprocess
        import time

        try:
            script_path = self.resolve_script_path(script_name)
            start_time = time.time()

            # Prepare environment with parameters as JSON
            env = os.environ.copy()
            env["SCRIPT_PARAMS"] = json.dumps(parameters)

            # Execute script
            result = subprocess.run(
                [script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout.strip(),
                    "duration_ms": duration_ms,
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip(),
                    "duration_ms": duration_ms,
                }

        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Script '{script_name}' not found",
                "duration_ms": 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Script execution timed out",
                "duration_ms": 30000,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "duration_ms": 0,
            }
