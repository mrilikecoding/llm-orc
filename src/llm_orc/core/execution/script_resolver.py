"""Script resolution and discovery for script agents."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class ScriptNotFoundError(FileNotFoundError):
    """Custom exception for script resolution errors with helpful guidance."""

    def __init__(self, script_ref: str, is_primitive: bool = False) -> None:
        """Initialize with script reference and guidance."""
        self.script_ref = script_ref
        self.is_primitive = is_primitive

        if is_primitive and script_ref.startswith("primitives/"):
            # Helpful guidance for missing library primitives
            message = (
                f"Primitive script '{script_ref}' not found. "
                f"To use library primitives:\n"
                f"  1. Initialize the library submodule: "
                f"git submodule update --init --recursive\n"
                f"  2. Or create a local implementation at "
                f".llm-orc/scripts/{script_ref}\n"
                f"  3. For tests, use TestPrimitiveFactory fixtures"
            )
        else:
            # Basic error message for non-primitive scripts
            message = f"Script not found: {script_ref}"

        super().__init__(message)


class ScriptResolver:
    """Resolves script references to executable paths with library support."""

    # Supported script file extensions
    SCRIPT_EXTENSIONS = (".py", ".sh", ".bash", ".js", ".rb")

    # Standard directory names
    SCRIPTS_DIR = "scripts"
    LLM_ORC_DIR = ".llm-orc"
    LIBRARY_DIR = "llm-orchestra-library"
    PRIMITIVES_DIR = "primitives"

    def __init__(self, search_paths: list[str] | None = None) -> None:
        """Initialize the script resolver with optional custom search paths."""
        self._cache: dict[str, str] = {}
        self._custom_search_paths = search_paths

    def _get_search_paths(self) -> list[str]:
        """Get search paths in priority order: local → library → system.

        Returns:
            List of search paths in priority order
        """
        if self._custom_search_paths:
            return self._custom_search_paths

        cwd = Path(os.getcwd())
        search_paths = []

        # Priority 0: Test primitives directory (for BDD tests)
        test_primitives_dir = os.environ.get("LLM_ORC_TEST_PRIMITIVES_DIR")
        if test_primitives_dir:
            search_paths.append(test_primitives_dir)

        # Priority 1: Local project paths
        search_paths.extend(
            [
                str(cwd / self.LLM_ORC_DIR / self.SCRIPTS_DIR),
                str(cwd / self.LLM_ORC_DIR),
                str(cwd),
            ]
        )

        # Priority 2: Library submodule paths
        library_base = cwd / self.LIBRARY_DIR
        if library_base.exists():
            search_paths.extend(
                [
                    str(library_base / self.SCRIPTS_DIR),
                    str(library_base / self.PRIMITIVES_DIR / "python"),
                    str(library_base / self.PRIMITIVES_DIR),
                    str(library_base),
                ]
            )

        return search_paths

    def resolve_script_path(self, script_ref: str) -> str:
        """Resolve script reference to executable path or inline content.

        Args:
            script_ref: Script reference - can be:
                - Relative path from search paths (e.g., "primitives/user_input.py")
                - Absolute path (e.g., "/usr/local/bin/analyzer")
                - Inline script content (backward compatibility)

        Returns:
            Resolved script path or inline content

        Raises:
            ScriptNotFoundError: If script file doesn't exist with helpful guidance
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
            raise ScriptNotFoundError(script_ref)

        # Check if it looks like a path (contains / or \ or has script extension)
        is_path = (
            "/" in script_ref
            or "\\" in script_ref
            or script_ref.endswith(self.SCRIPT_EXTENSIONS)
        )

        if is_path:
            # Try to resolve using library-aware search paths
            resolved = self._try_resolve_with_search_paths(script_ref)
            if resolved:
                return resolved

            # If it looks like a path but wasn't found, raise error with guidance
            is_primitive = script_ref.startswith("primitives/")
            raise ScriptNotFoundError(script_ref, is_primitive=is_primitive)

        # Fall back to treating it as inline content (backward compatibility)
        return script_ref

    def _try_resolve_with_search_paths(self, script_ref: str) -> str | None:
        """Try to resolve script using library-aware search paths.

        Args:
            script_ref: Relative script reference

        Returns:
            Resolved path or None if not found
        """
        search_paths = self._get_search_paths()

        for search_path in search_paths:
            search_dir = Path(search_path)

            # Try direct path in search directory
            candidate = search_dir / script_ref
            if candidate.exists():
                return str(candidate)

            # Try without "scripts/" prefix for backward compatibility
            scripts_prefix = f"{self.SCRIPTS_DIR}/"
            if script_ref.startswith(scripts_prefix):
                candidate_no_prefix = search_dir / script_ref.removeprefix(
                    scripts_prefix
                )
                if candidate_no_prefix.exists():
                    return str(candidate_no_prefix)

        return None

    def _try_resolve_relative_path(self, script_ref: str) -> str | None:
        """Legacy method for backward compatibility.

        Args:
            script_ref: Relative script reference

        Returns:
            Resolved path or None if not found
        """
        # Delegate to new library-aware resolution
        return self._try_resolve_with_search_paths(script_ref)

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
        scripts_dir = cwd / self.LLM_ORC_DIR / self.SCRIPTS_DIR

        if scripts_dir.exists():
            # Find all script files recursively
            for ext in self.SCRIPT_EXTENSIONS:
                pattern = f"*{ext}"
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
        except (FileNotFoundError, ScriptNotFoundError):
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

        except (FileNotFoundError, ScriptNotFoundError):
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
