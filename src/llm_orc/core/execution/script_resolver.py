"""Script resolution and discovery for script agents."""

import os
from pathlib import Path


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

        # Priority 1: Check .llm-orc/scripts/ directory
        llm_orc_path = cwd / ".llm-orc" / script_ref
        if llm_orc_path.exists():
            return str(llm_orc_path)

        # Priority 2: Check .llm-orc/ directly (for backward compatibility)
        llm_orc_direct = cwd / ".llm-orc" / script_ref.removeprefix("scripts/")
        if llm_orc_direct.exists():
            return str(llm_orc_direct)

        # Priority 3: Check current directory (fallback)
        current_path = cwd / script_ref
        if current_path.exists():
            return str(current_path)

        return None

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._cache.clear()
