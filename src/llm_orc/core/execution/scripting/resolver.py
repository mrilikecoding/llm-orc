"""Script resolution and discovery for script agents."""

import json
import os
import subprocess
import sys
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
            message = (
                f"Primitive script '{script_ref}' not found. "
                f"Primitives are included in the llm-orc package.\n"
                f"  1. Ensure llm-orc is installed: pip install -e .\n"
                f"  2. Or create a local implementation at "
                f".llm-orc/scripts/{script_ref}\n"
                f"  3. For tests, use TestPrimitiveFactory fixtures"
            )
        else:
            # Basic error message for non-primitive scripts
            message = f"Script not found: {script_ref}"

        super().__init__(message)


def _script_command(script_path: str) -> list[str]:
    """The argv for running ``script_path``: python scripts go through
    the interpreter running llm-orc, everything else runs as-is."""
    if Path(script_path).suffix.lower() in (".py", ".python"):
        if sys.executable and not getattr(sys, "frozen", False):
            return [sys.executable, script_path]
        return ["python3", script_path]
    return [script_path]


class ScriptResolver:
    """Resolves script references to executable paths with library support."""

    # Supported script file extensions
    SCRIPT_EXTENSIONS = (".py", ".sh", ".bash", ".js", ".rb")

    # Standard directory names
    SCRIPTS_DIR = "scripts"
    LLM_ORC_DIR = ".llm-orc"
    LIBRARY_DIR = "llm-orchestra-library"
    PRIMITIVES_DIR = "primitives"

    def __init__(
        self,
        search_paths: list[str] | None = None,
        project_dir: Path | None = None,
    ) -> None:
        """Initialize the script resolver with optional custom search paths."""
        self._cache: dict[str, str] = {}
        self._custom_search_paths = search_paths
        self._project_dir = project_dir

    def _get_search_paths(self) -> list[str]:
        """Get search paths in priority order: local → library → system.

        Returns:
            List of search paths in priority order
        """
        if self._custom_search_paths:
            return self._custom_search_paths

        base = self._project_dir or Path(os.getcwd())
        search_paths = []

        # Priority 0: Test primitives directory (for BDD tests)
        test_primitives_dir = os.environ.get("LLM_ORC_TEST_PRIMITIVES_DIR")
        if test_primitives_dir:
            search_paths.append(test_primitives_dir)

        # Priority 1: Local project paths
        search_paths.extend(
            [
                str(base / self.LLM_ORC_DIR / self.SCRIPTS_DIR),
                str(base / self.LLM_ORC_DIR),
                str(base),
            ]
        )

        # Priority 1.5: Installed package primitives
        # The parent of the primitives/ dir so that refs like
        # "primitives/user_interaction/get_user_input.py" resolve correctly
        package_primitives = Path(__file__).resolve().parents[3] / "primitives"
        if package_primitives.exists():
            search_paths.append(str(package_primitives.parent))

        # Priority 2: Library submodule paths
        library_base = base / self.LIBRARY_DIR
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

    def _has_path_syntax(self, script_ref: str) -> bool:
        """Whether a reference LOOKS like a path: a separator or a
        ``SCRIPT_EXTENSIONS`` suffix.

        No separate absolute-path check: every absolute path contains a
        separator on either platform, so one would be dead. Review round 3
        caught its removal surviving the whole suite, which is what dead
        defensive code looks like from the outside.
        """
        return (
            "/" in script_ref
            or "\\" in script_ref
            or script_ref.endswith(self.SCRIPT_EXTENSIONS)
        )

    def resolve_and_classify(self, script_ref: str) -> tuple[str, bool]:
        """Resolve AND classify ``script_ref`` from ONE observation of the
        filesystem, so a resolution and the decision that picks which
        subprocess shape runs cannot be taken at two different moments and
        disagree (#177 review round 1).

        Before this, a consumer that needed both called
        ``resolve_script_path`` and ``is_inline_content`` separately, each
        doing its OWN ``os.path.exists`` on a bare name. If the file
        vanished in the gap between those two calls, the resolve had
        already returned the bare name (found) while the LATER, fresh
        classification saw it gone and called it inline — so the bare
        name was handed to ``bash -c``, which runs whatever program
        shares that name on ``PATH`` rather than erroring. Measured true
        end to end, with a fresh ``ScriptAgent`` per execution (the
        production shape): a bare reference that classified as a file at
        the START of a call must stay a file for the REST of that call.

        Every consumer that needs both the resolved value and the
        classification — ``ScriptAgent``'s three execution sites and
        ``ScriptAgentRunner._cache_identity`` — calls this instead of
        pairing ``resolve_script_path`` with ``is_inline_content``.

        Returns:
            ``(resolved, is_file)``. ``is_file`` is authoritative for the
            rest of the caller's turn: an absence discovered later (the
            file removed between this call and the subprocess launch) is
            a run failure the subprocess itself reports, never a reason
            to reinterpret the reference as inline.

        Raises:
            ScriptNotFoundError: for an absolute or path-syntax reference
                that does not resolve to a file — unchanged from
                ``resolve_script_path``. A bare reference never raises:
                absent, it is inline content and answers ``(ref, False)``.
        """
        if os.path.isabs(script_ref):
            path = Path(script_ref)
            if path.exists():
                return str(path), True
            raise ScriptNotFoundError(script_ref)

        if self._has_path_syntax(script_ref):
            # Path syntax never falls back to inline (trap 3): the search
            # itself is the one observation, since each candidate is
            # necessarily probed with `.exists()` to find it.
            resolved = self._try_resolve_with_search_paths(script_ref)
            if resolved:
                return resolved, True
            is_primitive = script_ref.startswith("primitives/")
            raise ScriptNotFoundError(script_ref, is_primitive=is_primitive)

        # Bare name: the ONE stat that decides both the resolved value and
        # the classification together (trap 1: CWD wins, before any
        # search-path logic; unchanged when it names nothing -- trap 2).
        if os.path.exists(script_ref):
            return script_ref, True
        return script_ref, False

    def is_inline_content(self, script_ref: str) -> bool:
        """Whether this reference is inline script content rather than a file.

        A read-only classification for inspection (tests, tooling) — it
        answers on its own, without resolving. It does NOT raise for a
        path-syntax reference that resolves to nothing (unlike
        ``resolve_and_classify``, whose search there IS the resolution),
        so it stays a total function of the reference alone.

        A production consumer that needs to ACT on the classification —
        run a file, or name its bytes for a cache key — must call
        ``resolve_and_classify`` instead of pairing this with a separate
        ``resolve_script_path``: two independent calls take two
        independent snapshots of the filesystem, and #177 review round 1
        measured that gap as live (see ``resolve_and_classify``'s
        docstring).

        A reference is a FILE when it has path syntax (see
        ``_has_path_syntax``) OR when, bare, it names a file that exists
        relative to the process CWD. Otherwise it is inline content:
        ``resolve_script_path`` returns it verbatim and never touches a
        file for it.
        """
        if self._has_path_syntax(script_ref):
            return False
        # A bare name is a FILE when it names one relative to the process
        # CWD. This stat is the one extra filesystem call this predicate
        # costs on the inline path, same as it always has.
        return not os.path.exists(script_ref)

    def _resolve_uncached(self, script_ref: str) -> str:
        """Resolve script reference without using cache."""
        resolved, _ = self.resolve_and_classify(script_ref)
        return resolved

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

            # Try hyphen-to-underscore normalization
            normalized_ref = script_ref.replace("-", "_")
            if normalized_ref != script_ref:
                candidate_norm = search_dir / normalized_ref
                if candidate_norm.exists():
                    return str(candidate_norm)

            # Try without "scripts/" prefix for backward compatibility
            scripts_prefix = f"{self.SCRIPTS_DIR}/"
            if script_ref.startswith(scripts_prefix):
                candidate_no_prefix = search_dir / script_ref.removeprefix(
                    scripts_prefix
                )
                if candidate_no_prefix.exists():
                    return str(candidate_no_prefix)

        return None

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._cache.clear()

    def list_available_scripts(self) -> list[dict[str, str | None]]:
        """List available scripts from .llm-orc/scripts and package primitives.

        Returns:
            List of script dictionaries with name, path, and relative_path
        """
        scripts: list[dict[str, str | None]] = []
        cwd = Path(os.getcwd())
        scripts_dir = cwd / self.LLM_ORC_DIR / self.SCRIPTS_DIR

        if scripts_dir.exists():
            self._collect_local_scripts(scripts_dir, scripts)

        package_primitives = Path(__file__).resolve().parents[3] / "primitives"
        if package_primitives.exists():
            self._collect_package_primitives(package_primitives, scripts)

        return sorted(scripts, key=lambda x: x["display_name"] or "")

    def _collect_local_scripts(
        self,
        scripts_dir: Path,
        scripts: list[dict[str, str | None]],
    ) -> None:
        """Collect scripts from .llm-orc/scripts directory."""
        for ext in self.SCRIPT_EXTENSIONS:
            for script_file in scripts_dir.rglob(f"*{ext}"):
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

    @staticmethod
    def _collect_package_primitives(
        package_primitives: Path,
        scripts: list[dict[str, str | None]],
    ) -> None:
        """Collect primitives from the installed llm_orc package."""
        for script_file in package_primitives.rglob("*.py"):
            if script_file.name.startswith("__"):
                continue
            relative_path = script_file.relative_to(package_primitives)
            category = str(relative_path.parent)
            scripts.append(
                {
                    "name": script_file.name,
                    "display_name": f"primitives/{category}/{script_file.name}",
                    "path": str(script_file),
                    "relative_path": f"primitives/{category}",
                }
            )

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
        self,
        script_name: str,
        parameters: dict[str, str],
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Test script execution with given parameters.

        Args:
            script_name: Name of the script to test
            parameters: Dictionary of parameters for the script
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with execution results
        """

        try:
            script_path = self.resolve_script_path(script_name)
            start_time = time.time()

            # Prepare environment with parameters as JSON
            env = os.environ.copy()
            env["SCRIPT_PARAMS"] = json.dumps(parameters)

            # Execute script. A .py script runs under llm-orc's own
            # interpreter, matching the engine (#154): running it bare
            # relied on the shebang plus the exec bit, so any script
            # checked in as mode 644 (every serving script) failed with
            # Permission denied, and one that did run resolved python
            # through PATH — the same fragility, in the command users
            # reach for to debug a script.
            result = subprocess.run(
                _script_command(script_path),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
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
            timeout_ms = timeout * 1000
            return {
                "success": False,
                "output": "",
                "error": "Script execution timed out",
                "duration_ms": timeout_ms,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "duration_ms": 0,
            }
