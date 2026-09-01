"""Tests for script resolution and discovery."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_orc.core.execution.scripting.resolver import (
    ScriptNotFoundError,
    ScriptResolver,
)


@pytest.fixture(autouse=True)
def clear_test_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear test environment variables to prevent pollution from BDD tests."""
    monkeypatch.delenv("LLM_ORC_TEST_PRIMITIVES_DIR", raising=False)


class TestScriptResolver:
    """Test script resolver functionality."""

    def test_script_resolver_finds_scripts_in_llm_orc_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that script resolver finds scripts in .llm-orc/scripts/ directory."""
        # Create test directory structure
        llm_orc_dir = tmp_path / ".llm-orc"
        scripts_dir = llm_orc_dir / "scripts"
        primitives_dir = scripts_dir / "primitives"
        primitives_dir.mkdir(parents=True)

        # Create test script
        test_script = primitives_dir / "test_script.py"
        test_script.write_text("#!/usr/bin/env python3\nprint('Hello')")

        # Change to tmp directory for test
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # Test relative path resolution
            result = resolver.resolve_script_path("scripts/primitives/test_script.py")
            assert result == str(test_script)
            assert Path(result).exists()

    def test_script_resolver_handles_absolute_paths(self) -> None:
        """Test that script resolver handles absolute paths correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("#!/usr/bin/env python3\nprint('Absolute')")
            absolute_path = f.name

        try:
            resolver = ScriptResolver()
            result = resolver.resolve_script_path(absolute_path)
            assert result == absolute_path
            assert Path(result).exists()
        finally:
            Path(absolute_path).unlink(missing_ok=True)

    def test_script_resolver_falls_back_to_inline_content(self) -> None:
        """Test script resolver falls back to inline content for compatibility."""
        resolver = ScriptResolver()

        # Test with inline script content (no file path)
        inline_script = "echo 'This is inline content'"
        result = resolver.resolve_script_path(inline_script)
        assert result == inline_script

    def test_script_resolver_raises_for_missing_script(self, tmp_path: Path) -> None:
        """Test that script resolver raises error for missing script files."""
        # Change to tmp directory for test
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # Test with a path that looks like a file but doesn't exist
            with pytest.raises(FileNotFoundError, match="Script not found"):
                resolver.resolve_script_path("scripts/missing_script.py")

    def test_script_resolver_prioritizes_llm_orc_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that .llm-orc/scripts/ takes priority over other locations."""
        # Create .llm-orc script
        llm_orc_dir = tmp_path / ".llm-orc" / "scripts"
        llm_orc_dir.mkdir(parents=True)
        llm_orc_script = llm_orc_dir / "test.py"
        llm_orc_script.write_text("#!/usr/bin/env python3\nprint('llm-orc version')")

        # Create same-named script in current directory
        current_script = tmp_path / "scripts" / "test.py"
        current_script.parent.mkdir(parents=True)
        current_script.write_text("#!/usr/bin/env python3\nprint('current version')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # Should resolve to .llm-orc version
            result = resolver.resolve_script_path("scripts/test.py")
            assert result == str(llm_orc_script)
            content = Path(result).read_text()
            assert "llm-orc version" in content

    def test_script_resolver_handles_nested_paths(self, tmp_path: Path) -> None:
        """Test that script resolver handles nested directory paths."""
        # Create nested directory structure
        scripts_dir = tmp_path / ".llm-orc" / "scripts" / "primitives" / "network"
        scripts_dir.mkdir(parents=True)
        script = scripts_dir / "topology.py"
        script.write_text("#!/usr/bin/env python3\nprint('Topology')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            result = resolver.resolve_script_path(
                "scripts/primitives/network/topology.py"
            )
            assert result == str(script)
            assert Path(result).exists()

    def test_script_resolver_validates_script_extension(self, tmp_path: Path) -> None:
        """Test that script resolver validates allowed script extensions."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)

        # Create scripts with different extensions
        py_script = scripts_dir / "test.py"
        py_script.write_text("#!/usr/bin/env python3")

        sh_script = scripts_dir / "test.sh"
        sh_script.write_text("#!/bin/bash")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # Python scripts should work
            result = resolver.resolve_script_path("scripts/test.py")
            assert Path(result).exists()

            # Shell scripts should work
            result = resolver.resolve_script_path("scripts/test.sh")
            assert Path(result).exists()

    def test_script_resolver_caches_resolutions(self, tmp_path: Path) -> None:
        """Test that script resolver caches path resolutions for performance."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)
        script = scripts_dir / "cached.py"
        script.write_text("#!/usr/bin/env python3")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # First resolution
            result1 = resolver.resolve_script_path("scripts/cached.py")

            # Modify script to test cache (should not affect result)
            with patch.object(Path, "exists", return_value=False):
                # Second resolution should use cache
                result2 = resolver.resolve_script_path("scripts/cached.py")

            assert result1 == result2

    def test_list_available_scripts_empty_directory(self, tmp_path: Path) -> None:
        """Test list_available_scripts with empty scripts directory still finds
        package primitives."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            scripts = resolver.list_available_scripts()
            # Only package primitives (no local scripts)
            local_scripts = [
                s
                for s in scripts
                if not (s.get("relative_path") or "").startswith("primitives/")
            ]
            assert local_scripts == []

    def test_list_available_scripts_no_scripts_directory(self, tmp_path: Path) -> None:
        """Test list_available_scripts when scripts directory doesn't exist."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            scripts = resolver.list_available_scripts()
            local_scripts = [
                s
                for s in scripts
                if not (s.get("relative_path") or "").startswith("primitives/")
            ]
            assert local_scripts == []

    def test_list_available_scripts_includes_package_primitives(self) -> None:
        """Test list_available_scripts includes package primitives."""
        resolver = ScriptResolver()
        scripts = resolver.list_available_scripts()
        pkg_scripts = [
            s
            for s in scripts
            if (s.get("relative_path") or "").startswith("primitives/")
        ]
        assert len(pkg_scripts) >= 6
        names = [s["name"] for s in pkg_scripts]
        assert "get_user_input.py" in names
        assert "read_file.py" in names

    def test_list_available_scripts_with_various_extensions(
        self, tmp_path: Path
    ) -> None:
        """Test list_available_scripts finds all supported script extensions."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)

        # Create scripts with different extensions
        extensions = [".py", ".sh", ".bash", ".js", ".rb"]
        for ext in extensions:
            script = scripts_dir / f"test{ext}"
            script.write_text(f"#!/usr/bin/env python3\n# Test script {ext}")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            scripts = resolver.list_available_scripts()

            script_names = [s["name"] for s in scripts]
            for ext in extensions:
                assert f"test{ext}" in script_names

    def test_list_available_scripts_with_nested_structure(self, tmp_path: Path) -> None:
        """Test list_available_scripts with nested directory structure."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"

        # Create nested structure
        primitives_dir = scripts_dir / "primitives"
        network_dir = primitives_dir / "network"
        network_dir.mkdir(parents=True)

        # Create scripts at different levels
        root_script = scripts_dir / "root.py"
        root_script.write_text("# Root script")

        primitives_script = primitives_dir / "primitive.py"
        primitives_script.write_text("# Primitive script")

        network_script = network_dir / "topology.py"
        network_script.write_text("# Network script")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            scripts = resolver.list_available_scripts()

            # Filter to local scripts only for structure assertions
            local_scripts = [
                s
                for s in scripts
                if not (s.get("relative_path") or "").startswith("primitives/")
                or (s["path"] or "").startswith(str(tmp_path))
            ]
            assert len(local_scripts) >= 3

            # Sort by display_name to check order (expected: network < primitive < root)
            scripts_by_name = {
                s["display_name"]: s for s in scripts if s["display_name"] is not None
            }

            # Check that all expected scripts are present
            assert "primitives/primitive.py" in scripts_by_name
            assert "primitives/network/topology.py" in scripts_by_name
            assert "root.py" in scripts_by_name

            # Check structure for each script
            primitive_script_info: dict[str, str | None] = scripts_by_name[
                "primitives/primitive.py"
            ]
            assert primitive_script_info["name"] == "primitive.py"
            assert primitive_script_info["relative_path"] == "primitives"

            network_script_info: dict[str, str | None] = scripts_by_name[
                "primitives/network/topology.py"
            ]
            assert network_script_info["name"] == "topology.py"
            assert network_script_info["relative_path"] == "primitives/network"

            root_script_info: dict[str, str | None] = scripts_by_name["root.py"]
            assert root_script_info["name"] == "root.py"
            assert root_script_info["relative_path"] is None

    def test_get_script_info_existing_script(self, tmp_path: Path) -> None:
        """Test get_script_info for existing script."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)
        script = scripts_dir / "test.py"
        script.write_text("#!/usr/bin/env python3\nprint('Test')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            info = resolver.get_script_info("scripts/test.py")

            assert info is not None
            assert info["name"] == "scripts/test.py"
            assert info["path"] == str(script)
            assert "Script at" in info["description"]
            assert info["parameters"] == []

    def test_get_script_info_nonexistent_script(self, tmp_path: Path) -> None:
        """Test get_script_info for nonexistent script."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            info = resolver.get_script_info("scripts/missing.py")
            assert info is None

    def test_get_script_info_inline_content(self, tmp_path: Path) -> None:
        """Test get_script_info for inline script content."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            info = resolver.get_script_info("echo 'inline content'")

            # Should return None for inline content (no file path)
            assert info is None

    def test_get_script_info_absolute_path(self, tmp_path: Path) -> None:
        """Test get_script_info with absolute path."""
        script = tmp_path / "absolute_test.py"
        script.write_text("#!/usr/bin/env python3\nprint('Absolute')")

        resolver = ScriptResolver()
        info = resolver.get_script_info(str(script))

        assert info is not None
        assert info["name"] == str(script)
        assert info["path"] == str(script)
        assert "Script at" in info["description"]

    def test_test_script_successful_execution(self, tmp_path: Path) -> None:
        """Test test_script with successful script execution."""
        script = tmp_path / "success.py"
        script.write_text("#!/usr/bin/env python3\nprint('Success!')")
        script.chmod(0o755)

        resolver = ScriptResolver()
        result = resolver.test_script(str(script), {"key": "value"})

        assert result["success"] is True
        assert "Success!" in result["output"]
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)

    def test_test_script_failed_execution(self, tmp_path: Path) -> None:
        """Test test_script with failed script execution."""
        script = tmp_path / "failure.py"
        script.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(1)")
        script.chmod(0o755)

        resolver = ScriptResolver()
        result = resolver.test_script(str(script), {})

        assert result["success"] is False
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)

    def test_test_script_nonexistent(self, tmp_path: Path) -> None:
        """Test test_script with nonexistent script."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            result = resolver.test_script("missing.py", {})

            assert result["success"] is False
            assert "not found" in result["error"]
            assert result["duration_ms"] == 0

    def test_test_script_timeout(self, tmp_path: Path) -> None:
        """Test test_script with script timeout."""
        script = tmp_path / "timeout.py"
        script.write_text("#!/usr/bin/env python3\n")
        script.chmod(0o755)

        with patch(
            "llm_orc.core.execution.scripting.resolver.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=str(script), timeout=1),
        ):
            resolver = ScriptResolver()
            result = resolver.test_script(str(script), {}, timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"]
        assert result["duration_ms"] == 1000

    def test_test_script_parameters_passed(self, tmp_path: Path) -> None:
        """Test test_script passes parameters via environment."""
        script = tmp_path / "params.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import os, json\n"
            "params = json.loads(os.environ.get('SCRIPT_PARAMS', '{}'))\n"
            "print(f'Got: {params}')"
        )
        script.chmod(0o755)

        resolver = ScriptResolver()
        result = resolver.test_script(str(script), {"test": "value"})

        assert result["success"] is True
        assert "Got: {'test': 'value'}" in result["output"]

    def test_script_not_found_error_primitive_guidance(self) -> None:
        """Test ScriptNotFoundError provides helpful guidance for primitives."""
        error = ScriptNotFoundError("primitives/missing.py", is_primitive=True)

        error_msg = str(error)
        assert "Primitive script 'primitives/missing.py' not found" in error_msg
        assert "pip install" in error_msg
        assert ".llm-orc/scripts/primitives/missing.py" in error_msg
        assert "TestPrimitiveFactory fixtures" in error_msg

    def test_script_not_found_error_regular_script(self) -> None:
        """Test ScriptNotFoundError for regular scripts."""
        error = ScriptNotFoundError("missing.py", is_primitive=False)

        error_msg = str(error)
        assert "Script not found: missing.py" in error_msg
        assert "git submodule" not in error_msg

    def test_library_aware_resolution_with_library_submodule(
        self, tmp_path: Path
    ) -> None:
        """Test library-aware resolution finds scripts in library submodule."""
        # Create library submodule structure
        library_dir = tmp_path / "llm-orchestra-library"
        primitives_dir = library_dir / "primitives" / "python"
        primitives_dir.mkdir(parents=True)

        # Create script in library
        library_script = primitives_dir / "library_script.py"
        library_script.write_text("#!/usr/bin/env python3\nprint('Library')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            result = resolver.resolve_script_path("scripts/library_script.py")
            assert result == str(library_script)

    def test_library_aware_resolution_priority_order(self, tmp_path: Path) -> None:
        """Test that local scripts take priority over library scripts."""
        # Create local script
        local_dir = tmp_path / ".llm-orc" / "scripts"
        local_dir.mkdir(parents=True)
        local_script = local_dir / "same_name.py"
        local_script.write_text("#!/usr/bin/env python3\nprint('Local')")

        # Create library script with same name
        library_dir = tmp_path / "llm-orchestra-library"
        primitives_dir = library_dir / "primitives" / "python"
        primitives_dir.mkdir(parents=True)
        library_script = primitives_dir / "same_name.py"
        library_script.write_text("#!/usr/bin/env python3\nprint('Library')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            result = resolver.resolve_script_path("scripts/same_name.py")

            # Should resolve to local version
            assert result == str(local_script)
            content = Path(result).read_text()
            assert "Local" in content

    def test_clear_cache_functionality(self, tmp_path: Path) -> None:
        """Test that clear_cache actually clears the resolution cache."""
        scripts_dir = tmp_path / ".llm-orc" / "scripts"
        scripts_dir.mkdir(parents=True)
        script = scripts_dir / "cached.py"
        script.write_text("#!/usr/bin/env python3")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()

            # First resolution - populates cache
            resolver.resolve_script_path("scripts/cached.py")

            # Clear cache
            resolver.clear_cache()

            # Delete script to test cache is truly cleared
            script.unlink()

            # Should raise error since cache is cleared and file is gone
            with pytest.raises(ScriptNotFoundError):
                resolver.resolve_script_path("scripts/cached.py")

    def test_custom_search_paths(self, tmp_path: Path) -> None:
        """Test ScriptResolver with custom search paths."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        script = custom_dir / "custom_script.py"
        script.write_text("#!/usr/bin/env python3")

        resolver = ScriptResolver(search_paths=[str(custom_dir)])
        result = resolver.resolve_script_path("custom_script.py")
        assert result == str(script)

    def test_environment_variable_test_primitives_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ScriptResolver respects LLM_ORC_TEST_PRIMITIVES_DIR env var."""
        # Create test primitives directory
        test_primitives = tmp_path / "test_primitives"
        test_primitives.mkdir()
        script = test_primitives / "test_script.py"
        script.write_text("#!/usr/bin/env python3")

        # Set environment variable
        monkeypatch.setenv("LLM_ORC_TEST_PRIMITIVES_DIR", str(test_primitives))

        # Change to different directory
        with patch("os.getcwd", return_value=str(tmp_path / "other")):
            resolver = ScriptResolver()
            result = resolver.resolve_script_path("test_script.py")
            assert result == str(script)

    def test_package_primitives_search_path(self, tmp_path: Path) -> None:
        """Test that installed package primitives parent is in search paths."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            paths = resolver._get_search_paths()

            # Should contain the parent of the primitives/ package dir
            # so that refs like "primitives/..." resolve correctly
            package_parent = [p for p in paths if (Path(p) / "primitives").is_dir()]
            assert len(package_parent) >= 1

    def test_package_primitives_resolves_script(self, tmp_path: Path) -> None:
        """Test that primitives from installed package resolve correctly."""
        # Create a fake package primitives directory structure
        pkg_primitives = tmp_path / "pkg_primitives"
        user_interaction = pkg_primitives / "user_interaction"
        user_interaction.mkdir(parents=True)
        script = user_interaction / "get_user_input.py"
        script.write_text("#!/usr/bin/env python3\nprint('package')")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            # Patch the package primitives path
            with patch.object(
                resolver,
                "_get_search_paths",
                wraps=resolver._get_search_paths,
            ):
                # Add our fake primitives to custom search paths
                resolver._custom_search_paths = [str(pkg_primitives)]
                result = resolver.resolve_script_path(
                    "user_interaction/get_user_input.py"
                )
                assert result == str(script)

    def test_local_overrides_package_primitives(self, tmp_path: Path) -> None:
        """Test that local .llm-orc/scripts/ overrides package primitives."""
        # Create local script
        local_dir = tmp_path / ".llm-orc" / "scripts" / "primitives"
        local_dir.mkdir(parents=True)
        user_dir = local_dir / "user_interaction"
        user_dir.mkdir()
        local_script = user_dir / "get_user_input.py"
        local_script.write_text("# local version")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            result = resolver.resolve_script_path(
                "primitives/user_interaction/get_user_input.py"
            )
            assert result == str(local_script)
            assert "local version" in Path(result).read_text()

    def test_hyphen_to_underscore_normalization(self, tmp_path: Path) -> None:
        """Test hyphen-to-underscore normalization in script resolution."""
        # Create script with underscore path
        scripts_dir = tmp_path / ".llm-orc" / "scripts" / "primitives"
        underscore_dir = scripts_dir / "user_interaction"
        underscore_dir.mkdir(parents=True)
        script = underscore_dir / "get_user_input.py"
        script.write_text("#!/usr/bin/env python3")

        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            # Reference uses hyphens, file uses underscores
            result = resolver.resolve_script_path(
                "primitives/user-interaction/get_user_input.py"
            )
            assert result == str(script)

    def test_hyphen_normalization_no_false_positive(self, tmp_path: Path) -> None:
        """Test that normalization doesn't create false matches."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            resolver = ScriptResolver()
            with pytest.raises(ScriptNotFoundError):
                resolver.resolve_script_path("primitives/nonexistent-dir/missing.py")


class TestTestScriptInterpreter:
    """#154: ``llm-orc scripts test`` ran a .py script bare, relying on its
    shebang and the exec bit, so it failed with Permission denied on any
    script checked in as mode 644 (every serving script) and otherwise
    resolved python through PATH like the engine used to. Same invariant,
    same fix, so the repo has one way to run a python script."""

    def test_a_non_executable_python_script_still_runs(self, tmp_path: Path) -> None:
        script = tmp_path / ".llm-orc" / "scripts" / "probe.py"
        script.parent.mkdir(parents=True)
        script.write_text('import json\nprint(json.dumps({"ran": True}))\n')
        script.chmod(0o644)

        resolver = ScriptResolver(project_dir=tmp_path)
        result = resolver.test_script("probe.py", {})

        assert result["success"] is True, result
        assert "ran" in str(result["output"])

    def test_a_python_script_runs_under_the_host_interpreter(
        self, tmp_path: Path
    ) -> None:
        """The same guarantee the engine gives: an import that works
        in-process works here."""
        script = tmp_path / ".llm-orc" / "scripts" / "imports_llm_orc.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            'import json, sys, llm_orc\nprint(json.dumps({"exe": sys.executable}))\n'
        )
        script.chmod(0o644)

        resolver = ScriptResolver(project_dir=tmp_path)
        result = resolver.test_script("imports_llm_orc.py", {})

        assert result["success"] is True, result
        assert sys.executable in str(result["output"])


class TestOnePredicateFileVsInline:
    """#177: file-vs-inline was decided in three places by two rules, and
    they disagreed for a bare name that names a file in the process CWD
    -- the resolver called it content while ``ScriptAgent`` executed the
    file. ``resolve_and_classify`` is now the sole call and describes
    what actually executes; these pin the traps the widening has to
    respect.
    """

    def test_a_bare_cwd_file_classifies_the_same_as_scriptagent_would(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instrument 1. ``os.path.exists`` on the resolved reference is
        exactly how ``ScriptAgent`` decided file-vs-inline before one
        predicate existed, so ``resolve_and_classify``'s own answer has
        to agree -- including this one, where the two used to disagree."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "probe").write_text("#!/bin/bash\necho hi\n")
        resolver = ScriptResolver()

        resolved, is_file = resolver.resolve_and_classify("probe")
        scriptagent_would_execute_as_file = os.path.exists(resolved)

        assert is_file is scriptagent_would_execute_as_file, (
            "resolve_and_classify and a fresh os.path.exists disagree on file-vs-inline"
        )

    def test_a_bare_name_naming_nothing_stays_inline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trap 2, unchanged: a bare name that names nothing in the CWD is
        still inline content, resolved verbatim (never anchored -- only a
        FILE reference is anchored)."""
        monkeypatch.chdir(tmp_path)
        resolver = ScriptResolver()

        assert resolver.resolve_and_classify("probe") == ("probe", False)

    def test_a_dot_ts_bare_name_in_cwd_still_classifies_as_a_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instrument 4. ``.ts``/``.mjs`` (and ``.pl``/``.zsh``/``.ps1``/
        ``.R``, the issue's own list) are not in ``SCRIPT_EXTENSIONS``, so
        a bare ``probe.ts`` only classifies as a file through the
        CWD-existence clause -- the exact shape the issue measured.
        Anchored (``./probe.ts``), not verbatim (#177 review round 2
        finding 1)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "probe.ts").write_text("console.log('hi')\n")
        resolver = ScriptResolver()

        assert resolver.resolve_and_classify("probe.ts") == ("./probe.ts", True)

    def test_a_bare_name_present_in_both_cwd_and_llm_orc_scripts_keeps_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trap 1, a PRESERVATION guard, not a red-first pin (#177 review
        round 1 finding 2). Main already returned a bare name verbatim
        without ever consulting search paths, so this shape was never
        reachable pre-#177 and this assertion was already true on main --
        widening the predicate never sends a bare name through
        ``_try_resolve_with_search_paths`` at all (only path-syntax
        references search). Kept so a future reimplementation that DID
        route bare names through search paths could not silently let a
        same-named ``.llm-orc/scripts`` (or library) entry take over what
        a bare CWD name already executes."""
        llm_orc_scripts = tmp_path / ".llm-orc" / "scripts"
        llm_orc_scripts.mkdir(parents=True)
        (llm_orc_scripts / "probe").write_text("library version")
        (tmp_path / "probe").write_text("cwd version")

        monkeypatch.chdir(tmp_path)
        resolver = ScriptResolver(project_dir=tmp_path)

        result = resolver.resolve_script_path("probe")

        # Anchored (./probe), not verbatim (#177 review round 2 finding 1).
        assert result == "./probe"
        assert Path(result).read_text() == "cwd version"


class TestResolveAndClassify:
    """#177 review round 1 finding 1 (BLOCKER): the resolve and the
    classification used to come from two independent ``os.path.exists``
    calls, which a bare name's file could vanish between -- the resolve
    saw it and the later, separate classification did not, flipping a
    FILE reference to inline. ``resolve_and_classify`` is the one call
    every production consumer now uses instead, so there is only one
    observation to ask.

    Round 2 finding 1: one observation was not the whole fix. A bare FILE
    reference returned VERBATIM is still slashless, and handing a
    slashless name to an interpreter or execvp is subject to a PATH
    search if the file is gone by execution time -- measured true, a
    same-named PATH impostor ran and reported success. The FILE branch
    below is anchored (``./name``) rather than verbatim so a vanished
    file cannot be re-resolved that way.
    """

    def test_a_bare_cwd_file_is_one_call_not_two_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one-observation contract, directly: exactly one
        ``os.path.exists`` call for a bare reference, not the two a
        caller pairing a resolve with a separate classification predicate
        would make."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "probe").write_text("#!/bin/bash\necho hi\n")
        resolver = ScriptResolver()
        real_exists = os.path.exists
        calls: list[str] = []

        def counting_exists(path: str) -> bool:
            if path == "probe":
                calls.append(path)
            return real_exists(path)

        with patch("os.path.exists", counting_exists):
            resolved, is_file = resolver.resolve_and_classify("probe")

        assert (resolved, is_file) == ("./probe", True)
        assert len(calls) == 1, f"expected one stat, saw {len(calls)}"

    def test_a_bare_name_naming_nothing_answers_inline_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        resolver = ScriptResolver()

        assert resolver.resolve_and_classify("probe") == ("probe", False)

    def test_a_path_syntax_reference_that_resolves_nothing_raises(
        self, tmp_path: Path
    ) -> None:
        """Unlike a bare reference (a total, non-raising case), the
        search IS the resolution for path syntax, so not finding anything
        raises -- unchanged from ``resolve_script_path``."""
        resolver = ScriptResolver(project_dir=tmp_path)

        with pytest.raises(ScriptNotFoundError):
            resolver.resolve_and_classify("scripts/missing.py")

    def test_an_absolute_reference_that_does_not_exist_raises(
        self, tmp_path: Path
    ) -> None:
        resolver = ScriptResolver()

        with pytest.raises(ScriptNotFoundError):
            resolver.resolve_and_classify(str(tmp_path / "gone.py"))

    def test_an_absolute_reference_that_exists_classifies_as_a_file(
        self, tmp_path: Path
    ) -> None:
        script = tmp_path / "real.py"
        script.write_text("print('hi')\n")
        resolver = ScriptResolver()

        assert resolver.resolve_and_classify(str(script)) == (str(script), True)
