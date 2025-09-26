"""Tests for script resolution and discovery."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_orc.core.execution.script_resolver import ScriptResolver


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
