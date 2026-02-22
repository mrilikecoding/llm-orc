"""Tests for ProjectContext value object."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.mcp.project_context import ProjectContext


class TestProjectContextCreate:
    """Tests for ProjectContext.create factory method."""

    def test_create_with_none_path(self) -> None:
        """Create with None uses default ConfigurationManager."""
        ctx = ProjectContext.create(None)

        assert ctx.project_path is None
        assert ctx.config_manager is not None

    def test_create_with_string_path(self, tmp_path: Path) -> None:
        """Create with string path resolves to Path."""
        ctx = ProjectContext.create(str(tmp_path))

        assert ctx.project_path == tmp_path
        assert ctx.config_manager is not None

    def test_create_with_path_object(self, tmp_path: Path) -> None:
        """Create with Path object works."""
        ctx = ProjectContext.create(tmp_path)

        assert ctx.project_path == tmp_path

    def test_frozen_prevents_mutation(self, tmp_path: Path) -> None:
        """Frozen dataclass prevents attribute mutation."""
        ctx = ProjectContext.create(tmp_path)

        with pytest.raises(AttributeError):
            ctx.project_path = None  # type: ignore[misc]

    def test_create_resolves_relative_path(self, tmp_path: Path) -> None:
        """Create resolves relative paths."""
        ctx = ProjectContext.create(tmp_path / "." / "subdir" / "..")

        assert ctx.project_path == tmp_path
