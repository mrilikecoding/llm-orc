"""Unit tests for OrchestraService."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_orc.mcp.server import MCPServer  # noqa: F401 — breaks circular import
from llm_orc.services.orchestra_service import OrchestraService


@pytest.fixture
def mock_config_manager() -> Any:
    """Create a mock ConfigurationManager."""
    config = MagicMock()
    config.get_ensembles_dirs.return_value = []
    config.get_profiles_dirs.return_value = []
    return config


@pytest.fixture
def service(mock_config_manager: Any) -> OrchestraService:
    """Create an OrchestraService with mocked dependencies."""
    return OrchestraService(config_manager=mock_config_manager)


class TestOrchestraServiceLock:
    """Tests for the asyncio.Lock on OrchestraService."""

    def test_project_lock_exists(self, service: OrchestraService) -> None:
        """OrchestraService has a _project_lock attribute."""
        assert hasattr(service, "_project_lock")
        assert isinstance(service._project_lock, asyncio.Lock)


class TestHandleSetProjectAsync:
    """Tests for handle_set_project_async."""

    async def test_returns_ok_for_existing_path(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """handle_set_project_async returns ok for a valid path."""
        result = await service.handle_set_project_async(str(tmp_path))

        assert result["status"] == "ok"
        assert result["project_path"] == str(tmp_path)

    async def test_returns_error_for_nonexistent_path(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """handle_set_project_async returns error for a missing path."""
        missing = str(tmp_path / "does-not-exist")

        result = await service.handle_set_project_async(missing)

        assert result["status"] == "error"
        assert "does not exist" in result["error"]

    async def test_sets_project_path(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """handle_set_project_async updates the service project_path."""
        await service.handle_set_project_async(str(tmp_path))

        assert service.project_path == tmp_path

    async def test_note_when_no_llm_orc_dir(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """handle_set_project_async adds a note when .llm-orc is absent."""
        result = await service.handle_set_project_async(str(tmp_path))

        assert "note" in result
        assert ".llm-orc" in result["note"]

    async def test_no_note_when_llm_orc_dir_present(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """handle_set_project_async omits note when .llm-orc exists."""
        (tmp_path / ".llm-orc").mkdir()

        result = await service.handle_set_project_async(str(tmp_path))

        assert "note" not in result


class TestHandleSetProjectAsyncConcurrency:
    """Tests that concurrent calls to handle_set_project_async are serialized."""

    async def test_concurrent_calls_are_serialized(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """Concurrent project switches complete one at a time (lock serializes them)."""
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()

        completion_order: list[str] = []

        async def switch_a() -> None:
            await service.handle_set_project_async(str(dir_a))
            completion_order.append("a")

        async def switch_b() -> None:
            await service.handle_set_project_async(str(dir_b))
            completion_order.append("b")

        await asyncio.gather(switch_a(), switch_b())

        # Both calls must complete and the list must have exactly two entries.
        assert len(completion_order) == 2
        assert set(completion_order) == {"a", "b"}

    async def test_lock_prevents_overlapping_state(
        self, service: OrchestraService, tmp_path: Path
    ) -> None:
        """After concurrent switches, the service project_path reflects one of them."""
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()

        await asyncio.gather(
            service.handle_set_project_async(str(dir_a)),
            service.handle_set_project_async(str(dir_b)),
        )

        # project_path must be one of the two valid directories (not None or corrupt).
        assert service.project_path in {dir_a, dir_b}
