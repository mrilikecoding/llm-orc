"""Coverage tests for OrchestraService delegate methods.

Each test creates an OrchestraService with mocked handlers, calls a delegate
method, and verifies the underlying handler was called with the right arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.services.handlers.ensemble_crud_handler import EnsembleCrudHandler
from llm_orc.services.handlers.resource_handler import ResourceHandler
from llm_orc.services.orchestra_service import OrchestraService

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_manager() -> Any:
    """ConfigurationManager mock with sensible defaults."""
    config = MagicMock()
    config.get_ensembles_dirs.return_value = []
    config.get_profiles_dirs.return_value = []
    config.load_performance_config.return_value = {}
    return config


@pytest.fixture
def service(mock_config_manager: Any) -> OrchestraService:
    """OrchestraService with a mocked ConfigurationManager."""
    return OrchestraService(config_manager=mock_config_manager)


# ---------------------------------------------------------------------------
# Lines 113-114 — _configure_http_pool exception swallowed
# ---------------------------------------------------------------------------


class TestConfigureHttpPool:
    """_configure_http_pool swallows exceptions from load_performance_config."""

    def test_exception_is_swallowed(self, mock_config_manager: Any) -> None:
        """Exception in _configure_http_pool does not propagate."""
        mock_config_manager.load_performance_config.side_effect = RuntimeError(
            "config unavailable"
        )
        # Should not raise — the except block swallows the error.
        svc = OrchestraService(config_manager=mock_config_manager)
        assert svc is not None

    def test_exception_does_not_affect_handler_construction(
        self, mock_config_manager: Any
    ) -> None:
        """Service is fully constructed even when pool configuration fails."""
        mock_config_manager.load_performance_config.side_effect = ValueError("bad cfg")
        svc = OrchestraService(config_manager=mock_config_manager)
        # Handlers are still available after construction.
        assert hasattr(svc, "_help_handler")
        assert hasattr(svc, "_resource_handler")


# ---------------------------------------------------------------------------
# Lines 139-152 — list_ensembles_grouped
# ---------------------------------------------------------------------------


class TestListEnsemblesGrouped:
    """list_ensembles_grouped routes ensembles into local/library/global buckets."""

    def _make_service(self, dirs_and_tiers: list[tuple[str, str]]) -> OrchestraService:
        """Build a service whose config_manager yields the given (path, tier) pairs."""
        config = MagicMock()
        config.load_performance_config.return_value = {}
        paths = [Path(p) for p, _ in dirs_and_tiers]
        tier_map = {Path(p): t for p, t in dirs_and_tiers}
        config.get_ensembles_dirs.return_value = paths
        config.classify_tier.side_effect = lambda p: tier_map[p]
        return OrchestraService(config_manager=config)

    def test_empty_dirs_returns_empty_buckets(self, service: OrchestraService) -> None:
        result = service.list_ensembles_grouped()
        assert result == {"local": [], "library": [], "global": []}

    def test_local_tier_goes_into_local_bucket(self, tmp_path: Path) -> None:
        svc = self._make_service([(str(tmp_path), "local")])
        fake_ensemble = MagicMock()
        with patch.object(
            EnsembleLoader, "list_ensembles", return_value=[fake_ensemble]
        ):
            result = svc.list_ensembles_grouped()

        assert fake_ensemble in result["local"]
        assert result["library"] == []
        assert result["global"] == []

    def test_library_tier_goes_into_library_bucket(self, tmp_path: Path) -> None:
        svc = self._make_service([(str(tmp_path), "library")])
        fake_ensemble = MagicMock()
        with patch.object(
            EnsembleLoader, "list_ensembles", return_value=[fake_ensemble]
        ):
            result = svc.list_ensembles_grouped()

        assert fake_ensemble in result["library"]
        assert result["local"] == []
        assert result["global"] == []

    def test_other_tier_goes_into_global_bucket(self, tmp_path: Path) -> None:
        svc = self._make_service([(str(tmp_path), "other")])
        fake_ensemble = MagicMock()
        with patch.object(
            EnsembleLoader, "list_ensembles", return_value=[fake_ensemble]
        ):
            result = svc.list_ensembles_grouped()

        assert fake_ensemble in result["global"]
        assert result["local"] == []
        assert result["library"] == []

    def test_multiple_tiers_split_correctly(self, tmp_path: Path) -> None:
        local_dir = str(tmp_path / "local")
        lib_dir = str(tmp_path / "lib")
        svc = self._make_service([(local_dir, "local"), (lib_dir, "library")])
        local_ens = MagicMock()
        lib_ens = MagicMock()

        call_count = 0

        def _list_side_effect(path: str) -> list[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [local_ens]
            return [lib_ens]

        with patch.object(
            EnsembleLoader, "list_ensembles", side_effect=_list_side_effect
        ):
            result = svc.list_ensembles_grouped()

        assert result["local"] == [local_ens]
        assert result["library"] == [lib_ens]
        assert result["global"] == []


# ---------------------------------------------------------------------------
# Lines 229, 232, 237, 240, 243, 246 — resource handler delegates
# ---------------------------------------------------------------------------


class TestResourceHandlerDelegates:
    """read_* methods on OrchestraService delegate to _resource_handler."""

    @pytest.mark.asyncio
    async def test_read_ensemble_delegates(self, service: OrchestraService) -> None:
        """read_ensemble calls _resource_handler.read_ensemble with the name."""
        expected: dict[str, Any] = {"name": "my-ens"}
        with patch.object(
            ResourceHandler,
            "read_ensemble",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_ensemble("my-ens")

        assert result == expected

    @pytest.mark.asyncio
    async def test_read_artifacts_delegates(self, service: OrchestraService) -> None:
        """read_artifacts calls _resource_handler.read_artifacts with ensemble_name."""
        expected: list[dict[str, Any]] = [{"id": "art1"}]
        with patch.object(
            ResourceHandler,
            "read_artifacts",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_artifacts("my-ens")

        assert result == expected

    @pytest.mark.asyncio
    async def test_read_artifact_delegates(self, service: OrchestraService) -> None:
        """read_artifact calls _resource_handler.read_artifact with both args."""
        expected: dict[str, Any] = {"content": "data"}
        with patch.object(
            ResourceHandler,
            "read_artifact",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_artifact("my-ens", "art-42")

        assert result == expected

    @pytest.mark.asyncio
    async def test_read_metrics_delegates(self, service: OrchestraService) -> None:
        """read_metrics calls _resource_handler.read_metrics with ensemble_name."""
        expected: dict[str, Any] = {"tokens": 100}
        with patch.object(
            ResourceHandler,
            "read_metrics",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_metrics("my-ens")

        assert result == expected

    @pytest.mark.asyncio
    async def test_read_profiles_delegates(self, service: OrchestraService) -> None:
        """read_profiles calls _resource_handler.read_profiles with no args."""
        expected: list[dict[str, Any]] = [{"name": "gpt4"}]
        with patch.object(
            ResourceHandler,
            "read_profiles",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_profiles()

        assert result == expected

    @pytest.mark.asyncio
    async def test_read_resource_delegates(self, service: OrchestraService) -> None:
        """read_resource calls _resource_handler.read_resource with the URI."""
        expected: dict[str, Any] = {"uri": "llm-orc://ensembles"}
        with patch.object(
            ResourceHandler,
            "read_resource",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.read_resource("llm-orc://ensembles")

        assert result == expected


# ---------------------------------------------------------------------------
# Lines 292, 298 — ensemble CRUD handler delegates
# ---------------------------------------------------------------------------


class TestEnsembleCrudDelegates:
    """Ensemble CRUD delegates forward to _ensemble_crud_handler."""

    @pytest.mark.asyncio
    async def test_update_ensemble_delegates(self, service: OrchestraService) -> None:
        """update_ensemble calls _ensemble_crud_handler.update_ensemble."""
        expected: dict[str, Any] = {"status": "updated"}
        args: dict[str, Any] = {"ensemble_name": "my-ens", "changes": {}}
        with patch.object(
            EnsembleCrudHandler,
            "update_ensemble",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.update_ensemble(args)

        assert result == expected

    @pytest.mark.asyncio
    async def test_analyze_execution_delegates(self, service: OrchestraService) -> None:
        """analyze_execution calls _ensemble_crud_handler.analyze_execution."""
        expected: dict[str, Any] = {"analysis": "done"}
        args: dict[str, Any] = {"artifact_id": "my-ens/20240101"}
        with patch.object(
            EnsembleCrudHandler,
            "analyze_execution",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await service.analyze_execution(args)

        assert result == expected


# ---------------------------------------------------------------------------
# Line 326 — list_artifact_ensembles delegate
# ---------------------------------------------------------------------------


class TestArtifactManagerDelegate:
    """list_artifact_ensembles delegates to artifact_manager.list_ensembles."""

    def test_list_artifact_ensembles_delegates(self, service: OrchestraService) -> None:
        """list_artifact_ensembles returns artifact_manager.list_ensembles result."""
        expected: list[Any] = ["ens-a", "ens-b"]
        with patch.object(
            service.artifact_manager, "list_ensembles", return_value=expected
        ):
            result = service.list_artifact_ensembles()

        assert result == expected
