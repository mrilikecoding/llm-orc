"""Unit tests for ExecutionHandler.invoke status normalization."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_orc.mcp.handlers.execution_handler import ExecutionHandler


def _make_handler(ensemble_config: Any = None) -> ExecutionHandler:
    config_manager = MagicMock()
    config_manager.get_ensembles_dirs.return_value = ["/fake/ensembles"]

    ensemble_loader = MagicMock()
    ensemble_loader.find_ensemble.return_value = ensemble_config

    artifact_manager = MagicMock()

    return ExecutionHandler(
        config_manager=config_manager,
        ensemble_loader=ensemble_loader,
        artifact_manager=artifact_manager,
        get_executor_fn=MagicMock(),
        find_ensemble_fn=lambda name: ensemble_config,
    )


def _fake_ensemble(name: str = "test") -> Any:
    config = MagicMock()
    config.name = name
    config.agents = []
    return config


class TestInvokeStatusNormalization:
    """invoke translates internal status values to the API contract."""

    @pytest.mark.asyncio
    async def test_completed_maps_to_success(self) -> None:
        handler = _make_handler(_fake_ensemble())
        with patch(
            "llm_orc.core.execution.ensemble_execution.EnsembleExecutor"
        ) as mock_executor:
            mock_executor.return_value.execute = AsyncMock(
                return_value={"status": "completed", "results": {}, "synthesis": None}
            )

            result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_completed_with_errors_maps_to_error(self) -> None:
        handler = _make_handler(_fake_ensemble())
        with patch(
            "llm_orc.core.execution.ensemble_execution.EnsembleExecutor"
        ) as mock_executor:
            mock_executor.return_value.execute = AsyncMock(
                return_value={
                    "status": "completed_with_errors",
                    "results": {},
                    "synthesis": None,
                }
            )

            result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_status_passes_through(self) -> None:
        handler = _make_handler(_fake_ensemble())
        with patch(
            "llm_orc.core.execution.ensemble_execution.EnsembleExecutor"
        ) as mock_executor:
            mock_executor.return_value.execute = AsyncMock(
                return_value={
                    "status": "running",
                    "results": {},
                    "synthesis": None,
                }
            )

            result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "running"
