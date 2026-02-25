"""Unit tests for ExecutionHandler.invoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_orc.services.handlers.execution_handler import ExecutionHandler


def _make_handler(
    ensemble_config: Any = None,
    executor_execute_return: dict[str, Any] | None = None,
) -> ExecutionHandler:
    config_manager = MagicMock()
    config_manager.get_ensembles_dirs.return_value = ["/fake/ensembles"]

    ensemble_loader = MagicMock()
    ensemble_loader.find_ensemble.return_value = ensemble_config

    artifact_manager = MagicMock()

    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=executor_execute_return or {})

    return ExecutionHandler(
        config_manager=config_manager,
        ensemble_loader=ensemble_loader,
        artifact_manager=artifact_manager,
        get_executor_fn=lambda: mock_executor,
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
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {},
                "synthesis": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_completed_with_errors_maps_to_error(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed_with_errors",
                "results": {},
                "synthesis": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_status_passes_through(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "running",
                "results": {},
                "synthesis": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "running"


class TestInvokeInputFile:
    """invoke reads input_file when input is empty."""

    @pytest.mark.asyncio
    async def test_input_file_reads_content(self, tmp_path: Path) -> None:
        """input_file contents used when input is empty."""
        input_file = tmp_path / "data.txt"
        input_file.write_text("file content here")

        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {},
                "synthesis": None,
            },
        )
        await handler.invoke(
            {
                "ensemble_name": "test",
                "input": "",
                "input_file": str(input_file),
            }
        )

        # The executor should receive the file contents
        mock_exec: Any = handler._get_executor()
        mock_exec.execute.assert_called_once()
        call_args = mock_exec.execute.call_args
        assert call_args[0][1] == "file content here"

    @pytest.mark.asyncio
    async def test_input_takes_priority_over_input_file(self, tmp_path: Path) -> None:
        """Inline input takes priority over input_file."""
        input_file = tmp_path / "data.txt"
        input_file.write_text("file content")

        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {},
                "synthesis": None,
            },
        )
        await handler.invoke(
            {
                "ensemble_name": "test",
                "input": "inline content",
                "input_file": str(input_file),
            }
        )

        mock_exec: Any = handler._get_executor()
        call_args = mock_exec.execute.call_args
        assert call_args[0][1] == "inline content"

    @pytest.mark.asyncio
    async def test_input_file_not_found_raises(self) -> None:
        """Nonexistent input_file raises FileNotFoundError."""
        handler = _make_handler(_fake_ensemble())
        with pytest.raises(FileNotFoundError):
            await handler.invoke(
                {
                    "ensemble_name": "test",
                    "input": "",
                    "input_file": "/no/such/file.txt",
                }
            )
