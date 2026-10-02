"""Unit tests for ExecutionHandler.invoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_orc.core.config.closure import Closure
from llm_orc.services.handlers.execution_handler import ExecutionHandler
from llm_orc.services.handlers.provider_handler import Preflight


async def _runnable(*args: Any, **kwargs: Any) -> Preflight:
    """A gate that finds nothing unmet: these tests are about the result
    projection, the gate has its own pins."""
    return Preflight(closure=Closure([], {}), reports=[], providers={})


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
        preflight_fn=_runnable,
    )


def _fake_ensemble(name: str = "test") -> Any:
    config = MagicMock()
    config.name = name
    config.agents = []
    return config


class TestInvokeStatusNormalization:
    """invoke translates internal status values to the caller contract's
    "success"/"error" vocabulary plus has_errors (fail-closed-
    composition, caller contract) — the same mapping every surface
    (REST, MCP invoke, CLI JSON) uses, via caller_status."""

    @pytest.mark.asyncio
    async def test_completed_maps_to_success(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {},
                "deliverable": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "success"
        assert result["has_errors"] is False

    @pytest.mark.asyncio
    async def test_completed_with_errors_maps_to_error(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed_with_errors",
                "results": {},
                "deliverable": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "error"
        assert result["has_errors"] is True

    @pytest.mark.asyncio
    async def test_unrecognized_status_is_error_not_passed_through(self) -> None:
        """No caller ever sees a raw internal status string — only
        "success" or "error"."""
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "running",
                "results": {},
                "deliverable": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["status"] == "error"
        assert result["has_errors"] is True


class TestInvokeDeliverablePassthrough:
    """invoke carries the executor's deliverable contract (ADR-035 D1)."""

    @pytest.mark.asyncio
    async def test_deliverable_rides_the_projection(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {"synthesizer": {"status": "success", "response": "code"}},
                "deliverable": "code",
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["deliverable"] == "code"

    @pytest.mark.asyncio
    async def test_absent_deliverable_projects_none(self) -> None:
        handler = _make_handler(
            _fake_ensemble(),
            executor_execute_return={
                "status": "completed",
                "results": {},
                "deliverable": None,
            },
        )

        result = await handler.invoke({"ensemble_name": "test", "input": "hello"})

        assert result["deliverable"] is None


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
                "deliverable": None,
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
                "deliverable": None,
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


class _FakeReporter:
    """A minimal ProgressReporter, standing in for FastMCP's Context."""

    async def info(self, message: str) -> None:
        return None

    async def warning(self, message: str) -> None:
        return None

    async def error(self, message: str) -> None:
        return None

    async def report_progress(self, progress: int, total: int) -> None:
        return None


def _real_service(tmp_path: Path, agents: list[dict[str, Any]]) -> Any:
    """A real OrchestraService (real ConfigurationManager, real executor)
    wired to a temp project with one script-based ensemble — the actual
    code path the MCP ``invoke`` tool drives via ``_invoke_tool_with_
    streaming`` -> ``execute_streaming`` (fail-closed-composition,
    Doctrine 11: no mocked executor)."""
    import yaml

    from llm_orc.core.config.config_manager import ConfigurationManager
    from llm_orc.services.orchestra_service import OrchestraService

    ensembles_dir = tmp_path / ".llm-orc" / "ensembles"
    ensembles_dir.mkdir(parents=True)
    (ensembles_dir / "mcp-pin.yaml").write_text(
        yaml.dump(
            {"name": "mcp-pin", "description": "MCP outcome pin", "agents": agents}
        )
    )
    config_manager = ConfigurationManager(project_dir=tmp_path, provision=False)
    return OrchestraService(config_manager=config_manager)


class TestExecuteStreamingCallerContract:
    """Outcome pins for the caller contract (fail-closed-composition,
    Doctrine 11): the MCP ``invoke`` tool's real code path
    (``execute_streaming``, not the non-streaming ``invoke`` method) —
    a real executor running real script agents, not a mocked one."""

    @pytest.mark.asyncio
    async def test_clean_run_reports_success(self, tmp_path: Path) -> None:
        service = _real_service(
            tmp_path, [{"name": "answer", "script": "echo '{\"ok\": true}'"}]
        )

        result = await service.execute_streaming("mcp-pin", "hello", _FakeReporter())

        assert result["status"] == "success"
        assert result["has_errors"] is False
        assert result["deliverable"] is not None

    @pytest.mark.asyncio
    async def test_failed_terminal_reports_error(self, tmp_path: Path) -> None:
        service = _real_service(tmp_path, [{"name": "answer", "script": "exit 1"}])

        result = await service.execute_streaming("mcp-pin", "hello", _FakeReporter())

        assert result["status"] == "error"
        assert result["has_errors"] is True
        assert result["deliverable"] is None
