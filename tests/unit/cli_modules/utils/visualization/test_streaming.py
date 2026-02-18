"""Comprehensive tests for streaming execution module."""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.cli_modules.utils.visualization.streaming import (
    _run_text_json_execution,
    run_standard_execution,
    run_streaming_execution,
)


class TestRunStreamingExecution:
    """Test streaming execution functions."""

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming._run_text_json_execution")
    async def test_run_streaming_execution_json_format(
        self, mock_json_execution: Mock
    ) -> None:
        """Test running streaming execution with JSON output format."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        await run_streaming_execution(executor, ensemble_config, input_data, "json")

        mock_json_execution.assert_called_once_with(
            executor, ensemble_config, input_data, "json", True
        )

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming.Console")
    async def test_run_streaming_execution_rich_format(
        self, mock_console_class: Mock
    ) -> None:
        """Test running streaming execution with rich output format."""
        executor = AsyncMock()

        # Create async generator for streaming events
        async def mock_execute_streaming(config: Any, input_data: str) -> Any:
            yield {"type": "execution_started"}
            yield {
                "type": "execution_completed",
                "data": {"results": {}, "metadata": {}},
            }

        executor.execute_streaming = mock_execute_streaming

        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Setup status context manager properly
        mock_status = Mock()
        mock_console.status.return_value = mock_status
        mock_status.__enter__ = Mock(return_value=mock_status)
        mock_status.__exit__ = Mock(return_value=None)

        await run_streaming_execution(executor, ensemble_config, input_data, "rich")

        # Verify console status was used
        mock_console.status.assert_called_once()

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming.Console")
    async def test_run_streaming_execution_default_format(
        self, mock_console_class: Mock
    ) -> None:
        """Test running streaming execution with default format."""
        executor = AsyncMock()

        # Create async generator for streaming events
        async def mock_execute_streaming(config: Any, input_data: str) -> Any:
            yield {
                "type": "execution_completed",
                "data": {"results": {}, "metadata": {}},
            }

        executor.execute_streaming = mock_execute_streaming

        ensemble_config = Mock()
        ensemble_config.agents = []
        input_data = "Test input"

        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Setup status context manager properly
        mock_status = Mock()
        mock_console.status.return_value = mock_status
        mock_status.__enter__ = Mock(return_value=mock_status)
        mock_status.__exit__ = Mock(return_value=None)

        await run_streaming_execution(executor, ensemble_config, input_data)

        # Verify console status was used for default rich format
        mock_console.status.assert_called_once()


class TestRunStandardExecution:
    """Test standard execution functions."""

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming._display_json_results")
    async def test_run_standard_execution_json_format(
        self, mock_json_display: Mock
    ) -> None:
        """Test running standard execution with JSON output."""
        executor = AsyncMock()
        result = {
            "results": {"agent_a": {"status": "success"}},
            "metadata": {"duration": "5s"},
        }
        executor.execute = AsyncMock(return_value=result)
        ensemble_config = Mock()
        input_data = "Test input"

        await run_standard_execution(executor, ensemble_config, input_data, "json")

        executor.execute.assert_called_once_with(ensemble_config, input_data)
        mock_json_display.assert_called_once_with(result, ensemble_config)

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming.display_results")
    async def test_run_standard_execution_rich_format(self, mock_display: Mock) -> None:
        """Test running standard execution with rich output."""
        executor = AsyncMock()
        result = {
            "results": {"agent_a": {"status": "success"}},
            "metadata": {"duration": "5s"},
        }
        executor.execute = AsyncMock(return_value=result)
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        await run_standard_execution(executor, ensemble_config, input_data, "rich")

        executor.execute.assert_called_once_with(ensemble_config, input_data)
        mock_display.assert_called_once_with(
            {"agent_a": {"status": "success"}},
            {"duration": "5s"},
            [{"name": "agent_a"}],
            detailed=True,
        )


class TestRunTextJsonExecution:
    """Test text JSON execution."""

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming.click.echo")
    async def test_run_text_json_execution_success(self, mock_echo: Mock) -> None:
        """Test successful text JSON execution."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        # Mock streaming events for JSON output
        async def mock_execute_streaming(config: Any, data: str) -> Any:
            yield {"type": "agent_started", "agent_name": "agent_a"}
            yield {"type": "agent_completed", "agent_name": "agent_a"}

        executor.execute_streaming = mock_execute_streaming

        await _run_text_json_execution(
            executor, ensemble_config, input_data, "json", True
        )

        # Should output each event as JSON
        assert mock_echo.call_count == 2

        # Check first event JSON output
        first_call_args = mock_echo.call_args_list[0][0][0]
        first_event = json.loads(first_call_args)
        assert first_event["type"] == "agent_started"
        assert first_event["agent_name"] == "agent_a"

    @pytest.mark.asyncio
    @patch("llm_orc.cli_modules.utils.visualization.streaming.click.echo")
    async def test_run_text_json_execution_error(self, mock_echo: Mock) -> None:
        """Test text JSON execution with error."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        # Mock execute_streaming to raise an error
        async def mock_execute_streaming(config: Any, data: str) -> Any:
            raise Exception("Test error")
            yield  # This will never be reached but needed for async generator

        executor.execute_streaming = mock_execute_streaming

        await _run_text_json_execution(
            executor, ensemble_config, input_data, "json", True
        )

        mock_echo.assert_called_once()

        # Check that error JSON was output
        call_args = mock_echo.call_args[0][0]
        output_data = json.loads(call_args)
        assert "error" in output_data
        assert output_data["error"] == "Test error"


class TestComplexStreamingScenarios:
    """Test complex streaming execution scenarios."""

    @pytest.mark.asyncio
    async def test_json_streaming_execution_success(self) -> None:
        """Test JSON streaming execution with successful events."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        # Create async generator for events
        async def mock_execute_streaming(config: Any, data: str) -> Any:
            yield {"type": "agent_started", "agent_name": "agent_a"}
            yield {
                "type": "agent_completed",
                "agent_name": "agent_a",
                "result": "Success",
            }

        executor.execute_streaming = mock_execute_streaming

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming.click.echo"
        ) as mock_echo:
            await _run_text_json_execution(
                executor, ensemble_config, input_data, "json", True
            )

            # Should output each event as JSON
            assert mock_echo.call_count == 2

            # Check first event
            first_call = mock_echo.call_args_list[0][0][0]
            first_event = json.loads(first_call)
            assert first_event["type"] == "agent_started"
            assert first_event["agent_name"] == "agent_a"

    @pytest.mark.asyncio
    async def test_json_streaming_execution_error(self) -> None:
        """Test JSON streaming execution with error."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"

        async def mock_execute_streaming(config: Any, data: str) -> Any:
            raise Exception("Streaming error")
            yield  # This will never be reached, but needed for async generator

        executor.execute_streaming = mock_execute_streaming

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming.click.echo"
        ) as mock_echo:
            await _run_text_json_execution(
                executor, ensemble_config, input_data, "json", True
            )

            # Should output error event as JSON
            mock_echo.assert_called_once()
            error_call = mock_echo.call_args[0][0]
            error_event = json.loads(error_call)
            assert error_event["type"] == "error"
            assert error_event["error"] == "Streaming error"

    @pytest.mark.asyncio
    async def test_rich_streaming_execution_complete_flow(self) -> None:
        """Test rich streaming execution with complete event flow."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        input_data = "Test input"

        # Create async generator for complete execution flow
        async def mock_execute_streaming(config: Any, data: str) -> Any:
            yield {"type": "agent_started", "data": {"agent_name": "agent_a"}}
            yield {"type": "agent_completed", "data": {"agent_name": "agent_a"}}
            yield {"type": "agent_started", "data": {"agent_name": "agent_b"}}
            yield {"type": "agent_completed", "data": {"agent_name": "agent_b"}}
            yield {
                "type": "execution_completed",
                "data": {
                    "results": {
                        "agent_a": {"status": "success"},
                        "agent_b": {"status": "success"},
                    },
                    "metadata": {"duration": "5s"},
                },
            }

        executor.execute_streaming = mock_execute_streaming

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming.Console"
        ) as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            # Setup status context manager properly
            mock_status = Mock()
            mock_console.status.return_value = mock_status
            mock_status.__enter__ = Mock(return_value=mock_status)
            mock_status.__exit__ = Mock(return_value=None)

            await run_streaming_execution(executor, ensemble_config, input_data, "rich")

            # Should have updated the status display multiple times
            # Once for each event that updates status
            assert mock_status.update.call_count >= 4

    @pytest.mark.asyncio
    async def test_rich_streaming_execution_with_error(self) -> None:
        """Test rich streaming execution with error during execution."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        input_data = "Test input"

        async def mock_execute_streaming(config: Any, data: str) -> Any:
            yield {"type": "agent_started", "data": {"agent_name": "agent_a"}}
            raise Exception("Execution error")

        executor.execute_streaming = mock_execute_streaming

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming.Console"
        ) as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            # Setup status context manager properly
            mock_status = Mock()
            mock_console.status.return_value = mock_status
            mock_status.__enter__ = Mock(return_value=mock_status)
            mock_status.__exit__ = Mock(return_value=None)

            # This should not raise an exception - errors are caught and handled
            try:
                await run_streaming_execution(
                    executor, ensemble_config, input_data, "rich"
                )
            except Exception:
                pass  # Expected since we're mocking an error

            # Should have at least tried to create the status display
            mock_console.status.assert_called_once()


class TestNewHelperFunctions:
    """Test the new helper functions created during refactoring."""

    def test_handle_agent_progress_event(self) -> None:
        """Test _handle_agent_progress_event function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_agent_progress_event,
        )

        event = {
            "data": {
                "started_agent_names": ["agent_a"],
                "completed_agent_names": ["agent_b"],
            }
        }
        agent_statuses = {"agent_a": "pending", "agent_b": "running"}
        ensemble_config = Mock()
        ensemble_config.agents = [
            {"name": "agent_a"},
            {"name": "agent_b"},
        ]

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming"
            "._update_agent_status_by_names_from_lists"
        ) as mock_update:
            result = _handle_agent_progress_event(
                event, agent_statuses, ensemble_config
            )

            mock_update.assert_called_once()
            # Status should change if mock_update modifies agent_statuses
            assert isinstance(result, bool)

    def test_handle_agent_started_event(self) -> None:
        """Test _handle_agent_started_event function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_agent_started_event,
        )

        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses: dict[str, str] = {}

        result = _handle_agent_started_event(event, agent_statuses)

        assert result is True
        assert agent_statuses["agent_a"] == "running"

    def test_handle_agent_started_event_no_change(self) -> None:
        """Test _handle_agent_started_event when status doesn't change."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_agent_started_event,
        )

        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses = {"agent_a": "running"}

        result = _handle_agent_started_event(event, agent_statuses)

        assert result is False
        assert agent_statuses["agent_a"] == "running"

    def test_handle_agent_completed_event(self) -> None:
        """Test _handle_agent_completed_event function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_agent_completed_event,
        )

        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses: dict[str, str] = {}

        result = _handle_agent_completed_event(event, agent_statuses)

        assert result is True
        assert agent_statuses["agent_a"] == "completed"

    def test_handle_agent_failed_event(self) -> None:
        """Test _handle_agent_failed_event function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_agent_failed_event,
        )

        event = {"data": {"agent_name": "agent_a"}}
        agent_statuses: dict[str, str] = {}

        result = _handle_agent_failed_event(event, agent_statuses)

        assert result is True
        assert agent_statuses["agent_a"] == "failed"

    def test_handle_execution_completed_event(self) -> None:
        """Test _handle_execution_completed_event function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_execution_completed_event,
        )

        event = {
            "data": {
                "results": {
                    "agent_a": {
                        "status": "success",
                        "response": "Hello",
                    }
                },
                "metadata": {"duration": "5s"},
            }
        }
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        status = Mock()
        console = Mock()

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming._display_simple_results"
        ) as mock_simple:
            result = _handle_execution_completed_event(
                event, ensemble_config, status, console, detailed=False
            )

        assert result is False  # Should break event loop
        status.stop.assert_called_once()
        console.print.assert_called_with("")
        # When detailed=False, simple results should be displayed
        mock_simple.assert_called_once()

    def test_handle_execution_completed_event_detailed(self) -> None:
        """Test _handle_execution_completed_event with detailed=True."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _handle_execution_completed_event,
        )

        event = {
            "data": {
                "results": {"agent_a": {"status": "success"}},
                "metadata": {"duration": "5s"},
            }
        }
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        status = Mock()
        console = Mock()

        with patch(
            "llm_orc.cli_modules.utils.visualization.streaming"
            "._display_detailed_execution_results"
        ) as mock_display:
            result = _handle_execution_completed_event(
                event, ensemble_config, status, console, detailed=True
            )

            assert result is False
            status.stop.assert_called_once()
            console.print.assert_called_with("")
            mock_display.assert_called_once()

    def test_display_detailed_execution_results(self) -> None:
        """Test _display_detailed_execution_results function."""
        from llm_orc.cli_modules.utils.visualization.streaming import (
            _display_detailed_execution_results,
        )

        results = {"agent_a": {"status": "success"}}
        metadata = {"duration": "5s"}
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        results_console = Mock()

        with (
            patch(
                "llm_orc.cli_modules.utils.visualization.streaming.create_dependency_tree"
            ) as mock_tree,
            patch(
                "llm_orc.cli_modules.utils.visualization.results_display._process_agent_results"
            ) as mock_process,
            patch(
                "llm_orc.cli_modules.utils.visualization.results_display._display_agent_result"
            ) as mock_display_result,
            patch(
                "llm_orc.cli_modules.utils.visualization.results_display._format_performance_metrics"
            ) as mock_format_perf,
        ):
            mock_process.return_value = {
                "agent_a": {"status": "success", "response": "Hello"}
            }
            mock_format_perf.return_value = ["Performance: Good"]

            _display_detailed_execution_results(
                results, metadata, ensemble_config, results_console
            )

            mock_tree.assert_called_once()
            mock_process.assert_called_once_with(results)
            mock_display_result.assert_called_once()
            mock_format_perf.assert_called_once_with(metadata)
            results_console.print.assert_called()
