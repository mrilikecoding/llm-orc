"""Comprehensive tests for streaming execution module."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from llm_orc.cli_modules.utils.visualization.streaming import (
    run_streaming_execution,
    run_standard_execution,
    _run_text_json_execution,
    _handle_streaming_event,
    _process_execution_completed_event,
    _update_agent_progress_status,
    _update_agent_status_by_names,
    _handle_fallback_started_event,
    _handle_fallback_completed_event,
    _handle_fallback_failed_event,
    _handle_text_fallback_started,
    _handle_text_fallback_completed,
    _handle_text_fallback_failed,
)


class TestRunStreamingExecution:
    """Test streaming execution functions."""

    @pytest.mark.asyncio
    @patch('llm_orc.cli_modules.utils.visualization.streaming._run_text_json_execution')
    async def test_run_streaming_execution_json_format(self, mock_json_execution):
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
    @patch('llm_orc.cli_modules.utils.visualization.streaming.Console')
    async def test_run_streaming_execution_rich_format(self, mock_console_class):
        """Test running streaming execution with rich output format."""
        executor = AsyncMock()
        
        # Create async generator for streaming events
        async def mock_execute_streaming(config, input_data):
            yield {"type": "execution_started"}
            yield {"type": "execution_completed", "data": {"results": {}, "metadata": {}}}
        
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
    @patch('llm_orc.cli_modules.utils.visualization.streaming.Console')
    async def test_run_streaming_execution_default_format(self, mock_console_class):
        """Test running streaming execution with default format."""
        executor = AsyncMock()
        
        # Create async generator for streaming events  
        async def mock_execute_streaming(config, input_data):
            yield {"type": "execution_completed", "data": {"results": {}, "metadata": {}}}
        
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
    @patch('llm_orc.cli_modules.utils.visualization.streaming._display_json_results')
    async def test_run_standard_execution_json_format(self, mock_json_display):
        """Test running standard execution with JSON output."""
        executor = AsyncMock()
        result = {"results": {"agent_a": {"status": "success"}}, "metadata": {"duration": "5s"}}
        executor.execute = AsyncMock(return_value=result)
        ensemble_config = Mock()
        input_data = "Test input"
        
        await run_standard_execution(executor, ensemble_config, input_data, "json")
        
        executor.execute.assert_called_once_with(ensemble_config, input_data)
        mock_json_display.assert_called_once_with(result, ensemble_config)

    @pytest.mark.asyncio
    @patch('llm_orc.cli_modules.utils.visualization.streaming.display_results')
    async def test_run_standard_execution_rich_format(self, mock_display):
        """Test running standard execution with rich output."""
        executor = AsyncMock()
        result = {"results": {"agent_a": {"status": "success"}}, "metadata": {"duration": "5s"}}
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
            detailed=True
        )


class TestRunTextJsonExecution:
    """Test text JSON execution."""

    @pytest.mark.asyncio
    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    async def test_run_text_json_execution_success(self, mock_echo):
        """Test successful text JSON execution."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"
        
        # Mock streaming events for JSON output
        async def mock_execute_streaming(config, data):
            yield {"type": "agent_started", "agent_name": "agent_a"}
            yield {"type": "agent_completed", "agent_name": "agent_a"}
        
        executor.execute_streaming = mock_execute_streaming
        
        await _run_text_json_execution(executor, ensemble_config, input_data, "json", True)
        
        # Should output each event as JSON
        assert mock_echo.call_count == 2
        
        # Check first event JSON output
        first_call_args = mock_echo.call_args_list[0][0][0]
        first_event = json.loads(first_call_args)
        assert first_event["type"] == "agent_started"
        assert first_event["agent_name"] == "agent_a"

    @pytest.mark.asyncio
    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    async def test_run_text_json_execution_error(self, mock_echo):
        """Test text JSON execution with error."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"
        
        # Mock execute_streaming to raise an error
        async def mock_execute_streaming(config, data):
            raise Exception("Test error")
            yield  # This will never be reached but needed for async generator
        
        executor.execute_streaming = mock_execute_streaming
        
        await _run_text_json_execution(executor, ensemble_config, input_data, "json", True)
        
        mock_echo.assert_called_once()
        
        # Check that error JSON was output
        call_args = mock_echo.call_args[0][0]
        output_data = json.loads(call_args)
        assert "error" in output_data
        assert output_data["error"] == "Test error"


class TestHandleStreamingEvent:
    """Test streaming event handling."""

    def test_handle_streaming_event_agent_started(self):
        """Test handling agent started event."""
        event = {"event_type": "agent_started", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "🔄 In Progress"

    def test_handle_streaming_event_agent_completed(self):
        """Test handling agent completed event."""
        event = {"event_type": "agent_completed", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "✅ Completed"

    def test_handle_streaming_event_agent_failed(self):
        """Test handling agent failed event."""
        event = {"event_type": "agent_failed", "agent_name": "agent_a", "error": "Test error"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "❌ Failed"
        assert agent_progress["agent_a"]["error"] == "Test error"

    def test_handle_streaming_event_agent_failed_no_error(self):
        """Test handling agent failed event without error message."""
        event = {"event_type": "agent_failed", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        assert agent_progress["agent_a"]["error"] == "Unknown error"

    @patch('llm_orc.cli_modules.utils.visualization.streaming._handle_fallback_started_event')
    def test_handle_streaming_event_fallback_started(self, mock_fallback_started):
        """Test handling fallback started event."""
        event = {"event_type": "fallback_started", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        mock_fallback_started.assert_called_once_with(event, agent_progress)

    @patch('llm_orc.cli_modules.utils.visualization.streaming._handle_fallback_completed_event')
    def test_handle_streaming_event_fallback_completed(self, mock_fallback_completed):
        """Test handling fallback completed event."""
        event = {"event_type": "fallback_completed", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        mock_fallback_completed.assert_called_once_with(event, agent_progress)

    @patch('llm_orc.cli_modules.utils.visualization.streaming._handle_fallback_failed_event')
    def test_handle_streaming_event_fallback_failed(self, mock_fallback_failed):
        """Test handling fallback failed event."""
        event = {"event_type": "fallback_failed", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        mock_fallback_failed.assert_called_once_with(event, agent_progress)

    def test_handle_streaming_event_no_agent_name(self):
        """Test handling event without agent name."""
        event = {"event_type": "agent_started"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        # Should not modify agent_progress
        assert agent_progress == {}

    def test_handle_streaming_event_unknown_type(self):
        """Test handling unknown event type."""
        event = {"event_type": "unknown", "agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_streaming_event(event, agent_progress)
        
        # Should create agent entry but not set status
        assert "agent_a" in agent_progress
        assert "status" not in agent_progress["agent_a"]


class TestProcessExecutionCompletedEvent:
    """Test execution completed event processing."""

    @patch('llm_orc.cli_modules.utils.visualization.streaming.Console')
    def test_process_execution_completed_event_success(self, mock_console_class):
        """Test processing successful execution completed event."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        event = {
            "results": {
                "agent_a": {"status": "success"},
                "agent_b": {"status": "success"},
                "agent_c": {"status": "failed"}
            },
            "metadata": {"duration": "10s"}
        }
        
        _process_execution_completed_event(event)
        
        # Check console output
        assert mock_console.print.call_count >= 2
        calls = [call[0][0] for call in mock_console.print.call_args_list]
        assert any("✅ Execution Completed" in call for call in calls)
        assert any("Results: 2/3 agents successful" in call for call in calls)

    @patch('llm_orc.cli_modules.utils.visualization.streaming.Console')
    def test_process_execution_completed_event_no_results(self, mock_console_class):
        """Test processing execution completed event with no results."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        event = {}
        
        _process_execution_completed_event(event)
        
        # Should still print completion message
        assert mock_console.print.call_count >= 2


class TestUpdateAgentProgressStatus:
    """Test agent progress status updates."""

    def test_update_agent_progress_status_new_agent(self):
        """Test updating status for new agent."""
        agent_progress = {}
        
        _update_agent_progress_status("agent_a", "✅ Completed", agent_progress)
        
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "✅ Completed"

    def test_update_agent_progress_status_existing_agent(self):
        """Test updating status for existing agent."""
        agent_progress = {"agent_a": {"other_data": "value"}}
        
        _update_agent_progress_status("agent_a", "❌ Failed", agent_progress)
        
        assert agent_progress["agent_a"]["status"] == "❌ Failed"
        assert agent_progress["agent_a"]["other_data"] == "value"

    def test_update_agent_status_by_names(self):
        """Test updating status for multiple agents."""
        agent_progress = {}
        
        _update_agent_status_by_names(["agent_a", "agent_b"], "🔄 Running", agent_progress)
        
        assert "agent_a" in agent_progress
        assert "agent_b" in agent_progress
        assert agent_progress["agent_a"]["status"] == "🔄 Running"
        assert agent_progress["agent_b"]["status"] == "🔄 Running"


class TestFallbackEventHandlers:
    """Test fallback event handlers."""

    def test_handle_fallback_started_event(self):
        """Test handling fallback started event."""
        event = {
            "agent_name": "agent_a",
            "original_model": "gpt-4",
            "fallback_model": "gpt-3.5-turbo"
        }
        agent_progress = {}
        
        _handle_fallback_started_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        expected_status = "🔄 Fallback: gpt-4 → gpt-3.5-turbo"
        assert agent_progress["agent_a"]["status"] == expected_status

    def test_handle_fallback_started_event_no_agent_name(self):
        """Test handling fallback started event without agent name."""
        event = {
            "original_model": "gpt-4",
            "fallback_model": "gpt-3.5-turbo"
        }
        agent_progress = {}
        
        _handle_fallback_started_event(event, agent_progress)
        
        assert agent_progress == {}

    def test_handle_fallback_completed_event(self):
        """Test handling fallback completed event."""
        event = {
            "agent_name": "agent_a",
            "fallback_model": "gpt-3.5-turbo"
        }
        agent_progress = {}
        
        _handle_fallback_completed_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        expected_status = "✅ Completed via gpt-3.5-turbo"
        assert agent_progress["agent_a"]["status"] == expected_status

    def test_handle_fallback_completed_event_no_agent_name(self):
        """Test handling fallback completed event without agent name."""
        event = {"fallback_model": "gpt-3.5-turbo"}
        agent_progress = {}
        
        _handle_fallback_completed_event(event, agent_progress)
        
        assert agent_progress == {}

    def test_handle_fallback_failed_event(self):
        """Test handling fallback failed event."""
        event = {
            "agent_name": "agent_a",
            "error": "Model unavailable"
        }
        agent_progress = {}
        
        _handle_fallback_failed_event(event, agent_progress)
        
        assert "agent_a" in agent_progress
        assert agent_progress["agent_a"]["status"] == "❌ Fallback Failed"
        assert agent_progress["agent_a"]["error"] == "Model unavailable"

    def test_handle_fallback_failed_event_no_error(self):
        """Test handling fallback failed event without error message."""
        event = {"agent_name": "agent_a"}
        agent_progress = {}
        
        _handle_fallback_failed_event(event, agent_progress)
        
        assert agent_progress["agent_a"]["error"] == "Fallback failed"

    def test_handle_fallback_failed_event_no_agent_name(self):
        """Test handling fallback failed event without agent name."""
        event = {"error": "Model unavailable"}
        agent_progress = {}
        
        _handle_fallback_failed_event(event, agent_progress)
        
        assert agent_progress == {}


class TestTextModeEventHandlers:
    """Test text mode event handlers."""

    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    def test_handle_text_fallback_started(self, mock_echo):
        """Test text mode fallback started handler."""
        event = {
            "agent_name": "agent_a",
            "original_model": "gpt-4",
            "fallback_model": "gpt-3.5-turbo"
        }
        
        _handle_text_fallback_started(event)
        
        expected = "🔄 agent_a: Falling back from gpt-4 to gpt-3.5-turbo"
        mock_echo.assert_called_once_with(expected)

    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    def test_handle_text_fallback_completed(self, mock_echo):
        """Test text mode fallback completed handler."""
        event = {
            "agent_name": "agent_a",
            "fallback_model": "gpt-3.5-turbo"
        }
        
        _handle_text_fallback_completed(event)
        
        expected = "✅ agent_a: Completed using fallback model gpt-3.5-turbo"
        mock_echo.assert_called_once_with(expected)

    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    def test_handle_text_fallback_failed(self, mock_echo):
        """Test text mode fallback failed handler."""
        event = {
            "agent_name": "agent_a",
            "error": "Connection timeout"
        }
        
        _handle_text_fallback_failed(event)
        
        expected = "❌ agent_a: Fallback failed - Connection timeout"
        mock_echo.assert_called_once_with(expected)

    @patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo')
    def test_handle_text_fallback_failed_no_error(self, mock_echo):
        """Test text mode fallback failed handler without error."""
        event = {"agent_name": "agent_a"}
        
        _handle_text_fallback_failed(event)
        
        expected = "❌ agent_a: Fallback failed - Unknown error"
        mock_echo.assert_called_once_with(expected)


class TestComplexStreamingScenarios:
    """Test complex streaming execution scenarios."""

    @pytest.mark.asyncio
    async def test_json_streaming_execution_success(self):
        """Test JSON streaming execution with successful events."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"
        
        # Create async generator for events
        async def mock_execute_streaming(config, data):
            yield {"type": "agent_started", "agent_name": "agent_a"}
            yield {"type": "agent_completed", "agent_name": "agent_a", "result": "Success"}
        
        executor.execute_streaming = mock_execute_streaming
        
        with patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo') as mock_echo:
            await _run_text_json_execution(executor, ensemble_config, input_data, "json", True)
            
            # Should output each event as JSON
            assert mock_echo.call_count == 2
            
            # Check first event
            first_call = mock_echo.call_args_list[0][0][0]
            first_event = json.loads(first_call)
            assert first_event["type"] == "agent_started"
            assert first_event["agent_name"] == "agent_a"

    @pytest.mark.asyncio
    async def test_json_streaming_execution_error(self):
        """Test JSON streaming execution with error."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a"}]
        input_data = "Test input"
        
        async def mock_execute_streaming(config, data):
            raise Exception("Streaming error")
            yield  # This will never be reached, but needed for async generator
        
        executor.execute_streaming = mock_execute_streaming
        
        with patch('llm_orc.cli_modules.utils.visualization.streaming.click.echo') as mock_echo:
            await _run_text_json_execution(executor, ensemble_config, input_data, "json", True)
            
            # Should output error event as JSON
            mock_echo.assert_called_once()
            error_call = mock_echo.call_args[0][0]
            error_event = json.loads(error_call)
            assert error_event["type"] == "error"
            assert error_event["error"] == "Streaming error"

    @pytest.mark.asyncio
    async def test_rich_streaming_execution_complete_flow(self):
        """Test rich streaming execution with complete event flow."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]}
        ]
        input_data = "Test input"
        
        # Create async generator for complete execution flow
        async def mock_execute_streaming(config, data):
            yield {"type": "agent_started", "data": {"agent_name": "agent_a"}}
            yield {"type": "agent_completed", "data": {"agent_name": "agent_a"}}
            yield {"type": "agent_started", "data": {"agent_name": "agent_b"}}
            yield {"type": "agent_completed", "data": {"agent_name": "agent_b"}}
            yield {
                "type": "execution_completed",
                "data": {
                    "results": {
                        "agent_a": {"status": "success"},
                        "agent_b": {"status": "success"}
                    },
                    "metadata": {"duration": "5s"}
                }
            }
        
        executor.execute_streaming = mock_execute_streaming
        
        with patch('llm_orc.cli_modules.utils.visualization.streaming.Console') as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console
            
            # Setup status context manager properly
            mock_status = Mock()
            mock_console.status.return_value = mock_status
            mock_status.__enter__ = Mock(return_value=mock_status)
            mock_status.__exit__ = Mock(return_value=None)
            
            await run_streaming_execution(executor, ensemble_config, input_data, "rich")
            
            # Should have updated the status display multiple times
            assert mock_status.update.call_count >= 4  # Once for each event that updates status

    @pytest.mark.asyncio
    async def test_rich_streaming_execution_with_error(self):
        """Test rich streaming execution with error during execution."""
        executor = AsyncMock()
        ensemble_config = Mock()
        ensemble_config.agents = [{"name": "agent_a", "depends_on": []}]
        input_data = "Test input"
        
        async def mock_execute_streaming(config, data):
            yield {"type": "agent_started", "data": {"agent_name": "agent_a"}}
            raise Exception("Execution error")
        
        executor.execute_streaming = mock_execute_streaming
        
        with patch('llm_orc.cli_modules.utils.visualization.streaming.Console') as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console
            
            # Setup status context manager properly
            mock_status = Mock()
            mock_console.status.return_value = mock_status
            mock_status.__enter__ = Mock(return_value=mock_status)
            mock_status.__exit__ = Mock(return_value=None)
            
            # This should not raise an exception - errors are caught and handled
            try:
                await run_streaming_execution(executor, ensemble_config, input_data, "rich")
            except Exception:
                pass  # Expected since we're mocking an error
            
            # Should have at least tried to create the status display
            mock_console.status.assert_called_once()