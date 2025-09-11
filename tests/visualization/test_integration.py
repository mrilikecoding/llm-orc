"""Comprehensive tests for visualization integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.visualization.integration import VisualizationIntegratedExecutor


@pytest.fixture(autouse=True)
def mock_expensive_dependencies():
    """Mock expensive dependencies for all visualization integration tests."""
    with patch("llm_orc.core.execution.ensemble_execution.ConfigurationManager"):
        with patch("llm_orc.core.execution.ensemble_execution.CredentialStorage"):
            with patch("llm_orc.core.execution.ensemble_execution.ModelFactory"):
                yield


class TestVisualizationIntegratedExecutor:
    """Test visualization integrated executor."""

    @pytest.fixture
    def mock_viz_config(self) -> Mock:
        """Create mock visualization config."""
        config = Mock()
        config.enabled = True
        config.default_mode = "simple"
        config.simple = Mock()
        return config

    @pytest.fixture
    def mock_ensemble_config(self) -> Mock:
        """Create mock ensemble config."""
        config = Mock(spec=EnsembleConfig)
        config.name = "test-ensemble"
        config.agents = [
            {
                "name": "agent_a",
                "depends_on": [],
                "model_profile": "default",
            },
            {
                "name": "agent_b",
                "depends_on": ["agent_a"],
                "model_profile": "gpt-4",
            },
        ]
        return config

    @pytest.fixture
    def mock_stream_manager(self) -> Mock:
        """Create mock stream manager."""
        manager = Mock()
        mock_stream = Mock()
        mock_stream.emit = AsyncMock()
        manager.create_stream.return_value = mock_stream
        return manager

    def test_init_with_config(self, mock_viz_config: Mock) -> None:
        """Test initialization with provided config."""
        # When
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        # Then
        assert executor.viz_config == mock_viz_config
        assert executor.current_stream is None
        assert executor.current_execution_id is None

    @patch("llm_orc.visualization.integration.load_visualization_config")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    def test_init_without_config(
        self, mock_get_stream_manager: Mock, mock_load_config: Mock
    ) -> None:
        """Test initialization without provided config."""
        # Given
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        mock_manager = Mock()
        mock_get_stream_manager.return_value = mock_manager

        # When
        executor = VisualizationIntegratedExecutor()

        # Then
        assert executor.viz_config == mock_config
        assert executor.stream_manager == mock_manager
        mock_load_config.assert_called_once()
        mock_get_stream_manager.assert_called_once()

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_visualization_disabled(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
    ) -> None:
        """Test execution when visualization is disabled."""
        # Given
        mock_viz_config.enabled = False
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(executor, "execute", new=AsyncMock()) as mock_execute:
            mock_execute.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute.assert_called_once_with(mock_ensemble_config, "test input")

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_simple_visualization(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with simple visualization mode."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_viz_config.default_mode = "simple"
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_simple_visualization", new=AsyncMock()
        ) as mock_execute_simple:
            mock_execute_simple.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "simple"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_simple.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_terminal_visualization(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with terminal visualization mode."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_terminal_visualization", new=AsyncMock()
        ) as mock_execute_terminal:
            mock_execute_terminal.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "terminal"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_terminal.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_web_visualization(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with web visualization mode (fallback to terminal)."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_terminal_visualization", new=AsyncMock()
        ) as mock_execute_terminal:
            mock_execute_terminal.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "web"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_terminal.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_debug_visualization(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with debug visualization mode (fallback to terminal)."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_terminal_visualization", new=AsyncMock()
        ) as mock_execute_terminal:
            mock_execute_terminal.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "debug"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_terminal.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_minimal_visualization(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with minimal visualization mode."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_minimal_visualization", new=AsyncMock()
        ) as mock_execute_minimal:
            mock_execute_minimal.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "minimal"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_minimal.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_unknown_visualization_mode(
        self,
        mock_get_stream_manager: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with unknown visualization mode (defaults to simple)."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_simple_visualization", new=AsyncMock()
        ) as mock_execute_simple:
            mock_execute_simple.return_value = {"result": "success"}

            # When
            result = await executor.execute_with_visualization(
                mock_ensemble_config, "test input", "unknown_mode"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_simple.assert_called_once_with(
                mock_ensemble_config, "test input"
            )

    @patch("llm_orc.visualization.integration.TerminalVisualizer")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_terminal_visualization_success(
        self,
        mock_get_stream_manager: Mock,
        mock_terminal_visualizer_class: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test successful terminal visualization execution."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_visualizer = Mock()
        mock_visualizer.visualize_execution = AsyncMock()
        mock_visualizer.print_summary = Mock()
        mock_terminal_visualizer_class.return_value = mock_visualizer

        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_events", new=AsyncMock()
        ) as mock_execute_events:
            mock_execute_events.return_value = {"result": "success"}

            # When
            result = await executor._execute_with_terminal_visualization(
                mock_ensemble_config, "test input"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_events.assert_called_once_with(
                mock_ensemble_config, "test input"
            )
            mock_visualizer.print_summary.assert_called_once()

    @patch("llm_orc.visualization.integration.TerminalVisualizer")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_terminal_visualization_exception(
        self,
        mock_get_stream_manager: Mock,
        mock_terminal_visualizer_class: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test terminal visualization execution with exception."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_visualizer = Mock()
        mock_visualizer.visualize_execution = AsyncMock()
        mock_terminal_visualizer_class.return_value = mock_visualizer

        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_events", new=AsyncMock()
        ) as mock_execute_events:
            mock_execute_events.side_effect = RuntimeError("Execution failed")

            # When / Then
            with pytest.raises(RuntimeError, match="Execution failed"):
                await executor._execute_with_terminal_visualization(
                    mock_ensemble_config, "test input"
                )

    @patch("llm_orc.visualization.simple.SimpleVisualizer")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_simple_visualization_success(
        self,
        mock_get_stream_manager: Mock,
        mock_simple_visualizer_class: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test successful simple visualization execution."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_visualizer = Mock()
        mock_visualizer.visualize_execution = AsyncMock()
        mock_visualizer.print_summary = Mock()
        mock_simple_visualizer_class.return_value = mock_visualizer

        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_events", new=AsyncMock()
        ) as mock_execute_events:
            mock_execute_events.return_value = {"result": "success"}

            # When
            result = await executor._execute_with_simple_visualization(
                mock_ensemble_config, "test input"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_events.assert_called_once_with(
                mock_ensemble_config, "test input"
            )
            mock_visualizer.print_summary.assert_called_once()

    @patch("llm_orc.visualization.integration.TerminalVisualizer")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_minimal_visualization_terminal(
        self,
        mock_get_stream_manager: Mock,
        mock_terminal_visualizer_class: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test minimal visualization execution with terminal visualizer."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_visualizer = Mock()
        mock_visualizer.print_summary = Mock()
        mock_terminal_visualizer_class.return_value = mock_visualizer

        executor = VisualizationIntegratedExecutor(mock_viz_config)

        with patch.object(
            executor, "_execute_with_events", new=AsyncMock()
        ) as mock_execute_events:
            mock_execute_events.return_value = {"result": "success"}

            # When
            result = await executor._execute_with_minimal_visualization(
                mock_ensemble_config, "test input"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute_events.assert_called_once_with(
                mock_ensemble_config, "test input"
            )
            mock_visualizer.print_summary.assert_called_once()

    @patch("llm_orc.visualization.integration.EventFactory")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_events_success(
        self,
        mock_get_stream_manager: Mock,
        mock_event_factory: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test successful execution with events."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_stream = mock_stream_manager.create_stream.return_value

        mock_started_event = Mock()
        mock_completed_event = Mock()
        mock_event_factory.ensemble_started.return_value = mock_started_event
        mock_event_factory.ensemble_completed.return_value = mock_completed_event

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = mock_stream
        executor.current_execution_id = "test-execution-id"

        with patch.object(executor, "execute", new=AsyncMock()) as mock_execute:
            mock_execute.return_value = {"result": "success"}

            # When
            result = await executor._execute_with_events(
                mock_ensemble_config, "test input"
            )

            # Then
            assert result == {"result": "success"}
            mock_execute.assert_called_once_with(mock_ensemble_config, "test input")

            # Check events were emitted
            assert mock_stream.emit.call_count == 2
            mock_stream.emit.assert_any_call(mock_started_event)
            mock_stream.emit.assert_any_call(mock_completed_event)

    @patch("llm_orc.visualization.integration.EventFactory")
    @patch("llm_orc.visualization.integration.get_stream_manager")
    async def test_execute_with_events_failure(
        self,
        mock_get_stream_manager: Mock,
        mock_event_factory: Mock,
        mock_viz_config: Mock,
        mock_ensemble_config: Mock,
        mock_stream_manager: Mock,
    ) -> None:
        """Test execution with events when execution fails."""
        # Given
        mock_get_stream_manager.return_value = mock_stream_manager
        mock_stream = mock_stream_manager.create_stream.return_value

        mock_started_event = Mock()
        mock_failed_event = Mock()
        mock_event_factory.ensemble_started.return_value = mock_started_event
        mock_event_factory.ensemble_failed.return_value = mock_failed_event

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = mock_stream
        executor.current_execution_id = "test-execution-id"

        with patch.object(executor, "execute", new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = RuntimeError("Execution failed")

            # When / Then
            with pytest.raises(RuntimeError, match="Execution failed"):
                await executor._execute_with_events(mock_ensemble_config, "test input")

            # Check events were emitted
            assert mock_stream.emit.call_count == 2
            mock_stream.emit.assert_any_call(mock_started_event)
            mock_stream.emit.assert_any_call(mock_failed_event)

    async def test_execute_with_events_no_stream(
        self, mock_viz_config: Mock, mock_ensemble_config: Mock
    ) -> None:
        """Test execute with events when no stream is available."""
        # Given
        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = None

        # When / Then
        with pytest.raises(RuntimeError, match="No event stream available"):
            await executor._execute_with_events(mock_ensemble_config, "test input")

    def test_visualization_hook_no_stream(self, mock_viz_config: Mock) -> None:
        """Test visualization hook when no stream is available."""
        # Given
        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = None

        # When
        executor._visualization_hook("agent_started", {"agent_name": "test"})

        # Then - should not raise an error

    @patch("llm_orc.visualization.integration.asyncio.get_event_loop")
    def test_visualization_hook_with_running_loop(
        self, mock_get_loop: Mock, mock_viz_config: Mock
    ) -> None:
        """Test visualization hook with running event loop."""
        # Given
        mock_stream = Mock()
        mock_stream.emit = AsyncMock()
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        mock_loop.create_task = Mock()
        mock_get_loop.return_value = mock_loop

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = mock_stream
        executor.current_execution_id = "test-execution-id"

        with patch.object(
            executor, "_convert_performance_event", return_value=Mock()
        ) as mock_convert:
            # When
            executor._visualization_hook("agent_started", {"agent_name": "test"})

            # Then
            mock_convert.assert_called_once_with(
                "agent_started", {"agent_name": "test"}
            )
            mock_loop.create_task.assert_called_once()

    @patch("llm_orc.visualization.integration.asyncio.run")
    @patch("llm_orc.visualization.integration.asyncio.get_event_loop")
    def test_visualization_hook_without_running_loop(
        self, mock_get_loop: Mock, mock_asyncio_run: Mock, mock_viz_config: Mock
    ) -> None:
        """Test visualization hook without running event loop."""
        # Given
        mock_stream = Mock()
        mock_stream.emit = AsyncMock()
        mock_loop = Mock()
        mock_loop.is_running.return_value = False
        mock_get_loop.return_value = mock_loop

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = mock_stream
        executor.current_execution_id = "test-execution-id"

        with patch.object(
            executor, "_convert_performance_event", return_value=Mock()
        ) as mock_convert:
            # When
            executor._visualization_hook("agent_started", {"agent_name": "test"})

            # Then
            mock_convert.assert_called_once_with(
                "agent_started", {"agent_name": "test"}
            )
            mock_asyncio_run.assert_called_once()

    def test_visualization_hook_exception_handling(self, mock_viz_config: Mock) -> None:
        """Test visualization hook handles exceptions gracefully."""
        # Given
        mock_stream = Mock()
        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_stream = mock_stream
        executor.current_execution_id = "test-execution-id"

        with patch.object(
            executor,
            "_convert_performance_event",
            side_effect=Exception("Convert error"),
        ):
            # When / Then - should not raise an error
            executor._visualization_hook("agent_started", {"agent_name": "test"})

    @patch("llm_orc.visualization.integration.EventFactory")
    def test_convert_performance_event_agent_started(
        self, mock_event_factory: Mock, mock_viz_config: Mock
    ) -> None:
        """Test converting agent_started performance event."""
        # Given
        mock_event = Mock()
        mock_event_factory.agent_started.return_value = mock_event

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_execution_id = "test-execution-id"

        data = {
            "agent_name": "test_agent",
            "model": "gpt-4",
            "depends_on": ["agent_a"],
        }

        # When
        result = executor._convert_performance_event("agent_started", data)

        # Then
        assert result == mock_event
        mock_event_factory.agent_started.assert_called_once_with(
            agent_name="test_agent",
            ensemble_name="current_ensemble",
            execution_id="test-execution-id",
            model="gpt-4",
            depends_on=["agent_a"],
        )

    @patch("llm_orc.visualization.integration.EventFactory")
    def test_convert_performance_event_agent_completed(
        self, mock_event_factory: Mock, mock_viz_config: Mock
    ) -> None:
        """Test converting agent_completed performance event."""
        # Given
        mock_event = Mock()
        mock_event_factory.agent_completed.return_value = mock_event

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_execution_id = "test-execution-id"

        data = {
            "agent_name": "test_agent",
            "result": "Agent response",
            "duration_ms": 1500,
            "cost_usd": 0.05,
            "tokens_used": 100,
        }

        # When
        result = executor._convert_performance_event("agent_completed", data)

        # Then
        assert result == mock_event
        mock_event_factory.agent_completed.assert_called_once_with(
            agent_name="test_agent",
            ensemble_name="current_ensemble",
            execution_id="test-execution-id",
            result="Agent response",
            duration_ms=1500,
            cost_usd=0.05,
            tokens_used=100,
        )

    @patch("llm_orc.visualization.integration.EventFactory")
    def test_convert_performance_event_agent_failed(
        self, mock_event_factory: Mock, mock_viz_config: Mock
    ) -> None:
        """Test converting agent_failed performance event."""
        # Given
        mock_event = Mock()
        mock_event_factory.agent_failed.return_value = mock_event

        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_execution_id = "test-execution-id"

        data = {
            "agent_name": "test_agent",
            "error": "Agent execution failed",
            "duration_ms": 500,
        }

        # When
        result = executor._convert_performance_event("agent_failed", data)

        # Then
        assert result == mock_event
        mock_event_factory.agent_failed.assert_called_once_with(
            agent_name="test_agent",
            ensemble_name="current_ensemble",
            execution_id="test-execution-id",
            error="Agent execution failed",
            duration_ms=500,
        )

    def test_convert_performance_event_unknown_type(
        self, mock_viz_config: Mock
    ) -> None:
        """Test converting unknown performance event type."""
        # Given
        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_execution_id = "test-execution-id"

        # When
        result = executor._convert_performance_event("unknown_event", {})

        # Then
        assert result is None

    def test_convert_performance_event_missing_data(
        self, mock_viz_config: Mock
    ) -> None:
        """Test converting performance event with missing data."""
        # Given
        executor = VisualizationIntegratedExecutor(mock_viz_config)
        executor.current_execution_id = "test-execution-id"

        with patch("llm_orc.visualization.integration.EventFactory") as mock_factory:
            mock_event = Mock()
            mock_factory.agent_started.return_value = mock_event

            # When - data is missing some fields
            result = executor._convert_performance_event("agent_started", {})

            # Then
            assert result == mock_event
            mock_factory.agent_started.assert_called_once_with(
                agent_name="unknown",
                ensemble_name="current_ensemble",
                execution_id="test-execution-id",
                model="unknown",
                depends_on=[],
            )
