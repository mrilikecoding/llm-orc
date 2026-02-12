"""Tests for agent execution coordination with parallel processing."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.agent_executor import AgentExecutor


class TestAgentExecutor:
    """Test the agent executor."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_emit_performance_event = Mock()
        self.mock_resolve_model_profile = AsyncMock()
        self.mock_execute_agent_with_timeout = AsyncMock()
        self.mock_get_agent_input = Mock()

        # Default performance config
        self.performance_config = {
            "concurrency": {"max_concurrent_agents": 0},
            "execution": {"default_timeout": 60},
        }

        # Create executor
        self.executor = AgentExecutor(
            performance_config=self.performance_config,
            emit_performance_event=self.mock_emit_performance_event,
            resolve_model_profile_to_config=self.mock_resolve_model_profile,
            execute_agent_with_timeout=self.mock_execute_agent_with_timeout,
            get_agent_input=self.mock_get_agent_input,
        )

    def test_init(self) -> None:
        """Test executor initialization."""
        assert self.executor._performance_config == self.performance_config
        assert self.executor._emit_performance_event == self.mock_emit_performance_event
        assert (
            self.executor._resolve_model_profile_to_config
            == self.mock_resolve_model_profile
        )
        assert (
            self.executor._execute_agent_with_timeout
            == self.mock_execute_agent_with_timeout
        )
        assert self.executor._get_agent_input == self.mock_get_agent_input

    def test_get_effective_concurrency_limit_configured(self) -> None:
        """Test concurrency limit with explicit configuration."""
        # Set explicit limit
        config = self.executor._performance_config["concurrency"]
        config["max_concurrent_agents"] = 15

        result = self.executor.get_effective_concurrency_limit(20)
        assert result == 15

    def test_get_effective_concurrency_limit_small_ensemble(self) -> None:
        """Test concurrency limit for small ensembles (≤3 agents)."""
        result = self.executor.get_effective_concurrency_limit(2)
        assert result == 2

        result = self.executor.get_effective_concurrency_limit(3)
        assert result == 3

    def test_get_effective_concurrency_limit_medium_ensemble(self) -> None:
        """Test concurrency limit for medium ensembles (4-10 agents)."""
        result = self.executor.get_effective_concurrency_limit(5)
        assert result == 5

        result = self.executor.get_effective_concurrency_limit(10)
        assert result == 5

    def test_get_effective_concurrency_limit_large_ensemble(self) -> None:
        """Test concurrency limit for large ensembles (11-20 agents)."""
        result = self.executor.get_effective_concurrency_limit(15)
        assert result == 8

        result = self.executor.get_effective_concurrency_limit(20)
        assert result == 8

    def test_get_effective_concurrency_limit_very_large_ensemble(self) -> None:
        """Test concurrency limit for very large ensembles (>20 agents)."""
        result = self.executor.get_effective_concurrency_limit(25)
        assert result == 10

        result = self.executor.get_effective_concurrency_limit(100)
        assert result == 10

    def test_get_effective_concurrency_limit_invalid_config(self) -> None:
        """Test concurrency limit with invalid configuration."""
        # Set invalid config (non-positive integer)
        config = self.executor._performance_config["concurrency"]
        config["max_concurrent_agents"] = 0
        result = self.executor.get_effective_concurrency_limit(5)
        assert result == 5  # Should use default logic

        # Set non-integer config
        config = self.executor._performance_config["concurrency"]
        config["max_concurrent_agents"] = "invalid"
        result = self.executor.get_effective_concurrency_limit(5)
        assert result == 5  # Should use default logic

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_empty_list(self) -> None:
        """Test parallel execution with empty agent list."""
        config = EnsembleConfig(name="test", description="test", agents=[])
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        await self.executor.execute_agents_parallel(
            agents=[],
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Should return early without doing anything
        assert results_dict == {}
        assert agent_usage == {}
        self.mock_emit_performance_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_unlimited_path(self) -> None:
        """Test parallel execution taking unlimited concurrency path."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock the unlimited execution method
        with patch.object(self.executor, "execute_agents_unlimited") as mock_unlimited:
            await self.executor.execute_agents_parallel(
                agents=agents,
                input_data="test input",
                config=config,
                results_dict=results_dict,
                agent_usage=agent_usage,
            )

            mock_unlimited.assert_called_once_with(
                agents, "test input", config, results_dict, agent_usage
            )

    @pytest.mark.asyncio
    async def test_execute_agents_parallel_semaphore_path(self) -> None:
        """Test parallel execution taking semaphore concurrency path."""
        # Create more agents than limit (6 agents, should trigger semaphore path)
        agents = [{"name": f"agent{i}", "model_profile": "claude"} for i in range(6)]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock the semaphore execution method
        with patch.object(
            self.executor, "execute_agents_with_semaphore"
        ) as mock_semaphore:
            await self.executor.execute_agents_parallel(
                agents=agents,
                input_data="test input",
                config=config,
                results_dict=results_dict,
                agent_usage=agent_usage,
            )

            mock_semaphore.assert_called_once_with(
                agents, "test input", config, results_dict, agent_usage, 5
            )

    @pytest.mark.asyncio
    async def test_execute_agents_unlimited_success(self) -> None:
        """Test unlimited execution with successful agents."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.side_effect = lambda input_data, agent_name: (
            f"input for {agent_name}"
        )

        # Mock successful agent execution
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"tokens": 100, "cost": 0.01}
        self.mock_execute_agent_with_timeout.side_effect = [
            ("response1", mock_model),
            ("response2", mock_model),
        ]

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify results
        assert len(results_dict) == 2
        assert results_dict["agent1"]["response"] == "response1"
        assert results_dict["agent1"]["status"] == "success"
        assert results_dict["agent2"]["response"] == "response2"
        assert results_dict["agent2"]["status"] == "success"

        # Verify usage tracking
        assert len(agent_usage) == 2
        assert agent_usage["agent1"]["tokens"] == 100
        assert agent_usage["agent2"]["tokens"] == 100

        # Verify performance events (2 started, 2 completed)
        assert self.mock_emit_performance_event.call_count >= 4

    @pytest.mark.asyncio
    async def test_execute_agents_unlimited_with_agent_failure(self) -> None:
        """Test unlimited execution with agent failure."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.side_effect = lambda input_data, agent_name: (
            f"input for {agent_name}"
        )

        # Mock one success, one failure
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"tokens": 100, "cost": 0.01}
        self.mock_execute_agent_with_timeout.side_effect = [
            ("response1", mock_model),
            Exception("Agent execution failed"),
        ]

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify results - success and failure
        assert len(results_dict) == 2
        assert results_dict["agent1"]["response"] == "response1"
        assert results_dict["agent1"]["status"] == "success"
        assert results_dict["agent2"]["error"] == "Agent execution failed"
        assert results_dict["agent2"]["status"] == "failed"

        # Verify usage tracking - only successful agent
        assert len(agent_usage) == 1
        assert "agent1" in agent_usage

    @pytest.mark.asyncio
    async def test_execute_agents_unlimited_gather_exception(self) -> None:
        """Test unlimited execution with gather-level exception."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock asyncio.gather to raise an exception
        with patch("asyncio.gather", side_effect=Exception("Gather failed")):
            await self.executor.execute_agents_unlimited(
                agents=agents,
                input_data="test input",
                config=config,
                results_dict=results_dict,
                agent_usage=agent_usage,
            )

        # Should mark all agents as failed
        assert len(results_dict) == 2
        assert results_dict["agent1"]["error"] == "Gather failed"
        assert results_dict["agent1"]["status"] == "failed"
        assert results_dict["agent2"]["error"] == "Gather failed"
        assert results_dict["agent2"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_agents_with_semaphore_success(self) -> None:
        """Test semaphore-controlled execution with successful agents."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
            {"name": "agent3", "model_profile": "claude"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        max_concurrent = 2

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.side_effect = lambda input_data, agent_name: (
            f"input for {agent_name}"
        )

        # Mock successful agent execution
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"tokens": 100, "cost": 0.01}
        self.mock_execute_agent_with_timeout.side_effect = [
            ("response1", mock_model),
            ("response2", mock_model),
            ("response3", mock_model),
        ]

        await self.executor.execute_agents_with_semaphore(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
            max_concurrent=max_concurrent,
        )

        # Verify results
        assert len(results_dict) == 3
        for i in range(1, 4):
            assert results_dict[f"agent{i}"]["response"] == f"response{i}"
            assert results_dict[f"agent{i}"]["status"] == "success"

        # Verify usage tracking
        assert len(agent_usage) == 3

    @pytest.mark.asyncio
    async def test_execute_agents_with_semaphore_with_failure(self) -> None:
        """Test semaphore-controlled execution with agent failure."""
        agents = [
            {"name": "agent1", "model_profile": "claude"},
            {"name": "agent2", "model_profile": "gpt4"},
        ]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        max_concurrent = 2

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.side_effect = lambda input_data, agent_name: (
            f"input for {agent_name}"
        )

        # Mock one success, one failure
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"tokens": 100, "cost": 0.01}
        self.mock_execute_agent_with_timeout.side_effect = [
            ("response1", mock_model),
            Exception("Semaphore agent failed"),
        ]

        await self.executor.execute_agents_with_semaphore(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
            max_concurrent=max_concurrent,
        )

        # Verify mixed results
        assert len(results_dict) == 2
        assert results_dict["agent1"]["response"] == "response1"
        assert results_dict["agent1"]["status"] == "success"
        assert results_dict["agent2"]["error"] == "Semaphore agent failed"
        assert results_dict["agent2"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_agents_with_semaphore_gather_exception(self) -> None:
        """Test semaphore execution with gather-level exception."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}
        max_concurrent = 1

        # Mock asyncio.gather to raise an exception
        with patch(
            "asyncio.gather",
            side_effect=Exception("Semaphore gather failed"),
        ):
            await self.executor.execute_agents_with_semaphore(
                agents=agents,
                input_data="test input",
                config=config,
                results_dict=results_dict,
                agent_usage=agent_usage,
                max_concurrent=max_concurrent,
            )

        # Should mark all agents as failed
        assert results_dict["agent1"]["error"] == "Semaphore gather failed"
        assert results_dict["agent1"]["status"] == "failed"

    def test_process_agent_results_with_exceptions(self) -> None:
        """Test processing results with exceptions."""
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mix of exceptions and results
        agent_results = [
            Exception("Some error"),
            ("agent1", ("response1", Mock())),
            ("agent2", None),  # Error case
        ]

        self.executor.process_agent_results(agent_results, results_dict, agent_usage)

        # Should only process valid results
        assert len(results_dict) == 1
        assert results_dict["agent1"]["response"] == "response1"
        assert results_dict["agent1"]["status"] == "success"

    def test_process_agent_results_with_usage_tracking(self) -> None:
        """Test processing results with usage metrics."""
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock model with usage
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"tokens": 150, "cost": 0.02}

        # Mock model without usage
        mock_model_no_usage = Mock()
        mock_model_no_usage.get_last_usage.return_value = None

        agent_results = [
            ("agent1", ("response1", mock_model)),
            ("agent2", ("response2", mock_model_no_usage)),
            ("agent3", ("response3", None)),  # No model instance
        ]

        self.executor.process_agent_results(agent_results, results_dict, agent_usage)

        # Verify results
        assert len(results_dict) == 3
        for i in range(1, 4):
            assert results_dict[f"agent{i}"]["response"] == f"response{i}"
            assert results_dict[f"agent{i}"]["status"] == "success"

        # Verify usage tracking - only agent1 has usage
        assert len(agent_usage) == 1
        assert agent_usage["agent1"]["tokens"] == 150
        assert agent_usage["agent1"]["cost"] == 0.02

    @pytest.mark.asyncio
    async def test_timeout_configuration_from_enhanced_config(self) -> None:
        """Test timeout configuration from enhanced agent config."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock resolve to return specific timeout
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 120}
        self.mock_get_agent_input.return_value = "test input"

        # Mock successful execution
        mock_model = Mock()
        mock_model.get_last_usage.return_value = None
        self.mock_execute_agent_with_timeout.return_value = ("response", mock_model)

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify timeout was passed correctly
        self.mock_execute_agent_with_timeout.assert_called_once()
        call_args = self.mock_execute_agent_with_timeout.call_args
        assert call_args[0][2] == 120  # timeout parameter

    @pytest.mark.asyncio
    async def test_timeout_fallback_to_default(self) -> None:
        """Test timeout fallback to default configuration."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock resolve to return config without timeout
        self.mock_resolve_model_profile.return_value = {}
        self.mock_get_agent_input.return_value = "test input"

        # Mock successful execution
        mock_model = Mock()
        mock_model.get_last_usage.return_value = None
        self.mock_execute_agent_with_timeout.return_value = ("response", mock_model)

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify default timeout was used
        self.mock_execute_agent_with_timeout.assert_called_once()
        call_args = self.mock_execute_agent_with_timeout.call_args
        assert call_args[0][2] == 60  # default timeout from performance config

    @pytest.mark.asyncio
    async def test_agent_input_handling_string_input(self) -> None:
        """Test agent input handling with string input."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.return_value = "processed input"
        self.mock_execute_agent_with_timeout.return_value = ("response", None)

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="original input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify get_agent_input was called correctly
        self.mock_get_agent_input.assert_called_once_with("original input", "agent1")

    @pytest.mark.asyncio
    async def test_agent_input_handling_dict_input(self) -> None:
        """Test agent input handling with dictionary input."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.return_value = "agent specific input"
        self.mock_execute_agent_with_timeout.return_value = ("response", None)

        input_dict = {"agent1": "specific input for agent1"}

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data=input_dict,
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify get_agent_input was called with dict
        self.mock_get_agent_input.assert_called_once_with(input_dict, "agent1")

    @pytest.mark.asyncio
    async def test_performance_events_timing(self) -> None:
        """Test that performance events include proper timing."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.return_value = "test input"

        # Mock execution to simulate some delay
        async def mock_execution(*args: Any) -> tuple[str, None]:
            await asyncio.sleep(0.1)  # 100ms delay
            return ("response", None)

        self.mock_execute_agent_with_timeout.side_effect = mock_execution

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify performance events were emitted
        assert self.mock_emit_performance_event.call_count == 2

        # Check agent_started event
        start_call = self.mock_emit_performance_event.call_args_list[0]
        assert start_call[0][0] == "agent_started"
        assert start_call[0][1]["agent_name"] == "agent1"
        assert "timestamp" in start_call[0][1]

        # Check agent_completed event
        end_call = self.mock_emit_performance_event.call_args_list[1]
        assert end_call[0][0] == "agent_completed"
        assert end_call[0][1]["agent_name"] == "agent1"
        assert "timestamp" in end_call[0][1]
        assert "duration_ms" in end_call[0][1]
        assert end_call[0][1]["duration_ms"] >= 100  # At least 100ms due to sleep

    @pytest.mark.asyncio
    async def test_performance_events_on_error(self) -> None:
        """Test that performance events are emitted even on agent errors."""
        agents = [{"name": "agent1", "model_profile": "claude"}]
        config = EnsembleConfig(name="test", description="test", agents=agents)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock dependencies
        self.mock_resolve_model_profile.return_value = {"timeout_seconds": 30}
        self.mock_get_agent_input.return_value = "test input"
        self.mock_execute_agent_with_timeout.side_effect = Exception("Test error")

        await self.executor.execute_agents_unlimited(
            agents=agents,
            input_data="test input",
            config=config,
            results_dict=results_dict,
            agent_usage=agent_usage,
        )

        # Verify performance events were emitted even on error
        assert self.mock_emit_performance_event.call_count == 2

        # Check agent_completed event includes error
        end_call = self.mock_emit_performance_event.call_args_list[1]
        assert end_call[0][0] == "agent_completed"
        assert end_call[0][1]["agent_name"] == "agent1"
        assert end_call[0][1]["error"] == "Test error"
        assert "duration_ms" in end_call[0][1]
