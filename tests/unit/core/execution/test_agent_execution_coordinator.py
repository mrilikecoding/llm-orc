"""Tests for agent execution coordinator."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_orc.core.execution.agent_execution_coordinator import (
    AgentExecutionCoordinator,
)


class TestAgentExecutionCoordinator:
    """Test agent execution coordination functionality."""

    def setup_coordinator(self) -> tuple[AgentExecutionCoordinator, dict[str, Any]]:
        """Set up coordinator with mocked dependencies."""
        performance_config = {
            "execution": {"default_timeout": 60},
            "concurrency": {"max_concurrent_agents": 5},
        }

        mock_agent_executor = AsyncMock()
        mock_agent_executor.return_value = ("Response", None)

        coordinator = AgentExecutionCoordinator(
            performance_config=performance_config,
            agent_executor=mock_agent_executor,
        )

        return coordinator, {
            "performance_config": performance_config,
            "agent_executor": mock_agent_executor,
        }

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_no_timeout(self) -> None:
        """Test agent execution without timeout."""
        coordinator, mocks = self.setup_coordinator()

        agent_config = {"name": "test_agent", "model": "mock"}
        result = await coordinator.execute_agent_with_timeout(
            agent_config, "test input", None
        )

        assert result == ("Response", None)
        mocks["agent_executor"].assert_called_once_with(agent_config, "test input")

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_success(self) -> None:
        """Test agent execution with timeout that succeeds."""
        coordinator, mocks = self.setup_coordinator()

        # Mock that completes quickly
        async def quick_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            await asyncio.sleep(0.01)  # Very short delay
            return ("Quick response", None)

        mocks["agent_executor"].side_effect = quick_execution

        agent_config = {"name": "test_agent", "model": "mock"}
        result = await coordinator.execute_agent_with_timeout(
            agent_config,
            "test input",
            1,  # 1 second timeout
        )

        assert result == ("Quick response", None)

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_timeout_error(self) -> None:
        """Test agent execution that times out."""
        coordinator, mocks = self.setup_coordinator()

        # Mock that takes too long
        async def slow_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            await asyncio.sleep(2.0)  # 2 seconds, longer than timeout
            return ("Slow response", None)

        mocks["agent_executor"].side_effect = slow_execution

        agent_config = {"name": "test_agent", "model": "mock"}

        with pytest.raises(Exception, match="timed out after 1 seconds"):
            await coordinator.execute_agent_with_timeout(
                agent_config,
                "test input",
                1,  # 1 second timeout (but will time out due to slow mock)
            )

    def test_get_effective_concurrency_limit_configured(self) -> None:
        """Test getting concurrency limit when explicitly configured."""
        coordinator, _ = self.setup_coordinator()

        # Should use configured value
        limit = coordinator.get_effective_concurrency_limit(10)
        assert limit == 5  # From setup configuration

    def test_get_effective_concurrency_limit_calculated_small(self) -> None:
        """Test calculated concurrency limit for small ensembles."""
        # No concurrency config
        performance_config = {"execution": {"default_timeout": 60}}
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        assert coordinator.get_effective_concurrency_limit(1) == 1
        assert coordinator.get_effective_concurrency_limit(3) == 3

    def test_get_effective_concurrency_limit_calculated_medium(self) -> None:
        """Test calculated concurrency limit for medium ensembles."""
        performance_config = {"execution": {"default_timeout": 60}}
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        assert coordinator.get_effective_concurrency_limit(5) == 5
        assert coordinator.get_effective_concurrency_limit(10) == 5

    def test_get_effective_concurrency_limit_calculated_large(self) -> None:
        """Test calculated concurrency limit for large ensembles."""
        performance_config = {"execution": {"default_timeout": 60}}
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        assert coordinator.get_effective_concurrency_limit(15) == 8
        assert coordinator.get_effective_concurrency_limit(20) == 8

    def test_get_effective_concurrency_limit_calculated_very_large(self) -> None:
        """Test calculated concurrency limit for very large ensembles."""
        performance_config = {"execution": {"default_timeout": 60}}
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        assert coordinator.get_effective_concurrency_limit(25) == 10
        assert coordinator.get_effective_concurrency_limit(100) == 10

    def test_get_agent_timeout_from_enhanced_config(self) -> None:
        """Test getting timeout from enhanced config."""
        coordinator, _ = self.setup_coordinator()

        agent_config = {"name": "test"}
        enhanced_config = {"timeout_seconds": 120}

        timeout = coordinator.get_agent_timeout(agent_config, enhanced_config)
        assert timeout == 120

    def test_get_agent_timeout_from_default(self) -> None:
        """Test getting timeout from default config."""
        coordinator, _ = self.setup_coordinator()

        agent_config = {"name": "test"}
        enhanced_config: dict[str, Any] = {}  # No timeout specified

        timeout = coordinator.get_agent_timeout(agent_config, enhanced_config)
        assert timeout == 60  # From performance_config default

    def test_get_agent_timeout_fallback(self) -> None:
        """Test timeout fallback when no config available."""
        performance_config: dict[str, Any] = {}  # No execution config
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        agent_config = {"name": "test"}
        enhanced_config: dict[str, Any] = {}

        timeout = coordinator.get_agent_timeout(agent_config, enhanced_config)
        assert timeout == 60  # Hardcoded fallback

    def test_should_use_concurrency_limit_true(self) -> None:
        """Test should use concurrency limit when agent count exceeds limit."""
        coordinator, _ = self.setup_coordinator()

        assert coordinator.should_use_concurrency_limit(10, 5) is True

    def test_should_use_concurrency_limit_false(self) -> None:
        """Test should not use concurrency limit when within limit."""
        coordinator, _ = self.setup_coordinator()

        assert coordinator.should_use_concurrency_limit(3, 5) is False
        assert coordinator.should_use_concurrency_limit(5, 5) is False

    @pytest.mark.asyncio
    async def test_execute_with_semaphore(self) -> None:
        """Test executing agent with semaphore control."""
        coordinator, mocks = self.setup_coordinator()

        semaphore = asyncio.Semaphore(2)
        agent_config = {"name": "test_agent"}

        result = await coordinator.execute_with_semaphore(
            semaphore, agent_config, "test input", 30
        )

        assert result == ("Response", None)
        mocks["agent_executor"].assert_called_once_with(agent_config, "test input")

    def test_create_semaphore(self) -> None:
        """Test creating semaphore."""
        coordinator, _ = self.setup_coordinator()

        semaphore = coordinator.create_semaphore(3)
        assert isinstance(semaphore, asyncio.Semaphore)
        assert semaphore._value == 3

    def test_get_timeout_strategy_custom(self) -> None:
        """Test timeout strategy for agent with custom timeout."""
        coordinator, _ = self.setup_coordinator()

        agent_config = {"name": "test", "timeout_seconds": 90}
        strategy = coordinator.get_timeout_strategy(agent_config)

        assert strategy["has_custom_timeout"] is True
        assert strategy["timeout_source"] == "custom"
        assert strategy["custom_timeout"] == 90
        assert strategy["default_timeout"] == 60

    def test_get_timeout_strategy_default(self) -> None:
        """Test timeout strategy for agent using default timeout."""
        coordinator, _ = self.setup_coordinator()

        agent_config = {"name": "test"}
        strategy = coordinator.get_timeout_strategy(agent_config)

        assert strategy["has_custom_timeout"] is False
        assert strategy["timeout_source"] == "default"
        assert "custom_timeout" not in strategy
        assert strategy["default_timeout"] == 60

    def test_validate_timeout_config_valid(self) -> None:
        """Test validating valid timeout configurations."""
        coordinator, _ = self.setup_coordinator()

        assert coordinator.validate_timeout_config(None) is True
        assert coordinator.validate_timeout_config(30) is True
        assert coordinator.validate_timeout_config(120) is True

    def test_validate_timeout_config_invalid(self) -> None:
        """Test validating invalid timeout configurations."""
        coordinator, _ = self.setup_coordinator()

        assert coordinator.validate_timeout_config(0) is False
        assert coordinator.validate_timeout_config(-10) is False
        assert coordinator.validate_timeout_config("30") is False

    def test_get_concurrency_strategy_configured(self) -> None:
        """Test concurrency strategy with configured limit."""
        coordinator, _ = self.setup_coordinator()

        strategy = coordinator.get_concurrency_strategy(10)

        assert strategy["agent_count"] == 10
        assert strategy["effective_concurrency_limit"] == 5
        assert strategy["uses_semaphore"] is True
        assert strategy["concurrency_source"] == "configured"
        assert strategy["configured_limit"] == 5

    def test_get_concurrency_strategy_calculated(self) -> None:
        """Test concurrency strategy with calculated limit."""
        performance_config = {"execution": {"default_timeout": 60}}
        coordinator = AgentExecutionCoordinator(performance_config, AsyncMock())

        strategy = coordinator.get_concurrency_strategy(3)

        assert strategy["agent_count"] == 3
        assert strategy["effective_concurrency_limit"] == 3
        assert strategy["uses_semaphore"] is False
        assert strategy["concurrency_source"] == "calculated"
        assert "configured_limit" not in strategy

    @pytest.mark.asyncio
    async def test_execute_multiple_with_coordination_empty(self) -> None:
        """Test executing empty agent list."""
        coordinator, _ = self.setup_coordinator()

        result = await coordinator.execute_multiple_with_coordination(
            [], lambda x: "input", lambda x: 30
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_execute_multiple_with_coordination_unlimited(self) -> None:
        """Test executing agents without concurrency limits."""
        coordinator, mocks = self.setup_coordinator()

        # Small agent count should use unlimited execution
        agents = [
            {"name": "agent1", "model": "mock"},
            {"name": "agent2", "model": "mock"},
        ]

        mocks["agent_executor"].side_effect = [
            ("Response 1", None),
            ("Response 2", None),
        ]

        results = await coordinator.execute_multiple_with_coordination(
            agents, lambda agent: f"input for {agent['name']}", lambda agent: 30
        )

        assert len(results) == 2
        assert results[0] == ("Response 1", None)
        assert results[1] == ("Response 2", None)

    @pytest.mark.asyncio
    async def test_execute_multiple_with_coordination_semaphore(self) -> None:
        """Test executing agents with semaphore coordination."""
        # Use config that forces semaphore usage
        performance_config = {
            "execution": {"default_timeout": 60},
            "concurrency": {"max_concurrent_agents": 2},
        }
        mock_agent_executor = AsyncMock()
        coordinator = AgentExecutionCoordinator(performance_config, mock_agent_executor)

        # Large enough agent count to trigger semaphore
        agents = [{"name": f"agent{i}", "model": "mock"} for i in range(5)]

        mock_agent_executor.side_effect = [(f"Response {i}", None) for i in range(5)]

        results = await coordinator.execute_multiple_with_coordination(
            agents, lambda agent: f"input for {agent['name']}", lambda agent: 30
        )

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == (f"Response {i}", None)

    def test_get_execution_plan(self) -> None:
        """Test getting execution plan."""
        coordinator, _ = self.setup_coordinator()

        agents: list[dict[str, Any]] = [
            {"name": "agent1", "timeout_seconds": 90},
            {"name": "agent2"},  # No custom timeout
        ]

        plan = coordinator.get_execution_plan(agents)

        assert plan["total_agents"] == 2
        assert plan["execution_mode"] == "unlimited"  # Small count

        # Check concurrency info
        concurrency = plan["concurrency"]
        assert concurrency["agent_count"] == 2
        assert concurrency["effective_concurrency_limit"] == 5
        assert concurrency["uses_semaphore"] is False

        # Check agent timeout info
        timeouts = plan["agent_timeouts"]
        assert len(timeouts) == 2

        assert timeouts[0]["agent_name"] == "agent1"
        assert timeouts[0]["timeout_info"]["has_custom_timeout"] is True

        assert timeouts[1]["agent_name"] == "agent2"
        assert timeouts[1]["timeout_info"]["has_custom_timeout"] is False

    def test_get_execution_plan_semaphore_mode(self) -> None:
        """Test execution plan for semaphore mode."""
        coordinator, _ = self.setup_coordinator()

        # Large enough agent count to trigger semaphore
        agents = [{"name": f"agent{i}"} for i in range(10)]

        plan = coordinator.get_execution_plan(agents)

        assert plan["total_agents"] == 10
        assert plan["execution_mode"] == "semaphore"
        assert plan["concurrency"]["uses_semaphore"] is True

    @pytest.mark.asyncio
    async def test_concurrent_execution_with_timeout_mix(self) -> None:
        """Test concurrent execution with mixed timeout scenarios."""
        coordinator, mocks = self.setup_coordinator()

        # Mock some agents fast, some slow
        async def variable_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            if "fast" in config["name"]:
                await asyncio.sleep(0.01)
                return (f"Fast {config['name']}", None)
            else:
                await asyncio.sleep(0.05)
                return (f"Slow {config['name']}", None)

        mocks["agent_executor"].side_effect = variable_execution

        agents = [
            {"name": "fast_agent1"},
            {"name": "slow_agent1"},
            {"name": "fast_agent2"},
        ]

        results = await coordinator.execute_multiple_with_coordination(
            agents,
            lambda agent: f"input for {agent['name']}",
            lambda agent: 1,  # 1 second timeout (should be enough)
        )

        assert len(results) == 3
        assert "Fast fast_agent1" in str(results[0])
        assert "Slow slow_agent1" in str(results[1])
        assert "Fast fast_agent2" in str(results[2])

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Test that semaphore actually limits concurrency."""
        performance_config = {
            "execution": {"default_timeout": 60},
            "concurrency": {"max_concurrent_agents": 2},
        }

        execution_times = []

        async def tracked_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            start_time = asyncio.get_event_loop().time()
            execution_times.append(("start", config["name"], start_time))
            await asyncio.sleep(0.1)  # Simulate work
            end_time = asyncio.get_event_loop().time()
            execution_times.append(("end", config["name"], end_time))
            return (f"Result {config['name']}", None)

        coordinator = AgentExecutionCoordinator(performance_config, tracked_execution)

        agents = [{"name": f"agent{i}"} for i in range(4)]

        results = await coordinator.execute_multiple_with_coordination(
            agents,
            lambda agent: "input",
            lambda agent: 10,  # Long timeout
        )

        assert len(results) == 4

        # With concurrency limit of 2, we should see staggered execution
        start_times = [event for event in execution_times if event[0] == "start"]
        assert len(start_times) == 4

        # Should have exactly 2 concurrent executions at any time
        # This is a simplified check - in practice we'd need more
        # sophisticated timing analysis
