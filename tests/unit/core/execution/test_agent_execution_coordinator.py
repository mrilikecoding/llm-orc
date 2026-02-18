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

        async def quick_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            await asyncio.sleep(0.01)
            return ("Quick response", None)

        mocks["agent_executor"].side_effect = quick_execution

        agent_config = {"name": "test_agent", "model": "mock"}
        result = await coordinator.execute_agent_with_timeout(
            agent_config, "test input", 1
        )

        assert result == ("Quick response", None)

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_timeout_error(self) -> None:
        """Test agent execution that times out."""
        coordinator, mocks = self.setup_coordinator()

        async def slow_execution(
            config: dict[str, Any], input_data: str
        ) -> tuple[str, Any]:
            await asyncio.sleep(2.0)
            return ("Slow response", None)

        mocks["agent_executor"].side_effect = slow_execution

        agent_config = {"name": "test_agent", "model": "mock"}

        with pytest.raises(Exception, match="timed out after 1 seconds"):
            await coordinator.execute_agent_with_timeout(
                agent_config, "test input", 1
            )
