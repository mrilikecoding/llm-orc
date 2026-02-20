"""Tests for agent execution coordinator."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.core.execution.agent_execution_coordinator import (
    AgentExecutionCoordinator,
)
from llm_orc.schemas.agent_config import LlmAgentConfig


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

        agent_config = LlmAgentConfig(name="test_agent", model_profile="mock")
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

        agent_config = LlmAgentConfig(name="test_agent", model_profile="mock")
        result = await coordinator.execute_agent_with_timeout(
            agent_config, "test input", 1
        )

        assert result == ("Quick response", None)

    @pytest.mark.asyncio
    async def test_execute_agent_with_timeout_timeout_error(self) -> None:
        """Test agent execution that times out."""
        coordinator, mocks = self.setup_coordinator()

        async def mock_wait_for(coro: Any, *, timeout: float | None = None) -> Any:
            coro.close()
            raise TimeoutError

        agent_config = LlmAgentConfig(name="test_agent", model_profile="mock")

        with (
            patch(
                "llm_orc.core.execution.agent_execution_coordinator.asyncio.wait_for",
                side_effect=mock_wait_for,
            ),
            pytest.raises(Exception, match="timed out after 1 seconds"),
        ):
            await coordinator.execute_agent_with_timeout(agent_config, "test input", 1)
