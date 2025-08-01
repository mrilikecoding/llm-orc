"""Integration tests for AgentExecutor with adaptive resource management."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.adaptive_resource_manager import (
    AdaptiveResourceManager,
    SystemResourceMonitor,
)
from llm_orc.core.execution.agent_executor import AgentExecutor


class TestAgentExecutorAdaptiveIntegration:
    """Test integration of AgentExecutor with adaptive resource management."""

    @pytest.fixture
    def mock_performance_config(self) -> dict[str, Any]:
        """Create mock performance configuration."""
        return {
            "concurrency": {
                "adaptive_enabled": True,
                "base_limit": 5,
                "min_limit": 1,
                "max_limit": 10,
            }
        }

    @pytest.fixture
    def mock_functions(self) -> dict[str, Mock]:
        """Create mock functions for AgentExecutor dependencies."""
        return {
            "emit_performance_event": Mock(),
            "resolve_model_profile_to_config": AsyncMock(
                return_value={"timeout_seconds": 60}
            ),
            "execute_agent_with_timeout": AsyncMock(return_value=("response", None)),
            "get_agent_input": Mock(return_value="test input"),
        }

    @pytest.fixture
    def adaptive_executor(
        self, mock_performance_config: dict[str, Any], mock_functions: dict[str, Mock]
    ) -> AgentExecutor:
        """Create an AgentExecutor with adaptive resource management enabled."""
        executor = AgentExecutor(
            performance_config=mock_performance_config,
            emit_performance_event=mock_functions["emit_performance_event"],
            resolve_model_profile_to_config=mock_functions[
                "resolve_model_profile_to_config"
            ],
            execute_agent_with_timeout=mock_functions["execute_agent_with_timeout"],
            get_agent_input=mock_functions["get_agent_input"],
        )

        # Add adaptive resource manager
        monitor = SystemResourceMonitor(polling_interval=0.01)
        executor.adaptive_manager = AdaptiveResourceManager(
            base_limit=5, monitor=monitor, min_limit=1, max_limit=10
        )

        return executor

    @pytest.mark.asyncio
    async def test_adaptive_concurrency_limit_integration(
        self, adaptive_executor: AgentExecutor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that AgentExecutor uses adaptive concurrency limits."""
        # Mock the adaptive manager to return a specific limit
        assert adaptive_executor.adaptive_manager is not None
        mock_get_limit = AsyncMock(return_value=3)
        monkeypatch.setattr(
            adaptive_executor.adaptive_manager, "get_adaptive_limit", mock_get_limit
        )

        # This method should now use adaptive limit instead of static
        limit = await adaptive_executor.get_adaptive_concurrency_limit(5)

        assert limit == 3
        mock_get_limit.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_static_when_adaptive_disabled(
        self, mock_functions: dict[str, Mock]
    ) -> None:
        """Test fallback to static limits when adaptive is disabled."""
        static_config = {"concurrency": {"adaptive_enabled": False}}

        executor = AgentExecutor(
            performance_config=static_config,
            emit_performance_event=mock_functions["emit_performance_event"],
            resolve_model_profile_to_config=mock_functions[
                "resolve_model_profile_to_config"
            ],
            execute_agent_with_timeout=mock_functions["execute_agent_with_timeout"],
            get_agent_input=mock_functions["get_agent_input"],
        )

        # Should use the existing static method
        limit = executor.get_effective_concurrency_limit(5)
        assert limit == 5  # Small ensemble, should return agent count

    @pytest.mark.asyncio
    async def test_adaptive_manager_integration_with_execution(
        self,
        adaptive_executor: AgentExecutor,
        mock_functions: dict[str, Mock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that adaptive manager integrates with actual agent execution."""
        # Create test agents
        agents = [
            {"name": "agent1", "model_profile": "test"},
            {"name": "agent2", "model_profile": "test"},
            {"name": "agent3", "model_profile": "test"},
        ]

        config = Mock(spec=EnsembleConfig)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Mock adaptive manager to return low limit
        assert adaptive_executor.adaptive_manager is not None
        mock_get_limit = AsyncMock(return_value=2)
        monkeypatch.setattr(
            adaptive_executor.adaptive_manager, "get_adaptive_limit", mock_get_limit
        )

        # Execute agents - should use semaphore with adaptive limit
        await adaptive_executor.execute_agents_parallel(
            agents, "test input", config, results_dict, agent_usage
        )

        # Verify adaptive limit was called
        mock_get_limit.assert_called()

        # Verify all agents were executed (even with lower limit)
        assert len(results_dict) == 3
        for agent in agents:
            assert agent["name"] in results_dict
