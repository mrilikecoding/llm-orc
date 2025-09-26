"""Integration tests for AgentExecutor with simplified resource management."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.agent_executor import AgentExecutor


class TestAgentExecutorSimplifiedIntegration:
    """Test integration of AgentExecutor with simplified resource management."""

    @pytest.fixture
    def mock_performance_config(self) -> dict[str, Any]:
        """Create mock performance configuration."""
        return {
            "concurrency": {"max_concurrent_agents": 5},
            "execution": {"default_timeout": 60},
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
    def simple_executor(
        self, mock_performance_config: dict[str, Any], mock_functions: dict[str, Mock]
    ) -> AgentExecutor:
        """Create an AgentExecutor with simplified resource management."""
        return AgentExecutor(
            performance_config=mock_performance_config,
            emit_performance_event=mock_functions["emit_performance_event"],
            resolve_model_profile_to_config=mock_functions[
                "resolve_model_profile_to_config"
            ],
            execute_agent_with_timeout=mock_functions["execute_agent_with_timeout"],
            get_agent_input=mock_functions["get_agent_input"],
        )

    @pytest.mark.asyncio
    async def test_user_configured_concurrency_limit(
        self, simple_executor: AgentExecutor
    ) -> None:
        """Test that AgentExecutor uses user-configured concurrency limits directly."""
        # Test with configured limit
        limit = await simple_executor._get_concurrency_limit(10)
        assert limit == 5  # From config

        # Test with default behavior when no limit configured
        simple_executor._performance_config["concurrency"]["max_concurrent_agents"] = 0
        limit = await simple_executor._get_concurrency_limit(3)
        assert limit == 3  # Small ensemble, all agents

        limit = await simple_executor._get_concurrency_limit(15)
        assert limit == 8  # Large ensemble, capped

    def test_effective_concurrency_limit_behavior(
        self, simple_executor: AgentExecutor
    ) -> None:
        """Test effective concurrency limit calculation."""
        # Test with configured limit
        limit = simple_executor.get_effective_concurrency_limit(10)
        assert limit == 5  # From config

        # Test without configured limit - smart defaults
        simple_executor._performance_config["concurrency"]["max_concurrent_agents"] = 0

        assert simple_executor.get_effective_concurrency_limit(2) == 2  # Small
        assert simple_executor.get_effective_concurrency_limit(5) == 5  # Medium
        assert simple_executor.get_effective_concurrency_limit(15) == 8  # Large
        assert simple_executor.get_effective_concurrency_limit(100) == 10  # Very large

    @pytest.mark.asyncio
    async def test_simplified_execution_workflow(
        self,
        simple_executor: AgentExecutor,
        mock_functions: dict[str, Mock],
    ) -> None:
        """Test that simplified execution workflow works correctly."""
        # Create test agents
        agents = [
            {"name": "agent1", "model_profile": "test"},
            {"name": "agent2", "model_profile": "test"},
            {"name": "agent3", "model_profile": "test"},
        ]

        config = Mock(spec=EnsembleConfig)
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        # Execute agents - should use user-configured limits
        await simple_executor.execute_agents_parallel(
            agents, "test input", config, results_dict, agent_usage
        )

        # Verify all agents were executed
        assert len(results_dict) == 3
        for agent in agents:
            assert agent["name"] in results_dict

        # Verify performance events were emitted
        emit_calls = mock_functions["emit_performance_event"].call_args_list
        event_types = [call[0][0] for call in emit_calls]
        assert "using_configured_concurrency" in event_types

        # Verify adaptive stats were collected
        stats = simple_executor.get_adaptive_stats()
        assert stats["management_type"] == "user_configured"
        assert not stats["adaptive_used"]
        assert "execution_metrics" in stats
