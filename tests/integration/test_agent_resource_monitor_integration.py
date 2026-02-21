"""Tests for AgentResourceMonitor resource monitoring and metrics."""

from typing import Any

import pytest

from llm_orc.core.execution.agent_resource_monitor import AgentResourceMonitor


class TestAgentResourceMonitor:
    """Test AgentResourceMonitor monitoring and metrics functionality."""

    @pytest.fixture
    def executor(self) -> AgentResourceMonitor:
        """Create an AgentResourceMonitor instance."""
        config: dict[str, Any] = {
            "concurrency": {"max_concurrent_agents": 5},
            "execution": {"default_timeout": 60},
        }
        return AgentResourceMonitor(performance_config=config)

    def test_get_adaptive_stats_returns_execution_metrics(
        self, executor: AgentResourceMonitor
    ) -> None:
        """get_adaptive_stats always provides execution_metrics."""
        stats = executor.get_adaptive_stats()
        assert stats["management_type"] == "user_configured"
        assert not stats["adaptive_used"]
        assert "execution_metrics" in stats

    def test_phase_metrics_initially_empty(
        self, executor: AgentResourceMonitor
    ) -> None:
        """Phase metrics list starts empty."""
        assert executor._phase_metrics == []

    def test_monitor_initialized(self, executor: AgentResourceMonitor) -> None:
        """Resource monitor is created during init."""
        assert executor.monitor is not None
