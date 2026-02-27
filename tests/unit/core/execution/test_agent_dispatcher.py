"""Tests for AgentDispatcher max-concurrency support."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.execution.phases.agent_dispatcher import AgentDispatcher
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig


def _make_dispatcher(
    max_concurrent_agents: int = 0,
) -> AgentDispatcher:
    """Build a dispatcher with controllable concurrency config."""
    perf_config: dict[str, Any] = {
        "concurrency": {"max_concurrent_agents": max_concurrent_agents},
        "execution": {"default_timeout": 60},
    }

    coordinator = AsyncMock()
    coordinator.execute_agent_with_timeout = AsyncMock(
        return_value=("response", None, False)
    )

    dependency_resolver = Mock()
    dependency_resolver.is_fan_out_instance_config.return_value = False
    dependency_resolver.get_agent_input.side_effect = lambda inp, _name: inp

    progress_controller = Mock()
    progress_controller.update_agent_progress = AsyncMock()

    resolve_profile_fn = AsyncMock(return_value={"timeout_seconds": None})

    return AgentDispatcher(
        execution_coordinator=coordinator,
        dependency_resolver=dependency_resolver,
        progress_controller=progress_controller,
        emit_event_fn=lambda _e, _d: None,
        resolve_profile_fn=resolve_profile_fn,
        performance_config=perf_config,
    )


def _make_agents(n: int) -> list[AgentConfig]:
    return [LlmAgentConfig(name=f"agent-{i}", model_profile="local") for i in range(n)]


def _tracking_execute_factory() -> tuple[
    Any,  # the async callable
    dict[str, int],  # mutable counters: peak_concurrent, current_concurrent
]:
    """Create a tracking execute function and its shared counters."""
    counters = {"peak": 0, "current": 0}
    lock = asyncio.Lock()

    async def tracking_execute(
        config: Any, input_data: Any, timeout: Any
    ) -> tuple[str, None, bool]:
        async with lock:
            counters["current"] += 1
            counters["peak"] = max(counters["peak"], counters["current"])
        await asyncio.sleep(0.01)
        async with lock:
            counters["current"] -= 1
        return ("response", None, False)

    return tracking_execute, counters


class TestMaxConcurrentAgents:
    """Scenario: max_concurrent_agents limits parallel execution."""

    @pytest.mark.asyncio
    async def test_concurrency_limited_to_config_value(self) -> None:
        """With max_concurrent_agents=1, agents run sequentially."""
        dispatcher = _make_dispatcher(max_concurrent_agents=1)
        agents = _make_agents(3)
        tracking_execute, counters = _tracking_execute_factory()

        with patch.object(
            dispatcher._execution_coordinator,
            "execute_agent_with_timeout",
            tracking_execute,
        ):
            await dispatcher.execute_agents_in_phase(agents, "test input")

        assert counters["peak"] == 1

    @pytest.mark.asyncio
    async def test_unlimited_when_zero(self) -> None:
        """With max_concurrent_agents=0, all agents run in parallel."""
        dispatcher = _make_dispatcher(max_concurrent_agents=0)
        agents = _make_agents(3)
        tracking_execute, counters = _tracking_execute_factory()

        with patch.object(
            dispatcher._execution_coordinator,
            "execute_agent_with_timeout",
            tracking_execute,
        ):
            await dispatcher.execute_agents_in_phase(agents, "test input")

        assert counters["peak"] == 3

    @pytest.mark.asyncio
    async def test_unlimited_when_absent(self) -> None:
        """Missing concurrency config defaults to unlimited."""
        perf_config: dict[str, Any] = {
            "execution": {"default_timeout": 60},
        }

        coordinator = AsyncMock()
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=("response", None, False)
        )

        dependency_resolver = Mock()
        dependency_resolver.is_fan_out_instance_config.return_value = False
        dependency_resolver.get_agent_input.side_effect = lambda inp, _name: inp

        progress_controller = Mock()
        progress_controller.update_agent_progress = AsyncMock()

        dispatcher = AgentDispatcher(
            execution_coordinator=coordinator,
            dependency_resolver=dependency_resolver,
            progress_controller=progress_controller,
            emit_event_fn=lambda _e, _d: None,
            resolve_profile_fn=AsyncMock(return_value={"timeout_seconds": None}),
            performance_config=perf_config,
        )

        agents = _make_agents(3)
        tracking_execute, counters = _tracking_execute_factory()

        with patch.object(
            dispatcher._execution_coordinator,
            "execute_agent_with_timeout",
            tracking_execute,
        ):
            await dispatcher.execute_agents_in_phase(agents, "test input")

        assert counters["peak"] == 3

    @pytest.mark.asyncio
    async def test_concurrency_two_allows_two_parallel(self) -> None:
        """With max_concurrent_agents=2, peak is at most 2."""
        dispatcher = _make_dispatcher(max_concurrent_agents=2)
        agents = _make_agents(4)
        tracking_execute, counters = _tracking_execute_factory()

        with patch.object(
            dispatcher._execution_coordinator,
            "execute_agent_with_timeout",
            tracking_execute,
        ):
            await dispatcher.execute_agents_in_phase(agents, "test input")

        assert counters["peak"] <= 2

    @pytest.mark.asyncio
    async def test_all_agents_still_complete(self) -> None:
        """All agents complete even under concurrency limit."""
        dispatcher = _make_dispatcher(max_concurrent_agents=1)
        agents = _make_agents(3)

        results = await dispatcher.execute_agents_in_phase(agents, "test input")

        assert len(results) == 3
        assert all(r.status == "success" for r in results.values())
