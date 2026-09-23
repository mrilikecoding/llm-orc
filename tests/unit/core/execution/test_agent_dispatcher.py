"""Tests for AgentDispatcher max-concurrency support."""

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.execution.phases.agent_dispatcher import AgentDispatcher
from llm_orc.schemas.agent_config import (
    AgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    ScriptAgentConfig,
)


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


class TestFanOutInstanceBaseInput:
    """PR 203 round 2: fan-out instances run in phases past phase 0, where
    the per-agent input dict is keyed by the PRE-expansion agent name; an
    instance named processor[0] must read its base input under
    fan_out_original, not its own instance name."""

    @pytest.mark.asyncio
    async def test_instance_reads_base_input_by_original_name(self) -> None:
        dispatcher = _make_dispatcher()
        resolver = cast(Mock, dispatcher._dependency_resolver)
        resolver.is_fan_out_instance_config.return_value = True
        resolver.prepare_fan_out_instance_input.return_value = "prepared"

        instance = EnsembleAgentConfig(
            name="processor[0]",
            ensemble="pdf-processor",
            fan_out_chunk="a.pdf",
            fan_out_index=0,
            fan_out_total=2,
            fan_out_original="processor",
        )

        await dispatcher._execute_single_agent_in_phase(
            instance, {"processor": "the task"}
        )

        call = resolver.prepare_fan_out_instance_input.call_args
        assert call.args[1] == "the task"


class TestScriptAgentFailureShape:
    """A script agent's own response can report failure (non-zero exit,
    timeout, or an ``{"error": ...}`` payload written by a script that
    still exits 0 — web_searcher's convention) without the subprocess
    itself raising. Fail-closed-composition B2: that failure becomes the
    agent's status, not a "success" wrapping an error string a
    downstream LLM has to notice on its own."""

    @pytest.mark.asyncio
    async def test_error_shaped_response_fails_the_agent(self) -> None:
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(
                json.dumps({"error": "authentication_failed", "backend": "tavily"}),
                None,
                False,
            )
        )
        agent = ScriptAgentConfig(name="searcher", script="web_searcher.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["searcher"]
        assert result.status == "failed"
        assert result.error == "authentication_failed"

    @pytest.mark.asyncio
    async def test_success_flag_false_fails_the_agent(self) -> None:
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(json.dumps({"success": False}), None, False)
        )
        agent = ScriptAgentConfig(name="worker", script="worker.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["worker"]
        assert result.status == "failed"
        assert result.error == json.dumps({"success": False})

    @pytest.mark.asyncio
    async def test_successful_script_response_still_succeeds(self) -> None:
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(json.dumps({"results": ["a"]}), None, False)
        )
        agent = ScriptAgentConfig(name="searcher", script="web_searcher.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["searcher"]
        assert result.status == "success"
        assert result.response == json.dumps({"results": ["a"]})

    @pytest.mark.asyncio
    async def test_stderr_preserved_on_failure(self) -> None:
        """B2: fields alongside a failed script's own error/success keys
        survive on the failed record — stderr (#174's turn_trace reads
        it), not just error."""
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(
                json.dumps(
                    {
                        "success": False,
                        "error": "Script failed with exit code 3",
                        "stderr": "boom\n",
                    }
                ),
                None,
                False,
            )
        )
        agent = ScriptAgentConfig(name="a", script="fail.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["a"]
        assert result.status == "failed"
        assert result.error == "Script failed with exit code 3"
        assert result.error_payload == {"stderr": "boom\n"}

    @pytest.mark.asyncio
    async def test_backend_field_preserved_on_failure(self) -> None:
        """B2: web_searcher's own producer-specific field survives too,
        not just stderr."""
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(
                json.dumps({"error": "authentication_failed", "backend": "tavily"}),
                None,
                False,
            )
        )
        agent = ScriptAgentConfig(name="searcher", script="web_searcher.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["searcher"]
        assert result.error_payload == {"backend": "tavily"}

    @pytest.mark.asyncio
    async def test_no_error_payload_when_nothing_beyond_error_and_success(
        self,
    ) -> None:
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(json.dumps({"success": False}), None, False)
        )
        agent = ScriptAgentConfig(name="worker", script="worker.py")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        assert results["worker"].error_payload is None

    @pytest.mark.asyncio
    async def test_llm_agent_response_is_never_inspected_for_failure_shape(
        self,
    ) -> None:
        """An LLM agent's own text response happening to contain the word
        "error" must not be reinterpreted as agent failure — the B2
        contract is script-agent-only."""
        dispatcher = _make_dispatcher()
        coordinator = cast(AsyncMock, dispatcher._execution_coordinator)
        coordinator.execute_agent_with_timeout = AsyncMock(
            return_value=(
                json.dumps({"error": "not a failure, just prose"}),
                None,
                False,
            )
        )
        agent = LlmAgentConfig(name="writer", model_profile="local")

        results = await dispatcher.execute_agents_in_phase([agent], "test input")

        result = results["writer"]
        assert result.status == "success"
