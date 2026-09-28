"""Unit tests for PhaseResultProcessor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_orc.core.execution.phases.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.phases.phase_result_processor import PhaseResultProcessor
from llm_orc.core.execution.result_types import AgentResult
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


def _make_processor() -> tuple[PhaseResultProcessor, AsyncMock]:
    mock_request_processor = MagicMock(spec=AgentRequestProcessor)
    mock_request_processor.process_script_output_with_requests = AsyncMock(
        return_value={
            "source_agent": "agent1",
            "response_data": {},
            "agent_requests": [],
            "coordinated_agents": [],
        }
    )
    mock_usage_collector = MagicMock(spec=UsageCollector)
    events: list[Any] = []
    processor = PhaseResultProcessor(
        agent_request_processor=mock_request_processor,
        usage_collector=mock_usage_collector,
        emit_event_fn=lambda event_type, data: events.append((event_type, data)),
    )
    return processor, mock_request_processor.process_script_output_with_requests


def _llm_config(name: str = "llm-agent") -> LlmAgentConfig:
    return LlmAgentConfig(name=name, model_profile="some-profile")


def _script_config(name: str = "script-agent") -> ScriptAgentConfig:
    return ScriptAgentConfig(name=name, script="some_script")


def _success_result(response: str) -> AgentResult:
    result = MagicMock(spec=AgentResult)
    result.status = "success"
    result.response = response
    result.model_instance = None
    result.model_substituted = False
    result.error = None
    return result


class TestOutcomeStampingAndPayloadIsolation:
    """fail-closed-composition addendum 2026-09-23: PhaseResultProcessor
    is the one stamping point for LLM/script/ensemble/dispatch/loop node
    results, and the BLOCKER this closes — a failed script's payload
    landing in its own namespace, never merged over the record."""

    @pytest.mark.asyncio
    async def test_failed_result_stamps_failed_outcome_and_has_errors(self) -> None:
        processor, _ = _make_processor()
        agent_config = _script_config("worker")
        result = AgentResult(status="failed", error="boom")
        results_dict: dict[str, Any] = {}

        await processor.process_phase_results(
            {"worker": result}, results_dict=results_dict, phase_agents=[agent_config]
        )

        assert results_dict["worker"]["outcome"] == "failed"
        assert results_dict["worker"]["has_errors"] is True

    @pytest.mark.asyncio
    async def test_success_result_stamps_succeeded_outcome(self) -> None:
        processor, _ = _make_processor()
        agent_config = _script_config("worker")
        results_dict: dict[str, Any] = {}

        await processor.process_phase_results(
            {"worker": _success_result("ok")},
            results_dict=results_dict,
            phase_agents=[agent_config],
        )

        assert results_dict["worker"]["outcome"] == "succeeded"
        assert results_dict["worker"]["has_errors"] is False

    @pytest.mark.asyncio
    async def test_failed_scripts_colliding_payload_never_overwrites_the_record(
        self,
    ) -> None:
        """The exact shape from the addendum: a script prints {"success":
        false, "error": "x", "status": "success", "response":
        "FABRICATED"}; AgentDispatcher already strips error/success and
        hands the rest as payload={"status": "success", "response":
        "FABRICATED dossier"}. The stored record must still say failed."""
        processor, _ = _make_processor()
        agent_config = _script_config("worker")
        result = AgentResult(
            status="failed",
            error="search backend down",
            payload={"status": "success", "response": "FABRICATED dossier"},
        )
        results_dict: dict[str, Any] = {}

        await processor.process_phase_results(
            {"worker": result}, results_dict=results_dict, phase_agents=[agent_config]
        )

        stored = results_dict["worker"]
        assert stored["status"] == "failed"
        assert stored["response"] is None
        assert stored["error"] == "search backend down"
        assert stored["payload"] == {
            "status": "success",
            "response": "FABRICATED dossier",
        }
        assert stored["outcome"] == "failed"
        assert stored["has_errors"] is True


class TestProcessAgentRequestsGuard:
    """process_script_output_with_requests is only called for script agents."""

    @pytest.mark.asyncio
    async def test_skips_json_parsing_for_llm_agent_plain_text(self) -> None:
        """LLM agent plain-text response does not trigger JSON parsing."""
        processor, mock_parse = _make_processor()
        agent_name = "llm-agent"
        agent_config = _llm_config(agent_name)
        phase_results = {agent_name: _success_result("Authentication working")}

        await processor.process_phase_results(
            phase_results,
            results_dict={},
            phase_agents=[agent_config],
        )

        mock_parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_json_for_script_agent(self) -> None:
        """Script agent response triggers JSON parsing."""
        processor, mock_parse = _make_processor()
        agent_name = "script-agent"
        agent_config = _script_config(agent_name)
        phase_results = {agent_name: _success_result('{"output": "done"}')}

        await processor.process_phase_results(
            phase_results,
            results_dict={},
            phase_agents=[agent_config],
        )

        mock_parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_parsing_for_empty_response(self) -> None:
        """Empty response does not trigger JSON parsing even for script agents."""
        processor, mock_parse = _make_processor()
        agent_name = "script-agent"
        agent_config = _script_config(agent_name)
        phase_results = {agent_name: _success_result("")}

        await processor.process_phase_results(
            phase_results,
            results_dict={},
            phase_agents=[agent_config],
        )

        mock_parse.assert_not_called()
