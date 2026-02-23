"""Unit tests for PhaseResultProcessor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.phase_result_processor import PhaseResultProcessor
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
