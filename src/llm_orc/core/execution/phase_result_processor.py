"""Phase result processing for ensemble execution."""

import logging
from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.agent_request_processor import AgentRequestProcessor
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig

logger = logging.getLogger(__name__)


class PhaseResultProcessor:
    """Processes phase execution results, collecting usage and metadata.

    Handles storing agent results, processing agent requests from script
    output, collecting usage metrics, and emitting phase completion events.
    """

    def __init__(
        self,
        agent_request_processor: AgentRequestProcessor,
        usage_collector: UsageCollector,
        emit_event_fn: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self._agent_request_processor = agent_request_processor
        self._usage_collector = usage_collector
        self._emit_event = emit_event_fn
        self._ensemble_metadata: dict[str, Any] = {}

    def get_ensemble_metadata(self) -> dict[str, Any]:
        """Return collected ensemble metadata."""
        return self._ensemble_metadata

    async def process_phase_results(
        self,
        phase_results: dict[str, Any],
        results_dict: dict[str, Any],
        phase_agents: list[AgentConfig],
    ) -> bool:
        """Process parallel execution results and return if any errors occurred."""
        has_errors = False
        processed_agent_requests: list[dict[str, Any]] = []

        agent_configs = {agent.name: agent for agent in phase_agents}

        for agent_name, agent_result in phase_results.items():
            self._store_agent_result(results_dict, agent_name, agent_result)

            if agent_result["status"] == "failed":
                has_errors = True

            if agent_result["status"] == "success":
                await self._process_successful_agent_result(
                    agent_result,
                    agent_name,
                    results_dict,
                    processed_agent_requests,
                    phase_agents,
                    agent_configs,
                )

        self._store_agent_requests_metadata(processed_agent_requests)

        return has_errors

    def emit_phase_completed_event(
        self,
        phase_index: int,
        phase_agents: list[AgentConfig],
        results_dict: dict[str, Any],
    ) -> None:
        """Emit phase completion event with success/failure counts."""
        successful_agents = [
            a
            for a in phase_agents
            if results_dict.get(a.name, {}).get("status") == "success"
        ]
        failed_agents = [
            a
            for a in phase_agents
            if results_dict.get(a.name, {}).get("status") == "failed"
        ]

        self._emit_event(
            "phase_completed",
            {
                "phase_index": phase_index,
                "successful_agents": len(successful_agents),
                "failed_agents": len(failed_agents),
            },
        )

    def _store_agent_result(
        self,
        results_dict: dict[str, Any],
        agent_name: str,
        agent_result: dict[str, Any],
    ) -> None:
        """Store agent result in results dictionary."""
        results_dict[agent_name] = {
            "response": agent_result.get("response"),
            "status": agent_result["status"],
        }
        if agent_result["status"] == "failed":
            results_dict[agent_name]["error"] = agent_result["error"]

    async def _process_successful_agent_result(
        self,
        agent_result: dict[str, Any],
        agent_name: str,
        results_dict: dict[str, Any],
        processed_agent_requests: list[dict[str, Any]],
        phase_agents: list[AgentConfig],
        agent_configs: dict[str, AgentConfig],
    ) -> None:
        """Process successful agent result for requests and usage."""
        response = agent_result.get("response")
        if response and isinstance(response, str):
            await self._process_agent_requests(
                response,
                agent_name,
                results_dict,
                processed_agent_requests,
                phase_agents,
            )

        if agent_result["model_instance"] is not None:
            agent_config = agent_configs.get(agent_name)
            model_profile = (
                agent_config.model_profile
                if agent_config and isinstance(agent_config, LlmAgentConfig)
                else "unknown"
            )
            self._usage_collector.collect_agent_usage(
                agent_name, agent_result["model_instance"], model_profile
            )

    async def _process_agent_requests(
        self,
        response: str,
        agent_name: str,
        results_dict: dict[str, Any],
        processed_agent_requests: list[dict[str, Any]],
        phase_agents: list[AgentConfig],
    ) -> None:
        """Process agent requests from script output."""
        try:
            processed_result = (
                await self._agent_request_processor.process_script_output_with_requests(
                    response, agent_name, phase_agents
                )
            )

            if processed_result.get("agent_requests"):
                processed_agent_requests.extend(processed_result["agent_requests"])

            results_dict[agent_name]["agent_requests"] = processed_result.get(
                "agent_requests", []
            )

        except Exception:
            logger.warning(
                "Failed to process agent requests for %r", agent_name, exc_info=True
            )

    def _store_agent_requests_metadata(
        self, processed_agent_requests: list[dict[str, Any]]
    ) -> None:
        """Store processed agent requests in ensemble metadata."""
        if processed_agent_requests:
            self._ensemble_metadata["processed_agent_requests"] = (
                processed_agent_requests
            )
