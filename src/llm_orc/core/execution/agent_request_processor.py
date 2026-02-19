"""Agent request processor for dynamic parameter generation (ADR-001).

This module processes AgentRequest objects from ScriptAgentOutput to enable
inter-agent communication and dynamic parameter generation during ensemble execution.
"""

import json
from typing import Any

from llm_orc.core.execution.dependency_resolver import DependencyResolver
from llm_orc.schemas.script_agent import AgentRequest


class AgentRequestProcessor:
    """Processes AgentRequest objects for dynamic parameter generation."""

    def __init__(self, dependency_resolver: DependencyResolver) -> None:
        """Initialize with dependency resolver for agent coordination.

        Args:
            dependency_resolver: DependencyResolver instance for agent coordination
        """
        self._dependency_resolver = dependency_resolver

    def generate_dynamic_parameters(
        self, agent_request: AgentRequest, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate dynamic parameters for target agent based on AgentRequest.

        Args:
            agent_request: AgentRequest with target agent and parameters
            context: Optional context for parameter generation

        Returns:
            Dictionary of parameters for the target agent
        """
        if context is None:
            context = {}

        # For now, return the parameters as-is
        # Future enhancement: apply context-based transformations
        return agent_request.parameters

    def coordinate_agent_execution(
        self,
        agent_requests: list[AgentRequest],
        results_dict: dict[str, Any],
        phase_agents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Coordinate agent execution based on agent requests.

        Args:
            agent_requests: List of AgentRequest objects to process
            results_dict: Current execution results
            phase_agents: Available agents in current phase

        Returns:
            Updated agent configurations or execution plan
        """
        coordinated_agents = []

        for request in agent_requests:
            # Find matching agent in phase_agents
            for agent_config in phase_agents:
                agent_name = agent_config.get("name", "")
                agent_type = agent_config.get("type", "")

                # Match by type (for now, simple type matching)
                if (
                    request.target_agent_type in agent_name
                    or request.target_agent_type == agent_type
                ):
                    # Create updated agent config with dynamic parameters
                    updated_config = agent_config.copy()
                    updated_config["dynamic_parameters"] = (
                        self.generate_dynamic_parameters(request)
                    )
                    coordinated_agents.append(updated_config)
                    break

        return coordinated_agents

    async def process_script_output_with_requests(
        self,
        script_response: str,
        source_agent: str,
        current_phase_agents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process script output containing AgentRequest objects.

        Args:
            script_response: JSON response from script agent
            source_agent: Name of the source agent
            current_phase_agents: Available agents in current phase

        Returns:
            Processed result with agent_requests extracted and processed
        """
        try:
            response_data = json.loads(script_response)

            # Extract agent_requests if present
            agent_requests_data = response_data.get("agent_requests", [])
            agent_requests = [
                AgentRequest(**request_data) for request_data in agent_requests_data
            ]

            # Coordinate execution if requests exist
            coordinated_agents = []
            if agent_requests:
                coordinated_agents = self.coordinate_agent_execution(
                    agent_requests, {}, current_phase_agents
                )

            return {
                "source_agent": source_agent,
                "response_data": response_data,
                "agent_requests": agent_requests_data,
                "coordinated_agents": coordinated_agents,
            }

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to process script output JSON: {str(e)}") from e

