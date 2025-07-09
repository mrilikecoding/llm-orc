"""Ensemble execution with agent coordination."""

import asyncio
import time
from typing import Any

from llm_orc.ensemble_config import EnsembleConfig
from llm_orc.models import ModelInterface, OllamaModel
from llm_orc.orchestration import Agent
from llm_orc.roles import RoleDefinition
from llm_orc.script_agent import ScriptAgent


class EnsembleExecutor:
    """Executes ensembles of agents and coordinates their responses."""

    async def execute(self, config: EnsembleConfig, input_data: str) -> dict[str, Any]:
        """Execute an ensemble and return structured results."""
        start_time = time.time()

        # Initialize result structure
        result: dict[str, Any] = {
            "ensemble": config.name,
            "status": "running",
            "input": {"data": input_data},
            "results": {},
            "synthesis": None,
            "metadata": {"agents_used": len(config.agents), "started_at": start_time},
        }

        # Ensure results is properly typed
        results_dict: dict[str, Any] = result["results"]

        # Execute agents in phases: script agents first, then LLM agents
        has_errors = False
        agent_usage: dict[str, Any] = {}
        context_data = {}

        # Phase 1: Execute script agents to gather context
        script_agents = [a for a in config.agents if a.get("type") == "script"]
        for agent_config in script_agents:
            try:
                timeout = agent_config.get("timeout_seconds") or config.coordinator.get(
                    "timeout_seconds"
                )
                agent_result, model_instance = await self._execute_agent_with_timeout(
                    agent_config, input_data, timeout
                )
                results_dict[agent_config["name"]] = {
                    "response": agent_result,
                    "status": "success",
                }
                # Store script results as context for LLM agents
                context_data[agent_config["name"]] = agent_result
            except Exception as e:
                results_dict[agent_config["name"]] = {
                    "error": str(e),
                    "status": "failed",
                }
                has_errors = True

        # Phase 2: Execute LLM agents with context from script agents
        llm_agents = [a for a in config.agents if a.get("type") != "script"]

        # Prepare enhanced input for LLM agents
        enhanced_input = input_data
        if context_data:
            context_text = "\n\n".join(
                [f"=== {name} ===\n{data}" for name, data in context_data.items()]
            )
            enhanced_input = f"{input_data}\n\n{context_text}"

        # Execute LLM agents concurrently with enhanced input
        agent_tasks = []
        for agent_config in llm_agents:
            timeout = agent_config.get("timeout_seconds") or config.coordinator.get(
                "timeout_seconds"
            )
            task = self._execute_agent_with_timeout(
                agent_config, enhanced_input, timeout
            )
            agent_tasks.append((agent_config["name"], task))

        # Wait for all LLM agents to complete
        for agent_name, task in agent_tasks:
            try:
                agent_result, model_instance = await task
                results_dict[agent_name] = {
                    "response": agent_result,
                    "status": "success",
                }
                # Collect usage metrics (only for LLM agents)
                if model_instance is not None:
                    usage = model_instance.get_last_usage()
                    if usage:
                        agent_usage[agent_name] = usage
            except Exception as e:
                results_dict[agent_name] = {"error": str(e), "status": "failed"}
                has_errors = True

        # Synthesize results if coordinator is configured
        synthesis_usage = None
        if config.coordinator.get("synthesis_prompt"):
            try:
                synthesis_timeout = config.coordinator.get("synthesis_timeout_seconds")
                synthesis_result = await self._synthesize_results_with_timeout(
                    config, results_dict, synthesis_timeout
                )
                synthesis, synthesis_model = synthesis_result
                result["synthesis"] = synthesis
                synthesis_usage = synthesis_model.get_last_usage()
            except Exception as e:
                result["synthesis"] = f"Synthesis failed: {str(e)}"
                has_errors = True

        # Calculate usage totals
        usage_summary = self._calculate_usage_summary(agent_usage, synthesis_usage)

        # Finalize result
        end_time = time.time()
        result["status"] = "completed_with_errors" if has_errors else "completed"
        metadata_dict: dict[str, Any] = result["metadata"]
        metadata_dict["duration"] = f"{(end_time - start_time):.2f}s"
        metadata_dict["completed_at"] = end_time
        metadata_dict["usage"] = usage_summary

        return result

    async def _execute_agent(
        self, agent_config: dict[str, Any], input_data: str
    ) -> tuple[str, ModelInterface | None]:
        """Execute a single agent and return its response and model instance."""
        agent_type = agent_config.get("type", "llm")

        if agent_type == "script":
            # Execute script agent
            script_agent = ScriptAgent(agent_config["name"], agent_config)
            response = await script_agent.execute(input_data)
            return response, None  # Script agents don't have model instances
        else:
            # Execute LLM agent
            # Load role and model for this agent
            role = await self._load_role(agent_config["role"])
            model = await self._load_model(agent_config["model"])

            # Create agent
            agent = Agent(agent_config["name"], role, model)

            # Generate response
            response = await agent.respond_to_message(input_data)
            return response, model

    async def _load_role(self, role_name: str) -> RoleDefinition:
        """Load a role definition."""
        # For now, create a simple role
        # TODO: Load from role configuration files
        return RoleDefinition(
            name=role_name, prompt=f"You are a {role_name}. Provide helpful analysis."
        )

    async def _load_model(self, model_name: str) -> ModelInterface:
        """Load a model interface."""
        # For now, just create a mock model for testing
        # TODO: Implement proper model loading based on configuration
        if model_name.startswith("mock"):
            # Return a mock model that will be replaced in tests
            from unittest.mock import AsyncMock

            mock = AsyncMock(spec=ModelInterface)
            mock.generate_response.return_value = f"Response from {model_name}"
            return mock
        else:
            # Default to Ollama for now
            return OllamaModel(model_name="llama3")

    async def _synthesize_results(
        self, config: EnsembleConfig, agent_results: dict[str, Any]
    ) -> tuple[str, ModelInterface]:
        """Synthesize results from all agents."""
        synthesis_model = await self._get_synthesis_model()

        # Prepare synthesis prompt with agent results
        results_text = ""
        for agent_name, result in agent_results.items():
            if result["status"] == "success":
                results_text += f"\n{agent_name}: {result['response']}\n"
            else:
                results_text += f"\n{agent_name}: [Error: {result['error']}]\n"

        synthesis_prompt = (
            f"{config.coordinator['synthesis_prompt']}\n\nAgent Results:{results_text}"
        )

        # Generate synthesis
        response = await synthesis_model.generate_response(
            message="Please synthesize these results", role_prompt=synthesis_prompt
        )

        return response, synthesis_model

    async def _get_synthesis_model(self) -> ModelInterface:
        """Get model for synthesis."""
        # For now, use Ollama
        # TODO: Make this configurable
        return OllamaModel(model_name="llama3")

    def _calculate_usage_summary(
        self, agent_usage: dict[str, Any], synthesis_usage: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Calculate aggregated usage summary."""
        summary = {
            "agents": agent_usage,
            "totals": {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "total_duration_ms": 0,
                "agents_count": len(agent_usage),
            },
        }

        # Aggregate agent usage
        for usage in agent_usage.values():
            summary["totals"]["total_tokens"] += usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += usage.get("input_tokens", 0)
            summary["totals"]["total_output_tokens"] += usage.get("output_tokens", 0)
            summary["totals"]["total_cost_usd"] += usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += usage.get("duration_ms", 0)

        # Add synthesis usage
        if synthesis_usage:
            summary["synthesis"] = synthesis_usage
            summary["totals"]["total_tokens"] += synthesis_usage.get("total_tokens", 0)
            summary["totals"]["total_input_tokens"] += synthesis_usage.get(
                "input_tokens", 0
            )
            summary["totals"]["total_output_tokens"] += synthesis_usage.get(
                "output_tokens", 0
            )
            summary["totals"]["total_cost_usd"] += synthesis_usage.get("cost_usd", 0.0)
            summary["totals"]["total_duration_ms"] += synthesis_usage.get(
                "duration_ms", 0
            )

        return summary

    async def _execute_agent_with_timeout(
        self, agent_config: dict[str, Any], input_data: str, timeout_seconds: int | None
    ) -> tuple[str, ModelInterface | None]:
        """Execute an agent with optional timeout."""
        if timeout_seconds is None:
            # No timeout specified, execute normally
            return await self._execute_agent(agent_config, input_data)

        try:
            return await asyncio.wait_for(
                self._execute_agent(agent_config, input_data), timeout=timeout_seconds
            )
        except TimeoutError as e:
            raise Exception(
                f"Agent execution timed out after {timeout_seconds} seconds"
            ) from e

    async def _synthesize_results_with_timeout(
        self,
        config: EnsembleConfig,
        agent_results: dict[str, Any],
        timeout_seconds: int | None,
    ) -> tuple[str, ModelInterface]:
        """Synthesize results with optional timeout."""
        if timeout_seconds is None:
            # No timeout specified, execute normally
            return await self._synthesize_results(config, agent_results)

        try:
            return await asyncio.wait_for(
                self._synthesize_results(config, agent_results), timeout=timeout_seconds
            )
        except TimeoutError as e:
            raise Exception(
                f"Synthesis timed out after {timeout_seconds} seconds"
            ) from e
