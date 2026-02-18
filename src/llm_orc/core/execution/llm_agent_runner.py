"""LLM agent runner extracted from EnsembleExecutor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_orc.core.config.config_manager import (
    ConfigurationManager,
)
from llm_orc.core.config.roles import RoleDefinition
from llm_orc.core.execution.orchestration import Agent
from llm_orc.core.execution.usage_collector import (
    UsageCollector,
)
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.models.base import ModelInterface


class LlmAgentRunner:
    """Runs LLM agents with fallback and resource monitoring."""

    def __init__(
        self,
        model_factory: ModelFactory,
        config_manager: ConfigurationManager,
        usage_collector: UsageCollector,
        emit_event: Callable[[str, dict[str, Any]], None],
        classify_failure: Callable[[str], str],
    ) -> None:
        self._model_factory = model_factory
        self._config_manager = config_manager
        self._usage_collector = usage_collector
        self._emit_event = emit_event
        self._classify_failure = classify_failure

    async def execute(
        self,
        agent_config: dict[str, Any],
        input_data: str,
    ) -> tuple[str, ModelInterface | None]:
        """Execute LLM agent with fallback handling."""
        agent_name = agent_config["name"]

        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            role = await self._load_role_from_config(agent_config)
            model = await self._load_model_with_fallback(agent_config)
            agent = Agent(agent_name, role, model)

            self._usage_collector.sample_agent_resources(agent_name)

            try:
                response = await agent.respond_to_message(input_data)
                self._usage_collector.sample_agent_resources(agent_name)
                return response, model
            except Exception as e:
                return await self._handle_runtime_fallback(
                    agent_config, role, input_data, e
                )
        finally:
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _load_model_with_fallback(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Load model with fallback handling."""
        try:
            return await self._model_factory.load_model_from_agent_config(agent_config)
        except Exception as model_loading_error:
            return await self._handle_model_loading_fallback(
                agent_config, model_loading_error
            )

    async def _handle_model_loading_fallback(
        self,
        agent_config: dict[str, Any],
        model_loading_error: Exception,
    ) -> ModelInterface:
        """Handle model loading failure with fallback."""
        fallback_model = await self._model_factory.get_fallback_model(
            context=f"agent_{agent_config['name']}",
            original_profile=agent_config.get("model_profile"),
        )
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")

        failure_type = self._classify_failure(str(model_loading_error))
        self._emit_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config["name"],
                "failure_type": failure_type,
                "original_error": str(model_loading_error),
                "original_model_profile": agent_config.get("model_profile", "unknown"),
                "fallback_model_profile": None,
                "fallback_model_name": fallback_model_name,
            },
        )
        return fallback_model

    async def _handle_runtime_fallback(
        self,
        agent_config: dict[str, Any],
        role: RoleDefinition,
        input_data: str,
        error: Exception,
    ) -> tuple[str, ModelInterface]:
        """Handle runtime failure with fallback model."""
        fallback_model = await self._model_factory.get_fallback_model(
            context=f"agent_{agent_config['name']}"
        )
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")

        failure_type = self._classify_failure(str(error))
        self._emit_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config["name"],
                "failure_type": failure_type,
                "original_error": str(error),
                "original_model_profile": agent_config.get("model_profile", "unknown"),
                "fallback_model_profile": None,
                "fallback_model_name": fallback_model_name,
            },
        )

        fallback_agent = Agent(agent_config["name"], role, fallback_model)

        try:
            response = await fallback_agent.respond_to_message(input_data)
            self._emit_fallback_success_event(
                agent_config["name"],
                fallback_model,
                response,
            )
            return response, fallback_model
        except Exception as fallback_error:
            self._emit_fallback_failure_event(
                agent_config["name"],
                fallback_model_name,
                fallback_error,
            )
            raise fallback_error

    def _emit_fallback_success_event(
        self,
        agent_name: str,
        fallback_model: ModelInterface,
        response: str,
    ) -> None:
        """Emit fallback success event."""
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")
        response_preview = response[:100] + "..." if len(response) > 100 else response
        self._emit_event(
            "agent_fallback_completed",
            {
                "agent_name": agent_name,
                "fallback_model_name": fallback_model_name,
                "response_preview": response_preview,
            },
        )

    def _emit_fallback_failure_event(
        self,
        agent_name: str,
        fallback_model_name: str,
        fallback_error: Exception,
    ) -> None:
        """Emit fallback failure event."""
        fallback_failure_type = self._classify_failure(str(fallback_error))
        self._emit_event(
            "agent_fallback_failed",
            {
                "agent_name": agent_name,
                "failure_type": fallback_failure_type,
                "fallback_error": str(fallback_error),
                "fallback_model_name": fallback_model_name,
            },
        )

    async def _load_role_from_config(
        self, agent_config: dict[str, Any]
    ) -> RoleDefinition:
        """Load a role definition from agent configuration."""
        agent_name = agent_config["name"]

        enhanced_config = await self._resolve_model_profile_to_config(agent_config)

        if "system_prompt" in enhanced_config:
            prompt = enhanced_config["system_prompt"]
        else:
            prompt = f"You are a {agent_name}. Provide helpful analysis."

        return RoleDefinition(name=agent_name, prompt=prompt)

    async def _resolve_model_profile_to_config(
        self, agent_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve model profile and merge with agent config."""
        enhanced_config = agent_config.copy()

        if "model_profile" in agent_config:
            profiles = self._config_manager.get_model_profiles()
            profile_name = agent_config["model_profile"]
            if profile_name in profiles:
                profile_config = profiles[profile_name]
                enhanced_config = {
                    **profile_config,
                    **agent_config,
                }

        return enhanced_config
