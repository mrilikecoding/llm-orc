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
from llm_orc.core.models.model_factory import (
    ModelConfigurationError,
    ModelFactory,
)
from llm_orc.models.base import ModelInterface
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig


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
        agent_config: AgentConfig,
        input_data: str,
    ) -> tuple[str, ModelInterface | None, bool]:
        """Execute LLM agent with fallback handling.

        Returns:
            Tuple of (response, model_instance, model_substituted).
            model_substituted is True when a fallback model was used
            instead of the originally configured model.
        """
        agent_name = agent_config.name

        self._usage_collector.start_agent_resource_monitoring(agent_name)

        try:
            role = await self._load_role_from_config(agent_config)
            model, substituted = await self._load_model_with_fallback(agent_config)
            agent = Agent(agent_name, role, model)

            self._usage_collector.sample_agent_resources(agent_name)

            try:
                response = await agent.respond_to_message(input_data)
                self._usage_collector.sample_agent_resources(agent_name)
                return response, model, substituted
            except Exception as e:
                return await self._handle_runtime_fallback(
                    agent_config, role, input_data, e
                )
        finally:
            self._usage_collector.finalize_agent_resource_monitoring(agent_name)

    async def _load_model_with_fallback(
        self, agent_config: AgentConfig
    ) -> tuple[ModelInterface, bool]:
        """Load model with fallback handling.

        Returns:
            Tuple of (model, substituted) where substituted is True
            when a fallback model was used.
        """
        try:
            config_dict = agent_config.model_dump()
            model = await self._model_factory.load_model_from_agent_config(config_dict)
            return model, False
        except ModelConfigurationError:
            # A config error (e.g. think on a provider that doesn't
            # support it) is never fallback-eligible — substituting a
            # working model would report success on a broken config.
            raise
        except Exception as model_loading_error:
            fallback = await self._handle_model_loading_fallback(
                agent_config, model_loading_error
            )
            return fallback, True

    def _fallback_chain_inputs(
        self, agent_config: AgentConfig
    ) -> tuple[str | None, str | None]:
        """Extract the model_profile and agent-level fallback_model_profile
        that drive the explicit fallback chain."""
        if not isinstance(agent_config, LlmAgentConfig):
            return None, None
        return agent_config.model_profile, agent_config.fallback_model_profile

    def _generation_params(self, agent_config: AgentConfig) -> dict[str, Any]:
        """The agent-level generation params (SF2) carried into whichever
        fallback profile loads, the same way they reach a primary load."""
        if not isinstance(agent_config, LlmAgentConfig):
            return {
                "temperature": None,
                "max_tokens": None,
                "agent_options": None,
                "response_format": None,
            }
        return {
            "temperature": agent_config.temperature,
            "max_tokens": agent_config.max_tokens,
            "agent_options": agent_config.options,
            "response_format": agent_config.response_format,
        }

    async def _handle_model_loading_fallback(
        self,
        agent_config: AgentConfig,
        model_loading_error: Exception,
    ) -> ModelInterface:
        """Handle model loading failure with fallback."""
        model_profile, agent_fallback_profile = self._fallback_chain_inputs(
            agent_config
        )

        try:
            (
                fallback_model,
                fallback_profile_name,
            ) = await self._model_factory.get_fallback_model(
                context=f"agent_{agent_config.name}",
                original_profile=model_profile,
                agent_fallback_profile=agent_fallback_profile,
                **self._generation_params(agent_config),
            )
        except Exception as fallback_unavailable:
            raise model_loading_error from fallback_unavailable

        fallback_model_name = getattr(fallback_model, "model_name", "unknown")

        failure_type = self._classify_failure(str(model_loading_error))
        self._emit_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config.name,
                "failure_type": failure_type,
                "original_error": str(model_loading_error),
                "original_model_profile": model_profile or "unknown",
                "fallback_model_profile": fallback_profile_name,
                "fallback_model_name": fallback_model_name,
            },
        )
        return fallback_model

    async def _handle_runtime_fallback(
        self,
        agent_config: AgentConfig,
        role: RoleDefinition,
        input_data: str,
        error: Exception,
    ) -> tuple[str, ModelInterface, bool]:
        """Handle runtime failure by walking the explicit fallback chain.

        Each candidate that loads is tried in turn; a runtime failure on
        one candidate continues to the next hop instead of giving up
        after a single try. Exhausted (or nothing configured) re-raises
        the ORIGINAL runtime error, chained to the last failure seen.
        """
        model_profile, agent_fallback_profile = self._fallback_chain_inputs(
            agent_config
        )
        gen_params = self._generation_params(agent_config)

        last_error: Exception = error
        try:
            async for (
                fallback_model,
                fallback_profile_name,
            ) in self._model_factory.iter_fallback_chain(
                original_profile=model_profile,
                agent_fallback_profile=agent_fallback_profile,
                **gen_params,
            ):
                response, hop_error = await self._try_fallback_hop(
                    agent_config,
                    role,
                    input_data,
                    last_error,
                    model_profile,
                    fallback_model,
                    fallback_profile_name,
                )
                if hop_error is None:
                    assert response is not None
                    return response, fallback_model, True
                last_error = hop_error
        except Exception as chain_error:
            last_error = chain_error

        if last_error is error:
            raise error from ValueError(
                f"No fallback_model_profile configured for agent_"
                f"{agent_config.name}; fallback chain exhausted"
            )
        if isinstance(last_error, ModelConfigurationError):
            # A config error on a fallback hop (addendum 2026-09-23 NIT)
            # is an author mistake, not a routine "this candidate didn't
            # work" — it must be the recorded error (AgentDispatcher
            # records str(the raised exception)), not buried in
            # __cause__ behind the original runtime failure that
            # triggered the fallback walk in the first place.
            raise last_error from error
        raise error from last_error

    async def _try_fallback_hop(
        self,
        agent_config: AgentConfig,
        role: RoleDefinition,
        input_data: str,
        prior_error: Exception,
        model_profile: str | None,
        fallback_model: ModelInterface,
        fallback_profile_name: str,
    ) -> tuple[str, None] | tuple[None, Exception]:
        """Try generating a response with one fallback-chain candidate.

        Returns (response, None) on success, or (None, error) on
        failure — having already emitted the corresponding event — so
        the caller can continue to the next hop with the real error.
        """
        fallback_model_name = getattr(fallback_model, "model_name", "unknown")
        failure_type = self._classify_failure(str(prior_error))
        self._emit_event(
            "agent_fallback_started",
            {
                "agent_name": agent_config.name,
                "failure_type": failure_type,
                "original_error": str(prior_error),
                "original_model_profile": model_profile or "unknown",
                "fallback_model_profile": fallback_profile_name,
                "fallback_model_name": fallback_model_name,
            },
        )

        fallback_agent = Agent(agent_config.name, role, fallback_model)
        try:
            response = await fallback_agent.respond_to_message(input_data)
        except Exception as fallback_error:
            self._emit_fallback_failure_event(
                agent_config.name,
                fallback_model_name,
                fallback_error,
            )
            return None, fallback_error

        self._emit_fallback_success_event(
            agent_config.name,
            fallback_model,
            response,
        )
        return response, None

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

    async def _load_role_from_config(self, agent_config: AgentConfig) -> RoleDefinition:
        """Load a role definition from agent configuration."""
        agent_name = agent_config.name

        enhanced_config = await self._resolve_model_profile_to_config(agent_config)

        configured = enhanced_config.get("system_prompt")
        # key-presence, not truthiness: an explicitly-empty system_prompt
        # suppresses the generic fallback (issue #95)
        prompt = (
            configured
            if configured is not None
            else f"You are a {agent_name}. Provide helpful analysis."
        )

        return RoleDefinition(name=agent_name, prompt=prompt)

    async def _resolve_model_profile_to_config(
        self, agent_config: AgentConfig
    ) -> dict[str, Any]:
        """Resolve model profile and merge with agent config.

        Returns a plain dict — this is the boundary between Pydantic
        models and downstream code that still expects dicts (e.g.
        ModelFactory).
        """
        config_dict = agent_config.model_dump()

        if isinstance(agent_config, LlmAgentConfig) and agent_config.model_profile:
            profiles = self._config_manager.get_model_profiles()
            profile_name = agent_config.model_profile
            if profile_name in profiles:
                profile_config = profiles[profile_name]
                # Merge profile first, then let explicit (non-None) agent values
                # override. None values in config_dict mean "not set by agent".
                agent_overrides = {
                    k: v for k, v in config_dict.items() if v is not None
                }
                config_dict = {**config_dict, **profile_config, **agent_overrides}

        return config_dict
