"""Model factory for creating model instances based on configuration."""

import logging
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.models.anthropic import (
    ClaudeCLIModel,
    ClaudeModel,
)
from llm_orc.models.base import ModelInterface
from llm_orc.models.mock import MockModel
from llm_orc.models.openai_compat import OpenAICompatibleModel
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig

logger = logging.getLogger(__name__)


class ModelConfigurationError(ValueError):
    """An author config error — never a fallback-eligible failure.

    Distinguishes a structural mistake in the ensemble YAML (e.g.
    ``think`` set on a provider that doesn't speak llama-server's
    chat-template convention) from a runtime/availability failure
    (bad credentials, unreachable host) that the fallback chain exists
    to route around. Still a ``ValueError`` so existing ``except
    ValueError`` call sites are unaffected; callers that must not let
    a config error masquerade as a successful fallback substitution
    check for this type explicitly (``LlmAgentRunner``).
    """


class ModelFactory:
    """Factory for creating model instances based on configuration."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        credential_storage: CredentialStorage,
        *,
        execution_id: str | None = None,
    ) -> None:
        """Initialize the model factory.

        Args:
            config_manager: Configuration manager instance
            credential_storage: Credential storage instance
            execution_id: Stable identifier for the top-level ensemble
                execution this factory serves. Generated when omitted.
                Child executors share their parent's ModelFactory
                instance (ExecutorFactory.create_child_executor), so
                this id is naturally shared by every agent in the
                execution tree, including fan-out instances.
        """
        self._config_manager = config_manager
        self._credential_storage = credential_storage
        self.execution_id = execution_id or uuid.uuid4().hex

    async def load_model_from_agent_config(
        self, agent_config: dict[str, Any]
    ) -> ModelInterface:
        """Load a model based on agent configuration.

        Configuration can specify model_profile or model+provider.

        Args:
            agent_config: Agent configuration dictionary

        Returns:
            Configured model interface

        Raises:
            ValueError: If configuration is invalid
        """
        # Extract generation parameters from agent config
        temperature: float | None = agent_config.get("temperature")
        max_tokens: int | None = agent_config.get("max_tokens")
        agent_options: dict[str, Any] | None = agent_config.get("options")
        response_format: str | dict[str, Any] | None = agent_config.get(
            "response_format"
        )

        # Check if model_profile is specified (takes precedence)
        # Use .get() truthy check: model_dump() includes None values as keys
        if agent_config.get("model_profile"):
            return await self._load_profile(
                agent_config["model_profile"],
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            )

        # Fall back to explicit model+provider
        model: str | None = agent_config.get("model")
        provider: str | None = agent_config.get("provider")

        if not model:
            raise ValueError(
                "Agent configuration must specify either 'model_profile' or 'model'"
            )

        return await self.load_model(
            model,
            provider,
            temperature=temperature,
            max_tokens=max_tokens,
            options=agent_options,
            response_format=response_format,
        )

    async def _load_profile(
        self,
        profile_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> ModelInterface:
        """Load a named model profile — the one code path every profile
        load goes through, whether it is the agent's primary profile or a
        hop in its fallback chain.

        Resolves the profile's model+provider, merges the profile's own
        ``options`` with the caller's agent-level ``options`` (agent
        wins), and carries the profile's ``base_url``.
        """
        resolved_model, resolved_provider = self._config_manager.resolve_model_profile(
            profile_name
        )
        profile = self._config_manager.get_model_profile(profile_name)
        profile_options = (profile or {}).get("options")
        merged_options = _merge_options(profile_options, agent_options)
        base_url: str | None = (profile or {}).get("base_url")
        return await self.load_model(
            resolved_model,
            resolved_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            options=merged_options,
            response_format=response_format,
            base_url=base_url,
        )

    async def load_model(
        self,
        model_name: str,
        provider: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> ModelInterface:
        """Load a model interface based on authentication configuration.

        Args:
            model_name: Name of the model to load
            provider: Optional provider name
            temperature: Optional temperature for generation
            max_tokens: Optional max tokens for generation
            options: Optional provider-specific options (e.g. sampling params)
            response_format: Optional structured-output format (schema dict or 'json')
            base_url: Optional base URL for OpenAI-compatible endpoints

        Returns:
            Configured model interface

        Raises:
            ValueError: If model cannot be loaded
        """
        # Handle mock models for testing
        if model_name.startswith("mock"):
            return MockModel(model_name)

        _validate_think_option(provider, options)

        storage = self._credential_storage

        # Get authentication method
        auth_method = _resolve_authentication_method(model_name, provider, storage)

        if not auth_method:
            return _handle_no_authentication(
                model_name,
                provider,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
                response_format=response_format,
                base_url=base_url,
                execution_id=self.execution_id,
            )

        # Create authenticated model (cloud providers don't use options)
        return _create_authenticated_model(
            model_name,
            provider,
            auth_method,
            storage,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            execution_id=self.execution_id,
        )

    async def get_fallback_model(
        self,
        context: str = "general",
        original_profile: str | None = None,
        agent_fallback_profile: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> tuple[ModelInterface, str]:
        """Get a fallback model from the explicit fallback_model_profile chain.

        There is no implicit fallback: the agent-level
        ``fallback_model_profile`` is tried first, then the original
        profile's own ``fallback_model_profile`` chain. If neither
        yields a loadable model, the chain is exhausted and the caller
        is expected to fail the agent with its original error.

        Args:
            context: Context for fallback (for logging)
            original_profile: Original model profile that failed
            agent_fallback_profile: Agent-level fallback_model_profile
                override, tried before the profile's own chain
            temperature: The failed agent's temperature, carried into
                whichever fallback profile loads (SF2: a hop loads
                exactly as a primary profile would).
            max_tokens: The failed agent's max_tokens, carried likewise.
            agent_options: The failed agent's options, merged with the
                fallback profile's own options (agent wins).
            response_format: The failed agent's response_format,
                carried likewise.

        Returns:
            Tuple of (model, fallback_model_profile) naming the profile
            that actually loaded.

        Raises:
            ValueError: If the fallback chain is empty or exhausted.
        """
        if agent_fallback_profile:
            agent_result = await self._try_single_fallback(
                agent_fallback_profile,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            )
            if agent_result:
                return agent_result

        if original_profile:
            chain_result = await self._try_configurable_fallback(
                original_profile,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            )
            if chain_result:
                return chain_result

        raise ValueError(
            f"No fallback_model_profile configured for {context}; "
            "fallback chain exhausted"
        )

    async def _try_single_fallback(
        self,
        profile_name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> tuple[ModelInterface, str] | None:
        """Try loading one named fallback profile — the same code path
        (``_load_profile``) a primary profile load uses, so this hop
        carries the profile's base_url/options and the agent's
        generation params instead of dropping them.

        Returns:
            (model, profile_name) if successful, None if the profile
            doesn't resolve or load — callers continue down the chain.
        """
        try:
            model = await self._load_profile(
                profile_name,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            )
            return model, profile_name
        except (ValueError, KeyError):
            return None

    async def _try_configurable_fallback(
        self,
        original_profile: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> tuple[ModelInterface, str] | None:
        """The first loadable candidate in ``original_profile``'s
        fallback_model_profile chain, or None if the chain is exhausted.

        Args:
            original_profile: The original profile that failed
            temperature: Carried into whichever hop loads.
            max_tokens: Carried into whichever hop loads.
            agent_options: Merged with each hop's own options (agent wins).
            response_format: Carried into whichever hop loads.
        """
        async for result in self._iter_configurable_fallback_chain(
            original_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_options=agent_options,
            response_format=response_format,
        ):
            return result
        return None

    async def _iter_configurable_fallback_chain(
        self,
        original_profile: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[ModelInterface, str]]:
        """Yield every profile in ``original_profile``'s
        fallback_model_profile chain that successfully loads, in chain
        order — unlike ``_try_configurable_fallback``, which stops at
        the first one, this walks the whole chain so a caller (the
        runtime-failure path) can keep trying past a hop that loaded
        fine but failed to actually generate.
        """
        fallback_chain_visited: set[str] = set()
        current_profile = original_profile

        while current_profile:
            if current_profile in fallback_chain_visited:
                raise ValueError(
                    f"Cycle detected in fallback chain: {fallback_chain_visited}"
                )
            fallback_chain_visited.add(current_profile)

            profile_config = self._config_manager.get_model_profile(current_profile)
            if not profile_config:
                break

            fallback_profile_name = profile_config.get("fallback_model_profile")
            if not fallback_profile_name:
                break

            try:
                model = await self._load_profile(
                    fallback_profile_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    agent_options=agent_options,
                    response_format=response_format,
                )
                yield model, fallback_profile_name
            except (ValueError, KeyError):
                pass

            current_profile = fallback_profile_name

    async def iter_fallback_chain(
        self,
        original_profile: str | None = None,
        agent_fallback_profile: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        agent_options: dict[str, Any] | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[ModelInterface, str]]:
        """Yield every loadable candidate in the explicit fallback chain,
        in the same priority order as ``get_fallback_model`` (agent-level
        override first, then the original profile's own chain).

        For a caller that needs to keep trying past a runtime failure on
        an earlier candidate — ``get_fallback_model`` returns only the
        first one that loads, which is enough when a model-load failure
        is the trigger (load success there IS the outcome), but not when
        the trigger is a runtime failure: the first loadable model can
        still fail to generate, and the next candidate deserves a try
        too.
        """
        if agent_fallback_profile:
            agent_result = await self._try_single_fallback(
                agent_fallback_profile,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            )
            if agent_result:
                yield agent_result

        if original_profile:
            async for result in self._iter_configurable_fallback_chain(
                original_profile,
                temperature=temperature,
                max_tokens=max_tokens,
                agent_options=agent_options,
                response_format=response_format,
            ):
                yield result


LLAMA_SERVER_PROVIDER = "llama-server"
LLAMA_SERVER_URL_ENV = "LLAMA_SERVER_URL"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1"


def _llama_server_url() -> str:
    """The llama-server router's OpenAI-compatible base URL."""
    return os.environ.get(LLAMA_SERVER_URL_ENV, DEFAULT_LLAMA_SERVER_URL)


def _validate_think_option(
    provider: str | None, options: dict[str, Any] | None
) -> None:
    """Fail closed at load time when ``think`` targets a provider that
    doesn't speak llama-server's chat-template convention.

    ``think`` only has a home in ``OpenAICompatibleModel._apply_options``,
    which folds it into ``chat_template_kwargs.enable_thinking`` — a
    llama-server-specific request field. Other OpenAI-compatible
    providers (OpenCode Zen/Go, OpenAI proper, ...) reject an unknown
    field with an opaque request-time 400 (measured 2026-09-22 against
    OpenCode Go). Raising here, while the provider and the option are
    both still in hand, turns that into a load-time config error that
    names both.
    """
    if not options or "think" not in options:
        return
    if provider == LLAMA_SERVER_PROVIDER:
        return
    raise ModelConfigurationError(
        f"options.think is only supported for provider "
        f"'{LLAMA_SERVER_PROVIDER}' (got provider={provider!r}). Remove "
        "'think' from this profile/agent's options, or point it at a "
        "llama-server-backed profile."
    )


def validate_think_options_for_ensemble(
    agents: Sequence[AgentConfig],
    config_manager: ConfigurationManager,
) -> None:
    """Fail closed BEFORE any agent executes when an ensemble configures
    ``think`` against a provider that doesn't speak llama-server's
    chat-template convention (Invariant 14: structural errors are caught
    at load time and prevent execution — they must never reach the
    runtime fallback chain, which would silently substitute a working
    model and report success).

    Resolves each LLM agent's provider and effective options the same
    way ``ModelFactory.load_model_from_agent_config`` resolves them for
    a primary load: through ``model_profile`` (profile options merged
    with agent options, agent wins) when set, else the agent's inline
    ``model``/``provider``. Non-LLM agents (script, ensemble, loop,
    dispatch) are skipped. ``_validate_think_option`` on the direct
    ``ModelFactory.load_model`` path stays in place as defense in depth.
    """
    for agent in agents:
        if not isinstance(agent, LlmAgentConfig):
            continue

        provider: str | None
        options: dict[str, Any] | None

        if agent.model_profile:
            try:
                _, provider = config_manager.resolve_model_profile(agent.model_profile)
            except (ValueError, KeyError):
                # An unresolvable profile is a different structural
                # error, surfaced elsewhere — nothing to validate here.
                continue
            profile = config_manager.get_model_profile(agent.model_profile)
            profile_options = (profile or {}).get("options")
            options = _merge_options(profile_options, agent.options)
        else:
            provider = agent.provider
            options = agent.options

        _validate_think_option(provider, options)


def _is_openai_compatible(provider: str | None) -> bool:
    """Check if a provider is openai-compatible (exact or scoped)."""
    if not provider:
        return False
    return provider == "openai-compatible" or provider.startswith("openai-compatible/")


def _resolve_authentication_method(
    model_name: str,
    provider: str | None,
    storage: CredentialStorage,
) -> str | None:
    """Resolve authentication method for model loading.

    Args:
        model_name: Name of the model to load
        provider: Optional provider name
        storage: Credential storage instance

    Returns:
        Authentication method string or None if not found
    """
    lookup_key = provider if provider else model_name
    return storage.get_auth_method(lookup_key)


def _create_authenticated_model(
    model_name: str,
    provider: str | None,
    auth_method: str,
    storage: CredentialStorage,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    execution_id: str | None = None,
) -> ModelInterface:
    """Create authenticated model based on authentication method.

    Args:
        model_name: Name of the model to load
        provider: Optional provider name
        auth_method: Authentication method (api_key or oauth)
        storage: Credential storage instance
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation
        base_url: Optional base URL for OpenAI-compatible endpoints
        execution_id: Stable id for the top-level ensemble execution

    Returns:
        Configured model interface

    Raises:
        ValueError: If credentials are missing or unknown method
    """
    lookup_key = provider if provider else model_name

    if auth_method == "api_key":
        api_key = storage.get_api_key(lookup_key)
        if not api_key:
            raise ValueError(f"No API key found for {lookup_key}")
        return _create_api_key_model(
            model_name,
            api_key,
            provider,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            execution_id=execution_id,
        )

    else:
        raise ValueError(f"Unknown authentication method: {auth_method}")


def _handle_no_authentication(
    model_name: str,
    provider: str | None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    options: dict[str, Any] | None = None,
    response_format: str | dict[str, Any] | None = None,
    base_url: str | None = None,
    execution_id: str | None = None,
) -> ModelInterface:
    """Handle cases when no authentication is configured.

    Args:
        model_name: Name of the model
        provider: Optional provider name
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation
        options: Optional provider-specific options forwarded to local models
        response_format: Optional structured-output format (schema dict or 'json')
        base_url: Optional base URL for OpenAI-compatible endpoints
        execution_id: Stable id for the top-level ensemble execution

    Returns:
        Model interface for providers that don't require auth

    Raises:
        ValueError: If the provider requires authentication
    """
    if provider == LLAMA_SERVER_PROVIDER:
        # llama-server (#90): OpenAI-compatible transport, no auth; the
        # router's URL comes from the profile or the environment.
        return OpenAICompatibleModel(
            model_name=model_name,
            base_url=base_url or _llama_server_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            response_format=response_format,
            execution_id=execution_id,
        )
    elif _is_openai_compatible(provider):
        return OpenAICompatibleModel(
            model_name=model_name,
            base_url=base_url or "https://api.openai.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            response_format=response_format,
            execution_id=execution_id,
        )
    elif provider:
        raise ValueError(
            f"No authentication configured for provider "
            f"'{provider}' with model '{model_name}'. "
            f"Run 'llm-orc auth setup' to configure "
            f"authentication."
        )
    else:
        logger.info(
            "No provider specified for '%s', treating as a llama-server model",
            model_name,
        )
        return OpenAICompatibleModel(
            model_name=model_name,
            base_url=base_url or _llama_server_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            response_format=response_format,
            execution_id=execution_id,
        )


def _create_api_key_model(
    model_name: str,
    api_key: str,
    provider: str | None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
    execution_id: str | None = None,
) -> ModelInterface:
    """Create model using API key authentication.

    Args:
        model_name: Name of the model
        api_key: API key for authentication
        provider: Optional provider name
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation
        base_url: Optional base URL for OpenAI-compatible endpoints
        execution_id: Stable id for the top-level ensemble execution

    Returns:
        Configured model interface
    """
    if model_name == "claude-cli" or api_key.startswith("/"):
        return ClaudeCLIModel(
            claude_path=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "google-gemini":
        from llm_orc.models.google import GeminiModel

        return GeminiModel(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif _is_openai_compatible(provider):
        return OpenAICompatibleModel(
            model_name=model_name,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            execution_id=execution_id,
        )
    else:
        return ClaudeModel(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _merge_options(
    profile_options: dict[str, Any] | None,
    agent_options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge profile and agent options dicts. Agent keys win."""
    if not profile_options and not agent_options:
        return None
    return {**(profile_options or {}), **(agent_options or {})}
