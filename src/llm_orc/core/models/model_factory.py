"""Model factory for creating model instances based on configuration."""

import logging
from typing import Any

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.models.anthropic import (
    ClaudeCLIModel,
    ClaudeModel,
    OAuthClaudeModel,
)
from llm_orc.models.base import ModelInterface
from llm_orc.models.mock import MockModel
from llm_orc.models.ollama import OllamaModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances based on configuration."""

    def __init__(
        self,
        config_manager: ConfigurationManager,
        credential_storage: CredentialStorage,
    ) -> None:
        """Initialize the model factory.

        Args:
            config_manager: Configuration manager instance
            credential_storage: Credential storage instance
        """
        self._config_manager = config_manager
        self._credential_storage = credential_storage

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
        ollama_format: str | dict[str, Any] | None = agent_config.get("ollama_format")

        # Check if model_profile is specified (takes precedence)
        # Use .get() truthy check: model_dump() includes None values as keys
        if agent_config.get("model_profile"):
            profile_name = agent_config["model_profile"]
            resolved_model, resolved_provider = (
                self._config_manager.resolve_model_profile(profile_name)
            )
            # Merge profile options with agent options (agent wins)
            profile = self._config_manager.get_model_profile(profile_name)
            profile_options = (profile or {}).get("options")
            merged_options = _merge_options(profile_options, agent_options)
            return await self.load_model(
                resolved_model,
                resolved_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                options=merged_options,
                ollama_format=ollama_format,
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
            ollama_format=ollama_format,
        )

    async def load_model(
        self,
        model_name: str,
        provider: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        ollama_format: str | dict[str, Any] | None = None,
    ) -> ModelInterface:
        """Load a model interface based on authentication configuration.

        Args:
            model_name: Name of the model to load
            provider: Optional provider name
            temperature: Optional temperature for generation
            max_tokens: Optional max tokens for generation
            options: Optional provider-specific options (e.g. Ollama options)

        Returns:
            Configured model interface

        Raises:
            ValueError: If model cannot be loaded
        """
        # Handle mock models for testing
        if model_name.startswith("mock"):
            return MockModel(model_name)

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
                ollama_format=ollama_format,
            )

        # Create authenticated model (cloud providers don't use options)
        return _create_authenticated_model(
            model_name,
            provider,
            auth_method,
            storage,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def get_fallback_model(
        self,
        context: str = "general",
        original_profile: str | None = None,
    ) -> ModelInterface:
        """Get a fallback model with configurable fallback support.

        Args:
            context: Context for fallback (for logging)
            original_profile: Original model profile that failed

        Returns:
            Fallback model interface
        """
        if original_profile:
            model = await self._try_configurable_fallback(original_profile)
            if model:
                return model

        return await self._try_legacy_fallback()

    async def _try_configurable_fallback(
        self, original_profile: str
    ) -> ModelInterface | None:
        """Try configurable fallback chain for a given profile.

        Args:
            original_profile: The original profile that failed

        Returns:
            Model if successful, None if fallback chain exhausted
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
                resolved_model, resolved_provider = (
                    self._config_manager.resolve_model_profile(fallback_profile_name)
                )
                return await self.load_model(resolved_model, resolved_provider)
            except (ValueError, KeyError):
                current_profile = fallback_profile_name
                continue

        return None

    async def _try_legacy_fallback(self) -> ModelInterface:
        """Try legacy fallback system.

        Returns:
            Model interface (guaranteed to return something)
        """
        project_config = self._config_manager.load_project_config()
        default_models = project_config.get("project", {}).get("default_models", {})

        fallback_profile = default_models.get("test")

        if fallback_profile and isinstance(fallback_profile, str):
            try:
                resolved_model, resolved_provider = (
                    self._config_manager.resolve_model_profile(fallback_profile)
                )
                if resolved_provider == "ollama":
                    try:
                        return await self.load_model(resolved_model, resolved_provider)
                    except (ValueError, OSError):
                        logger.warning(
                            "Failed to load fallback profile %r",
                            fallback_profile,
                            exc_info=True,
                        )
            except (ValueError, KeyError):
                pass

        fallback_model = default_models.get("fallback", "llama3")
        fallback_provider = default_models.get("fallback_provider", "ollama")
        try:
            return await self.load_model(fallback_model, fallback_provider)
        except (ValueError, OSError):
            return OllamaModel(model_name=fallback_model)


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
) -> ModelInterface:
    """Create authenticated model based on authentication method.

    Args:
        model_name: Name of the model to load
        provider: Optional provider name
        auth_method: Authentication method (api_key or oauth)
        storage: Credential storage instance
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation

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
        )

    elif auth_method == "oauth":
        oauth_token = storage.get_oauth_token(lookup_key)
        if not oauth_token:
            raise ValueError(f"No OAuth token found for {lookup_key}")
        return _create_oauth_model(
            oauth_token,
            storage,
            lookup_key,
            temperature=temperature,
            max_tokens=max_tokens,
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
    ollama_format: str | dict[str, Any] | None = None,
) -> ModelInterface:
    """Handle cases when no authentication is configured.

    Args:
        model_name: Name of the model
        provider: Optional provider name
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation
        options: Optional provider-specific options forwarded to OllamaModel

    Returns:
        Model interface for providers that don't require auth

    Raises:
        ValueError: If the provider requires authentication
    """
    if provider == "ollama":
        return OllamaModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            ollama_format=ollama_format,
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
            "No provider specified for '%s', treating as local Ollama model",
            model_name,
        )
        return OllamaModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            ollama_format=ollama_format,
        )


def _create_api_key_model(
    model_name: str,
    api_key: str,
    provider: str | None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelInterface:
    """Create model using API key authentication.

    Args:
        model_name: Name of the model
        api_key: API key for authentication
        provider: Optional provider name
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation

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
    else:
        return ClaudeModel(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _create_oauth_model(
    oauth_token: dict[str, Any],
    storage: CredentialStorage,
    model_name: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelInterface:
    """Create model using OAuth authentication.

    Args:
        oauth_token: OAuth token dictionary
        storage: Credential storage instance
        model_name: Name of the model
        temperature: Optional temperature for generation
        max_tokens: Optional max tokens for generation

    Returns:
        Configured OAuth model interface
    """
    client_id = oauth_token.get("client_id")
    if not client_id and model_name == "anthropic-claude-pro-max":
        client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    return OAuthClaudeModel(
        access_token=oauth_token["access_token"],
        refresh_token=oauth_token.get("refresh_token"),
        client_id=client_id,
        credential_storage=storage,
        provider_key=model_name,
        expires_at=oauth_token.get("expires_at"),
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
