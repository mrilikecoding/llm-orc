"""Model factory for creating model instances based on configuration."""

from typing import Any
from unittest.mock import AsyncMock

import click

from llm_orc.core.auth.authentication import AuthenticationManager, CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.models import (
    ClaudeCLIModel,
    ClaudeModel,
    ModelInterface,
    OAuthClaudeModel,
    OllamaModel,
)


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
        # Check if model_profile is specified (takes precedence)
        if "model_profile" in agent_config:
            profile_name = agent_config["model_profile"]
            resolved_model, resolved_provider = (
                self._config_manager.resolve_model_profile(profile_name)
            )
            return await self.load_model(resolved_model, resolved_provider)

        # Fall back to explicit model+provider
        model: str | None = agent_config.get("model")
        provider: str | None = agent_config.get("provider")

        if not model:
            raise ValueError(
                "Agent configuration must specify either 'model_profile' or 'model'"
            )

        return await self.load_model(model, provider)

    async def load_model(
        self, model_name: str, provider: str | None = None
    ) -> ModelInterface:
        """Load a model interface based on authentication configuration.

        Args:
            model_name: Name of the model to load
            provider: Optional provider name

        Returns:
            Configured model interface

        Raises:
            ValueError: If model cannot be loaded
        """
        # Handle mock models for testing
        if model_name.startswith("mock"):
            mock = AsyncMock(spec=ModelInterface)
            mock.generate_response.return_value = f"Response from {model_name}"
            return mock

        storage = self._credential_storage

        try:
            # Get authentication method for the provider configuration
            # Use provider if specified, otherwise use model_name as lookup key
            lookup_key = provider if provider else model_name
            auth_method = storage.get_auth_method(lookup_key)

            if not auth_method:
                # Prompt user to set up authentication if not configured
                if _should_prompt_for_auth(model_name):
                    auth_configured = _prompt_auth_setup(model_name, storage)
                    if auth_configured:
                        # Retry model loading after auth setup
                        return await self.load_model(model_name, provider)

                # Handle based on provider
                if provider == "ollama":
                    # Expected behavior for Ollama - no auth needed
                    return OllamaModel(model_name=model_name)
                elif provider:
                    # Other providers require authentication
                    raise ValueError(
                        f"No authentication configured for provider '{provider}' "
                        f"with model '{model_name}'. "
                        f"Run 'llm-orc auth setup' to configure authentication."
                    )
                else:
                    # No provider specified, fallback to Ollama
                    click.echo(
                        f"ℹ️  No provider specified for '{model_name}', "
                        f"treating as local Ollama model"
                    )
                    return OllamaModel(model_name=model_name)

            if auth_method == "api_key":
                lookup_key = provider if provider else model_name
                api_key = storage.get_api_key(lookup_key)
                if not api_key:
                    raise ValueError(f"No API key found for {lookup_key}")

                # Check if this is a claude-cli configuration
                # (stored as api_key but path-like)
                if model_name == "claude-cli" or api_key.startswith("/"):
                    return ClaudeCLIModel(claude_path=api_key)
                elif provider == "google-gemini":
                    from llm_orc.models import GeminiModel

                    return GeminiModel(api_key=api_key, model=model_name)
                else:
                    # Assume it's an Anthropic API key for Claude
                    return ClaudeModel(api_key=api_key)

            elif auth_method == "oauth":
                lookup_key = provider if provider else model_name
                oauth_token = storage.get_oauth_token(lookup_key)
                if not oauth_token:
                    raise ValueError(f"No OAuth token found for {lookup_key}")

                # Use stored client_id or fallback for anthropic-claude-pro-max
                client_id = oauth_token.get("client_id")
                if not client_id and lookup_key == "anthropic-claude-pro-max":
                    client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

                return OAuthClaudeModel(
                    access_token=oauth_token["access_token"],
                    refresh_token=oauth_token.get("refresh_token"),
                    client_id=client_id,
                    credential_storage=storage,
                    provider_key=model_name,
                    expires_at=oauth_token.get("expires_at"),
                )

            else:
                raise ValueError(f"Unknown authentication method: {auth_method}")

        except Exception as e:
            # Fallback: use configured default model or treat as Ollama
            click.echo(f"⚠️  Failed to load model '{model_name}': {str(e)}")
            if model_name in ["llama3", "llama2"]:  # Known local models
                click.echo(f"🔄 Treating '{model_name}' as local Ollama model")
                return OllamaModel(model_name=model_name)
            else:
                # For unknown models, use configured fallback
                click.echo(f"🔄 Using configured fallback instead of '{model_name}'")
                return await self.get_fallback_model("general")

    async def get_fallback_model(self, context: str = "general") -> ModelInterface:
        """Get a fallback model - always use free local model for reliability.

        Args:
            context: Context for fallback (for logging)

        Returns:
            Fallback model interface
        """
        # Load project configuration to get default models
        project_config = self._config_manager.load_project_config()
        default_models = project_config.get("project", {}).get("default_models", {})

        # Log that we're falling back
        click.echo(f"⚠️  Falling back to free local model for {context}")

        # Always prefer free local models for fallback reliability
        # Look for a "test" profile first (typically free/local)
        fallback_profile = default_models.get("test")

        if fallback_profile and isinstance(fallback_profile, str):
            try:
                # Try to resolve the test profile to get a free local model
                resolved_model, resolved_provider = (
                    self._config_manager.resolve_model_profile(fallback_profile)
                )
                # Only use if it's a local/free provider (ollama)
                if resolved_provider == "ollama":
                    click.echo(
                        f"🔄 Using configured free local model '{fallback_profile}' "
                        f"→ {resolved_model}"
                    )
                    try:
                        return await self.load_model(resolved_model, resolved_provider)
                    except Exception as e:
                        click.echo(
                            f"❌ Failed to load configured fallback "
                            f"'{fallback_profile}': {e}"
                        )
                        # Don't raise immediately, try hardcoded fallback first
                else:
                    click.echo(
                        f"⚠️  Configured test profile '{fallback_profile}' uses "
                        f"{resolved_provider}, not ollama"
                    )
            except (ValueError, KeyError) as e:
                click.echo(
                    f"⚠️  Configured test profile '{fallback_profile}' not found: {e}"
                )

        # Last resort: hardcoded free local fallback
        click.echo("🔄 Using hardcoded fallback: llama3 via ollama")
        try:
            return await self.load_model("llama3", "ollama")
        except Exception as e:
            click.echo(f"❌ Hardcoded fallback model 'llama3' failed: {e}")
            # For tests and when Ollama is not available, return basic model
            # In production, this would indicate a serious configuration issue
            click.echo("🆘 Creating basic Ollama model as last resort")
            return OllamaModel(model_name="llama3")

    def setup_authentication(self, model_name: str, storage: CredentialStorage) -> bool:
        """Set up authentication for a model.

        Args:
            model_name: Name of the model to set up authentication for
            storage: Credential storage instance

        Returns:
            True if authentication was set up successfully
        """
        try:
            auth_manager = AuthenticationManager(storage)

            if model_name == "anthropic-api":
                return _setup_anthropic_api_auth(storage)
            elif model_name == "anthropic-claude-pro-max":
                return _setup_anthropic_oauth_auth(auth_manager, model_name)
            elif model_name == "claude-cli":
                return _setup_claude_cli_auth(storage)
            else:
                click.echo(
                    f"Don't know how to set up authentication for '{model_name}'"
                )
                return False

        except Exception as e:
            click.echo(f"Failed to set up authentication: {str(e)}")
            return False


def _should_prompt_for_auth(model_name: str) -> bool:
    """Check if we should prompt for authentication setup for this provider."""
    # Only prompt for known providers that typically require setup
    known_providers = {
        "anthropic-api",
        "anthropic-claude-pro-max",
        "claude-cli",
        "openai-api",
        "google-gemini",
    }
    return model_name in known_providers


def _prompt_auth_setup(model_name: str, storage: CredentialStorage) -> bool:
    """Prompt user to set up authentication for a model."""
    if click.confirm(
        f"No authentication found for '{model_name}'. Set it up now?",
        default=True,
    ):
        factory = ModelFactory(ConfigurationManager(), storage)
        return factory.setup_authentication(model_name, storage)
    return False


def _setup_anthropic_api_auth(storage: CredentialStorage) -> bool:
    """Set up Anthropic API key authentication."""
    api_key = click.prompt("Enter your Anthropic API key", hide_input=True)
    storage.store_api_key("anthropic-api", api_key)
    click.echo("✅ Anthropic API key configured")
    return True


def _setup_anthropic_oauth_auth(
    auth_manager: AuthenticationManager, provider_key: str
) -> bool:
    """Set up Anthropic OAuth authentication."""
    try:
        from llm_orc.core.auth.authentication import AnthropicOAuthFlow

        oauth_flow = AnthropicOAuthFlow.create_with_guidance()

        if auth_manager.authenticate_oauth(
            provider_key, oauth_flow.client_id, oauth_flow.client_secret
        ):
            click.echo("✅ Anthropic OAuth configured")
            return True
        else:
            click.echo("❌ OAuth authentication failed")
            return False

    except Exception as e:
        click.echo(f"❌ OAuth setup failed: {str(e)}")
        return False


def _setup_claude_cli_auth(storage: CredentialStorage) -> bool:
    """Set up Claude CLI authentication."""
    import shutil

    # Check if claude command is available
    claude_path = shutil.which("claude")
    if not claude_path:
        click.echo("❌ Claude CLI not found. Please install the Claude CLI from:")
        click.echo("   https://docs.anthropic.com/en/docs/claude-code")
        return False

    # Store claude-cli configuration
    storage.store_api_key("claude-cli", claude_path)
    click.echo("✅ Claude CLI authentication configured")
    click.echo(f"   Using local claude command at: {claude_path}")
    return True
