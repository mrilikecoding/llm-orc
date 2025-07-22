"""Tests for ModelFactory."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import (
    ModelFactory,
    _prompt_auth_setup,
    _setup_anthropic_api_auth,
    _setup_anthropic_oauth_auth,
    _setup_claude_cli_auth,
    _should_prompt_for_auth,
)
from llm_orc.models.anthropic import ClaudeCLIModel, ClaudeModel, OAuthClaudeModel
from llm_orc.models.ollama import OllamaModel


class TestModelFactory:
    """Test the ModelFactory class."""

    @pytest.fixture
    def mock_config_manager(self) -> Mock:
        """Create a mock configuration manager."""
        manager = Mock(spec=ConfigurationManager)
        manager.resolve_model_profile.return_value = ("claude-3-sonnet", "anthropic")
        manager.load_project_config.return_value = {
            "project": {"default_models": {"test": "test-local"}}
        }
        return manager

    @pytest.fixture
    def mock_credential_storage(self) -> Mock:
        """Create a mock credential storage."""
        storage = Mock(spec=CredentialStorage)
        storage.get_auth_method.return_value = None
        storage.get_api_key.return_value = None
        storage.get_oauth_token.return_value = None
        return storage

    @pytest.fixture
    def model_factory(
        self, mock_config_manager: Mock, mock_credential_storage: Mock
    ) -> ModelFactory:
        """Create a ModelFactory instance with mocked dependencies."""
        return ModelFactory(mock_config_manager, mock_credential_storage)

    def test_init(
        self, mock_config_manager: Mock, mock_credential_storage: Mock
    ) -> None:
        """Test ModelFactory initialization."""
        factory = ModelFactory(mock_config_manager, mock_credential_storage)

        assert factory._config_manager == mock_config_manager
        assert factory._credential_storage == mock_credential_storage

    async def test_load_model_from_agent_config_with_model_profile(
        self, model_factory: ModelFactory, mock_config_manager: Mock
    ) -> None:
        """Test loading model with model_profile configuration."""
        agent_config = {"model_profile": "claude-sonnet"}
        mock_config_manager.resolve_model_profile.return_value = (
            "claude-3-sonnet",
            "anthropic",
        )

        # Mock the load_model method to return a specific model
        with patch.object(
            model_factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_config_manager.resolve_model_profile.assert_called_once_with(
                "claude-sonnet"
            )
            mock_load.assert_called_once_with("claude-3-sonnet", "anthropic")
            assert result is not None

    async def test_load_model_from_agent_config_with_model_and_provider(
        self, model_factory: ModelFactory
    ) -> None:
        """Test loading model with explicit model and provider."""
        agent_config = {"model": "claude-3-opus", "provider": "anthropic"}

        with patch.object(
            model_factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_load.assert_called_once_with("claude-3-opus", "anthropic")
            assert result is not None

    async def test_load_model_from_agent_config_with_model_only(
        self, model_factory: ModelFactory
    ) -> None:
        """Test loading model with only model specified."""
        agent_config = {"model": "claude-3-haiku"}

        with patch.object(
            model_factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_load.assert_called_once_with("claude-3-haiku", None)
            assert result is not None

    async def test_load_model_from_agent_config_missing_model(
        self, model_factory: ModelFactory
    ) -> None:
        """Test error when neither model_profile nor model is specified."""
        agent_config = {"provider": "anthropic"}  # Only provider, no model

        with pytest.raises(
            ValueError, match="must specify either 'model_profile' or 'model'"
        ):
            await model_factory.load_model_from_agent_config(agent_config)

    async def test_load_model_mock_model(self, model_factory: ModelFactory) -> None:
        """Test loading mock models for testing."""
        model = await model_factory.load_model("mock-test-model")

        assert hasattr(model, "generate_response")
        # Mock should return specific response
        response = await model.generate_response("test", "system prompt")
        assert response == "Response from mock-test-model"

    async def test_load_model_no_auth_ollama_provider(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading model with ollama provider and no auth."""
        mock_credential_storage.get_auth_method.return_value = None

        model = await model_factory.load_model("llama3", "ollama")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "llama3"

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_no_auth_other_provider_fallback(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback when no auth configured for non-ollama provider."""
        mock_credential_storage.get_auth_method.return_value = None

        # Mock fallback model to avoid deep recursion
        with patch.object(
            model_factory, "get_fallback_model", return_value=OllamaModel("llama3")
        ):
            model = await model_factory.load_model("claude-3-sonnet", "anthropic")

            assert isinstance(model, OllamaModel)
            mock_echo.assert_any_call(
                "⚠️  Failed to load model 'claude-3-sonnet': "
                "No authentication configured for provider 'anthropic' "
                "with model 'claude-3-sonnet'. Run 'llm-orc auth setup' "
                "to configure authentication."
            )

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_no_auth_no_provider_fallback_ollama(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback to ollama when no provider and no auth."""
        mock_credential_storage.get_auth_method.return_value = None

        model = await model_factory.load_model("some-model")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "some-model"
        mock_echo.assert_called_with(
            "ℹ️  No provider specified for 'some-model', treating as local Ollama model"
        )

    async def test_load_model_api_key_auth_claude_cli(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading claude-cli model with API key auth."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "/usr/local/bin/claude"

        model = await model_factory.load_model("claude-cli")

        assert isinstance(model, ClaudeCLIModel)
        mock_credential_storage.get_api_key.assert_called_with("claude-cli")

    async def test_load_model_api_key_auth_path_like(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading model with path-like API key (treated as claude-cli)."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "/some/path/claude"

        model = await model_factory.load_model("some-model")

        assert isinstance(model, ClaudeCLIModel)

    async def test_load_model_api_key_auth_google_gemini(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading Google Gemini model with API key auth."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "google-api-key"

        # Patch the dynamic import inside the method
        with patch("llm_orc.models.google.GeminiModel") as mock_gemini:
            mock_instance = Mock()
            mock_gemini.return_value = mock_instance

            model = await model_factory.load_model("gemini-pro", "google-gemini")

            assert model == mock_instance
            mock_gemini.assert_called_once_with(
                api_key="google-api-key", model="gemini-pro"
            )

    async def test_load_model_api_key_auth_anthropic_default(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading Anthropic model with API key auth (default case)."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "anthropic-api-key"

        model = await model_factory.load_model("claude-3-sonnet", "anthropic")

        assert isinstance(model, ClaudeModel)
        mock_credential_storage.get_api_key.assert_called_with("anthropic")

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_api_key_auth_missing_key(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback when API key auth configured but key not found."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = None

        # Mock fallback model to avoid deep recursion
        with patch.object(
            model_factory, "get_fallback_model", return_value=OllamaModel("llama3")
        ):
            model = await model_factory.load_model("claude-3-sonnet", "anthropic")

            assert isinstance(model, OllamaModel)
            mock_echo.assert_any_call(
                "⚠️  Failed to load model 'claude-3-sonnet': "
                "No API key found for anthropic"
            )

    async def test_load_model_oauth_auth_with_client_id(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test loading model with OAuth authentication and client_id."""
        mock_credential_storage.get_auth_method.return_value = "oauth"
        mock_credential_storage.get_oauth_token.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "client_id": "test-client-id",
            "expires_at": 1234567890,
        }

        model = await model_factory.load_model("claude-pro", "anthropic")

        assert isinstance(model, OAuthClaudeModel)
        assert model.access_token == "access-token"
        assert model.client_id == "test-client-id"

    async def test_load_model_oauth_auth_anthropic_claude_pro_max(
        self, model_factory: ModelFactory, mock_credential_storage: Mock
    ) -> None:
        """Test OAuth auth for anthropic-claude-pro-max with fallback client_id."""
        mock_credential_storage.get_auth_method.return_value = "oauth"
        mock_credential_storage.get_oauth_token.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            # No client_id in token
        }

        model = await model_factory.load_model("anthropic-claude-pro-max")

        assert isinstance(model, OAuthClaudeModel)
        assert model.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_oauth_auth_missing_token(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback when OAuth auth configured but token not found."""
        mock_credential_storage.get_auth_method.return_value = "oauth"
        mock_credential_storage.get_oauth_token.return_value = None

        # Mock fallback model to avoid deep recursion
        with patch.object(
            model_factory, "get_fallback_model", return_value=OllamaModel("llama3")
        ):
            model = await model_factory.load_model("claude-pro", "anthropic")

            assert isinstance(model, OllamaModel)
            mock_echo.assert_any_call(
                "⚠️  Failed to load model 'claude-pro': "
                "No OAuth token found for anthropic"
            )

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_unknown_auth_method(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback with unknown authentication method."""
        mock_credential_storage.get_auth_method.return_value = "unknown-auth"

        # Mock fallback model to avoid deep recursion
        with patch.object(
            model_factory, "get_fallback_model", return_value=OllamaModel("llama3")
        ):
            model = await model_factory.load_model("some-model")

            assert isinstance(model, OllamaModel)
            mock_echo.assert_any_call(
                "⚠️  Failed to load model 'some-model': "
                "Unknown authentication method: unknown-auth"
            )

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_exception_fallback_known_local(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback to ollama for known local models when exception occurs."""
        mock_credential_storage.get_auth_method.side_effect = Exception("Auth error")

        model = await model_factory.load_model("llama3")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "llama3"
        mock_echo.assert_any_call("⚠️  Failed to load model 'llama3': Auth error")
        mock_echo.assert_any_call("🔄 Treating 'llama3' as local Ollama model")

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_load_model_exception_fallback_unknown_model(
        self,
        mock_echo: Mock,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback to get_fallback_model for unknown models."""
        mock_credential_storage.get_auth_method.side_effect = Exception("Auth error")

        with patch.object(
            model_factory, "get_fallback_model", return_value=AsyncMock()
        ) as mock_fallback:
            model = await model_factory.load_model("unknown-model")

            assert model is not None
            mock_fallback.assert_called_once_with("general")
            mock_echo.assert_any_call(
                "⚠️  Failed to load model 'unknown-model': Auth error"
            )
            mock_echo.assert_any_call(
                "🔄 Using configured fallback instead of 'unknown-model'"
            )

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_get_fallback_model_with_configured_test_profile_ollama(
        self, mock_echo: Mock, model_factory: ModelFactory, mock_config_manager: Mock
    ) -> None:
        """Test fallback model with configured test profile using ollama."""
        mock_config_manager.load_project_config.return_value = {
            "project": {"default_models": {"test": "test-local"}}
        }
        mock_config_manager.resolve_model_profile.return_value = ("llama3", "ollama")

        with patch.object(
            model_factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            await model_factory.get_fallback_model("test-context")

            mock_config_manager.resolve_model_profile.assert_called_once_with(
                "test-local"
            )
            mock_load.assert_called_once_with("llama3", "ollama")
            mock_echo.assert_any_call(
                "⚠️  Falling back to free local model for test-context"
            )

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_get_fallback_model_with_configured_test_profile_non_ollama(
        self, mock_echo: Mock, model_factory: ModelFactory, mock_config_manager: Mock
    ) -> None:
        """Test fallback when configured test profile uses non-ollama provider."""
        mock_config_manager.load_project_config.return_value = {
            "project": {"default_models": {"test": "expensive-model"}}
        }
        mock_config_manager.resolve_model_profile.return_value = (
            "claude-3-opus",
            "anthropic",
        )

        with patch.object(
            model_factory, "load_model", return_value=OllamaModel("llama3")
        ) as mock_load:
            await model_factory.get_fallback_model()

            # Should skip the anthropic profile and use hardcoded fallback
            mock_echo.assert_any_call(
                "⚠️  Configured test profile 'expensive-model' uses "
                "anthropic, not ollama"
            )
            mock_echo.assert_any_call("🔄 Using hardcoded fallback: llama3 via ollama")
            # Final load_model call should be for hardcoded fallback
            assert mock_load.call_args_list[-1] == (("llama3", "ollama"),)

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_get_fallback_model_no_configured_profile(
        self, mock_echo: Mock, model_factory: ModelFactory, mock_config_manager: Mock
    ) -> None:
        """Test fallback when no test profile configured."""
        mock_config_manager.load_project_config.return_value = {"project": {}}

        with patch.object(
            model_factory, "load_model", return_value=OllamaModel("llama3")
        ) as mock_load:
            await model_factory.get_fallback_model()

            mock_echo.assert_any_call("🔄 Using hardcoded fallback: llama3 via ollama")
            mock_load.assert_called_with("llama3", "ollama")

    @patch("llm_orc.core.models.model_factory.click.echo")
    async def test_get_fallback_model_hardcoded_fallback_fails(
        self, mock_echo: Mock, model_factory: ModelFactory, mock_config_manager: Mock
    ) -> None:
        """Test last resort when even hardcoded fallback fails."""
        mock_config_manager.load_project_config.return_value = {"project": {}}

        with patch.object(
            model_factory, "load_model", side_effect=Exception("Ollama not available")
        ):
            model = await model_factory.get_fallback_model()

            assert isinstance(model, OllamaModel)
            assert model.model_name == "llama3"
            mock_echo.assert_any_call(
                "❌ Hardcoded fallback model 'llama3' failed: Ollama not available"
            )
            mock_echo.assert_any_call("🆘 Creating basic Ollama model as last resort")

    def test_setup_authentication_anthropic_api(
        self, model_factory: ModelFactory
    ) -> None:
        """Test setting up anthropic API authentication."""
        mock_storage = Mock()

        with patch(
            "llm_orc.core.models.model_factory._setup_anthropic_api_auth",
            return_value=True,
        ) as mock_setup:
            result = model_factory.setup_authentication("anthropic-api", mock_storage)

            assert result is True
            mock_setup.assert_called_once_with(mock_storage)

    def test_setup_authentication_anthropic_oauth(
        self, model_factory: ModelFactory
    ) -> None:
        """Test setting up anthropic OAuth authentication."""
        mock_storage = Mock()

        with (
            patch(
                "llm_orc.core.models.model_factory.AuthenticationManager"
            ) as mock_auth_manager,
            patch(
                "llm_orc.core.models.model_factory._setup_anthropic_oauth_auth",
                return_value=True,
            ) as mock_setup,
        ):
            mock_auth_instance = Mock()
            mock_auth_manager.return_value = mock_auth_instance

            result = model_factory.setup_authentication(
                "anthropic-claude-pro-max", mock_storage
            )

            assert result is True
            mock_auth_manager.assert_called_once_with(mock_storage)
            mock_setup.assert_called_once_with(
                mock_auth_instance, "anthropic-claude-pro-max"
            )

    def test_setup_authentication_claude_cli(self, model_factory: ModelFactory) -> None:
        """Test setting up Claude CLI authentication."""
        mock_storage = Mock()

        with patch(
            "llm_orc.core.models.model_factory._setup_claude_cli_auth",
            return_value=True,
        ) as mock_setup:
            result = model_factory.setup_authentication("claude-cli", mock_storage)

            assert result is True
            mock_setup.assert_called_once_with(mock_storage)

    def test_setup_authentication_unknown_model(
        self, model_factory: ModelFactory
    ) -> None:
        """Test setting up authentication for unknown model."""
        mock_storage = Mock()

        with patch("llm_orc.core.models.model_factory.click.echo") as mock_echo:
            result = model_factory.setup_authentication("unknown-model", mock_storage)

            assert result is False
            mock_echo.assert_called_once_with(
                "Don't know how to set up authentication for 'unknown-model'"
            )

    def test_setup_authentication_exception(self, model_factory: ModelFactory) -> None:
        """Test exception handling in authentication setup."""
        mock_storage = Mock()

        with (
            patch(
                "llm_orc.core.models.model_factory._setup_anthropic_api_auth",
                side_effect=Exception("Setup error"),
            ),
            patch("llm_orc.core.models.model_factory.click.echo") as mock_echo,
        ):
            result = model_factory.setup_authentication("anthropic-api", mock_storage)

            assert result is False
            mock_echo.assert_called_once_with(
                "Failed to set up authentication: Setup error"
            )


class TestModuleFunctions:
    """Test module-level functions."""

    def test_should_prompt_for_auth_known_provider(self) -> None:
        """Test prompting for known providers."""
        known_providers = [
            "anthropic-api",
            "anthropic-claude-pro-max",
            "claude-cli",
            "openai-api",
            "google-gemini",
        ]

        for provider in known_providers:
            assert _should_prompt_for_auth(provider) is True

    def test_should_prompt_for_auth_unknown_provider(self) -> None:
        """Test not prompting for unknown providers."""
        assert _should_prompt_for_auth("unknown-provider") is False
        assert _should_prompt_for_auth("ollama") is False

    @patch("llm_orc.core.models.model_factory.click.confirm")
    def test_prompt_auth_setup_confirmed(self, mock_confirm: Mock) -> None:
        """Test auth setup when user confirms."""
        mock_confirm.return_value = True
        mock_storage = Mock()

        with (
            patch("llm_orc.core.models.model_factory.ConfigurationManager"),
            patch(
                "llm_orc.core.models.model_factory.ModelFactory"
            ) as mock_factory_class,
        ):
            mock_factory = Mock()
            mock_factory.setup_authentication.return_value = True
            mock_factory_class.return_value = mock_factory

            result = _prompt_auth_setup("anthropic-api", mock_storage)

            assert result is True
            mock_confirm.assert_called_once_with(
                "No authentication found for 'anthropic-api'. Set it up now?",
                default=True,
            )
            mock_factory.setup_authentication.assert_called_once_with(
                "anthropic-api", mock_storage
            )

    @patch("llm_orc.core.models.model_factory.click.confirm")
    def test_prompt_auth_setup_declined(self, mock_confirm: Mock) -> None:
        """Test auth setup when user declines."""
        mock_confirm.return_value = False
        mock_storage = Mock()

        result = _prompt_auth_setup("anthropic-api", mock_storage)

        assert result is False
        mock_confirm.assert_called_once()

    @patch("llm_orc.core.models.model_factory.click.prompt")
    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_anthropic_api_auth(self, mock_echo: Mock, mock_prompt: Mock) -> None:
        """Test Anthropic API key setup."""
        mock_prompt.return_value = "test-api-key"
        mock_storage = Mock()

        result = _setup_anthropic_api_auth(mock_storage)

        assert result is True
        mock_prompt.assert_called_once_with(
            "Enter your Anthropic API key", hide_input=True
        )
        mock_storage.store_api_key.assert_called_once_with(
            "anthropic-api", "test-api-key"
        )
        mock_echo.assert_called_once_with("✅ Anthropic API key configured")

    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_anthropic_oauth_auth_success(self, mock_echo: Mock) -> None:
        """Test successful Anthropic OAuth setup."""
        mock_auth_manager = Mock()
        mock_auth_manager.authenticate_oauth.return_value = True

        with patch(
            "llm_orc.core.auth.authentication.AnthropicOAuthFlow"
        ) as mock_oauth_flow_class:
            mock_oauth_flow = Mock()
            mock_oauth_flow.client_id = "test-client-id"
            mock_oauth_flow.client_secret = "test-client-secret"
            mock_oauth_flow_class.create_with_guidance.return_value = mock_oauth_flow

            result = _setup_anthropic_oauth_auth(mock_auth_manager, "provider-key")

            assert result is True
            mock_auth_manager.authenticate_oauth.assert_called_once_with(
                "provider-key", "test-client-id", "test-client-secret"
            )
            mock_echo.assert_called_once_with("✅ Anthropic OAuth configured")

    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_anthropic_oauth_auth_failed(self, mock_echo: Mock) -> None:
        """Test failed Anthropic OAuth setup."""
        mock_auth_manager = Mock()
        mock_auth_manager.authenticate_oauth.return_value = False

        with patch(
            "llm_orc.core.auth.authentication.AnthropicOAuthFlow"
        ) as mock_oauth_flow_class:
            mock_oauth_flow = Mock()
            mock_oauth_flow.client_id = "test-client-id"
            mock_oauth_flow.client_secret = "test-client-secret"
            mock_oauth_flow_class.create_with_guidance.return_value = mock_oauth_flow

            result = _setup_anthropic_oauth_auth(mock_auth_manager, "provider-key")

            assert result is False
            mock_echo.assert_called_once_with("❌ OAuth authentication failed")

    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_anthropic_oauth_auth_exception(self, mock_echo: Mock) -> None:
        """Test Anthropic OAuth setup with exception."""
        mock_auth_manager = Mock()

        with patch(
            "llm_orc.core.auth.authentication.AnthropicOAuthFlow"
        ) as mock_oauth_flow_class:
            mock_oauth_flow_class.create_with_guidance.side_effect = Exception(
                "OAuth error"
            )

            result = _setup_anthropic_oauth_auth(mock_auth_manager, "provider-key")

            assert result is False
            mock_echo.assert_called_once_with("❌ OAuth setup failed: OAuth error")

    @patch("shutil.which")
    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_claude_cli_auth_success(
        self, mock_echo: Mock, mock_which: Mock
    ) -> None:
        """Test successful Claude CLI setup."""
        mock_which.return_value = "/usr/local/bin/claude"
        mock_storage = Mock()

        result = _setup_claude_cli_auth(mock_storage)

        assert result is True
        mock_which.assert_called_once_with("claude")
        mock_storage.store_api_key.assert_called_once_with(
            "claude-cli", "/usr/local/bin/claude"
        )
        mock_echo.assert_any_call("✅ Claude CLI authentication configured")
        mock_echo.assert_any_call(
            "   Using local claude command at: /usr/local/bin/claude"
        )

    @patch("shutil.which")
    @patch("llm_orc.core.models.model_factory.click.echo")
    def test_setup_claude_cli_auth_not_found(
        self, mock_echo: Mock, mock_which: Mock
    ) -> None:
        """Test Claude CLI setup when CLI not found."""
        mock_which.return_value = None
        mock_storage = Mock()

        result = _setup_claude_cli_auth(mock_storage)

        assert result is False
        mock_which.assert_called_once_with("claude")
        mock_echo.assert_any_call(
            "❌ Claude CLI not found. Please install the Claude CLI from:"
        )
        mock_echo.assert_any_call("   https://docs.anthropic.com/en/docs/claude-code")
        mock_storage.store_api_key.assert_not_called()
