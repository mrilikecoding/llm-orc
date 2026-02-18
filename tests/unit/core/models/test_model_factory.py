"""Tests for ModelFactory."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import (
    ModelFactory,
    _create_api_key_model,
    _create_authenticated_model,
    _create_oauth_model,
    _handle_mock_models,
    _handle_no_authentication,
    _resolve_authentication_method,
)
from llm_orc.models.anthropic import (
    ClaudeCLIModel,
    ClaudeModel,
    OAuthClaudeModel,
)
from llm_orc.models.mock import MockModel
from llm_orc.models.ollama import OllamaModel


class TestModelFactory:
    """Test the ModelFactory class."""

    @pytest.fixture
    def mock_config_manager(self) -> Mock:
        """Create a mock configuration manager."""
        manager = Mock(spec=ConfigurationManager)
        manager.resolve_model_profile.return_value = (
            "claude-3-sonnet",
            "anthropic",
        )
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
        self,
        mock_config_manager: Mock,
        mock_credential_storage: Mock,
    ) -> ModelFactory:
        """Create a ModelFactory with mocked dependencies."""
        return ModelFactory(mock_config_manager, mock_credential_storage)

    def test_init(
        self,
        mock_config_manager: Mock,
        mock_credential_storage: Mock,
    ) -> None:
        """Test ModelFactory initialization."""
        factory = ModelFactory(mock_config_manager, mock_credential_storage)

        assert factory._config_manager == mock_config_manager
        assert factory._credential_storage == mock_credential_storage

    async def test_load_model_from_agent_config_with_model_profile(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test loading model with model_profile."""
        agent_config = {"model_profile": "claude-sonnet"}
        mock_config_manager.resolve_model_profile.return_value = (
            "claude-3-sonnet",
            "anthropic",
        )

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_config_manager.resolve_model_profile.assert_called_once_with(
                "claude-sonnet"
            )
            mock_load.assert_called_once_with(
                "claude-3-sonnet",
                "anthropic",
                temperature=None,
                max_tokens=None,
            )
            assert result is not None

    async def test_load_model_from_agent_config_with_model_and_provider(
        self, model_factory: ModelFactory
    ) -> None:
        """Test loading model with explicit model and provider."""
        agent_config = {
            "model": "claude-3-opus",
            "provider": "anthropic",
        }

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_load.assert_called_once_with(
                "claude-3-opus",
                "anthropic",
                temperature=None,
                max_tokens=None,
            )
            assert result is not None

    async def test_load_model_from_agent_config_with_model_only(
        self, model_factory: ModelFactory
    ) -> None:
        """Test loading model with only model specified."""
        agent_config = {"model": "claude-3-haiku"}

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            result = await model_factory.load_model_from_agent_config(agent_config)

            mock_load.assert_called_once_with(
                "claude-3-haiku",
                None,
                temperature=None,
                max_tokens=None,
            )
            assert result is not None

    async def test_load_model_from_agent_config_forwards_params(
        self, model_factory: ModelFactory
    ) -> None:
        """Test that temperature and max_tokens are forwarded."""
        agent_config = {
            "model": "llama2",
            "provider": "ollama",
            "temperature": 0.7,
            "max_tokens": 500,
        }

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            await model_factory.load_model_from_agent_config(agent_config)

            mock_load.assert_called_once_with(
                "llama2",
                "ollama",
                temperature=0.7,
                max_tokens=500,
            )

    async def test_load_model_from_agent_config_missing_model(
        self, model_factory: ModelFactory
    ) -> None:
        """Test error when neither model_profile nor model."""
        agent_config = {"provider": "anthropic"}

        with pytest.raises(
            ValueError,
            match="must specify either 'model_profile' or 'model'",
        ):
            await model_factory.load_model_from_agent_config(agent_config)

    async def test_load_model_mock_model(self, model_factory: ModelFactory) -> None:
        """Test loading mock models for testing."""
        model = await model_factory.load_model("mock-test-model")

        assert isinstance(model, MockModel)
        assert hasattr(model, "generate_response")
        response = await model.generate_response("test", "system prompt")
        assert "test" in response.lower()

    async def test_load_model_no_auth_ollama_provider(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test loading model with ollama provider and no auth."""
        mock_credential_storage.get_auth_method.return_value = None

        model = await model_factory.load_model("llama3", "ollama")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "llama3"

    async def test_load_model_ollama_with_params(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test temperature/max_tokens forwarded to OllamaModel."""
        mock_credential_storage.get_auth_method.return_value = None

        model = await model_factory.load_model(
            "llama3",
            "ollama",
            temperature=0.5,
            max_tokens=200,
        )

        assert isinstance(model, OllamaModel)
        assert model.temperature == 0.5
        assert model.max_tokens == 200

    async def test_load_model_api_key_claude_with_params(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test temperature/max_tokens forwarded to ClaudeModel."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "test-key"

        model = await model_factory.load_model(
            "claude-3-sonnet",
            "anthropic",
            temperature=0.8,
            max_tokens=1500,
        )

        assert isinstance(model, ClaudeModel)
        assert model.temperature == 0.8
        assert model.max_tokens == 1500

    async def test_load_model_no_auth_other_provider_exception(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exception when no auth for non-ollama provider."""
        mock_credential_storage.get_auth_method.return_value = None

        with pytest.raises(ValueError, match=r"No authentication configured"):
            await model_factory.load_model("claude-3-sonnet", "anthropic")

    async def test_load_model_no_auth_no_provider_fallback_ollama(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test fallback to ollama when no provider and no auth."""
        mock_credential_storage.get_auth_method.return_value = None

        model = await model_factory.load_model("some-model")

        assert isinstance(model, OllamaModel)
        assert model.model_name == "some-model"

    async def test_load_model_api_key_auth_claude_cli(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test loading claude-cli model with API key auth."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "/usr/local/bin/claude"

        model = await model_factory.load_model("claude-cli")

        assert isinstance(model, ClaudeCLIModel)
        mock_credential_storage.get_api_key.assert_called_with("claude-cli")

    async def test_load_model_api_key_auth_path_like(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test path-like API key treated as claude-cli."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "/some/path/claude"

        model = await model_factory.load_model("some-model")

        assert isinstance(model, ClaudeCLIModel)

    async def test_load_model_api_key_auth_google_gemini(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test loading Google Gemini model with API key auth."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "google-api-key"

        with patch("llm_orc.models.google.GeminiModel") as mock_gemini:
            mock_instance = Mock()
            mock_gemini.return_value = mock_instance

            model = await model_factory.load_model("gemini-pro", "google-gemini")

            assert model == mock_instance
            mock_gemini.assert_called_once_with(
                api_key="google-api-key",
                model="gemini-pro",
                temperature=None,
                max_tokens=None,
            )

    async def test_load_model_api_key_auth_anthropic_default(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test loading Anthropic model with API key auth."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = "anthropic-api-key"

        model = await model_factory.load_model("claude-3-sonnet", "anthropic")

        assert isinstance(model, ClaudeModel)
        mock_credential_storage.get_api_key.assert_called_with("anthropic")

    async def test_load_model_api_key_auth_missing_key(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exception when API key configured but not found."""
        mock_credential_storage.get_auth_method.return_value = "api_key"
        mock_credential_storage.get_api_key.return_value = None

        with pytest.raises(ValueError, match=r"No API key found"):
            await model_factory.load_model("claude-3-sonnet", "anthropic")

    async def test_load_model_oauth_auth_with_client_id(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test loading model with OAuth and client_id."""
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
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test OAuth for anthropic-claude-pro-max fallback."""
        mock_credential_storage.get_auth_method.return_value = "oauth"
        mock_credential_storage.get_oauth_token.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
        }

        model = await model_factory.load_model("anthropic-claude-pro-max")

        assert isinstance(model, OAuthClaudeModel)
        assert model.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    async def test_load_model_oauth_auth_missing_token(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exception when OAuth configured but no token."""
        mock_credential_storage.get_auth_method.return_value = "oauth"
        mock_credential_storage.get_oauth_token.return_value = None

        with pytest.raises(ValueError, match=r"No OAuth token found"):
            await model_factory.load_model("claude-pro", "anthropic")

    async def test_load_model_unknown_auth_method(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exception with unknown authentication method."""
        mock_credential_storage.get_auth_method.return_value = "unknown-auth"

        with pytest.raises(
            ValueError,
            match=r"Unknown authentication method",
        ):
            await model_factory.load_model("some-model")

    async def test_load_model_exception_known_local(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exceptions propagated for known local models."""
        mock_credential_storage.get_auth_method.side_effect = Exception("Auth error")

        with pytest.raises(Exception, match=r"Auth error"):
            await model_factory.load_model("llama3")

    async def test_load_model_exception_unknown_model(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
    ) -> None:
        """Test exceptions propagated for unknown models."""
        mock_credential_storage.get_auth_method.side_effect = Exception("Auth error")

        with pytest.raises(Exception, match=r"Auth error"):
            await model_factory.load_model("unknown-model")

    async def test_get_fallback_model_with_configured_test_profile_ollama(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test fallback with configured test profile (ollama)."""
        mock_config_manager.load_project_config.return_value = {
            "project": {"default_models": {"test": "test-local"}}
        }
        mock_config_manager.resolve_model_profile.return_value = (
            "llama3",
            "ollama",
        )

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            await model_factory.get_fallback_model("test-context")

            mock_config_manager.resolve_model_profile.assert_called_once_with(
                "test-local"
            )
            mock_load.assert_called_once_with("llama3", "ollama")

    async def test_get_fallback_model_with_configured_test_profile_non_ollama(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test fallback with non-ollama configured profile."""
        mock_config_manager.load_project_config.return_value = {
            "project": {"default_models": {"test": "expensive-model"}}
        }
        mock_config_manager.resolve_model_profile.return_value = (
            "claude-3-opus",
            "anthropic",
        )

        with patch.object(
            model_factory,
            "load_model",
            return_value=OllamaModel("llama3"),
        ) as mock_load:
            await model_factory.get_fallback_model()

            assert mock_load.call_args_list[-1] == (("llama3", "ollama"),)

    async def test_get_fallback_model_no_configured_profile(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test fallback when no test profile configured."""
        mock_config_manager.load_project_config.return_value = {"project": {}}

        with patch.object(
            model_factory,
            "load_model",
            return_value=OllamaModel("llama3"),
        ) as mock_load:
            await model_factory.get_fallback_model()

            mock_load.assert_called_with("llama3", "ollama")

    async def test_get_fallback_model_hardcoded_fallback_fails(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test last resort when hardcoded fallback fails."""
        mock_config_manager.load_project_config.return_value = {"project": {}}

        with patch.object(
            model_factory,
            "load_model",
            side_effect=Exception("Ollama not available"),
        ):
            model = await model_factory.get_fallback_model()

            assert isinstance(model, OllamaModel)
            assert model.model_name == "llama3"

    async def test_get_fallback_model_with_configurable_fallback_profile(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test fallback using configurable fallback_model_profile."""
        mock_config_manager.get_model_profile.return_value = {
            "model": "claude-sonnet-4",
            "provider": "anthropic-claude-pro-max",
            "fallback_model_profile": "micro-local",
        }

        mock_config_manager.resolve_model_profile.return_value = (
            "qwen3:0.6b",
            "ollama",
        )

        with patch.object(
            model_factory,
            "load_model",
            return_value=OllamaModel("qwen3:0.6b"),
        ) as mock_load:
            model = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="claude-pro-max",
            )

            mock_config_manager.get_model_profile.assert_called_with("claude-pro-max")
            mock_config_manager.resolve_model_profile.assert_called_with("micro-local")
            mock_load.assert_called_with("qwen3:0.6b", "ollama")
            assert isinstance(model, OllamaModel)

    async def test_get_fallback_model_with_cascading_fallbacks(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test cascading fallbacks: A -> B -> C -> default."""
        profile_configs = {
            "claude-pro-max": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-claude-pro-max",
                "fallback_model_profile": "micro-local",
            },
            "micro-local": {
                "model": "qwen3:0.6b",
                "provider": "ollama",
                "fallback_model_profile": "tiny-local",
            },
            "tiny-local": {
                "model": "llama3",
                "provider": "ollama",
            },
        }

        mock_config_manager.get_model_profile.side_effect = lambda profile: (
            profile_configs.get(profile)
        )

        mock_config_manager.load_project_config.return_value = {"project": {}}

        model_load_calls: list[tuple[str, str]] = []

        def mock_load_side_effect(model: str, provider: str) -> OllamaModel:
            model_load_calls.append((model, provider))
            if len(model_load_calls) <= 2:
                raise Exception("Model failed")
            return OllamaModel("llama3")

        resolve_calls: list[str] = []

        def mock_resolve_side_effect(
            profile: str,
        ) -> tuple[str, str]:
            resolve_calls.append(profile)
            if profile == "micro-local":
                return ("qwen3:0.6b", "ollama")
            elif profile == "tiny-local":
                return ("llama3", "ollama")
            else:
                raise ValueError(f"Unknown profile: {profile}")

        mock_config_manager.resolve_model_profile.side_effect = mock_resolve_side_effect

        with patch.object(
            model_factory,
            "load_model",
            side_effect=mock_load_side_effect,
        ):
            model = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="claude-pro-max",
            )

            assert len(model_load_calls) == 3
            assert model_load_calls[0] == (
                "qwen3:0.6b",
                "ollama",
            )
            assert model_load_calls[1] == (
                "llama3",
                "ollama",
            )
            assert model_load_calls[2] == (
                "llama3",
                "ollama",
            )
            assert isinstance(model, OllamaModel)

    async def test_get_fallback_model_prevents_cycles(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test that fallback cycles are detected."""
        profile_configs = {
            "profile-a": {
                "model": "model-a",
                "provider": "provider-a",
                "fallback_model_profile": "profile-b",
            },
            "profile-b": {
                "model": "model-b",
                "provider": "provider-b",
                "fallback_model_profile": "profile-c",
            },
            "profile-c": {
                "model": "model-c",
                "provider": "provider-c",
                "fallback_model_profile": "profile-a",
            },
        }

        mock_config_manager.get_model_profile.side_effect = lambda profile: (
            profile_configs.get(profile)
        )

        def mock_resolve_side_effect(
            profile: str,
        ) -> tuple[str, str]:
            config = profile_configs.get(profile)
            if config:
                return (config["model"], config["provider"])
            raise ValueError(f"Profile {profile} not found")

        mock_config_manager.resolve_model_profile.side_effect = mock_resolve_side_effect

        with patch.object(
            model_factory,
            "load_model",
            side_effect=Exception("Model load failed"),
        ):
            with pytest.raises(
                ValueError,
                match="Cycle detected in fallback chain",
            ):
                await model_factory.get_fallback_model(
                    context="agent_test",
                    original_profile="profile-a",
                )


class TestLoadModelHelperMethods:
    """Test helper methods extracted from load_model."""

    def test_handle_mock_models(self) -> None:
        """Test mock model handling."""
        result = _handle_mock_models("mock-test-model")

        assert isinstance(result, MockModel)
        assert result.name == "mock-test-model"

    def test_handle_no_authentication_ollama_fallback(
        self,
    ) -> None:
        """Test no auth handler with no-provider Ollama fallback."""
        result = _handle_no_authentication("llama3", None)

        assert isinstance(result, OllamaModel)
        assert result.model_name == "llama3"

    def test_handle_no_authentication_ollama_provider(
        self,
    ) -> None:
        """Test no auth handler with explicit Ollama provider."""
        result = _handle_no_authentication(
            "llama3",
            "ollama",
            temperature=0.5,
            max_tokens=200,
        )

        assert isinstance(result, OllamaModel)
        assert result.model_name == "llama3"
        assert result.temperature == 0.5
        assert result.max_tokens == 200

    def test_handle_no_authentication_other_provider_raises(
        self,
    ) -> None:
        """Test no auth handler raises for non-ollama providers."""
        with pytest.raises(ValueError, match="No authentication configured"):
            _handle_no_authentication("claude-3-sonnet", "anthropic")

    def test_create_api_key_model_claude_cli(self) -> None:
        """Test API key model creation for Claude CLI."""
        result = _create_api_key_model("claude-cli", "/path/to/claude", None)

        assert isinstance(result, ClaudeCLIModel)
        assert result.claude_path == "/path/to/claude"

    def test_create_oauth_model(self) -> None:
        """Test OAuth model creation."""
        oauth_token = {
            "access_token": "test-token",
            "refresh_token": "refresh-token",
            "expires_at": 1234567890,
        }
        storage = Mock()

        result = _create_oauth_model(oauth_token, storage, "claude-pro")

        assert isinstance(result, OAuthClaudeModel)
        assert result.access_token == "test-token"

    def test_resolve_authentication_method_with_provider(
        self,
    ) -> None:
        """Test auth resolution with explicit provider."""
        storage = Mock()
        storage.get_auth_method.return_value = "oauth"

        result = _resolve_authentication_method("claude-3-sonnet", "anthropic", storage)

        assert result == "oauth"
        storage.get_auth_method.assert_called_once_with("anthropic")

    def test_resolve_authentication_method_without_provider(
        self,
    ) -> None:
        """Test auth resolution using model name as lookup."""
        storage = Mock()
        storage.get_auth_method.return_value = "api_key"

        result = _resolve_authentication_method("claude-3-sonnet", None, storage)

        assert result == "api_key"
        storage.get_auth_method.assert_called_once_with("claude-3-sonnet")

    def test_resolve_authentication_method_no_auth(
        self,
    ) -> None:
        """Test auth resolution when no auth is found."""
        storage = Mock()
        storage.get_auth_method.return_value = None

        result = _resolve_authentication_method("claude-3-sonnet", "anthropic", storage)

        assert result is None
        storage.get_auth_method.assert_called_once_with("anthropic")

    def test_create_authenticated_model_api_key(self) -> None:
        """Test authenticated model creation with API key."""
        storage = Mock()
        storage.get_api_key.return_value = "test-api-key"

        with patch(
            "llm_orc.core.models.model_factory._create_api_key_model"
        ) as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            result = _create_authenticated_model(
                "claude-3-sonnet",
                "anthropic",
                "api_key",
                storage,
            )

        assert result == mock_model
        mock_create.assert_called_once_with(
            "claude-3-sonnet",
            "test-api-key",
            "anthropic",
            temperature=None,
            max_tokens=None,
        )

    def test_create_authenticated_model_oauth(self) -> None:
        """Test authenticated model creation with OAuth."""
        storage = Mock()
        oauth_token = {"access_token": "test-token"}
        storage.get_oauth_token.return_value = oauth_token

        with patch(
            "llm_orc.core.models.model_factory._create_oauth_model"
        ) as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            result = _create_authenticated_model(
                "claude-pro",
                "anthropic",
                "oauth",
                storage,
            )

        assert result == mock_model
        mock_create.assert_called_once_with(
            oauth_token,
            storage,
            "anthropic",
            temperature=None,
            max_tokens=None,
        )

    def test_create_authenticated_model_no_api_key(
        self,
    ) -> None:
        """Test authenticated model when API key is missing."""
        storage = Mock()
        storage.get_api_key.return_value = None

        with pytest.raises(
            ValueError,
            match="No API key found for anthropic",
        ):
            _create_authenticated_model(
                "claude-3-sonnet",
                "anthropic",
                "api_key",
                storage,
            )

    def test_create_authenticated_model_unknown_method(
        self,
    ) -> None:
        """Test authenticated model with unknown auth method."""
        storage = Mock()

        with pytest.raises(
            ValueError,
            match="Unknown authentication method: unknown",
        ):
            _create_authenticated_model(
                "claude-3-sonnet",
                "anthropic",
                "unknown",
                storage,
            )
