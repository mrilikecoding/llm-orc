"""Tests for ModelFactory."""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.models.model_factory import (
    ModelConfigurationError,
    ModelFactory,
    _create_api_key_model,
    _create_authenticated_model,
    _handle_no_authentication,
    _is_openai_compatible,
    _merge_options,
    _resolve_authentication_method,
    validate_think_options_for_ensemble,
)
from llm_orc.models.anthropic import (
    ClaudeCLIModel,
    ClaudeModel,
)
from llm_orc.models.mock import MockModel
from llm_orc.models.openai_compat import OpenAICompatibleModel
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


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
        mock_config_manager.get_model_profile.return_value = None

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
                options=None,
                response_format=None,
                base_url=None,
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
                options=None,
                response_format=None,
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
                options=None,
                response_format=None,
            )
            assert result is not None

    async def test_load_model_from_agent_config_forwards_params(
        self, model_factory: ModelFactory
    ) -> None:
        """Test that temperature and max_tokens are forwarded."""
        agent_config = {
            "model": "llama2",
            "provider": "llama-server",
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
                "llama-server",
                temperature=0.7,
                max_tokens=500,
                options=None,
                response_format=None,
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
        """Test exception when no auth for cloud provider."""
        mock_credential_storage.get_auth_method.return_value = None

        with pytest.raises(ValueError, match=r"No authentication configured"):
            await model_factory.load_model("claude-3-sonnet", "anthropic")

    async def test_load_model_no_auth_no_provider_defaults_to_llama_server(
        self,
        model_factory: ModelFactory,
        mock_credential_storage: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No provider and no auth means the local router (#90)."""
        mock_credential_storage.get_auth_method.return_value = None
        monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)

        model = await model_factory.load_model("some-model")

        assert isinstance(model, OpenAICompatibleModel)
        assert model.model_name == "some-model"
        assert model.base_url == "http://127.0.0.1:8080/v1"

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

    async def test_get_fallback_model_raises_when_chain_exhausted(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """No original_profile and no agent_fallback_profile: no implicit
        fallback exists, so the chain is empty and the call raises."""
        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            with pytest.raises(ValueError, match="fallback chain exhausted"):
                await model_factory.get_fallback_model("test-context")

            mock_load.assert_not_called()

    async def test_get_fallback_model_raises_when_profile_chain_ends(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """original_profile has no fallback_model_profile of its own and
        there is no agent-level override: the chain is exhausted, not a
        hardcoded default."""
        mock_config_manager.get_model_profile.return_value = {
            "model": "qwen3-8b",
            "provider": "llama-server",
        }

        with patch.object(
            model_factory,
            "load_model",
            return_value=AsyncMock(),
        ) as mock_load:
            with pytest.raises(ValueError, match="fallback chain exhausted"):
                await model_factory.get_fallback_model(
                    context="agent_test", original_profile="local-llama"
                )

            mock_load.assert_not_called()

    async def test_get_fallback_model_with_configurable_fallback_profile(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test fallback using the failed profile's own fallback_model_profile."""
        mock_config_manager.get_model_profile.return_value = {
            "model": "claude-sonnet-4",
            "provider": "anthropic-api",
            "fallback_model_profile": "micro-local",
        }

        mock_config_manager.resolve_model_profile.return_value = (
            "qwen3-0.6b",
            "llama-server",
        )

        with patch.object(
            model_factory,
            "load_model",
            return_value=OpenAICompatibleModel("qwen3-0.6b"),
        ) as mock_load:
            model, fallback_profile = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="premium-claude",
            )

            mock_config_manager.get_model_profile.assert_any_call("premium-claude")
            mock_config_manager.resolve_model_profile.assert_called_with("micro-local")
            mock_load.assert_called_with(
                "qwen3-0.6b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=None,
                base_url=None,
            )
            assert isinstance(model, OpenAICompatibleModel)
            assert fallback_profile == "micro-local"

    async def test_get_fallback_model_agent_level_tried_first(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """The agent-level fallback_model_profile is tried before the
        original profile's own fallback_model_profile chain."""
        mock_config_manager.resolve_model_profile.return_value = (
            "qwen3-0.6b",
            "llama-server",
        )
        mock_config_manager.get_model_profile.return_value = None

        with patch.object(
            model_factory,
            "load_model",
            return_value=OpenAICompatibleModel("qwen3-0.6b"),
        ) as mock_load:
            model, fallback_profile = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="premium-claude",
                agent_fallback_profile="my-explicit-fallback",
            )

            mock_config_manager.resolve_model_profile.assert_called_once_with(
                "my-explicit-fallback"
            )
            mock_load.assert_called_once_with(
                "qwen3-0.6b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=None,
                base_url=None,
            )
            assert fallback_profile == "my-explicit-fallback"
            assert isinstance(model, OpenAICompatibleModel)
            # The original profile's own chain is never consulted - only
            # the agent-level fallback profile's own config is read.
            mock_config_manager.get_model_profile.assert_called_once_with(
                "my-explicit-fallback"
            )

    async def test_get_fallback_model_agent_level_failure_falls_through(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """When the agent-level fallback_model_profile doesn't resolve,
        the original profile's own chain is still tried."""
        mock_config_manager.get_model_profile.return_value = {
            "model": "claude-sonnet-4",
            "provider": "anthropic-api",
            "fallback_model_profile": "micro-local",
        }

        def mock_resolve_side_effect(profile: str) -> tuple[str, str]:
            if profile == "my-explicit-fallback":
                raise ValueError("Model profile 'my-explicit-fallback' not found")
            return ("qwen3-0.6b", "llama-server")

        mock_config_manager.resolve_model_profile.side_effect = mock_resolve_side_effect

        with patch.object(
            model_factory,
            "load_model",
            return_value=OpenAICompatibleModel("qwen3-0.6b"),
        ) as mock_load:
            model, fallback_profile = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="premium-claude",
                agent_fallback_profile="my-explicit-fallback",
            )

            mock_load.assert_called_once_with(
                "qwen3-0.6b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=None,
                base_url=None,
            )
            assert fallback_profile == "micro-local"
            assert isinstance(model, OpenAICompatibleModel)

    async def test_get_fallback_model_with_cascading_fallbacks(
        self,
        model_factory: ModelFactory,
        mock_config_manager: Mock,
    ) -> None:
        """Test cascading fallbacks entirely within the profile chain:
        A -> B (load fails) -> C (load succeeds). No legacy fallback
        exists to catch an exhausted chain."""
        profile_configs = {
            "premium-claude": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-api",
                "fallback_model_profile": "micro-local",
            },
            "micro-local": {
                "model": "qwen3-0.6b",
                "provider": "llama-server",
                "fallback_model_profile": "tiny-local",
            },
            "tiny-local": {
                "model": "qwen3-8b",
                "provider": "llama-server",
            },
        }

        mock_config_manager.get_model_profile.side_effect = lambda profile: (
            profile_configs.get(profile)
        )

        model_load_calls: list[tuple[str, str]] = []

        def mock_load_side_effect(
            model: str, provider: str, **_kwargs: Any
        ) -> OpenAICompatibleModel:
            model_load_calls.append((model, provider))
            if len(model_load_calls) == 1:
                raise ValueError("Model failed")
            return OpenAICompatibleModel("qwen3-8b")

        def mock_resolve_side_effect(
            profile: str,
        ) -> tuple[str, str]:
            if profile == "micro-local":
                return ("qwen3-0.6b", "llama-server")
            elif profile == "tiny-local":
                return ("qwen3-8b", "llama-server")
            else:
                raise ValueError(f"Unknown profile: {profile}")

        mock_config_manager.resolve_model_profile.side_effect = mock_resolve_side_effect

        with patch.object(
            model_factory,
            "load_model",
            side_effect=mock_load_side_effect,
        ):
            model, fallback_profile = await model_factory.get_fallback_model(
                context="agent_test",
                original_profile="premium-claude",
            )

            assert len(model_load_calls) == 2
            assert model_load_calls[0] == ("qwen3-0.6b", "llama-server")
            assert model_load_calls[1] == ("qwen3-8b", "llama-server")
            assert fallback_profile == "tiny-local"
            assert isinstance(model, OpenAICompatibleModel)

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
            side_effect=ValueError("Model load failed"),
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

    def test_handle_no_authentication_no_provider_is_llama_server(
        self,
    ) -> None:
        """No provider means the local router (#90)."""
        result = _handle_no_authentication("qwen3-8b", None)

        assert isinstance(result, OpenAICompatibleModel)
        assert result.model_name == "qwen3-8b"

    def test_handle_no_authentication_other_provider_raises(
        self,
    ) -> None:
        """Test no auth handler raises for cloud providers."""
        with pytest.raises(ValueError, match="No authentication configured"):
            _handle_no_authentication("claude-3-sonnet", "anthropic")

    def test_create_api_key_model_claude_cli(self) -> None:
        """Test API key model creation for Claude CLI."""
        result = _create_api_key_model("claude-cli", "/path/to/claude", None)

        assert isinstance(result, ClaudeCLIModel)
        assert result.claude_path == "/path/to/claude"

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
            base_url=None,
            execution_id=None,
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


class TestMergeOptions:
    """Test the _merge_options helper."""

    def test_both_none(self) -> None:
        assert _merge_options(None, None) is None

    def test_profile_only(self) -> None:
        assert _merge_options({"num_ctx": 8192}, None) == {"num_ctx": 8192}

    def test_agent_only(self) -> None:
        assert _merge_options(None, {"top_k": 20}) == {"top_k": 20}

    def test_agent_wins_on_conflict(self) -> None:
        result = _merge_options({"top_k": 40, "num_ctx": 8192}, {"top_k": 20})
        assert result == {"top_k": 20, "num_ctx": 8192}

    def test_merge_distinct_keys(self) -> None:
        result = _merge_options({"num_ctx": 8192}, {"top_k": 20})
        assert result == {"num_ctx": 8192, "top_k": 20}


class TestModelFactoryExecutionId:
    """Scenario: ModelFactory.execution_id identifies one top-level
    ensemble execution, for the OpenCode Go session header (#90 plan §C).
    Two agents sharing a ModelFactory instance must see the SAME id;
    two ModelFactory instances (two executions) must see DIFFERENT ones
    (Doctrine 11)."""

    def test_generates_an_id_when_omitted(self) -> None:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)

        factory = ModelFactory(config_manager, credential_storage)

        assert isinstance(factory.execution_id, str)
        assert factory.execution_id

    def test_honors_a_provided_id(self) -> None:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)

        factory = ModelFactory(
            config_manager, credential_storage, execution_id="exec-fixed"
        )

        assert factory.execution_id == "exec-fixed"

    def test_two_factories_get_different_ids(self) -> None:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)

        first = ModelFactory(config_manager, credential_storage)
        second = ModelFactory(config_manager, credential_storage)

        assert first.execution_id != second.execution_id

    async def test_two_agents_from_one_factory_get_the_same_id_on_the_model(
        self,
    ) -> None:
        """Two ``load_model`` calls from the same factory (two agents in
        one execution) both hand the same execution_id to their model
        instances."""
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        factory = ModelFactory(config_manager, credential_storage)

        first = await factory.load_model("qwen3-8b", "llama-server")
        second = await factory.load_model("qwen3-14b", "llama-server")

        assert isinstance(first, OpenAICompatibleModel)
        assert isinstance(second, OpenAICompatibleModel)
        assert first._execution_id == factory.execution_id
        assert second._execution_id == factory.execution_id


class TestOptionsPassThrough:
    """Scenario: options threaded from config to the router-backed model."""

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_load_model_from_agent_config_passes_options(
        self, factory: ModelFactory
    ) -> None:
        """Options from agent config dict reach load_model."""
        agent_config = {
            "model": "qwen3:8b",
            "provider": "llama-server",
            "options": {"num_ctx": 8192, "top_k": 20},
        }

        with patch.object(factory, "load_model", return_value=AsyncMock()) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "qwen3:8b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options={"num_ctx": 8192, "top_k": 20},
                response_format=None,
            )

    async def test_load_model_from_agent_config_no_options(
        self, factory: ModelFactory
    ) -> None:
        """No options field passes None (backward compat)."""
        agent_config = {"model": "llama3", "provider": "llama-server"}

        with patch.object(factory, "load_model", return_value=AsyncMock()) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "llama3",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=None,
            )

    async def test_load_model_from_agent_config_merges_profile_options(
        self, factory: ModelFactory
    ) -> None:
        """Profile options merged with agent options, agent wins."""
        config_mock = cast(Mock, factory._config_manager)
        config_mock.resolve_model_profile.return_value = (
            "qwen3:8b",
            "llama-server",
        )
        config_mock.get_model_profile.return_value = {
            "model": "qwen3:8b",
            "provider": "llama-server",
            "options": {"num_ctx": 8192, "top_k": 40},
        }

        agent_config = {
            "model_profile": "analyst-qwen",
            "options": {"top_k": 20, "top_p": 0.8},
        }

        with patch.object(factory, "load_model", return_value=AsyncMock()) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "qwen3:8b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options={"num_ctx": 8192, "top_k": 20, "top_p": 0.8},
                response_format=None,
                base_url=None,
            )

    async def test_load_model_forwards_options_to_llama_server(
        self, factory: ModelFactory
    ) -> None:
        """load_model passes options to the router-backed model."""
        model = await factory.load_model(
            "qwen3-8b",
            "llama-server",
            options={"num_ctx": 8192},
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model._options == {"num_ctx": 8192}

    async def test_load_model_cloud_provider_ignores_options(
        self, factory: ModelFactory
    ) -> None:
        """Cloud providers don't break when options is passed."""
        storage_mock = cast(Mock, factory._credential_storage)
        storage_mock.get_auth_method.return_value = "api_key"
        storage_mock.get_api_key.return_value = "test-key"

        model = await factory.load_model(
            "claude-3-sonnet",
            "anthropic",
            options={"num_ctx": 8192},
        )

        assert isinstance(model, ClaudeModel)


class TestThinkOptionProviderGuard:
    """Scenario: `think` only has a home in llama-server's chat-template
    convention (OpenAICompatibleModel._apply_options folds it into
    ``chat_template_kwargs``). Every other OpenAI-compatible provider —
    OpenCode Zen/Go included — rejects an unrecognized
    ``chat_template_kwargs`` field with an opaque request-time 400
    (measured 2026-09-22 against
    ``https://opencode.ai/zen/go/v1/chat/completions``). The guard fails
    at load time instead, naming the provider and the option.
    """

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_think_on_llama_server_loads(self, factory: ModelFactory) -> None:
        model = await factory.load_model(
            "qwen3-8b", "llama-server", options={"think": False}
        )

        assert isinstance(model, OpenAICompatibleModel)

    async def test_think_on_zen_provider_raises_at_load(
        self, factory: ModelFactory
    ) -> None:
        with pytest.raises(ValueError, match="think"):
            await factory.load_model(
                "minimax-m2.5",
                "openai-compatible/zen",
                options={"think": False},
            )

    async def test_think_on_zen_provider_error_names_the_provider(
        self, factory: ModelFactory
    ) -> None:
        with pytest.raises(ValueError, match="openai-compatible/zen"):
            await factory.load_model(
                "minimax-m2.5",
                "openai-compatible/zen",
                options={"think": False},
            )

    async def test_think_on_authenticated_provider_raises_at_load(
        self, factory: ModelFactory
    ) -> None:
        """The guard also covers the authenticated (API-key) load path —
        a real Zen/Go profile carries an API key, so this is the path it
        actually takes."""
        storage_mock = cast(Mock, factory._credential_storage)
        storage_mock.get_auth_method.return_value = "api_key"
        storage_mock.get_api_key.return_value = "test-key"

        with pytest.raises(ValueError, match="think"):
            await factory.load_model(
                "minimax-m2.5",
                "openai-compatible/zen",
                options={"think": False},
            )

    async def test_think_true_also_guarded(self, factory: ModelFactory) -> None:
        """Presence of the key triggers the guard, not its truthiness."""
        with pytest.raises(ValueError, match="think"):
            await factory.load_model(
                "minimax-m2.5", "openai-compatible/zen", options={"think": True}
            )

    async def test_think_absent_does_not_raise_for_other_providers(
        self, factory: ModelFactory
    ) -> None:
        model = await factory.load_model(
            "gpt-4o", "openai-compatible", options={"seed": 11}
        )

        assert isinstance(model, OpenAICompatibleModel)


class TestValidateThinkOptionsForEnsemble:
    """Scenario (SF1): ``think`` vs. provider is a structural error caught
    before any agent executes — never laundered through the runtime
    fallback chain into a quiet substitution (Invariant 14).
    """

    @pytest.fixture
    def config_manager(self) -> Mock:
        return Mock(spec=ConfigurationManager)

    def test_think_on_hosted_profile_raises_before_execution(
        self, config_manager: Mock
    ) -> None:
        config_manager.resolve_model_profile.return_value = (
            "minimax-m2.5",
            "openai-compatible/zen",
        )
        config_manager.get_model_profile.return_value = {
            "model": "minimax-m2.5",
            "provider": "openai-compatible/zen",
            "options": {"think": False},
            "fallback_model_profile": "local-llama",
        }
        agents = [
            LlmAgentConfig(name="decomposer", model_profile="hosted-thinking"),
        ]

        with pytest.raises(ModelConfigurationError, match="think"):
            validate_think_options_for_ensemble(agents, config_manager)

    def test_think_in_agent_level_options_also_raises(
        self, config_manager: Mock
    ) -> None:
        config_manager.resolve_model_profile.return_value = (
            "minimax-m2.5",
            "openai-compatible/zen",
        )
        config_manager.get_model_profile.return_value = {
            "model": "minimax-m2.5",
            "provider": "openai-compatible/zen",
        }
        agents = [
            LlmAgentConfig(
                name="decomposer",
                model_profile="hosted",
                options={"think": False},
            ),
        ]

        with pytest.raises(ModelConfigurationError, match="think"):
            validate_think_options_for_ensemble(agents, config_manager)

    def test_think_on_llama_server_profile_does_not_raise(
        self, config_manager: Mock
    ) -> None:
        config_manager.resolve_model_profile.return_value = (
            "qwen3-8b",
            "llama-server",
        )
        config_manager.get_model_profile.return_value = {
            "model": "qwen3-8b",
            "provider": "llama-server",
            "options": {"think": False},
        }
        agents = [LlmAgentConfig(name="worker", model_profile="local-qwen")]

        validate_think_options_for_ensemble(agents, config_manager)

    def test_think_on_inline_model_provider_raises(self, config_manager: Mock) -> None:
        agents = [
            LlmAgentConfig(
                name="worker",
                model="minimax-m2.5",
                provider="openai-compatible/zen",
                options={"think": False},
            ),
        ]

        with pytest.raises(ModelConfigurationError, match="think"):
            validate_think_options_for_ensemble(agents, config_manager)

    def test_unresolvable_profile_is_skipped_not_raised(
        self, config_manager: Mock
    ) -> None:
        config_manager.resolve_model_profile.side_effect = ValueError(
            "Model profile 'ghost' not found"
        )
        agents = [LlmAgentConfig(name="worker", model_profile="ghost")]

        validate_think_options_for_ensemble(agents, config_manager)

    def test_non_llm_agents_are_skipped(self, config_manager: Mock) -> None:
        agents = [ScriptAgentConfig(name="setup", script="echo hi")]

        validate_think_options_for_ensemble(agents, config_manager)


class TestResponseFormatPassThrough:
    """Scenario: response_format threaded from agent config to the local model."""

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_response_format_extracted_from_agent_config(
        self, factory: ModelFactory
    ) -> None:
        """response_format from agent config dict reaches load_model."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        agent_config = {
            "model": "qwen3:14b",
            "provider": "llama-server",
            "response_format": schema,
        }

        with patch.object(factory, "load_model", return_value=AsyncMock()) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "qwen3:14b",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=schema,
            )

    async def test_response_format_none_when_absent(
        self, factory: ModelFactory
    ) -> None:
        """No response_format passes None (backward compat)."""
        agent_config = {"model": "llama3", "provider": "llama-server"}

        with patch.object(factory, "load_model", return_value=AsyncMock()) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "llama3",
                "llama-server",
                temperature=None,
                max_tokens=None,
                options=None,
                response_format=None,
            )

    async def test_load_model_forwards_format_to_llama_server(
        self, factory: ModelFactory
    ) -> None:
        """load_model passes the format to the router-backed model."""
        schema = {"type": "object"}
        model = await factory.load_model(
            "qwen3-14b",
            "llama-server",
            response_format=schema,
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model._response_format == schema

    async def test_load_model_string_format_to_llama_server(
        self, factory: ModelFactory
    ) -> None:
        """load_model passes string 'json' format to the router-backed model."""
        model = await factory.load_model(
            "qwen3-14b",
            "llama-server",
            response_format="json",
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model._response_format == "json"


class TestIsOpenAICompatible:
    """Test the _is_openai_compatible helper."""

    def test_exact_match(self) -> None:
        assert _is_openai_compatible("openai-compatible") is True

    def test_scoped_match(self) -> None:
        assert _is_openai_compatible("openai-compatible/openrouter") is True

    def test_none(self) -> None:
        assert _is_openai_compatible(None) is False

    def test_other_provider(self) -> None:
        assert _is_openai_compatible("anthropic") is False

    def test_partial_match(self) -> None:
        assert _is_openai_compatible("openai") is False


class TestOpenAICompatibleRouting:
    """Test openai-compatible provider routing through factory."""

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_no_auth_openai_compatible(self, factory: ModelFactory) -> None:
        """openai-compatible with no auth creates OpenAICompatibleModel."""
        model = await factory.load_model(
            "deepseek-coder-v2",
            "openai-compatible",
            base_url="http://localhost:8000/v1",
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model.model_name == "deepseek-coder-v2"
        assert model.base_url == "http://localhost:8000/v1"
        assert model.api_key is None

    async def test_no_auth_openai_compatible_default_base_url(
        self, factory: ModelFactory
    ) -> None:
        """openai-compatible with no base_url defaults to OpenAI."""
        model = await factory.load_model("gpt-4o", "openai-compatible")

        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "https://api.openai.com/v1"

    async def test_api_key_openai_compatible(self, factory: ModelFactory) -> None:
        """openai-compatible with API key creates authenticated model."""
        storage = cast(Mock, factory._credential_storage)
        storage.get_auth_method.return_value = "api_key"
        storage.get_api_key.return_value = "sk-test-key"

        model = await factory.load_model(
            "gpt-4o",
            "openai-compatible",
            base_url="https://api.openai.com/v1",
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model.api_key == "sk-test-key"
        assert model.base_url == "https://api.openai.com/v1"

    async def test_openai_compatible_with_params(self, factory: ModelFactory) -> None:
        """Temperature and max_tokens forwarded to OpenAICompatibleModel."""
        model = await factory.load_model(
            "deepseek-coder-v2",
            "openai-compatible",
            temperature=0.5,
            max_tokens=1000,
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model.temperature == 0.5
        assert model.max_tokens == 1000

    async def test_scoped_provider_routes_to_openai_compatible(
        self, factory: ModelFactory
    ) -> None:
        """Scoped provider like openai-compatible/openrouter routes correctly."""
        model = await factory.load_model(
            "meta-llama/llama-3.1-70b",
            "openai-compatible/openrouter",
            base_url="https://openrouter.ai/api/v1",
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "https://openrouter.ai/api/v1"

    async def test_base_url_threaded_from_profile(self, factory: ModelFactory) -> None:
        """base_url from profile config reaches the model."""
        config_mock = cast(Mock, factory._config_manager)
        config_mock.resolve_model_profile.return_value = (
            "deepseek-coder-v2",
            "openai-compatible",
        )
        config_mock.get_model_profile.return_value = {
            "model": "deepseek-coder-v2",
            "provider": "openai-compatible",
            "base_url": "http://localhost:8000/v1",
        }

        agent_config = {"model_profile": "local-deepseek"}

        model = await factory.load_model_from_agent_config(agent_config)

        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "http://localhost:8000/v1"

    def test_handle_no_auth_openai_compatible(self) -> None:
        """_handle_no_authentication creates OpenAICompatibleModel."""
        result = _handle_no_authentication(
            "gpt-4o",
            "openai-compatible",
            base_url="https://api.openai.com/v1",
        )

        assert isinstance(result, OpenAICompatibleModel)
        assert result.api_key is None

    def test_create_api_key_model_openai_compatible(self) -> None:
        """_create_api_key_model creates OpenAICompatibleModel."""
        result = _create_api_key_model(
            "gpt-4o",
            "sk-test-key",
            "openai-compatible",
            base_url="https://api.openai.com/v1",
        )

        assert isinstance(result, OpenAICompatibleModel)
        assert result.api_key == "sk-test-key"


class TestFallbackHopLoadsProfileConfig:
    """Scenario (SF2): a fallback-chain hop loads the fallback profile
    exactly as a primary profile would be loaded — same base_url, same
    options, same agent-level generation params. Reviewer probe: a
    fallback profile with base_url http://10.9.9.9:8080/v1 produced a
    model with base_url http://127.0.0.1:8080/v1 (the router default)
    and options None — the hop was calling load_model(model, provider)
    bare, dropping everything the profile and the agent configured.
    """

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_configurable_chain_hop_carries_base_url_and_options(
        self, factory: ModelFactory
    ) -> None:
        config_mock = cast(Mock, factory._config_manager)
        profile_configs: dict[str, dict[str, Any]] = {
            "premium-claude": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-api",
                "fallback_model_profile": "remote-llama",
            },
            "remote-llama": {
                "model": "qwen3-8b",
                "provider": "llama-server",
                "base_url": "http://10.9.9.9:8080/v1",
                "options": {"num_ctx": 4096},
            },
        }
        config_mock.get_model_profile.side_effect = lambda name: profile_configs.get(
            name
        )
        config_mock.resolve_model_profile.side_effect = lambda name: (
            "qwen3-8b",
            "llama-server",
        )

        model, fallback_profile = await factory.get_fallback_model(
            context="agent_test", original_profile="premium-claude"
        )

        assert fallback_profile == "remote-llama"
        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "http://10.9.9.9:8080/v1"
        assert model._options == {"num_ctx": 4096}

    async def test_agent_level_fallback_hop_carries_base_url_and_options(
        self, factory: ModelFactory
    ) -> None:
        config_mock = cast(Mock, factory._config_manager)
        config_mock.get_model_profile.return_value = {
            "model": "qwen3-8b",
            "provider": "llama-server",
            "base_url": "http://10.9.9.9:8080/v1",
            "options": {"num_ctx": 4096},
        }
        config_mock.resolve_model_profile.return_value = ("qwen3-8b", "llama-server")

        model, fallback_profile = await factory.get_fallback_model(
            context="agent_test",
            agent_fallback_profile="local-fallback",
        )

        assert fallback_profile == "local-fallback"
        assert isinstance(model, OpenAICompatibleModel)
        assert model.base_url == "http://10.9.9.9:8080/v1"
        assert model._options == {"num_ctx": 4096}

    async def test_fallback_hop_carries_agent_generation_params(
        self, factory: ModelFactory
    ) -> None:
        config_mock = cast(Mock, factory._config_manager)
        config_mock.get_model_profile.return_value = {
            "model": "qwen3-8b",
            "provider": "llama-server",
        }
        config_mock.resolve_model_profile.return_value = ("qwen3-8b", "llama-server")

        model, _ = await factory.get_fallback_model(
            context="agent_test",
            agent_fallback_profile="local-fallback",
            temperature=0.3,
            max_tokens=222,
            agent_options={"top_k": 5},
            response_format="json",
        )

        assert isinstance(model, OpenAICompatibleModel)
        assert model.temperature == 0.3
        assert model.max_tokens == 222
        assert model._options == {"top_k": 5}
        assert model._response_format == "json"


class TestIterFallbackChain:
    """Scenario: iter_fallback_chain yields every loadable candidate, in
    the same priority order get_fallback_model uses (agent-level
    override first, then the profile's own chain) — not just the
    first one that loads. This is what lets the runtime-failure path
    keep trying past a candidate that loaded fine but failed to
    generate."""

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_yields_agent_level_then_the_full_profile_chain(
        self, factory: ModelFactory
    ) -> None:
        config_mock = cast(Mock, factory._config_manager)
        profile_configs: dict[str, dict[str, Any]] = {
            "premium-claude": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-api",
                "fallback_model_profile": "chain-a",
            },
            "chain-a": {
                "model": "qwen3-0.6b",
                "provider": "llama-server",
                "fallback_model_profile": "chain-b",
            },
            "chain-b": {"model": "qwen3-8b", "provider": "llama-server"},
        }
        config_mock.get_model_profile.side_effect = profile_configs.get
        resolved = {
            "agent-fb": ("minimax", "llama-server"),
            "chain-a": ("qwen3-0.6b", "llama-server"),
            "chain-b": ("qwen3-8b", "llama-server"),
        }
        config_mock.resolve_model_profile.side_effect = lambda name: resolved[name]

        names = [
            profile_name
            async for _model, profile_name in factory.iter_fallback_chain(
                original_profile="premium-claude",
                agent_fallback_profile="agent-fb",
            )
        ]

        assert names == ["agent-fb", "chain-a", "chain-b"]

    async def test_a_hop_that_fails_to_load_is_skipped_not_stopped_at(
        self, factory: ModelFactory
    ) -> None:
        """chain-a fails to resolve; the walk continues to chain-b
        instead of stopping (this is the multi-hop behavior
        _try_configurable_fallback intentionally does NOT have — it
        stops at the first success)."""
        config_mock = cast(Mock, factory._config_manager)
        profile_configs: dict[str, dict[str, Any]] = {
            "premium-claude": {
                "model": "claude-sonnet-4",
                "provider": "anthropic-api",
                "fallback_model_profile": "chain-a",
            },
            "chain-a": {
                "model": "gone",
                "provider": "llama-server",
                "fallback_model_profile": "chain-b",
            },
            "chain-b": {"model": "qwen3-8b", "provider": "llama-server"},
        }
        config_mock.get_model_profile.side_effect = profile_configs.get

        def resolve(name: str) -> tuple[str, str]:
            if name == "chain-a":
                raise ValueError("Model profile 'chain-a' not found")
            return "qwen3-8b", "llama-server"

        config_mock.resolve_model_profile.side_effect = resolve

        names = [
            profile_name
            async for _model, profile_name in factory.iter_fallback_chain(
                original_profile="premium-claude"
            )
        ]

        assert names == ["chain-b"]

    async def test_cycle_is_still_detected(self, factory: ModelFactory) -> None:
        config_mock = cast(Mock, factory._config_manager)
        profile_configs: dict[str, dict[str, Any]] = {
            "profile-a": {
                "model": "model-a",
                "provider": "provider-a",
                "fallback_model_profile": "profile-b",
            },
            "profile-b": {
                "model": "model-b",
                "provider": "provider-b",
                "fallback_model_profile": "profile-a",
            },
        }
        config_mock.get_model_profile.side_effect = profile_configs.get
        config_mock.resolve_model_profile.side_effect = ValueError("always fails")

        with pytest.raises(ValueError, match="Cycle detected in fallback chain"):
            async for _ in factory.iter_fallback_chain(original_profile="profile-a"):
                pass
