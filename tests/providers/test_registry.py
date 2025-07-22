"""Comprehensive tests for provider registry."""

from llm_orc.providers.registry import (
    AuthMethod,
    ProviderInfo,
    ProviderRegistry,
    provider_registry,
)


class TestProviderInfo:
    """Test ProviderInfo dataclass and properties."""

    def test_provider_info_supports_oauth(self) -> None:
        """Test supports_oauth property with OAuth method."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.OAUTH],
        )
        assert provider.supports_oauth is True

    def test_provider_info_supports_oauth_false(self) -> None:
        """Test supports_oauth property without OAuth method."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.API_KEY],
        )
        assert provider.supports_oauth is False

    def test_provider_info_supports_api_key(self) -> None:
        """Test supports_api_key property with API key method."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.API_KEY],
        )
        assert provider.supports_api_key is True

    def test_provider_info_supports_api_key_false(self) -> None:
        """Test supports_api_key property without API key method."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.OAUTH],
        )
        assert provider.supports_api_key is False

    def test_provider_info_requires_auth_true(self) -> None:
        """Test requires_auth property when auth is required."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.API_KEY],
        )
        assert provider.requires_auth is True

    def test_provider_info_requires_auth_false(self) -> None:
        """Test requires_auth property when no auth required."""
        provider = ProviderInfo(
            key="test",
            display_name="Test",
            description="Test provider",
            auth_methods=[AuthMethod.NONE],
        )
        assert provider.requires_auth is False


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    def test_provider_registry_initialization(self) -> None:
        """Test registry initializes with default providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()

        # Should have at least the default providers
        assert len(providers) >= 4
        provider_keys = [p.key for p in providers]
        assert "anthropic-api" in provider_keys
        assert "anthropic-claude-pro-max" in provider_keys
        assert "google-gemini" in provider_keys
        assert "ollama" in provider_keys

    def test_register_custom_provider(self) -> None:
        """Test registering a custom provider."""
        registry = ProviderRegistry()
        custom_provider = ProviderInfo(
            key="custom",
            display_name="Custom Provider",
            description="Custom test provider",
            auth_methods=[AuthMethod.API_KEY],
        )

        registry.register(custom_provider)
        retrieved = registry.get_provider("custom")
        assert retrieved is not None
        assert retrieved.key == "custom"
        assert retrieved.display_name == "Custom Provider"

    def test_get_provider_existing(self) -> None:
        """Test getting an existing provider."""
        registry = ProviderRegistry()
        provider = registry.get_provider("anthropic-api")
        assert provider is not None
        assert provider.key == "anthropic-api"

    def test_get_provider_nonexistent(self) -> None:
        """Test getting a non-existent provider."""
        registry = ProviderRegistry()
        provider = registry.get_provider("nonexistent")
        assert provider is None

    def test_get_oauth_providers(self) -> None:
        """Test filtering providers that support OAuth."""
        registry = ProviderRegistry()
        oauth_providers = registry.get_oauth_providers()

        # Should include anthropic-claude-pro-max
        oauth_keys = [p.key for p in oauth_providers]
        assert "anthropic-claude-pro-max" in oauth_keys

        # All returned providers should support OAuth
        for provider in oauth_providers:
            assert provider.supports_oauth is True

    def test_get_api_key_providers(self) -> None:
        """Test filtering providers that support API keys."""
        registry = ProviderRegistry()
        api_key_providers = registry.get_api_key_providers()

        # Should include anthropic-api and google-gemini
        api_key_keys = [p.key for p in api_key_providers]
        assert "anthropic-api" in api_key_keys
        assert "google-gemini" in api_key_keys

        # All returned providers should support API keys
        for provider in api_key_providers:
            assert provider.supports_api_key is True

    def test_get_no_auth_providers(self) -> None:
        """Test filtering providers that don't require auth."""
        registry = ProviderRegistry()
        no_auth_providers = registry.get_no_auth_providers()

        # Should include ollama
        no_auth_keys = [p.key for p in no_auth_providers]
        assert "ollama" in no_auth_keys

        # All returned providers should not require auth
        for provider in no_auth_providers:
            assert provider.requires_auth is False

    def test_provider_exists_true(self) -> None:
        """Test provider_exists with existing provider."""
        registry = ProviderRegistry()
        assert registry.provider_exists("anthropic-api") is True

    def test_provider_exists_false(self) -> None:
        """Test provider_exists with non-existent provider."""
        registry = ProviderRegistry()
        assert registry.provider_exists("nonexistent") is False

    def test_list_providers(self) -> None:
        """Test listing all providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()
        assert len(providers) >= 4
        assert all(isinstance(p, ProviderInfo) for p in providers)


class TestGlobalRegistry:
    """Test global provider registry instance."""

    def test_global_registry_exists(self) -> None:
        """Test that global registry is properly initialized."""
        assert provider_registry is not None
        assert isinstance(provider_registry, ProviderRegistry)

        # Should have default providers
        providers = provider_registry.list_providers()
        assert len(providers) >= 4
