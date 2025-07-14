"""Provider registry for authentication and model management."""

from dataclasses import dataclass
from enum import Enum


class AuthMethod(Enum):
    """Supported authentication methods."""

    API_KEY = "api_key"
    OAUTH = "oauth"
    NONE = "none"


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""

    key: str  # Internal identifier (e.g., "anthropic")
    display_name: str  # User-friendly name (e.g., "Anthropic (Claude)")
    description: str  # Brief description of capabilities
    auth_methods: list[AuthMethod]  # Supported authentication methods
    emoji: str = ""  # Optional emoji for display
    oauth_provider_key: str | None = None  # OAuth provider identifier if different
    requires_subscription: bool = False  # True if OAuth requires paid subscription

    @property
    def supports_oauth(self) -> bool:
        """Check if provider supports OAuth authentication."""
        return AuthMethod.OAUTH in self.auth_methods

    @property
    def supports_api_key(self) -> bool:
        """Check if provider supports API key authentication."""
        return AuthMethod.API_KEY in self.auth_methods

    @property
    def requires_auth(self) -> bool:
        """Check if provider requires any authentication."""
        return AuthMethod.NONE not in self.auth_methods


class ProviderRegistry:
    """Registry of available LLM providers and their capabilities."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderInfo] = {}
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register the default set of supported providers."""
        # Anthropic - supports both OAuth and API key
        self.register(
            ProviderInfo(
                key="anthropic",
                display_name="Anthropic (Claude)",
                description="Supports Claude Pro/Max + API Key",
                auth_methods=[AuthMethod.OAUTH, AuthMethod.API_KEY],
                emoji="🎯",
                oauth_provider_key="anthropic-claude-pro-max",
                requires_subscription=True,
            )
        )

        # Google - API key only
        self.register(
            ProviderInfo(
                key="google",
                display_name="Google (Gemini)",
                description="API Key only",
                auth_methods=[AuthMethod.API_KEY],
                emoji="🔍",
            )
        )

        # Ollama - no auth needed
        self.register(
            ProviderInfo(
                key="ollama",
                display_name="Ollama (Local models)",
                description="No authentication needed",
                auth_methods=[AuthMethod.NONE],
                emoji="🏠",
            )
        )

    def register(self, provider_info: ProviderInfo) -> None:
        """Register a new provider."""
        self._providers[provider_info.key] = provider_info

    def get_provider(self, key: str) -> ProviderInfo | None:
        """Get provider info by key."""
        return self._providers.get(key)

    def list_providers(self) -> list[ProviderInfo]:
        """List all registered providers."""
        return list(self._providers.values())

    def get_oauth_providers(self) -> list[ProviderInfo]:
        """Get providers that support OAuth authentication."""
        return [p for p in self._providers.values() if p.supports_oauth]

    def get_api_key_providers(self) -> list[ProviderInfo]:
        """Get providers that support API key authentication."""
        return [p for p in self._providers.values() if p.supports_api_key]

    def get_no_auth_providers(self) -> list[ProviderInfo]:
        """Get providers that don't require authentication."""
        return [p for p in self._providers.values() if not p.requires_auth]

    def provider_exists(self, key: str) -> bool:
        """Check if a provider exists in the registry."""
        return key in self._providers


# Global provider registry instance
provider_registry = ProviderRegistry()
