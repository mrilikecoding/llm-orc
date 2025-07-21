"""Interactive menu system for LLM Orchestra CLI."""

from typing import Any

import click

from .providers.registry import provider_registry


class MenuOption:
    """Represents a menu option with title, description, and value."""

    def __init__(
        self, title: str, description: str = "", value: Any = None, emoji: str = ""
    ):
        self.title = title
        self.description = description
        self.value = value or title
        self.emoji = emoji

    def display(self, index: int) -> str:
        """Display the option with number and formatting."""
        prefix = f"{self.emoji} " if self.emoji else ""
        if self.description:
            return f"  {index}. {prefix}{self.title}\n     {self.description}"
        return f"  {index}. {prefix}{self.title}"


class InteractiveMenu:
    """Interactive menu system for CLI applications."""

    def __init__(self, title: str, options: list[MenuOption]):
        self.title = title
        self.options = options

    def show(self, default: int | None = None) -> Any:
        """Display menu and return selected option value."""
        click.echo(f"\n{self.title}")
        click.echo("=" * len(self.title))
        click.echo()

        # Display options
        for i, option in enumerate(self.options, 1):
            click.echo(option.display(i))
            click.echo()

        # Get user choice
        choices = [str(i) for i in range(1, len(self.options) + 1)]
        prompt_text = f"Choice [1-{len(self.options)}]"

        choice = click.prompt(
            prompt_text,
            type=click.Choice(choices),
            default=str(default) if default else None,
        )

        return self.options[int(choice) - 1].value


class AuthMenus:
    """Authentication-specific menu implementations."""

    @staticmethod
    def provider_selection() -> str:
        """Show provider selection menu with dynamic provider list."""
        providers = provider_registry.list_providers()

        options = []
        for provider in providers:
            options.append(
                MenuOption(
                    provider.display_name,
                    provider.description,
                    provider.key,
                    "✨",
                )
            )

        # Note: Custom provider option removed - only show supported providers

        menu = InteractiveMenu("🚀 Select a Provider to Configure", options)
        result = menu.show(default=1)
        return str(result)

    # Note: anthropic_auth_method removed - now handled by specific provider keys
    # (anthropic-api and anthropic-claude-pro-max)

    @staticmethod
    def get_auth_method_for_provider(provider_key: str) -> str:
        """Show authentication method selection for any provider."""
        provider = provider_registry.get_provider(provider_key)
        if not provider:
            raise ValueError(f"Provider {provider_key} not found in registry")

        if not provider.requires_auth:
            return "none"

        options = []

        if provider.supports_oauth:
            oauth_description = "• OAuth authentication"
            if provider.requires_subscription:
                oauth_description += (
                    f"\n     • Requires {provider.display_name} subscription"
                )
            options.append(MenuOption("OAuth", oauth_description, "oauth", "🔐"))

        if provider.supports_api_key:
            options.append(
                MenuOption(
                    "API Key",
                    "• API key authentication\n     • Direct API access",
                    "api-key",
                    "🔑",
                )
            )

        if len(options) == 1:
            # Only one method available, use it automatically
            return str(options[0].value)

        if len(options) > 1:
            options.append(
                MenuOption(
                    "Both methods",
                    "• Set up multiple authentication options",
                    "both",
                    "🔄",
                )
            )

        menu = InteractiveMenu(
            f"🎯 Choose Authentication Method for {provider.display_name}", options
        )
        result = menu.show(default=1)
        return str(result)

    @staticmethod
    def auth_list_actions(providers: list[str]) -> tuple[str, str | None]:
        """Show auth list with action menu."""
        if not providers:
            click.echo("🔍 No authentication providers configured")
            click.echo("Run 'llm-orc auth setup' to get started.")
            return "setup", None

        # First show provider status
        click.echo("🔐 Configured Authentication Providers\n")
        for i, provider in enumerate(providers, 1):
            # Get provider info for better display
            provider_info = provider_registry.get_provider(provider)
            display_name = provider_info.display_name if provider_info else provider
            emoji = provider_info.emoji if provider_info else ""

            status = "✅ Active"  # TODO: Get actual status
            click.echo(f"  {i}. {emoji} {display_name} ({status})")
        click.echo()

        # Show action menu
        options = [
            MenuOption(
                "Test authentication", "Verify a provider is working", "test", "🧪"
            ),
            MenuOption(
                "Add new provider", "Configure additional authentication", "add", "➕"
            ),
            MenuOption(
                "Remove provider", "Delete authentication credentials", "remove", "🗑️"
            ),
            MenuOption(
                "View details", "Show detailed provider information", "details", "📋"
            ),
            MenuOption("Refresh tokens", "Update OAuth tokens", "refresh", "🔄"),
            MenuOption("Quit", "Exit authentication management", "quit", "🚪"),
        ]

        menu = InteractiveMenu("Select Action", options)
        action = menu.show(default=6)

        if action in ["test", "remove", "details", "refresh"]:
            # Need to select which provider
            provider_options = []
            for provider in providers:
                provider_info = provider_registry.get_provider(provider)
                display_name = provider_info.display_name if provider_info else provider
                provider_options.append(
                    MenuOption(
                        display_name, f"Perform action on {display_name}", provider
                    )
                )

            provider_menu = InteractiveMenu(
                f"Select Provider to {action.title()}", provider_options
            )
            selected_provider = provider_menu.show()
            return action, selected_provider

        return action, None

    @staticmethod
    def troubleshooting_menu() -> str:
        """Show guided troubleshooting menu."""
        options = [
            MenuOption(
                "Authentication failed",
                "Token expired or invalid credentials",
                "auth-failed",
                "❌",
            ),
            MenuOption(
                "Browser authentication not working",
                "OAuth flow issues with browser",
                "browser-issues",
                "🌐",
            ),
            MenuOption(
                "Token refresh problems",
                "OAuth token refresh failures",
                "token-refresh",
                "🔄",
            ),
            MenuOption(
                "Provider not found",
                "No authentication configured",
                "no-provider",
                "🔍",
            ),
            MenuOption(
                "Permission errors",
                "File system or configuration issues",
                "permissions",
                "🔒",
            ),
            MenuOption("Other issue", "General troubleshooting guidance", "other", "🛠️"),
        ]

        menu = InteractiveMenu("🔧 What problem are you experiencing?", options)
        result = menu.show()
        return str(result)


def confirm_action(message: str, default: bool = False) -> bool:
    """Show confirmation dialog."""
    return click.confirm(f"⚠️  {message}", default=default)


def show_success(message: str) -> None:
    """Show success message with formatting."""
    click.echo(f"✅ {message}")


def show_error(message: str) -> None:
    """Show error message with formatting."""
    click.echo(f"❌ {message}")


def show_info(message: str) -> None:
    """Show info message with formatting."""
    click.echo(f"ℹ️  {message}")


def show_working(message: str) -> None:
    """Show working/progress message."""
    click.echo(f"🔄 {message}")
