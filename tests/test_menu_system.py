"""Tests for the interactive menu system."""

from unittest.mock import Mock, patch

import pytest

from llm_orc.menu_system import (
    AuthMenus,
    InteractiveMenu,
    MenuOption,
    confirm_action,
    show_error,
    show_info,
    show_success,
    show_working,
)


class TestMenuOption:
    """Test MenuOption class."""

    def test_init_basic(self) -> None:
        """Test basic menu option initialization."""
        option = MenuOption("Test Title", "Test Description", "test_value", "🔥")

        assert option.title == "Test Title"
        assert option.description == "Test Description"
        assert option.value == "test_value"
        assert option.emoji == "🔥"

    def test_init_without_value(self) -> None:
        """Test menu option initialization without explicit value."""
        option = MenuOption("Test Title", "Test Description")

        assert option.title == "Test Title"
        assert option.description == "Test Description"
        assert option.value == "Test Title"  # Should default to title
        assert option.emoji == ""

    def test_init_empty_value(self) -> None:
        """Test menu option initialization with empty value."""
        option = MenuOption("Test Title", "Test Description", "", "🔥")

        assert option.value == "Test Title"  # Should default to title when empty

    def test_display_with_description(self) -> None:
        """Test menu option display with description."""
        option = MenuOption("Test Title", "Test Description", "test_value", "🔥")

        display = option.display(1)

        assert "1. 🔥 Test Title" in display
        assert "Test Description" in display

    def test_display_without_description(self) -> None:
        """Test menu option display without description."""
        option = MenuOption("Test Title", "", "test_value", "🔥")

        display = option.display(2)

        assert display == "  2. 🔥 Test Title"

    def test_display_without_emoji(self) -> None:
        """Test menu option display without emoji."""
        option = MenuOption("Test Title", "Test Description")

        display = option.display(3)

        assert "3. Test Title" in display
        assert "Test Description" in display
        assert "🔥" not in display


class TestInteractiveMenu:
    """Test InteractiveMenu class."""

    def test_init(self) -> None:
        """Test menu initialization."""
        options = [
            MenuOption("Option 1", "Description 1"),
            MenuOption("Option 2", "Description 2"),
        ]
        menu = InteractiveMenu("Test Menu", options)

        assert menu.title == "Test Menu"
        assert menu.options == options

    @patch("click.echo")
    @patch("click.prompt")
    def test_show_menu_with_default(self, mock_prompt: Mock, mock_echo: Mock) -> None:
        """Test showing menu with default selection."""
        options = [
            MenuOption("Option 1", "First option", "value1"),
            MenuOption("Option 2", "Second option", "value2"),
        ]
        menu = InteractiveMenu("Test Menu", options)

        # Mock user selecting option 2
        mock_prompt.return_value = "2"

        result = menu.show(default=1)

        assert result == "value2"

        # Verify prompt was called with correct parameters
        mock_prompt.assert_called_once()
        call_args = mock_prompt.call_args
        assert "Choice [1-2]" in str(call_args)
        assert call_args[1]["default"] == "1"

    @patch("click.echo")
    @patch("click.prompt")
    def test_show_menu_without_default(
        self, mock_prompt: Mock, mock_echo: Mock
    ) -> None:
        """Test showing menu without default selection."""
        options = [
            MenuOption("Option 1", "", "value1"),
        ]
        menu = InteractiveMenu("Test Menu", options)

        # Mock user selecting option 1
        mock_prompt.return_value = "1"

        result = menu.show()

        assert result == "value1"

        # Verify prompt was called without default
        call_args = mock_prompt.call_args
        assert call_args[1]["default"] is None

    @patch("click.echo")
    @patch("click.prompt")
    def test_show_menu_displays_title_and_options(
        self, mock_prompt: Mock, mock_echo: Mock
    ) -> None:
        """Test that menu displays title and options correctly."""
        options = [
            MenuOption("Option 1", "First option", "value1", "🔥"),
            MenuOption("Option 2", "Second option", "value2", "⚡"),
        ]
        menu = InteractiveMenu("Test Menu Title", options)

        mock_prompt.return_value = "1"

        menu.show()

        # Check that title and separator were displayed
        assert mock_echo.call_count >= 4  # Title, separator, 2 options, empty line
        # Check that mock_echo was called with title
        title_call_found = False
        separator_call_found = False
        for call in mock_echo.call_args_list:
            if call.args and "Test Menu Title" in str(call.args[0]):
                title_call_found = True
            if call.args and "=" * len("Test Menu Title") in str(call.args[0]):
                separator_call_found = True
        assert title_call_found
        assert separator_call_found


class TestAuthMenus:
    """Test AuthMenus class functionality."""

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_provider_selection(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test provider selection menu."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.description = "Test Description"
        mock_provider.key = "test-provider"

        mock_registry.list_providers.return_value = [mock_provider]

        # Mock user selection
        mock_prompt.return_value = "1"

        result = AuthMenus.provider_selection()

        assert result == "test-provider"

    @patch("llm_orc.menu_system.provider_registry")
    def test_get_auth_method_for_provider_no_auth(self, mock_registry: Mock) -> None:
        """Test auth method selection for provider that requires no auth."""
        mock_provider = Mock()
        mock_provider.requires_auth = False

        mock_registry.get_provider.return_value = mock_provider

        result = AuthMenus.get_auth_method_for_provider("test-provider")

        assert result == "none"

    @patch("llm_orc.menu_system.provider_registry")
    def test_get_auth_method_for_provider_not_found(self, mock_registry: Mock) -> None:
        """Test auth method selection for non-existent provider."""
        mock_registry.get_provider.return_value = None

        with pytest.raises(ValueError, match="Provider test-provider not found"):
            AuthMenus.get_auth_method_for_provider("test-provider")

    @patch("llm_orc.menu_system.provider_registry")
    def test_get_auth_method_single_oauth(self, mock_registry: Mock) -> None:
        """Test auth method selection when only OAuth is available."""
        mock_provider = Mock()
        mock_provider.requires_auth = True
        mock_provider.supports_oauth = True
        mock_provider.supports_api_key = False
        mock_provider.requires_subscription = False
        mock_provider.display_name = "Test Provider"

        mock_registry.get_provider.return_value = mock_provider

        result = AuthMenus.get_auth_method_for_provider("test-provider")

        assert result == "oauth"

    @patch("llm_orc.menu_system.provider_registry")
    def test_get_auth_method_single_api_key(self, mock_registry: Mock) -> None:
        """Test auth method selection when only API key is available."""
        mock_provider = Mock()
        mock_provider.requires_auth = True
        mock_provider.supports_oauth = False
        mock_provider.supports_api_key = True
        mock_provider.display_name = "Test Provider"

        mock_registry.get_provider.return_value = mock_provider

        result = AuthMenus.get_auth_method_for_provider("test-provider")

        assert result == "api-key"

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_get_auth_method_multiple_options(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth method selection when multiple options are available."""
        mock_provider = Mock()
        mock_provider.requires_auth = True
        mock_provider.supports_oauth = True
        mock_provider.supports_api_key = True
        mock_provider.requires_subscription = True
        mock_provider.display_name = "Test Provider"

        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting OAuth (option 1)
        mock_prompt.return_value = "1"

        result = AuthMenus.get_auth_method_for_provider("test-provider")

        assert result == "oauth"

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_get_auth_method_both_methods(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test selecting 'both methods' option."""
        mock_provider = Mock()
        mock_provider.requires_auth = True
        mock_provider.supports_oauth = True
        mock_provider.supports_api_key = True
        mock_provider.requires_subscription = False
        mock_provider.display_name = "Test Provider"

        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting "Both methods" (option 3)
        mock_prompt.return_value = "3"

        result = AuthMenus.get_auth_method_for_provider("test-provider")

        assert result == "both"

    @patch("click.echo")
    def test_auth_list_actions_no_providers(self, mock_echo: Mock) -> None:
        """Test auth list actions when no providers are configured."""
        action, provider = AuthMenus.auth_list_actions([])

        assert action == "setup"
        assert provider is None

        # Verify appropriate message was shown
        message_found = False
        for call in mock_echo.call_args_list:
            if call.args and "No authentication providers configured" in str(
                call.args[0]
            ):
                message_found = True
                break
        assert message_found

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_with_providers_quit(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions with providers, selecting quit."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting quit (option 6)
        mock_prompt.return_value = "6"

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "quit"
        assert provider is None

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_with_providers_test(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions with providers, selecting test."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting test (option 1), then provider (option 1)
        mock_prompt.side_effect = ["1", "1"]

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "test"
        assert provider == "test-provider"

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_add_provider(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions selecting add new provider."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting add (option 2)
        mock_prompt.return_value = "2"

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "add"
        assert provider is None

    @patch("click.prompt")
    @patch("click.echo")
    def test_troubleshooting_menu(self, mock_echo: Mock, mock_prompt: Mock) -> None:
        """Test troubleshooting menu selection."""
        # Mock user selecting "Authentication failed" (option 1)
        mock_prompt.return_value = "1"

        result = AuthMenus.troubleshooting_menu()

        assert result == "auth-failed"

    @patch("click.prompt")
    @patch("click.echo")
    def test_troubleshooting_menu_browser_issues(
        self, mock_echo: Mock, mock_prompt: Mock
    ) -> None:
        """Test troubleshooting menu selecting browser issues."""
        # Mock user selecting "Browser authentication not working" (option 2)
        mock_prompt.return_value = "2"

        result = AuthMenus.troubleshooting_menu()

        assert result == "browser-issues"

    @patch("click.prompt")
    @patch("click.echo")
    def test_troubleshooting_menu_other_issue(
        self, mock_echo: Mock, mock_prompt: Mock
    ) -> None:
        """Test troubleshooting menu selecting other issue."""
        # Mock user selecting "Other issue" (option 6)
        mock_prompt.return_value = "6"

        result = AuthMenus.troubleshooting_menu()

        assert result == "other"


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("click.confirm")
    def test_confirm_action_default_false(self, mock_confirm: Mock) -> None:
        """Test confirm action with default False."""
        mock_confirm.return_value = True

        result = confirm_action("Delete this item?")

        assert result is True
        mock_confirm.assert_called_once_with("⚠️  Delete this item?", default=False)

    @patch("click.confirm")
    def test_confirm_action_default_true(self, mock_confirm: Mock) -> None:
        """Test confirm action with default True."""
        mock_confirm.return_value = False

        result = confirm_action("Save changes?", default=True)

        assert result is False
        mock_confirm.assert_called_once_with("⚠️  Save changes?", default=True)

    @patch("click.echo")
    def test_show_success(self, mock_echo: Mock) -> None:
        """Test show success message."""
        show_success("Operation completed")

        mock_echo.assert_called_once_with("✅ Operation completed")

    @patch("click.echo")
    def test_show_error(self, mock_echo: Mock) -> None:
        """Test show error message."""
        show_error("Something went wrong")

        mock_echo.assert_called_once_with("❌ Something went wrong")

    @patch("click.echo")
    def test_show_info(self, mock_echo: Mock) -> None:
        """Test show info message."""
        show_info("Information message")

        mock_echo.assert_called_once_with("ℹ️  Information message")

    @patch("click.echo")
    def test_show_working(self, mock_echo: Mock) -> None:
        """Test show working message."""
        show_working("Processing...")

        mock_echo.assert_called_once_with("🔄 Processing...")


class TestProviderRegistryIntegration:
    """Test integration with provider registry."""

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_missing_provider_info(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions when provider info is missing."""
        # Mock provider registry returning None for provider info
        mock_registry.get_provider.return_value = None

        # Mock user selecting quit
        mock_prompt.return_value = "6"

        action, provider = AuthMenus.auth_list_actions(["unknown-provider"])

        assert action == "quit"
        assert provider is None

        # Verify the provider was displayed with fallback info
        provider_found = False
        for call in mock_echo.call_args_list:
            if call.args and "unknown-provider" in str(call.args[0]):
                provider_found = True
                break
        assert provider_found

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_remove_action(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions selecting remove provider."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting remove (option 3), then provider (option 1)
        mock_prompt.side_effect = ["3", "1"]

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "remove"
        assert provider == "test-provider"

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_details_action(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions selecting view details."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting details (option 4), then provider (option 1)
        mock_prompt.side_effect = ["4", "1"]

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "details"
        assert provider == "test-provider"

    @patch("llm_orc.menu_system.provider_registry")
    @patch("click.prompt")
    @patch("click.echo")
    def test_auth_list_actions_refresh_action(
        self, mock_echo: Mock, mock_prompt: Mock, mock_registry: Mock
    ) -> None:
        """Test auth list actions selecting refresh tokens."""
        # Mock provider registry
        mock_provider = Mock()
        mock_provider.display_name = "Test Provider"
        mock_provider.emoji = "🔥"
        mock_registry.get_provider.return_value = mock_provider

        # Mock user selecting refresh (option 5), then provider (option 1)
        mock_prompt.side_effect = ["5", "1"]

        action, provider = AuthMenus.auth_list_actions(["test-provider"])

        assert action == "refresh"
        assert provider == "test-provider"
