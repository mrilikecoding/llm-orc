"""Common CLI utility functions."""

from typing import Any

import click


def handle_cli_error(operation: str, error: Exception) -> None:
    """Handle CLI errors with consistent formatting.

    Args:
        operation: Description of the operation that failed
        error: The exception that occurred

    Raises:
        click.ClickException: Always raises with formatted error message
    """
    raise click.ClickException(f"{operation} failed: {error}") from error


def confirm_destructive_action(message: str) -> bool:
    """Confirm a destructive action with the user.

    Args:
        message: Message to display for confirmation

    Returns:
        True if user confirms, False otherwise
    """
    return click.confirm(f"⚠️  {message}", default=False)


def echo_success(message: str) -> None:
    """Echo a success message with consistent formatting.

    Args:
        message: Success message to display
    """
    click.echo(f"✅ {message}")


def echo_error(message: str) -> None:
    """Echo an error message with consistent formatting.

    Args:
        message: Error message to display
    """
    click.echo(f"❌ {message}")


def echo_info(message: str) -> None:
    """Echo an info message with consistent formatting.

    Args:
        message: Info message to display
    """
    click.echo(f"ℹ️  {message}")


def echo_warning(message: str) -> None:
    """Echo a warning message with consistent formatting.

    Args:
        message: Warning message to display
    """
    click.echo(f"⚠️  {message}")


def validate_required_param(value: Any, param_name: str) -> None:
    """Validate that a required parameter has a value.

    Args:
        value: Parameter value to check
        param_name: Name of parameter for error message

    Raises:
        click.ClickException: If value is None or empty
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise click.ClickException(f"Missing required parameter: {param_name}")


def format_list_output(
    items: list[str], prefix: str = "  ", empty_message: str = "No items found"
) -> None:
    """Format and display a list of items.

    Args:
        items: List of items to display
        prefix: Prefix for each item (default: "  ")
        empty_message: Message to show if list is empty
    """
    if not items:
        echo_info(empty_message)
        return

    for item in items:
        click.echo(f"{prefix}{item}")


def format_key_value_output(
    data: dict[str, Any], prefix: str = "  ", separator: str = ": "
) -> None:
    """Format and display key-value pairs.

    Args:
        data: Dictionary of key-value pairs to display
        prefix: Prefix for each line (default: "  ")
        separator: Separator between key and value (default: ": ")
    """
    for key, value in data.items():
        click.echo(f"{prefix}{key}{separator}{value}")


def truncate_string(text: str, max_length: int = 80) -> str:
    """Truncate a string to a maximum length with ellipsis.

    Args:
        text: String to truncate
        max_length: Maximum length (default: 80)

    Returns:
        Truncated string with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
