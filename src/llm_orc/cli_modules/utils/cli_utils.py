"""Common CLI utility functions."""

import click


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
