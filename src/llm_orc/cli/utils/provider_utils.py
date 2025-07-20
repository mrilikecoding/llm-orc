"""Provider utility functions for CLI operations."""


def get_provider_status_symbol(provider: str, available_providers: set[str]) -> str:
    """Get status symbol for a provider based on availability.

    Args:
        provider: Provider name
        available_providers: Set of available provider names

    Returns:
        Status symbol (✅ for available, ❌ for unavailable)
    """
    return "✅" if provider in available_providers else "❌"


def format_provider_display(provider: str, available: bool) -> str:
    """Format provider name with availability status.

    Args:
        provider: Provider name
        available: Whether provider is available

    Returns:
        Formatted provider display string
    """
    symbol = "✅" if available else "❌"
    return f"{symbol} {provider}"


def format_provider_list(
    providers: list[str], available_providers: set[str]
) -> list[str]:
    """Format a list of providers with their availability status.

    Args:
        providers: List of provider names
        available_providers: Set of available provider names

    Returns:
        List of formatted provider display strings
    """
    return [
        format_provider_display(provider, provider in available_providers)
        for provider in providers
    ]


def get_success_status_symbol(success: bool) -> str:
    """Get status symbol for success/failure operations.

    Args:
        success: Whether operation was successful

    Returns:
        Status symbol (✅ for success, ❌ for failure)
    """
    return "✅" if success else "❌"


def format_operation_result(
    operation: str, success: bool, details: str | None = None
) -> str:
    """Format operation result with status symbol.

    Args:
        operation: Operation description
        success: Whether operation was successful
        details: Optional additional details

    Returns:
        Formatted operation result string
    """
    symbol = get_success_status_symbol(success)
    result = f"{symbol} {operation}"
    if details:
        result += f": {details}"
    return result
