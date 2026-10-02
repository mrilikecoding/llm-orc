"""Named remotes: resolve a ``--remote`` value to a base URL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_orc.core.config.config_manager import ConfigurationManager


class RemoteError(ValueError):
    """A remote value that cannot be resolved to a URL."""


def resolve_remote(value: str, config: ConfigurationManager) -> str:
    """The base URL for ``value``: a URL as given, or a configured name."""
    if "://" in value:
        return value.rstrip("/")
    remotes = config.remotes()
    if value not in remotes:
        known = ", ".join(sorted(remotes)) or None
        detail = f"known remotes: {known}" if known else "no remotes are configured"
        raise RemoteError(f"Unknown remote '{value}'; {detail}")
    return remotes[value]
