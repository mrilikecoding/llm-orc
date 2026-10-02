"""Named remotes: resolve a ``--remote`` value to a base URL."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from llm_orc.core.config.config_manager import ConfigurationManager


class RemoteError(ValueError):
    """A remote value that cannot be resolved to a URL."""


def resolve_remote(value: str, config: ConfigurationManager) -> str:
    """The base URL for ``value``: a URL as given, or a configured name.

    A URL must be ``http`` or ``https`` with a host and, when it has one, a
    port from 1 to 65535; anything else raises ``RemoteError``.
    """
    if "://" in value:
        return _checked_url(value, None)
    remotes = config.remotes()
    if value not in remotes:
        known = ", ".join(sorted(remotes)) or None
        detail = f"known remotes: {known}" if known else "no remotes are configured"
        raise RemoteError(f"Unknown remote '{value}'; {detail}")
    return _checked_url(remotes[value], f"remote '{value}'")


def _checked_url(url: str, source: str | None) -> str:
    """``url`` without a trailing slash, or ``RemoteError`` saying what is
    wrong with it (``source`` names the configured remote it came from)."""
    where = f"{source}: " if source else ""
    for char in url:
        if not char.isprintable() or char.isspace():
            raise RemoteError(
                f"{where}{url!r} contains a character that is not allowed "
                f"in a URL: {char!r}"
            )
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise RemoteError(f"{where}{url!r} is not a usable URL: {e}") from e
    if parts.scheme not in ("http", "https"):
        raise RemoteError(
            f"{where}{url!r} must use http or https, not {parts.scheme!r}"
        )
    for mark, name in (("?", "query"), ("#", "fragment")):
        if mark in url:
            raise RemoteError(f"{where}{url!r} has a {name}; give the base URL alone")
    if not parts.hostname:
        raise RemoteError(f"{where}{url!r} has no host")
    try:
        port = parts.port
    except ValueError:
        port = 0
    if port is not None and port < 1:
        raise RemoteError(f"{where}{url!r} has a port that is not 1-65535")
    return url.rstrip("/")
