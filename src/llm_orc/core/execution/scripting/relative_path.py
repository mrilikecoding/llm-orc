"""The relative-path rule shared by run requests and script file blocks."""

from __future__ import annotations


def check_relative(key: str, what: str) -> None:
    """A relative path: no ``..``, no leading ``/``, no backslash, no
    empty or ``.`` segment."""
    if not key or "\\" in key or "\0" in key or key.startswith("/"):
        raise ValueError(f"{what} {key!r} is not a relative path")
    if any(segment in ("", ".", "..") for segment in key.split("/")):
        raise ValueError(f"{what} {key!r} is not a relative path")
