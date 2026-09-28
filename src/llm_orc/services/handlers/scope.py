"""Write scope for CRUD handlers.

Reads merge every configuration layer (project, library, global). Writes
target exactly one scope. ``project`` is the default and is today's
behavior; ``global`` writes under ``ConfigurationManager.global_config_dir``
so a serve running from a plain directory has a writable home that
survives upgrades (#196).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

Scope = Literal["project", "global"]
SCOPES: tuple[str, ...] = ("project", "global")


def parse_scope(arguments: dict[str, Any]) -> Scope:
    """Read ``scope`` from tool arguments.

    An absent key means ``project``. An explicit value, including
    ``None``, must be one of the two scopes.
    """
    if "scope" not in arguments:
        return "project"
    raw = arguments["scope"]
    if raw not in SCOPES:
        raise ValueError(f"scope must be one of {list(SCOPES)}, got {raw!r}")
    return cast(Scope, raw)


def find_in_scope(
    *,
    name: str,
    filename: str,
    scope: Scope,
    scope_dir: Path | None,
    search_dirs: list[Path],
    classify: Callable[[Path], str],
    label: str,
) -> Path:
    """Locate ``filename`` in ``scope_dir`` only.

    A name that exists in another layer raises with that layer named, so
    a caller who asked for the wrong scope learns where the file lives
    instead of silently touching nothing, or the wrong file.
    """
    if scope_dir is not None:
        target = scope_dir / filename
        if target.exists():
            return target
    for directory in search_dirs:
        other = Path(directory) / filename
        if other.exists():
            raise ValueError(
                f"{label} '{name}' is not in scope '{scope}'; "
                f"it lives in the {classify(other)} tier at {other}"
            )
    raise ValueError(f"{label} not found: {name}")
