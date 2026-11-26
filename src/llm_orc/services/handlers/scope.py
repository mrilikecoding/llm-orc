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

from llm_orc.core.execution.scripting.relative_path import check_relative

Scope = Literal["project", "global"]
SCOPES: tuple[str, ...] = ("project", "global")


class NotFoundInScopeError(ValueError):
    """No file of that name exists in any layer. A ``ValueError`` so callers
    that catch the base type keep working; REST maps it to 404."""


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


def resolve_in_dir(*, scope_dir: Path, filename: str, scope: str, label: str) -> Path:
    """``scope_dir / filename``, refused unless it stays inside ``scope_dir``.

    The name behind ``filename`` comes from a tool or REST caller, so
    ``../`` segments and absolute paths must not reach a file outside the
    scope's directory. The check resolves the parent directory (symlinks
    included) and compares it with the resolved scope directory; the
    leaf is left alone so a symlinked ensemble still resolves to itself.
    A name with a separator that stays inside (a nested ensemble, a
    script category) is fine. A name the file system cannot take (too
    long) is refused the same way, so no caller sees an ``OSError``.
    The error names no absolute path.
    """
    target = scope_dir / filename
    try:
        inside = target.parent.resolve().is_relative_to(scope_dir.resolve())
        target.exists()
    except OSError as e:
        raise ValueError(
            f"{label} name is not allowed: the file system cannot take it "
            f"({e.strerror})"
        ) from e
    if not inside:
        raise ValueError(
            f"{label} name is not allowed: it resolves outside the "
            f"'{scope}' scope directory"
        )
    return target


def check_write_target(*, path: Path, scope_dir: Path, scope: str, label: str) -> None:
    """Refuse a write to ``path`` when it resolves outside ``scope_dir``.

    ``resolve_in_dir`` leaves the leaf alone, so a leaf that is a symlink
    (dangling too, which ``exists()`` reports as absent) would carry a
    create, update or copy to wherever it points. A link that stays
    inside the directory is fine.
    """
    if not path.resolve().is_relative_to(scope_dir.resolve()):
        raise ValueError(
            f"{label} target is not allowed: it is a link that resolves "
            f"outside the '{scope}' scope directory"
        )


def resolve_names_in_dir(base_dir: Path, *names: str, label: str) -> Path:
    """``base_dir / names``, where every name is a relative path with no
    ``..``, ``.`` or empty segment and the parent stays inside ``base_dir``
    once symlinks are resolved.

    For a directory addressed by ids (an artifact is ``<ensemble>/<id>``):
    a name that climbs, or resolves to ``base_dir`` itself, would make a
    delete or a listing act on something other than what the id names.
    The error names no absolute path.
    """
    for name in names:
        try:
            check_relative(name, label)
        except ValueError as e:
            raise ValueError(f"{label} name is not allowed: {e}") from e
    target = base_dir.joinpath(*names)
    if not target.parent.resolve().is_relative_to(base_dir.resolve()):
        raise ValueError(
            f"{label} name is not allowed: it resolves outside its directory"
        )
    return target


def refuse_link(path: Path, label: str) -> None:
    """Refuse ``path`` when it is a symlink.

    ``resolve_names_in_dir`` checks the parent only, so a leaf that links
    to a directory elsewhere passes it. Removing or listing a directory
    tree through such a link would act on the target.
    """
    if path.is_symlink():
        raise ValueError(f"{label} name is not allowed: it is a link")


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
        target = resolve_in_dir(
            scope_dir=scope_dir, filename=filename, scope=scope, label=label
        )
        if target.exists():
            return target
    for directory in search_dirs:
        other = resolve_in_dir(
            scope_dir=Path(directory), filename=filename, scope=scope, label=label
        )
        if other.exists():
            raise ValueError(
                f"{label} '{name}' is not in scope '{scope}'; "
                f"it lives in the {classify(other)} tier at {other}"
            )
    raise NotFoundInScopeError(f"{label} not found: {name}")
