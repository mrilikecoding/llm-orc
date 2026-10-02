"""One-run injection: the request and what it writes (Arc 4).

A run request names its root (an installed ensemble, or one inline) and
may carry child ensembles, profiles, scripts, bindings and a ``pull``
flag. ``materialize`` writes the injected parts into a run directory
shaped like every other config tier; ``apply_bindings`` adds one layer
profile per binding. Every path that reaches the disk is validated
before the first write and checked to resolve inside the run directory,
so the engine's own writes and its cleanup stay inside it (spec Arc 4
re-cut, rulings 3 and 9).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_orc.core.config.config_manager import ConfigurationManager

_FILE_MODE = 0o755


class RunRequestError(ValueError):
    """The request is malformed: the ``invalid_request`` carrier."""


def _check_relative(key: str, what: str) -> None:
    """A relative path: no ``..``, no leading ``/``, no backslash, no
    empty or ``.`` segment."""
    if not key or "\\" in key or "\0" in key or key.startswith("/"):
        raise ValueError(f"{what} {key!r} is not a relative path")
    if any(segment in ("", ".", "..") for segment in key.split("/")):
        raise ValueError(f"{what} {key!r} is not a relative path")


def _check_plain(key: str, what: str) -> None:
    """A name that becomes one file name: a relative path with no
    separator."""
    _check_relative(key, what)
    if "/" in key:
        raise ValueError(f"{what} {key!r} must not contain '/'")


class RunRequest(BaseModel):
    """What one run is asked to do. Unknown keys are an error."""

    model_config = ConfigDict(extra="forbid")

    ensemble_name: str | None = None
    ensemble: dict[str, Any] | None = None
    ensembles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scripts: dict[str, str] = Field(default_factory=dict)
    bind: dict[str, str] = Field(default_factory=dict)
    pull: bool = False
    input: str = ""

    @model_validator(mode="after")
    def _validate(self) -> RunRequest:
        if (self.ensemble is None) == (self.ensemble_name is None):
            raise ValueError("give exactly one of ensemble and ensemble_name")
        for key in self.ensembles:
            _check_relative(key, "ensemble name")
        for key in (*self.profiles, *self.bind):
            _check_plain(key, "profile name")
        for key in self.scripts:
            _check_relative(key, "script key")
        if self.ensemble is not None:
            self._validate_root(self.ensemble)
        twice = sorted(set(self.profiles) & set(self.bind))
        if twice:
            raise ValueError(f"profile {twice[0]!r} is defined twice (inline and bind)")
        return self

    def _validate_root(self, root: dict[str, Any]) -> None:
        name = root.get("name")
        if not isinstance(name, str):
            raise ValueError("an inline ensemble needs a name")
        _check_relative(name, "ensemble name")
        if name in self.ensembles:
            raise ValueError(f"ensemble {name!r} is defined twice (root and child)")

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> RunRequest:
        """Validate ``data``; every failure is a ``RunRequestError``."""
        try:
            return cls.model_validate(dict(data))
        except ValidationError as e:
            raise RunRequestError(_first_message(e)) from e

    @property
    def needs_layer(self) -> bool:
        """Whether anything is injected, so the run needs a layer."""
        return bool(
            self.ensemble is not None
            or self.ensembles
            or self.profiles
            or self.scripts
            or self.bind
        )


def _first_message(error: ValidationError) -> str:
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    message = str(first["msg"]).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


def _target(run_dir: Path, *parts: str) -> Path:
    """``run_dir / parts``, refused when it resolves outside ``run_dir``."""
    path = run_dir.joinpath(*parts)
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise RunRequestError(f"{'/'.join(parts)!r} resolves outside the run directory")
    return path


def _dump(data: Mapping[str, Any]) -> str:
    try:
        return yaml.safe_dump(dict(data))
    except yaml.YAMLError as e:
        raise RunRequestError(f"definition is not plain data: {e}") from e


def materialize(request: RunRequest, run_dir: Path) -> Path | None:
    """Write the injected ensembles, profiles and scripts into ``run_dir``.

    Returns the root's path when the root is inline. Everything is
    validated and every target checked before the first write.
    """
    writes: list[tuple[Path, str, int | None]] = []
    root_path: Path | None = None
    if request.ensemble is not None:
        root_path = _target(run_dir, "ensembles", f"{request.ensemble['name']}.yaml")
        writes.append((root_path, _dump(request.ensemble), None))
    for key, definition in request.ensembles.items():
        data = {"name": key.rsplit("/", 1)[-1], **definition}
        path = _target(run_dir, "ensembles", f"{key}.yaml")
        writes.append((path, _dump(data), None))
    for key, definition in request.profiles.items():
        path = _target(run_dir, "profiles", f"{key}.yaml")
        writes.append((path, _dump({**definition, "name": key}), None))
    for key, source in request.scripts.items():
        writes.append((_target(run_dir, key), source, _FILE_MODE))
    for path, text, mode in writes:
        _write(path, text, mode)
    return root_path


def _write(path: Path, text: str, mode: int | None) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if mode is not None:
            path.chmod(mode)
    except OSError as e:
        raise RunRequestError(f"cannot write {path.name!r}: {e}") from e


def apply_bindings(
    bind: Mapping[str, str], view: ConfigurationManager, run_dir: Path
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Write a layer profile ``a`` carrying ``b``'s definition for each
    ``a: b``, ``b`` read from the view as it stood before any binding
    was written (one hop: a target is never itself rebound).

    Returns the applied map and the unmet ``(a, b)`` pairs; an unmet
    pair writes nothing.
    """
    known = dict(view.get_model_profiles())
    applied: dict[str, str] = {}
    unmet: list[tuple[str, str]] = []
    for key, target in bind.items():
        definition = known.get(target)
        if definition is None:
            unmet.append((key, target))
            continue
        path = _target(run_dir, "profiles", f"{key}.yaml")
        _write(path, _dump({**definition, "name": key}), None)
        applied[key] = target
    return applied, unmet
