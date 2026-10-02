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

import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.execution.scripting.resolver import ScriptResolver

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


def _check_reachable(key: str) -> None:
    """A script key the resolver reads as a path, so the file written at
    the key is the one an agent referencing the key runs. Without path
    syntax the resolver takes the reference for inline shell content."""
    if not ScriptResolver().has_path_syntax(key):
        raise ValueError(
            f"script key {key!r} has no path syntax (needs a '/' or a script "
            f"extension: {', '.join(ScriptResolver.SCRIPT_EXTENSIONS)})"
        )


def _folded(path: str) -> tuple[str, ...]:
    """The segments of ``path`` as a case-folding, Unicode-normalizing
    disk (macOS) sees them: two spellings that fold equal are one name.
    The canonical caseless form is NFD(casefold(NFD(x)))."""
    decomposed = unicodedata.normalize("NFD", path)
    folded = unicodedata.normalize("NFD", decomposed.casefold())
    return tuple(folded.split("/"))


def _check_distinct(paths: Iterable[tuple[str, str]]) -> None:
    """No two ``(path, label)`` pairs meet on the disk: equal after
    folding, or one a directory prefix of the other."""
    files: dict[tuple[str, ...], str] = {}
    directories: dict[tuple[str, ...], str] = {}
    for path, label in paths:
        segments = _folded(path)
        meets = files.get(segments) or directories.get(segments)
        for end in range(1, len(segments)):
            meets = meets or files.get(segments[:end])
            directories.setdefault(segments[:end], label)
        if meets:
            raise ValueError(
                f"{label} collides with {meets}: names must be distinct ignoring case"
            )
        files[segments] = label


def _check_one_reference_one_script(keys: Iterable[str]) -> None:
    """No two script keys are reachable from the same reference. The
    resolver also tries a reference without its leading ``scripts/`` and
    with hyphens as underscores, and searches ``<run>/scripts`` before
    ``<run>``, so keys that fold to one form let the first win silently.
    The forms compare as the disk folds them (see ``_folded``)."""
    seen: dict[str, str] = {}
    for key in keys:
        folded = "/".join(_folded(key))
        form = ScriptResolver.underscored(ScriptResolver.unprefixed(folded))
        if form in seen:
            raise ValueError(
                f"script {key!r} and script {seen[form]!r} are reached by the "
                "same reference: script keys must be distinct"
            )
        seen[form] = key


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
            _check_reachable(key)
        _check_one_reference_one_script(self.scripts)
        if self.ensemble is not None:
            self._validate_root(self.ensemble)
        twice = sorted(set(self.profiles) & set(self.bind))
        if twice:
            raise ValueError(f"profile {twice[0]!r} is defined twice (inline and bind)")
        _check_distinct(self._layer_paths())
        return self

    def _layer_paths(self) -> list[tuple[str, str]]:
        """Every path the request writes into the run layer, with a label
        naming the request key it came from."""
        paths: list[tuple[str, str]] = []
        if self.ensemble is not None:
            name = self.ensemble["name"]
            paths.append((f"ensembles/{name}.yaml", f"ensemble {name!r}"))
        paths.extend((f"ensembles/{k}.yaml", f"ensemble {k!r}") for k in self.ensembles)
        paths.extend((f"profiles/{k}.yaml", f"profile {k!r}") for k in self.profiles)
        paths.extend((f"profiles/{k}.yaml", f"bind key {k!r}") for k in self.bind)
        paths.extend((k, f"script {k!r}") for k in self.scripts)
        return paths

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
    writes: list[tuple[Path, str, int | None, str]] = []
    root_path: Path | None = None
    if request.ensemble is not None:
        name = request.ensemble["name"]
        root_path = _target(run_dir, "ensembles", f"{name}.yaml")
        writes.append((root_path, _dump(request.ensemble), None, name))
    for key, definition in request.ensembles.items():
        data = {"name": key.rsplit("/", 1)[-1], **definition}
        path = _target(run_dir, "ensembles", f"{key}.yaml")
        writes.append((path, _dump(data), None, key))
    for key, definition in request.profiles.items():
        path = _target(run_dir, "profiles", f"{key}.yaml")
        writes.append((path, _dump({**definition, "name": key}), None, key))
    for key, source in request.scripts.items():
        writes.append((_target(run_dir, key), source, _FILE_MODE, key))
    for path, text, mode, key in writes:
        _write(path, text, mode, key)
    return root_path


def _write(path: Path, text: str, mode: int | None, key: str) -> None:
    """Write ``text`` at ``path``; a failure names the request ``key``
    and the OS reason, never the path (it holds the run directory)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if mode is not None:
            path.chmod(mode)
    except OSError as e:
        raise RunRequestError(f"cannot write {key!r}: {e.strerror or e}") from e


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
        _write(path, _dump({**definition, "name": key}), None, key)
        applied[key] = target
    return applied, unmet
