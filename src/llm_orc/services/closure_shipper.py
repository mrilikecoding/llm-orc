"""The closure shipper: a named local root as one run request (Arc 5).

The request carries the root and every child that resolves locally as
the parsed YAML of the file the loader read, and every script and listed
file with path syntax as its text, so another serve runs the caller's
closure. The walk uses the run's own lookups (``run_view``), never a
second one.
"""

from __future__ import annotations

import json
import os
import posixpath
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.config.closure import Closure, walk_closure
from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig
from llm_orc.core.execution.scripting.relative_path import check_relative
from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.services.handlers.run_request import (
    RunRequest,
    RunRequestError,
    script_key_form,
)
from llm_orc.services.handlers.run_view import run_view
from llm_orc.services.handlers.script_files import LocatedScript, ScriptFileLocator


class ShipError(ValueError):
    """The closure cannot be shipped; the message names the reference."""


def ship_closure(
    root_name: str,
    *,
    find_root: Callable[[str], EnsembleConfig | None],
    config_manager: ConfigurationManager,
    project_dir: Path | None,
    with_profiles: Sequence[str] = (),
    bind: Mapping[str, str] | None = None,
    pull: bool = False,
    persist: str | None = None,
    input_text: str = "",
) -> dict[str, Any]:
    """The run request for ``root_name`` and its closure, in the shape
    ``RunRequest`` takes (``persist`` only when given)."""
    root = find_root(root_name)
    if root is None:
        raise ShipError(f"ensemble {root_name!r} was not found")
    view = run_view(config_manager, project_dir)
    children: dict[str, EnsembleConfig] = {}

    def find_child(reference: str) -> EnsembleConfig | None:
        child = view.find_child(reference)
        if child is not None:
            children[reference] = child
        return child

    locator = ScriptFileLocator(view.resolver)
    closure = walk_closure(
        root,
        find_child,
        view.profiles,
        root_ref=root_name,
        script_files=locator,
    )
    request: dict[str, Any] = {
        "ensemble": _as_written(root_name, root),
        "ensembles": {
            reference: _as_written(reference, child)
            for reference, child in children.items()
        },
        "profiles": _profile_definitions(with_profiles, view.profiles),
        "scripts": _script_texts(closure, locator.found, view.resolver),
        "bind": dict(bind or {}),
        "pull": pull,
        "input": input_text,
    }
    _require_valid(request)
    if persist is not None:
        request["persist"] = persist
    return request


def _require_valid(request: Mapping[str, Any]) -> None:
    """The request passes the validator the remote will run it through."""
    try:
        RunRequest.parse(request)
    except RunRequestError as e:
        raise ShipError(f"the request would be refused: {e}") from e


def _profile_definitions(
    names: Sequence[str], profiles: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in profiles:
            raise ShipError(f"profile {name!r} does not exist locally")
        definition = dict(profiles[name])
        _require_plain_data(f"profile {name!r}", definition)
        definitions[name] = definition
    return definitions


def _as_written(reference: str, config: EnsembleConfig) -> dict[str, Any]:
    if config.source_path is None:
        raise ShipError(f"ensemble {reference!r} has no file to ship")
    data = yaml.safe_load(Path(config.source_path).read_text())
    _require_plain_data(f"ensemble {reference!r}", data)
    return dict(data)


def _require_plain_data(what: str, data: Any) -> None:
    """``data`` survives a JSON round trip unchanged. A date, a set, a
    key that is not a string or a NaN would reach the remote as
    something else, or not reach it."""
    try:
        same = json.loads(json.dumps(data, allow_nan=False)) == data
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ShipError(
            f"{what} is not plain data: a date, set, non-string key or "
            "non-finite number would not reach the remote unchanged"
        )


def _script_texts(
    closure: Closure, found: Sequence[LocatedScript], resolver: ScriptResolver
) -> dict[str, str]:
    """Each script and listed file the walk found, at the key the remote's
    resolver reaches the reference by.

    The remote searches ``<run>/scripts`` then ``<run>``, trying a
    reference as written, with hyphens as underscores, and without a
    leading ``scripts/``. The key is ``scripts/`` and the last segments
    of the local file's path, as many as the reference has without its
    ``scripts/``: whichever form found the file locally, one of the
    three finds it there, and the file keeps its real name. A listed
    file sits beside its owner's key.
    """
    _refuse_absolute(closure)
    texts: dict[str, str] = {}
    owners: dict[str, tuple[str, Path]] = {}  # key form -> label, file
    keys: dict[str, str] = {}  # reference -> key
    for item in found:
        label = _label(item)
        key = _key(item, label, keys, resolver)
        keys[item.dep.name] = key
        form = script_key_form(key)
        if form in owners:
            _require_same_file(label, item.path, owners[form])
            continue
        owners[form] = (label, item.path)
        texts[key] = _utf8(label, item.path)
    return texts


def _refuse_absolute(closure: Closure) -> None:
    for dep in closure.dependencies:
        if dep.kind == "script" and os.path.isabs(dep.name):
            raise ShipError(
                f"script {dep.name!r} is an absolute path, which names "
                "nothing on another host"
            )


def _label(item: LocatedScript) -> str:
    dep = item.dep
    if dep.beside is None:
        return f"script {dep.name!r}"
    return f"file {dep.listed!r} listed by script {dep.beside!r}"


def _key(
    item: LocatedScript,
    label: str,
    keys: Mapping[str, str],
    resolver: ScriptResolver,
) -> str:
    dep = item.dep
    if dep.beside is not None:
        assert dep.listed is not None
        return posixpath.join(posixpath.dirname(keys[dep.beside]), dep.listed)
    if not resolver.has_path_syntax(dep.name):
        raise ShipError(
            f"{label} is a file in the working directory; the remote would "
            "read the name as inline shell"
        )
    try:
        check_relative(dep.name, "script reference")
    except ValueError as e:
        raise ShipError(f"{label}: {e}") from e
    count = len(ScriptResolver.unprefixed(dep.name).split("/"))
    return "scripts/" + "/".join(item.path.parts[-count:])


def _require_same_file(label: str, path: Path, seen: tuple[str, Path]) -> None:
    other, other_path = seen
    if path != other_path:
        raise ShipError(
            f"{label} and {other} are different local files that the remote "
            "would reach by the same reference"
        )


def _utf8(label: str, path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise ShipError(f"{label} is not UTF-8 text") from e
