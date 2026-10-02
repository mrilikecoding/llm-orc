"""Bundles: a persisted closure is a stored run request (Arc 5, ruling 9).

A bundle is the request that ran, minus ``input``, ``pull`` and
``persist``, as one JSON file at ``<global config>/bundles/<root>.json``.
Nothing is copied into a tier: a bundle is only ever seen through the
run layer a run materializes from it, so its children and scripts
resolve for its own root and shadow nothing on the host.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_orc.services.handlers.run_request import (
    RunRequest,
    RunRequestError,
    check_plain,
)

#: What a listing calls a root that a bundle holds.
BUNDLE_SOURCE = "bundle"
_SUFFIX = ".json"
_TEMP_SUFFIX = ".tmp"
_STORED_KEYS = {"ensemble", "ensembles", "profiles", "scripts", "bind"}


class BundleError(ValueError):
    """A bundle cannot be read or written."""


def stored_form(request: RunRequest) -> dict[str, Any]:
    """What a bundle holds of ``request``: the closure, not the call."""
    return request.model_dump(include=_STORED_KEYS)


def not_a_tier_ensemble(name: str) -> str:
    """The plain answer of an operation that only a tier ensemble has."""
    return (
        f"{name!r} is a bundle (a stored run request), not a tier ensemble: "
        "run it by name, or delete it with scope 'global'"
    )


class BundleStore:
    """The bundles under one global config directory.

    ``root`` is read at each call, so the store follows its config
    manager.
    """

    def __init__(self, root: Callable[[], Path]) -> None:
        self._root = root

    @property
    def directory(self) -> Path:
        return Path(self._root()) / "bundles"

    def _path(self, name: str) -> Path | None:
        """The file of ``name``; None when it is not one plain file name."""
        try:
            check_plain(name, "bundle name")
        except ValueError:
            return None
        return self.directory / f"{name}{_SUFFIX}"

    def _entry(self, name: str) -> Path | None:
        """The file of ``name`` when the directory holds an entry named
        exactly ``<name>.json``. A case-folding disk opens ``pack.json``
        for ``PACK.json``; the directory listing is what tells them apart."""
        path = self._path(name)
        if path is None:
            return None
        try:
            held = path.name in os.listdir(self.directory)
        except OSError:
            return None
        return path if held and path.is_file() else None

    def _entry_named(self, name: str) -> Path | None:
        """``_entry`` unless the file readably holds a root named
        otherwise. An unreadable file stays a bundle: it can be deleted
        and a read of it says what is wrong."""
        path = self._entry(name)
        if path is None:
            return None
        try:
            root = json.loads(path.read_text())["ensemble"]["name"]
        except (OSError, ValueError, KeyError, TypeError):
            return path
        return path if root == name else None

    def has(self, name: str) -> bool:
        return self._entry_named(name) is not None

    def names(self) -> list[str]:
        """The names of the bundle files, sorted."""
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob(f"*{_SUFFIX}") if p.is_file())

    def read(self, name: str) -> dict[str, Any] | None:
        """The stored request of ``name``; None when no bundle holds it.

        Raises ``BundleError`` naming the bundle when its file does not
        parse or does not validate as a request for this root.
        """
        path = self._entry(name)
        if path is None:
            return None
        try:
            stored = json.loads(path.read_text())
            request = RunRequest.parse(stored)
        except (OSError, ValueError, RunRequestError) as e:
            raise BundleError(f"bundle {name!r} is not valid: {e}") from e
        if set(stored) - _STORED_KEYS or request.ensemble is None:
            raise BundleError(f"bundle {name!r} is not valid: not a stored closure")
        if request.ensemble["name"] != name:
            raise BundleError(
                f"bundle {name!r} is not valid: its root is named "
                f"{request.ensemble['name']!r}"
            )
        return dict(stored)

    def write(self, name: str, stored: dict[str, Any]) -> None:
        """Store ``stored`` as ``name``, replacing a bundle of that name.

        Written to a temp file in the same directory and renamed into
        place: a failure at any step leaves the previous bundle as it was
        and no file the listing would read.
        """
        path = self._path(name)
        if path is None:
            raise BundleError(f"bundle name {name!r} must be one plain file name")
        try:
            text = json.dumps(stored, indent=2, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise BundleError(f"bundle {name!r} is not plain data: {e}") from e
        try:
            self._replace(path, text)
        except OSError as e:
            raise BundleError(f"cannot write bundle {name!r}: {e.strerror or e}") from e

    def _replace(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(
            dir=path.parent, prefix=".bundle-", suffix=_TEMP_SUFFIX
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(text)
            os.replace(temp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp)
            raise

    def delete(self, name: str) -> bool:
        """Remove ``name``; whether a bundle was there."""
        path = self._entry_named(name)
        if path is None:
            return False
        path.unlink()
        return True
