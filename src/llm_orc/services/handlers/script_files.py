"""Where a script and the files it lists are, for the closure walk.

A script is resolved by the run's own ``ScriptResolver``. A file it
lists in its llm-orc block is looked for beside the resolved script and
nowhere else: asking the resolver's search paths would let a host file
at the same relative path in another tier stand in for a file the
script's own directory lacks (Arc 5 ruling 8, Arc 4 ruling 7).
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.config.closure import Dependency, ScriptListing
from llm_orc.core.execution.scripting.files_block import ListedFiles, listed_files
from llm_orc.core.execution.scripting.resolver import (
    ScriptNotFoundError,
    ScriptResolver,
)

_ABSENT = ScriptListing(False)


class ScriptFileLocator:
    """Reads the block of each script the walk hands it. The walk visits
    a script before the files it lists, so an owner's path is known when
    its listed file is asked for."""

    def __init__(self, resolver: ScriptResolver) -> None:
        self._resolver = resolver
        self._located: dict[str, Path] = {}

    def __call__(self, dep: Dependency) -> ScriptListing:
        path = self._locate(dep)
        if path is None:
            return _ABSENT
        self._located[dep.name] = path
        return ScriptListing(True, _read_block(path))

    def _locate(self, dep: Dependency) -> Path | None:
        if dep.beside is None or dep.listed is None:
            return self._resolve(dep.name)
        owner = self._located.get(dep.beside)
        if owner is None:
            return None
        candidate = owner.parent / dep.listed
        return candidate if candidate.is_file() else None

    def _resolve(self, reference: str) -> Path | None:
        try:
            resolved, is_file = self._resolver.resolve_and_classify(reference)
        except ScriptNotFoundError:
            return None
        path = Path(resolved)
        return path if is_file and path.is_file() else None


def _read_block(path: Path) -> ListedFiles:
    try:
        return listed_files(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return ListedFiles()
