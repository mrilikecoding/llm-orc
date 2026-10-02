"""A script's ``/// llm-orc`` block: the files it needs beside it.

The block is the form PEP 723 reserves for tools. The comment leader is
``#`` or ``//``, the body is TOML with one key, ``files``, a list of
relative paths under the script's own directory (spec: docs/plans/
2026-09-28-remote-delegation.md, Arc 5 re-cut ruling 8)::

    # /// llm-orc
    # files = ["_helpers.py"]
    # ///

Nothing is inferred from imports: what is listed is what a closure
carries, and a script with no block stands alone.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import Any

from llm_orc.core.execution.scripting.relative_path import check_relative

_OPENING = re.compile(r"(?m)^(?:#|//) /// llm-orc$")
_BLOCK = re.compile(
    r"(?m)^(?P<lead>#|//) /// llm-orc$\s"
    r"(?P<body>(?:^(?P=lead)(?: .*)?$\s)+?)"
    r"^(?P=lead) ///$"
)


@dataclass(frozen=True)
class ListedFiles:
    """The paths a block lists, or the reason the block cannot be read."""

    paths: tuple[str, ...] = ()
    error: str | None = None


def listed_files(source: str) -> ListedFiles:
    """The first ``llm-orc`` block of ``source``: no block lists nothing,
    a block that does not parse or breaks the path rule is an error."""
    text = source.replace("\r\n", "\n")
    match = _BLOCK.search(text)
    if match is None:
        if _OPENING.search(text):
            return ListedFiles(error="the llm-orc block is never closed with '# ///'")
        return ListedFiles()
    lead = match["lead"]
    body = "\n".join(line[len(lead) + 1 :] for line in match["body"].splitlines())
    try:
        data = tomllib.loads(body)
    except tomllib.TOMLDecodeError as e:
        return ListedFiles(error=f"the llm-orc block is not valid TOML: {e}")
    return _paths_of(data)


def _paths_of(data: dict[str, Any]) -> ListedFiles:
    unknown = sorted(set(data) - {"files"})
    if unknown:
        return ListedFiles(error=f"the llm-orc block has unknown keys: {unknown}")
    files = data.get("files", [])
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        return ListedFiles(error="'files' in the llm-orc block must be a list of paths")
    try:
        for path in files:
            check_relative(path, "listed file")
    except ValueError as e:
        return ListedFiles(error=str(e))
    return ListedFiles(paths=tuple(files))
