"""The packaged serving project: the repo's ``.llm-orc/`` shipped in the wheel (#196).

Resolution order:

1. ``LLM_ORC_SERVING_PROJECT_DIR``: a path uses that directory as the
   packaged tier and must carry the serving ensemble (a wrong path is a
   loud error, not an empty tier); an empty value disables the tier (the
   test suite's default, see ``tests/conftest.py``).
2. ``llm_orc/serving_project/`` next to this package: a wheel install.
3. The checkout's ``.llm-orc/``, two levels above ``src/llm_orc``: an
   editable install. In a checkout this is also the local project, and
   ``ConfigurationManager`` lists it once.

A candidate counts only when it carries ``ensembles/agentic-serving/serving.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

SERVING_PROJECT_ENV = "LLM_ORC_SERVING_PROJECT_DIR"
SERVING_MARKER = Path("ensembles") / "agentic-serving" / "serving.yaml"

_PACKAGE_DIR = Path(__file__).resolve().parents[2]


def has_serving_ensemble(directory: Path) -> bool:
    """True when ``directory`` is a serving project (carries the marker)."""
    return (directory / SERVING_MARKER).is_file()


def packaged_serving_project_dir() -> Path | None:
    """The packaged serving project, or None when there is none."""
    override = os.environ.get(SERVING_PROJECT_ENV)
    if override is not None:
        if override == "":
            return None
        path = Path(override).resolve()
        if not has_serving_ensemble(path):
            raise FileNotFoundError(
                f"{SERVING_PROJECT_ENV}={override!r} has no {SERVING_MARKER}"
            )
        return path
    for candidate in (
        _PACKAGE_DIR / "serving_project",
        _PACKAGE_DIR.parent.parent / ".llm-orc",
    ):
        if has_serving_ensemble(candidate):
            return candidate
    return None
