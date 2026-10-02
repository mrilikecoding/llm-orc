"""The packaged scripts declare the files they need (Arc 5, ruling 8).

Nothing is inferred at run time, so a script that imports a sibling
without listing it works from a checkout and fails on a host that was
shipped only what the blocks name. This holds every packaged script's
static sibling imports inside its block, and every listed file real.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llm_orc.core.execution.scripting.files_block import listed_files

SCRIPTS = Path(__file__).resolve().parents[4] / ".llm-orc" / "scripts"
PYTHON_SCRIPTS = sorted(
    p for p in SCRIPTS.rglob("*.py") if "__pycache__" not in p.parts
)


def _sibling_imports(script: Path) -> set[str]:
    """File names beside ``script`` that its source imports statically."""
    siblings = {p.stem for p in script.parent.glob("*.py")} | {
        p.name for p in script.parent.iterdir() if p.is_dir()
    }
    imported: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imported.add(node.module.split(".")[0])
    return {
        f"{name}.py" if (script.parent / f"{name}.py").exists() else name
        for name in imported & siblings
    }


def _id(script: Path) -> str:
    return str(script.relative_to(SCRIPTS))


def test_the_scripts_are_found() -> None:
    assert len(PYTHON_SCRIPTS) > 40


@pytest.mark.parametrize("script", PYTHON_SCRIPTS, ids=_id)
def test_a_block_reads_and_lists_files_that_exist(script: Path) -> None:
    listed = listed_files(script.read_text())

    assert listed.error is None
    for path in listed.paths:
        assert (script.parent / path).is_file(), path


@pytest.mark.parametrize("script", PYTHON_SCRIPTS, ids=_id)
def test_every_static_sibling_import_is_listed(script: Path) -> None:
    listed = set(listed_files(script.read_text()).paths)

    assert _sibling_imports(script) <= listed


def test_a_script_that_runs_a_sibling_by_path_lists_it() -> None:
    script = SCRIPTS / "agentic_serving" / "accept_executor.py"

    assert "accept_executor_runner.py" in listed_files(script.read_text()).paths
