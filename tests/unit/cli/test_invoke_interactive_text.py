"""``llm-orc invoke --output-format text`` on an ensemble with an interactive
script goes through the streaming module's text branch. The real command
through ``CliRunner``; the script is named like an interactive one but
never prompts."""

# ruff: noqa: F811  (imported fixtures are redefined as test parameters)
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_orc.cli import cli
from tests.unit.services.test_one_run_injection import (  # noqa: F401
    _ensemble,
    listing,
    project,
    state_dir,
)

_OK = (
    "import json, sys\n"
    "sys.stdin.read()\n"
    'print(json.dumps({"success": True, "data": "interactive done"}))\n'
)
_FAIL = (
    "import json, sys\n"
    "sys.stdin.read()\n"
    'print(json.dumps({"success": False, "error": "boom"}))\n'
    "sys.exit(1)\n"
)


def _write(project: Path, source: str) -> None:
    scripts = project / ".llm-orc" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "get_user_input.py").write_text(source)
    _ensemble(
        project / ".llm-orc", "ask", [{"name": "ask", "script": "get_user_input.py"}]
    )


@pytest.fixture(autouse=True)
def in_project(project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(project)
    return project


def test_a_successful_interactive_run_exits_0(in_project: Path) -> None:
    _write(in_project, _OK)

    result = CliRunner().invoke(cli, ["invoke", "ask", "hi", "--output-format", "text"])

    assert "interactive done" in result.output, result.output
    assert result.exit_code == 0, result.output


def test_a_failing_interactive_run_exits_1(in_project: Path) -> None:
    _write(in_project, _FAIL)

    result = CliRunner().invoke(cli, ["invoke", "ask", "hi", "--output-format", "text"])

    assert result.exit_code == 1, result.output
