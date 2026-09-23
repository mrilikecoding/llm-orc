"""Outcome pins for the caller contract (fail-closed-composition, Doctrine
11): drive the REAL `llm-orc invoke` CLI command end to end — a real
executor running real script agents, through Click's own command
dispatch — for a clean run, a failed-terminal run, and a when:-skip-only
run. No mocked executor or service: these pin what a shell script
actually sees (exit code, parsed stdout), not an internal call shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner, Result

from llm_orc.cli import cli


def _write_ensemble(
    tmp_path: Path, filename: str, agents: list[dict[str, object]]
) -> None:
    import yaml

    (tmp_path / filename).write_text(
        yaml.dump(
            {
                "name": filename.removesuffix(".yaml"),
                "description": "caller contract outcome pin",
                "agents": agents,
            }
        )
    )


def _invoke_json(tmp_path: Path, ensemble_name: str) -> Result:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "invoke",
            ensemble_name,
            "hello",
            "--config-dir",
            str(tmp_path),
            "--output-format",
            "json",
        ],
    )
    return result


class TestCliInvokeCleanRun:
    """(a) A clean run: exit 0, status success, has_errors False, a
    real deliverable."""

    def test_clean_run(self, tmp_path: Path) -> None:
        _write_ensemble(
            tmp_path,
            "clean.yaml",
            [{"name": "answer", "script": "echo '{\"ok\": true}'"}],
        )

        result = _invoke_json(tmp_path, "clean")

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "success"
        assert payload["has_errors"] is False
        assert payload["deliverable"] is not None
        assert "ok" in payload["deliverable"]


class TestCliInvokeFailedTerminal:
    """(b) A failed terminal: nonzero exit, status error, deliverable
    null."""

    def test_failed_terminal(self, tmp_path: Path) -> None:
        _write_ensemble(
            tmp_path,
            "failed.yaml",
            [{"name": "answer", "script": "exit 1"}],
        )

        result = _invoke_json(tmp_path, "failed")

        assert result.exit_code != 0
        payload = json.loads(result.output)
        assert payload["status"] == "error"
        assert payload["has_errors"] is True
        assert payload["deliverable"] is None


class TestCliInvokeWhenSkipOnly:
    """(c) A when:-false skip, with no dependency failure anywhere: the
    run is still success — a plain conditional skip is not an error."""

    def test_when_skip_only_run_is_success(self, tmp_path: Path) -> None:
        _write_ensemble(
            tmp_path,
            "when_skip.yaml",
            [
                {"name": "answer", "script": "echo '{\"ok\": true}'"},
                {
                    "name": "skipped",
                    # never a ${dep.field} match against a real dependency,
                    # so this is a plain when:-false skip, never a
                    # dependency-cascade skip (fail-closed-composition
                    # rule 1 does not apply -- "skipped" has no depends_on)
                    "when": '${nope.field} == "yes"',
                    "script": "echo '{\"unused\": true}'",
                },
            ],
        )

        result = _invoke_json(tmp_path, "when_skip")

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "success"
        assert payload["has_errors"] is False
        assert payload["results"]["skipped"]["status"] == "skipped"
        assert "reason" not in payload["results"]["skipped"]
        assert payload["deliverable"] is not None
