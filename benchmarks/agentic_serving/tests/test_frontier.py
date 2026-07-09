"""Unit tests for the frontier arm (deterministic; CI-safe).

The live Sonnet dispatch is an in-session step (the Agent tool, model=sonnet —
§7); there is no autonomous module to unit-test. These cover the pure parts: the
subagent prompt and scoring a populated workspace into a CellResult. Run with the
llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.agentic_serving import frontier
from benchmarks.agentic_serving.model import Cell

_CONV = "def c_to_f(c):\n    return c\n"


def _cell(deliverables: tuple[str, ...]) -> Cell:
    return Cell(
        name="t",
        horizon=2,
        complexity=2,
        prompt="Create the files described below.",
        expected_deliverables=deliverables,
    )


class TestFrontierPrompt:
    def test_names_every_deliverable(self) -> None:
        prompt = frontier.frontier_prompt(_cell(("a.py", "b.py")))
        assert "a.py" in prompt
        assert "b.py" in prompt

    def test_instructs_writing_into_the_workspace(self) -> None:
        prompt = frontier.frontier_prompt(_cell(("a.py",)))
        assert "directory" in prompt.lower()

    def test_carries_the_cell_task(self) -> None:
        prompt = frontier.frontier_prompt(_cell(("a.py",)))
        assert "Create the files described below." in prompt


class TestFrontierScoreCell:
    def test_clean_workspace_passes_at_n1(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "converters.py").write_text(_CONV)
        (ws / "cli.py").write_text("from converters import c_to_f\n")
        result = frontier.score_cell(ws, _cell(("converters.py", "cli.py")))
        assert result.n == 1
        assert result.passed
        assert result.records[0].terminated_clean is None  # loop-property N/A

    def test_missing_deliverable_fails(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "a.py").write_text("x = 1\n")
        result = frontier.score_cell(ws, _cell(("a.py", "b.py")))
        assert not result.passed
