"""Unit tests for the benchmark scorer (deterministic; CI-safe).

Run with the llm_orc coverage gate disabled (the benchmark is not llm_orc):
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.

Fixture log lines carry only the fields the scorer parses (``action``,
``shape``, ``delegated``, ``judgment_verdict``).
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.agentic_serving.model import Cell
from benchmarks.agentic_serving.scorer import score, score_frontier

# A clean session: two generation writes delegated, then a COMPLETE finish.
_GEN = "turn decision: action=write shape=generation delegated=code-generator\n"
_FINISH = "turn decision: action=finish shape=carry judgment_verdict=COMPLETE\n"
_CLEAN_LOG = _GEN + _GEN + _FINISH

# A coherent sibling pair: a defined function + a file that uses it.
_CONV = "def c_to_f(c):\n    return c\n"


def _cell(deliverables: tuple[str, ...]) -> Cell:
    return Cell(
        name="t",
        horizon=2,
        complexity=2,
        prompt="...",
        expected_deliverables=deliverables,
    )


def _ws(tmp_path: Path, files: dict[str, str]) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "opencode.json").write_text("{}")  # config the scorer ignores
    (ws / ".hidden").write_text("x")  # dotfile the scorer ignores
    for name, content in files.items():
        (ws / name).write_text(content)
    return ws


class TestFormValidity:
    def test_valid_py_json_md_pass(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"m.py": _CONV, "c.json": '{"a": 1}\n', "README.md": "# notes\n"},
        )
        assert score(ws, _CLEAN_LOG, _cell(("m.py",))).form_valid

    def test_unparseable_py_fails(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"m.py": "def f(: bad\nprose\n"})
        rec = score(ws, _CLEAN_LOG, _cell(("m.py",)))
        assert not rec.form_valid
        assert not rec.passed

    def test_invalid_json_fails(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"c.json": "{not valid,}"})
        assert not score(ws, _CLEAN_LOG, _cell(("c.json",))).form_valid

    def test_md_is_never_a_form_failure(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"README.md": "```bash\nfoo\n```\nprose"})
        assert score(ws, _CLEAN_LOG, _cell(("README.md",))).form_valid


class TestConvergence:
    def test_all_deliverables_present(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n", "b.py": "y = 2\n"})
        assert score(ws, _CLEAN_LOG, _cell(("a.py", "b.py"))).converged

    def test_missing_deliverable_fails(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        rec = score(ws, _CLEAN_LOG, _cell(("a.py", "b.py")))
        assert not rec.converged
        assert not rec.passed


class TestContentCoherence:
    def test_real_from_import_is_coherent(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"converters.py": _CONV, "cli.py": "from converters import c_to_f\n"},
        )
        assert score(ws, _CLEAN_LOG, _cell(("cli.py",))).content_coherent

    def test_invented_from_import_is_incoherent(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"converters.py": _CONV, "cli.py": "from converters import nope\n"},
        )
        rec = score(ws, _CLEAN_LOG, _cell(("cli.py",)))
        assert not rec.content_coherent
        assert not rec.passed

    def test_invented_attribute_is_incoherent(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {
                "converters.py": _CONV,
                "cli.py": "import converters\nconverters.nope()\n",
            },
        )
        assert not score(ws, _CLEAN_LOG, _cell(("cli.py",))).content_coherent

    def test_real_attribute_is_coherent(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {
                "converters.py": _CONV,
                "cli.py": "import converters\nconverters.c_to_f(0)\n",
            },
        )
        assert score(ws, _CLEAN_LOG, _cell(("cli.py",))).content_coherent

    def test_star_import_is_uncheckable_not_failure(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"converters.py": _CONV, "cli.py": "from converters import *\n"},
        )
        rec = score(ws, _CLEAN_LOG, _cell(("cli.py",)))
        assert rec.content_coherent
        assert any("un-checkable" in n for n in rec.notes)


class TestTermination:
    def test_complete_finish_is_clean(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score(ws, _CLEAN_LOG, _cell(("a.py",))).terminated_clean

    def test_zombie_tail_is_not_clean(self, tmp_path: Path) -> None:
        zombie = _GEN + "turn decision: action=write judgment_verdict=REMAINING\n"
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        rec = score(ws, zombie, _cell(("a.py",)))
        assert not rec.terminated_clean
        assert not rec.passed

    def test_no_decisions_is_not_clean(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert not score(ws, "", _cell(("a.py",))).terminated_clean


class TestReportedMetrics:
    def test_delegation_rate_over_generation_turns(self, tmp_path: Path) -> None:
        log = (
            _GEN
            + "turn decision: action=write shape=generation delegated=-\n"
            + _FINISH
        )
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score(ws, log, _cell(("a.py",))).delegation_rate == 0.5

    def test_delegation_rate_none_without_generation(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score(ws, _FINISH, _cell(("a.py",))).delegation_rate is None

    def test_escalation_detected(self, tmp_path: Path) -> None:
        log = _CLEAN_LOG + "form escalation: re-dispatch destination=cli.py\n"
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score(ws, log, _cell(("a.py",))).escalated

    def test_churn_counts_extra_write_turns(self, tmp_path: Path) -> None:
        log = _GEN + _GEN + _GEN + _FINISH  # 3 write turns, 1 file → churn 2
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score(ws, log, _cell(("a.py",))).churn == 2


class TestPassedComposition:
    def test_all_hard_signals_pass(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"converters.py": _CONV, "test_c.py": "from converters import c_to_f\n"},
        )
        rec = score(ws, _CLEAN_LOG, _cell(("converters.py", "test_c.py")))
        assert rec.form_valid
        assert rec.converged
        assert rec.content_coherent
        assert rec.terminated_clean
        assert rec.passed


class TestFrontierScoring:
    """The §7 frontier arm: a one-shot subagent workspace, no serve-log.

    Loop-termination is a property of the cheap arm's loop; a one-shot model has
    no loop, so it is N/A (``None``) and does not gate ``passed`` (§4 / §7).
    """

    def test_termination_is_na_without_a_log(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        assert score_frontier(ws, _cell(("a.py",))).terminated_clean is None

    def test_passes_on_the_three_file_signals(self, tmp_path: Path) -> None:
        ws = _ws(
            tmp_path,
            {"converters.py": _CONV, "cli.py": "from converters import c_to_f\n"},
        )
        rec = score_frontier(ws, _cell(("converters.py", "cli.py")))
        assert rec.form_valid
        assert rec.converged
        assert rec.content_coherent
        assert rec.terminated_clean is None
        assert rec.passed  # n/a termination does not block a clean file set

    def test_bad_form_still_fails(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "def f(: bad\n"})
        rec = score_frontier(ws, _cell(("a.py",)))
        assert not rec.form_valid
        assert not rec.passed

    def test_missing_deliverable_still_fails(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        rec = score_frontier(ws, _cell(("a.py", "b.py")))
        assert not rec.converged
        assert not rec.passed

    def test_log_only_metrics_are_none(self, tmp_path: Path) -> None:
        ws = _ws(tmp_path, {"a.py": "x = 1\n"})
        rec = score_frontier(ws, _cell(("a.py",)))
        assert rec.delegation_rate is None
        assert rec.churn is None
        assert not rec.escalated
