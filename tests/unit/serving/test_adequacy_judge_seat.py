"""Wiring tests for the extracted adequacy-judge seat (#84).

The adequacy judge lives in ONE ensemble (`adequacy-judge`), referenced by
build-gated-round the way test-writer / code-generator are — so the fixtures
harness (benchmarks/judge_adequacy) and the future #98 test-writing shape
exercise the exact seat the round runs, and prompt tuning happens in one
file. Isolation is preserved: the round's reference keeps
``input_scope: dependencies`` (the judge reads the executor's echoed
contract only, never the shape's context-threaded base input).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_orc.core.config.ensemble_config import _find_ensemble_in_dirs

REPO = Path(__file__).resolve().parents[3]
ENSEMBLES = REPO / ".llm-orc" / "ensembles"


def _load(name: str) -> Any:
    return _find_ensemble_in_dirs(name, [str(ENSEMBLES)])


def test_adequacy_judge_is_its_own_ensemble_with_the_prompt() -> None:
    config = _load("adequacy-judge")
    assert config is not None
    (judge,) = config.agents
    assert judge.name == "judge"
    assert judge.options == {"think": False}
    assert "test-adequacy reviewer" in judge.system_prompt
    assert "tests_adequate" in judge.system_prompt


def test_round_references_the_judge_ensemble_with_isolation() -> None:
    config = _load("build-gated-round")
    assert config is not None
    judge = next(a for a in config.agents if a.name == "judge")
    assert judge.ensemble == "adequacy-judge"
    assert judge.input_scope == "dependencies"
    assert judge.depends_on == ["executor"]


def test_the_judge_prompt_lives_in_exactly_one_ensemble_file() -> None:
    """Prompt tuning (#84) must not fork the seat: no other ensemble file
    carries the adequacy prompt."""
    carriers = [
        path
        for path in (ENSEMBLES / "agentic-serving").glob("*.yaml")
        if "test-adequacy reviewer" in path.read_text()
    ]
    assert [path.name for path in carriers] == ["adequacy-judge.yaml"]
