"""Wiring tests for the adequacy seats (#84).

After the 2026-07-09 measurement, the gated round's adequacy signal is the
DETERMINISTIC checker script (adequacy_check.py) — the model seat it
replaced stays defined in ONE ensemble (`adequacy-judge`) as the harness's
measurement subject and a catalog seat, with its prompt in exactly one
file.
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


def test_round_fills_the_judge_seat_with_the_deterministic_checker() -> None:
    config = _load("build-gated-round")
    assert config is not None
    judge = next(a for a in config.agents if a.name == "judge")
    assert judge.script == "scripts/agentic_serving/adequacy_check.py"
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
