"""Hermetic tests for the judge-adequacy measurement harness (#84).

The harness itself (benchmarks/judge_adequacy/run.py) drives the live
adequacy-judge seat and is not run in CI; these tests pin its deterministic
mechanics — fixture loading and labeling, verdict parsing, and the
false-reject / false-accept summary math — plus fixture sanity: every
fixture's code/tests actually execute as labeled through the REAL accept
executor.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO / "benchmarks" / "judge_adequacy"
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"


def _harness() -> Any:
    spec = importlib.util.spec_from_file_location(
        "judge_adequacy_run", HARNESS_DIR / "run.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixtures_load_with_labels_and_classes() -> None:
    harness = _harness()
    fixtures = harness.load_fixtures(HARNESS_DIR / "fixtures.yaml")
    assert len(fixtures) >= 12
    classes = {f["class"] for f in fixtures}
    assert classes == {
        "adequate-clear",
        "adequate-terse",
        "inadequate-trivial",
        "inadequate-tautological",
        "failing-but-adequate",
    }
    for fixture in fixtures:
        for key in ("name", "requirement", "code", "tests", "expected_adequate"):
            assert key in fixture, fixture.get("name", "?")
        assert isinstance(fixture["expected_adequate"], bool)


def test_summary_math_computes_frr_and_far_per_class() -> None:
    harness = _harness()
    samples = [
        {"class": "adequate-clear", "expected_adequate": True, "tests_adequate": True},
        {"class": "adequate-clear", "expected_adequate": True, "tests_adequate": False},
        {
            "class": "inadequate-trivial",
            "expected_adequate": False,
            "tests_adequate": False,
        },
        {
            "class": "inadequate-trivial",
            "expected_adequate": False,
            "tests_adequate": True,
        },
    ]
    summary = harness.summarize(samples)
    assert summary["classes"]["adequate-clear"]["false_reject_rate"] == 0.5
    assert summary["classes"]["inadequate-trivial"]["false_accept_rate"] == 0.5
    assert summary["overall"]["false_reject_rate"] == 0.5
    assert summary["overall"]["false_accept_rate"] == 0.5
    assert summary["overall"]["samples"] == 4


def test_verdict_parsing_peels_the_sub_ensemble_envelope() -> None:
    """The judge runs as a sub-ensemble seat; the harness parses the same
    nested result the gate peels."""
    harness = _harness()
    child = {
        "ensemble": "judge-fixture-run",
        "results": {
            "executor": {"response": "{}", "status": "success"},
            "judge": {
                "response": json.dumps(
                    {
                        "ensemble": "adequacy-judge",
                        "status": "completed",
                        "results": {
                            "judge": {
                                "response": '{"tests_adequate": false, '
                                '"reason": "trivial"}',
                                "status": "success",
                            }
                        },
                    }
                ),
                "status": "success",
            },
        },
    }
    verdict = harness.parse_verdict(child)
    assert verdict == {"tests_adequate": False, "reason": "trivial"}


@pytest.mark.parametrize(
    "klass",
    [
        "adequate-clear",
        "adequate-terse",
        "inadequate-trivial",
        "inadequate-tautological",
        "failing-but-adequate",
    ],
)
def test_fixture_sanity_code_and_tests_execute_as_labeled(klass: str) -> None:
    """Every fixture runs through the REAL accept executor and lands on its
    labeled tests_pass — a fixture whose tests error out or land backwards
    would poison the measurement."""
    harness = _harness()
    fixtures = [
        f
        for f in harness.load_fixtures(HARNESS_DIR / "fixtures.yaml")
        if f["class"] == klass
    ]
    assert fixtures
    for fixture in fixtures:
        contract = {
            "requirement": fixture["requirement"],
            "code": fixture["code"],
            "tests": fixture["tests"],
        }
        out = subprocess.run(
            [sys.executable, str(EXECUTOR)],
            input=json.dumps(contract),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        result = json.loads(out)
        assert result["tests_pass"] is fixture["expected_tests_pass"], fixture["name"]
        if fixture["class"] != "inadequate-trivial":
            assert result["n_tests"] > 0, fixture["name"]
