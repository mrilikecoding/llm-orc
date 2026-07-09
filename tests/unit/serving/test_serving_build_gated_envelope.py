"""Unit tests for the build-gated ``envelope`` node (WP-D8).

The build-gated shape's terminal: it carries the code deliverable (artifacts[0])
plus the accept-gate verdict (accept / reason / signals) in the ADR-024
diagnostics, so the serving marshal consumes a faithful structured artifact and
the client sees the accept/another-round decision (ODP-2). The surviving ADR-024
container shape is preserved; the retired calibration subfields stay absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ENVELOPE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "build_gated_envelope.py"


def _sub_ensemble_response(terminal_text: str) -> str:
    node = {"response": terminal_text, "status": "success"}
    return json.dumps(
        {
            "ensemble": "code-generator",
            "status": "completed",
            "results": {"synthesizer": node},
        }
    )


def _envelope(code_terminal: str, gate_verdict: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
                "accept_gate": {"response": json.dumps(gate_verdict)},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(ENVELOPE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_envelope_carries_code_and_accept_verdict() -> None:
    env = _envelope(
        "def f():\n    return 1",
        {"accept": True, "tests_pass": True, "tests_adequate": True, "reason": "ok"},
    )
    assert env["status"] == "success"
    assert env["artifacts"][0]["content"] == "def f():\n    return 1"
    assert env["primary"] == "def f():\n    return 1"
    diag = env["diagnostics"]
    assert diag["accept"] is True
    assert diag["tests_pass"] is True
    assert diag["tests_adequate"] is True


def test_envelope_carries_reject_verdict_and_reason() -> None:
    env = _envelope(
        "def f():\n    return 1",
        {
            "accept": False,
            "tests_pass": True,
            "tests_adequate": False,
            "reason": "tests inadequate to verify the requirement",
        },
    )
    diag = env["diagnostics"]
    assert diag["accept"] is False
    assert "inadequate" in diag["accept_reason"]


def test_envelope_extracts_code_from_fenced_seat_output() -> None:
    env = _envelope(
        "Here it is:\n```python\nx = 1\n```",
        {"accept": True, "reason": "ok"},
    )
    assert env["artifacts"][0]["content"] == "x = 1"


def test_envelope_drops_a_seat_emitted_test_fence_from_the_deliverable() -> None:
    """The shipped artifact must be the code, not code + an embedded copy of
    the tests (two-fence seat output; live finding 2026-07-09)."""
    env = _envelope(
        "Code:\n```python\nx = 1\n```\nTests:\n"
        "```python\ndef test_x():\n    assert x == 1\n```",
        {"accept": True, "reason": "ok"},
    )
    assert env["artifacts"][0]["content"] == "x = 1"


def test_envelope_preserves_adr024_container_without_retired_subfields() -> None:
    env = _envelope("x = 1", {"accept": True, "reason": "ok"})
    for field in ("primary", "structured", "artifacts", "diagnostics"):
        assert field in env
    assert "calibration_verdict" not in env["diagnostics"]
    assert "audit_findings" not in env["diagnostics"]


def _envelope_with_input(
    code_terminal: str,
    gate_verdict: dict[str, Any],
    base_input: str,
    executor_report: str = "",
    executor_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executor_result: dict[str, Any] = {
        "tests_pass": False,
        "report": executor_report,
    }
    executor_result.update(executor_extra or {})
    payload = json.dumps(
        {
            "input_data": base_input,
            "dependencies": {
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
                "accept_gate": {"response": json.dumps(gate_verdict)},
                "executor": {"response": json.dumps(executor_result)},
            },
        }
    )
    out = subprocess.run(
        [sys.executable, str(ENVELOPE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_reject_envelope_composes_a_retry_input_for_the_next_round() -> None:
    """The bounded retry round (issue #89, ADR-048 §5): the loop primitive's
    carry REPLACES the next iteration's input, so a rejected round's envelope
    composes diagnostics.retry_input = the original turn plus the failure
    report — the next round regenerates with the evidence in hand."""
    env = _envelope_with_input(
        "def f():\n    return 1",
        {"accept": False, "tests_pass": False, "reason": "tests did not pass"},
        base_input="Write f() in f.py",
        executor_report="test_f: AssertionError()",
    )
    retry = env["diagnostics"]["retry_input"]
    assert "Write f() in f.py" in retry
    assert "test_f: AssertionError()" in retry


def test_reject_with_ran_adequate_tests_carries_them_held_in_retry_input() -> None:
    """The TDD retry loop (issue #100): when the round's tests collected and
    ran (n_tests > 0) and the judge passed them, the reject carry embeds the
    tests under the HELD TESTS sentinel — round 2 holds them as the spec and
    regenerates only the code."""
    env = _envelope_with_input(
        "def f():\n    return 2",
        {
            "accept": False,
            "tests_pass": False,
            "tests_adequate": True,
            "reason": "tests did not pass",
        },
        base_input="Write f() in f.py",
        executor_report="test_f: AssertionError()",
        executor_extra={"n_tests": 2, "tests": "def test_f():\n    assert f() == 1"},
    )
    retry = env["diagnostics"]["retry_input"]
    assert "Write f() in f.py" in retry
    assert "[HELD TESTS" in retry
    assert "def test_f():\n    assert f() == 1" in retry


def test_reject_with_uncollected_tests_regenerates_fresh() -> None:
    """Tests that never collected (n_tests == 0) are not a spec — the carry
    stays the fresh-regeneration form."""
    env = _envelope_with_input(
        "def f():\n    return 2",
        {
            "accept": False,
            "tests_pass": False,
            "tests_adequate": True,
            "reason": "tests did not pass",
        },
        base_input="Write f() in f.py",
        executor_report="no test_* functions or TestCase classes found",
        executor_extra={"n_tests": 0, "tests": "garbage"},
    )
    retry = env["diagnostics"]["retry_input"]
    assert "[HELD TESTS" not in retry
    assert "Write f() in f.py" in retry


def test_reject_with_inadequate_tests_regenerates_fresh() -> None:
    """Judge-rejected tests are not a spec worth holding."""
    env = _envelope_with_input(
        "def f():\n    return 2",
        {
            "accept": False,
            "tests_pass": True,
            "tests_adequate": False,
            "reason": "tests inadequate to verify the requirement",
        },
        base_input="Write f() in f.py",
        executor_extra={"n_tests": 2, "tests": "def test_f():\n    assert True"},
    )
    assert "[HELD TESTS" not in env["diagnostics"]["retry_input"]


def test_envelope_marks_a_held_round_in_diagnostics() -> None:
    """Observability for the TDD retry loop: the round's mode is only
    knowable live from the envelope, so a held round (gather dep carries
    held=true) stamps diagnostics.held_round."""
    payload = json.dumps(
        {
            "input_data": "Write f() in f.py",
            "dependencies": {
                "code_writer": {"response": _sub_ensemble_response("x = 1")},
                "accept_gate": {"response": json.dumps({"accept": True})},
                "gather": {"response": json.dumps({"held": True})},
            },
        }
    )
    out = subprocess.run(
        [sys.executable, str(ENVELOPE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    env = json.loads(out)
    assert env["diagnostics"]["held_round"] is True


def test_envelope_marks_a_fresh_round_without_gather_held() -> None:
    env = _envelope("x = 1", {"accept": True, "reason": "ok"})
    assert env["diagnostics"]["held_round"] is False


def test_accept_envelope_has_no_retry_input() -> None:
    env = _envelope_with_input(
        "def f():\n    return 1",
        {"accept": True, "tests_pass": True, "tests_adequate": True, "reason": "ok"},
        base_input="Write f() in f.py",
    )
    assert env["diagnostics"]["accept"] is True
    assert "retry_input" not in env["diagnostics"]
