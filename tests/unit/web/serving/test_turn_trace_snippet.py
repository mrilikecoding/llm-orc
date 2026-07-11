"""Unit tests for the turn-trace snippet cap.

The trace clips node responses to a short snippet by default so the JSONL
stays readable; ``LLM_ORC_SERVE_TRACE_SNIPPET`` raises the cap for live
diagnosis sessions (two 2026-07-09 loop investigations dead-ended on the
280-char clip hiding the round envelope's diagnostics).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_orc.web.serving.turn_trace import build_turn_trace


def _trace_response(length: int) -> str:
    result = {"results": {"node": {"status": "success", "response": "x" * length}}}
    trace = build_turn_trace("serving", result)
    response: str = trace["nodes"][0]["response"]
    return response


def test_snippet_clips_at_the_default_cap() -> None:
    assert len(_trace_response(5000)) == 281  # 280 + ellipsis


def test_snippet_cap_is_env_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ORC_SERVE_TRACE_SNIPPET", "5000")
    assert len(_trace_response(4000)) == 4000


def test_unparseable_snippet_env_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ORC_SERVE_TRACE_SNIPPET", "lots")
    assert len(_trace_response(5000)) == 281


def test_seat_envelope_diagnostics_survive_the_snippet_cap() -> None:
    """Issue #114: the accept verdict fields (accept, held_round,
    tests_pass/adequate) are small and structured — the snippet cap must
    not cost them. A battery post-mortem could not answer "did the held
    round fire?" because the clip ate the envelope."""
    import json

    envelope = {
        "output": {
            "status": "success",
            "primary": "def add(a, b): return a + b" + "\n# pad" * 200,
            "diagnostics": {
                "ensemble": "build-gated",
                "accept": False,
                "accept_reason": "tests did not pass",
                "tests_pass": False,
                "tests_adequate": True,
                "held_round": True,
                "retry_input": "x" * 5000,
            },
        }
    }
    child_result = {"results": {"round": {"response": json.dumps(envelope)}}}
    result = {
        "results": {"seat": {"status": "success", "response": json.dumps(child_result)}}
    }

    trace = build_turn_trace("serving", result)

    seat_nodes = trace["nodes"][0]["seat"]
    diagnostics = seat_nodes[0]["diagnostics"]
    assert diagnostics["accept"] is False
    assert diagnostics["held_round"] is True
    assert diagnostics["tests_adequate"] is True
    assert diagnostics["accept_reason"] == "tests did not pass"
    # prose-sized fields still clip — only the structure is exempt
    assert len(diagnostics["retry_input"]) <= 281


def test_emit_never_propagates_a_trace_build_failure(tmp_path: Path) -> None:
    """PR #116 review: 'tracing must never break the serve' — a hostile
    child response (pathologically nested JSON raises RecursionError at
    parse) must not kill the request path. emit degrades to a stub trace."""
    import json

    from llm_orc.web.serving.turn_trace import emit_turn_trace

    hostile = "[" * 10050 + "]" * 10050
    child_result = {"results": {"round": {"response": hostile}}}
    result = {
        "results": {"seat": {"status": "success", "response": json.dumps(child_result)}}
    }

    trace = emit_turn_trace("serving", result, tmp_path)

    assert trace["ensemble"] == "serving"
