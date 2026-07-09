"""Unit tests for the turn-trace snippet cap.

The trace clips node responses to a short snippet by default so the JSONL
stays readable; ``LLM_ORC_SERVE_TRACE_SNIPPET`` raises the cap for live
diagnosis sessions (two 2026-07-09 loop investigations dead-ended on the
280-char clip hiding the round envelope's diagnostics).
"""

from __future__ import annotations

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
