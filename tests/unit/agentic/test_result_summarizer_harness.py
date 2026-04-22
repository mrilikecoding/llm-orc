"""Unit tests for Result Summarizer Harness (WP-D Group 1).

The Harness is interposed by Orchestrator Tool Dispatch on the
``invoke_ensemble`` return path (system design Amendment #3). These
tests exercise the module in isolation with a stubbed
``SummarizerInvoker``; Tool Dispatch wiring is covered in Group 3,
and the end-to-end unsummarized-result-unreachable property is
covered by the FC-8 static test in Group 5.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_orc.agentic.result_summarizer_harness import (
    RawOutputPassthrough,
    ResultSummarizerHarness,
    SummarizationFailure,
    SummarizationSuccess,
)


class _StubInvoker:
    """Handwritten test double for the ``SummarizerInvoker`` Protocol."""

    def __init__(
        self,
        *,
        returns: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._returns: dict[str, Any] = (
            returns if returns is not None else {"synthesis": "summary text"}
        )
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        if self._raises is not None:
            raise self._raises
        return self._returns


@pytest.mark.asyncio
async def test_raw_output_flag_returns_passthrough_without_invoking_summarizer() -> (
    None
):
    invoker = _StubInvoker()
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    raw = {
        "results": {"a": "x"},
        "synthesis": "ensemble-self-synthesis",
        "status": "ok",
    }
    result = await harness.summarize(raw, raw_output=True)

    assert isinstance(result, RawOutputPassthrough)
    assert result.content == raw
    assert invoker.calls == [], "summarizer must not run when raw_output is set"


@pytest.mark.asyncio
async def test_summarize_invokes_configured_summarizer_with_json_input() -> None:
    invoker = _StubInvoker(returns={"synthesis": "condensed output"})
    harness = ResultSummarizerHarness(invoker=invoker, summarizer_name="my-summarizer")

    raw = {"results": {"agent_a": "output"}, "status": "ok"}
    await harness.summarize(raw, raw_output=False)

    assert len(invoker.calls) == 1
    call = invoker.calls[0]
    assert call["ensemble_name"] == "my-summarizer"
    # Input is a JSON-serialized representation of the raw result.
    assert isinstance(call["input"], str)
    assert json.loads(call["input"]) == raw


@pytest.mark.asyncio
async def test_summarize_returns_success_with_synthesis_text() -> None:
    invoker = _StubInvoker(
        returns={"synthesis": "The ensemble found X.", "results": {}}
    )
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationSuccess)
    assert result.summary == "The ensemble found X."


@pytest.mark.asyncio
async def test_summarize_returns_failure_when_invoker_raises_value_error() -> None:
    invoker = _StubInvoker(raises=ValueError("summarizer ensemble not found"))
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="missing-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationFailure)
    assert "summarizer ensemble not found" in result.reason


@pytest.mark.asyncio
async def test_summarize_returns_failure_when_synthesis_field_missing() -> None:
    invoker = _StubInvoker(returns={"results": {"a": "x"}, "status": "ok"})
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationFailure)
    assert "agentic-result-summarizer" in result.reason
    assert "no summary text" in result.reason


@pytest.mark.asyncio
async def test_summarize_returns_failure_when_synthesis_is_empty_string() -> None:
    invoker = _StubInvoker(returns={"synthesis": "", "results": {}})
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationFailure)


@pytest.mark.asyncio
async def test_summarize_returns_failure_when_synthesis_is_not_string() -> None:
    invoker = _StubInvoker(returns={"synthesis": {"nested": "dict"}, "results": {}})
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationFailure)


@pytest.mark.asyncio
async def test_summarize_uses_single_agent_response_when_synthesis_absent() -> None:
    """Default summarizer is a single-agent ensemble; synthesis is None.

    llm-orc's dependency-based execution model never populates synthesis
    (see core/execution/results_processor.finalize_result). A single-agent
    summarizer ensemble puts its output in results[agent_name]["response"].
    """
    invoker = _StubInvoker(
        returns={
            "synthesis": None,
            "results": {"summarizer": {"response": "A short summary."}},
        }
    )
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationSuccess)
    assert result.summary == "A short summary."


@pytest.mark.asyncio
async def test_summarize_prefers_synthesis_over_single_agent_response() -> None:
    """When synthesis is populated, it wins over the results fallback."""
    invoker = _StubInvoker(
        returns={
            "synthesis": "synthesized",
            "results": {"summarizer": {"response": "agent response"}},
        }
    )
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationSuccess)
    assert result.summary == "synthesized"


@pytest.mark.asyncio
async def test_summarize_fallback_requires_exactly_one_agent() -> None:
    """Multi-agent results without synthesis → SummarizationFailure.

    The fallback is deliberately narrow: exactly one agent. Multi-agent
    ensembles must populate synthesis to disambiguate which agent's
    response carries the summary.
    """
    invoker = _StubInvoker(
        returns={
            "synthesis": None,
            "results": {
                "agent_a": {"response": "first"},
                "agent_b": {"response": "second"},
            },
        }
    )
    harness = ResultSummarizerHarness(
        invoker=invoker, summarizer_name="agentic-result-summarizer"
    )

    result = await harness.summarize({"results": {}}, raw_output=False)

    assert isinstance(result, SummarizationFailure)
