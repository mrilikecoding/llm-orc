"""Result Summarizer Harness — interposed on invoke_ensemble return path.

Per ``docs/agentic-serving/system-design.md`` §Result Summarizer Harness
(L2) and §Integration Contracts (Orchestrator Tool Dispatch → Result
Summarizer Harness, Result Summarizer Harness → Ensemble Engine).

The Harness is the structural enforcement of AS-7 (result summarization
is a correctness requirement) and ADR-004 (unsummarized results must not
reach the orchestrator's context by any path). Orchestrator Tool Dispatch
calls ``summarize`` on every successful ``invoke_ensemble`` result; the
Runtime never sees raw ensemble output unless the invoked ensemble's
``raw_output`` flag is set — the explicit escape hatch per ADR-004.

Design Amendment #3 places the Harness on the Tool Dispatch side, not the
Runtime side. The Runtime is unaware of summarization; the Orchestrator
LLM's reasoning surface stays "I emit tool calls and observe results."

The Harness invokes a summarizer ensemble via the same ensemble-operations
facade Tool Dispatch uses (reused Protocol shape, named here for intent).
A summarizer failure produces a typed ``SummarizationFailure`` — the raw
result is still persisted to the ensemble's artifact (Invariant 9 —
artifact persistence lives in Ensemble Engine and is untouched by the
Harness), but the orchestrator receives a tool error observation, never
the raw dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class SummarizerInvoker(Protocol):
    """Narrow facade for invoking the summarizer ensemble.

    ``OrchestraService`` satisfies this structurally; tests pass a
    handwritten double. Named for intent — the Harness invokes one
    ensemble (the summarizer) with a JSON-serialized raw result as
    input and expects a synthesis string back.
    """

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SummarizationSuccess:
    """A summary that will enter the orchestrator's context."""

    summary: str


@dataclass(frozen=True)
class RawOutputPassthrough:
    """Escape-hatch (ADR-004): raw ensemble result bypasses summarization.

    Honored only when the invoked ensemble's ``raw_output`` flag is set.
    The operator takes responsibility for the content being small enough
    to not cause context rot in the orchestrator's reasoning surface.
    """

    content: dict[str, Any]


@dataclass(frozen=True)
class SummarizationFailure:
    """The summarizer ensemble failed.

    The raw result is still persisted to the ensemble's execution
    artifact by the Ensemble Engine (Invariant 9). The orchestrator
    receives a typed tool error, not the raw dict — Tool Dispatch
    converts this into a ``ToolCallError(kind="summarization_failed")``.
    """

    reason: str


SummarizationResult = SummarizationSuccess | RawOutputPassthrough | SummarizationFailure


class ResultSummarizerHarness:
    """Interposes summarization on the invoke_ensemble return path."""

    def __init__(self, *, invoker: SummarizerInvoker, summarizer_name: str) -> None:
        self._invoker = invoker
        self._summarizer_name = summarizer_name

    async def summarize(
        self,
        raw_result: dict[str, Any],
        *,
        raw_output: bool,
    ) -> SummarizationResult:
        """Produce a summary, pass raw through, or report failure."""
        if raw_output:
            return RawOutputPassthrough(content=raw_result)
        try:
            invocation = await self._invoker.invoke(
                {
                    "ensemble_name": self._summarizer_name,
                    "input": json.dumps(raw_result, default=str),
                }
            )
        except ValueError as exc:
            return SummarizationFailure(reason=str(exc))
        summary = _extract_summary(invocation)
        if summary is None:
            return SummarizationFailure(
                reason=(
                    f"summarizer ensemble '{self._summarizer_name}' "
                    "returned no summary text (checked synthesis and "
                    "single-agent response)"
                )
            )
        return SummarizationSuccess(summary=summary)


def _extract_summary(invocation: dict[str, Any]) -> str | None:
    """Pull the summary text from a summarizer ensemble's output.

    The contract is forgiving so the operator can shape the summarizer
    ensemble naturally:

    1. If the ensemble populated ``synthesis`` with a non-empty string,
       that is the summary. This is the preferred shape.
    2. Otherwise, if the ensemble has exactly one agent result with a
       non-empty ``response`` field, use that response. This matches
       the default single-agent summarizer shape — llm-orc's dependency
       based execution model leaves ``synthesis`` unpopulated for
       single-agent ensembles (see ``core/execution/results_processor``
       ``finalize_result``).
    3. Otherwise, return ``None`` — caller raises ``SummarizationFailure``.
    """
    synthesis = invocation.get("synthesis")
    if isinstance(synthesis, str) and synthesis:
        return synthesis

    results = invocation.get("results")
    if isinstance(results, dict) and len(results) == 1:
        only_result = next(iter(results.values()))
        if isinstance(only_result, dict):
            response = only_result.get("response")
            if isinstance(response, str) and response:
                return response
    return None
