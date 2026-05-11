"""Orchestrator Runtime — the ReAct loop behind the serving layer.

Per ``docs/agentic-serving/system-design.md`` §Orchestrator Runtime
(L2) and §Integration Contracts (Serving Layer → Orchestrator Runtime,
Orchestrator Runtime → Budget Controller, Orchestrator Runtime →
Orchestrator Tool Dispatch).

The Runtime drives the orchestrator LLM through a tool-calling loop:

1. Check Budget before each iteration (FC-10; AS-3).
2. Call the LLM with the session's messages and the five-tool schema.
3. If the LLM emits tool calls, dispatch each through Orchestrator
   Tool Dispatch, feed the results back as observations, and loop.
4. If the LLM emits a stop response with no tool calls, terminate
   with ``Completion(finish_reason="stop")``.
5. On Budget exhaustion, terminate with an explicit ContentDelta
   explaining the exhaustion followed by ``Completion("length")`` —
   the termination is not silent.

FC-4: the Runtime imports only BudgetController and a dispatcher
Protocol — no Plexus, no config, no Autonomy, no Calibration. Result
summarization is interposed by Orchestrator Tool Dispatch (WP-D adds
the dedicated Harness). The ``OrchestratorLLM`` Protocol keeps the
Runtime decoupled from any particular model client — WP-C tests pass
a scripted double; production wiring passes a tool-calling adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from llm_orc.agentic.budget_controller import (
    BudgetCheckExhausted,
    BudgetController,
)
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    InternalToolCallInFlight,
    InternalToolCallResult,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    PhantomToolCallError,
    ToolCallResult,
    ToolCallSuccess,
)
from llm_orc.agentic.session_start import ChatMessage, SessionContext
from llm_orc.models.base import ToolCall, ToolCallingResponse


class OrchestratorLLM(Protocol):
    """Tool-calling LLM interface the Runtime drives.

    ``ModelInterface`` satisfies this Protocol structurally — any
    provider that implements ``generate_with_tools`` (opt-in per the
    ``supports_tool_calling`` flag) works here. The Protocol keeps
    the Runtime decoupled from ``ModelInterface``'s abstract
    ``name`` and ``generate_response`` surface so tests can pass
    minimal doubles.
    """

    async def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCallingResponse: ...


class ToolDispatcher(Protocol):
    """Minimum Tool Dispatch surface the Runtime calls."""

    async def dispatch(
        self, call: InternalToolCall, *, session_id: str = ""
    ) -> ToolCallResult: ...

    def validate_response(
        self,
        response_text: str,
        tool_call_names: tuple[str, ...],
        *,
        session_id: str = "",
    ) -> PhantomToolCallError | None:
        """Run ADR-017 structural validation on an orchestrator response."""
        ...


def _build_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-compatible tool schemas for the closed five-tool set.

    Argument schemas are intentionally minimal for WP-C — the LLM sees
    enough to emit valid calls; richer schemas can land with each tool's
    wiring WP (G for compose_ensemble, I for query_knowledge /
    record_outcome).
    """
    specs = {
        "invoke_ensemble": {
            "description": "Execute an ensemble by name with input text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the library ensemble to invoke.",
                    },
                    "input": {
                        "type": "string",
                        "description": "Task input text passed to the ensemble.",
                    },
                },
                "required": ["name", "input"],
            },
        },
        "compose_ensemble": {
            "description": (
                "Compose a new ensemble from library primitives. Not yet wired."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "list_ensembles": {
            "description": "List ensembles available in the library.",
            "parameters": {"type": "object", "properties": {}},
        },
        "query_knowledge": {
            "description": (
                "Query the knowledge graph for prior outcomes. Not yet wired."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        "record_outcome": {
            "description": ("Record a routing decision or outcome. Not yet wired."),
            "parameters": {"type": "object", "properties": {}},
        },
    }
    # Ensure the closed set is exactly the keys we advertise.
    assert set(specs) == TOOL_NAMES
    return [
        {
            "type": "function",
            "function": {"name": name, **spec},
        }
        for name, spec in specs.items()
    ]


class OrchestratorRuntime:
    """Drives the orchestrator's ReAct loop, yielding streaming chunks."""

    def __init__(
        self,
        *,
        llm: OrchestratorLLM,
        budget: BudgetController,
        tool_dispatch: ToolDispatcher,
        system_prompt: str = "",
    ) -> None:
        """Construct a Runtime for one session turn.

        ``system_prompt`` is prepended as a leading ``role: system``
        message on every LLM iteration when non-empty. Teaches the
        orchestrator LLM the five-internal-tool surface (ADR-003),
        Option C's one-kind-per-turn discipline, and the
        ``needs_client_tool`` retry convention (roadmap ODP #8
        mechanism i). Prepends *ahead* of any client-supplied
        ``role: system`` message so the orchestrator's discipline
        survives competing client guidance. Empty string is a no-op —
        used by tests and by deployments that want no orchestrator-
        side prompt.
        """
        self._llm = llm
        self._budget = budget
        self._tool_dispatch = tool_dispatch
        self._system_prompt = system_prompt

    async def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]:
        """Run the ReAct loop for this session turn.

        Tool surface advertised to the LLM is the union of the closed
        internal five (ADR-003) and the client-declared ``tools[]`` from
        ``SessionContext`` (Option C, Client Tool Surface Commitment). On
        each iteration the Runtime routes by ``TOOL_NAMES`` membership:
        names inside dispatch internally; names outside close the turn
        with :class:`ClientToolCall` (``finish_reason: tool_calls`` on the
        wire), so the client executes the tool and resumes the Session on
        the next request.
        """
        state = context.state
        messages: list[dict[str, Any]] = [
            _session_message_to_llm(m) for m in context.messages
        ]
        if self._system_prompt:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
        tools = _build_tool_schemas() + list(context.tools)
        # Client-declared tool names — only these route to ClientToolCall.
        # Names that appear in neither ``TOOL_NAMES`` nor ``client_tool_names``
        # fall through to Tool Dispatch, which returns ``unknown_tool`` — so
        # AS-6 closure (no script authorship) stays enforced regardless of
        # what the LLM hallucinates.
        client_tool_names = _client_tool_names(context.tools)

        while True:
            exhaustion_chunks = self._budget_exhaustion_chunks(state)
            if exhaustion_chunks is not None:
                yield exhaustion_chunks[0]
                yield exhaustion_chunks[1]
                return

            response = await self._llm.generate_with_tools(
                messages=messages, tools=tools
            )
            state.record_iteration(response.usage.total_tokens)

            if self._record_phantom_if_detected(
                messages, response, state.identity.value
            ):
                continue

            chunks, signal = await self._chunks_for_response(
                messages, response, client_tool_names, state.identity.value
            )
            for chunk in chunks:
                yield chunk
            if signal == "return":
                return

    async def _chunks_for_response(
        self,
        messages: list[dict[str, Any]],
        response: ToolCallingResponse,
        client_tool_names: frozenset[str],
        session_id: str,
    ) -> tuple[list[OrchestratorChunk], Literal["return", "continue"]]:
        """Compute chunks + control signal for a (non-phantom) LLM response.

        Returns ``(chunks, "return")`` for terminal cases — bare stop
        response or pure client-tool delegation — so the caller's
        ``run`` loop yields the chunks then returns. Returns
        ``(chunks, "continue")`` for the in-loop cases (mixed batch
        rejection or internal-tool dispatch); the caller yields and
        loops to the next iteration.

        Buffering chunks (rather than yielding directly) keeps the
        outer ``run`` method's cognitive complexity below the
        repository's complexipy ceiling without changing the chunk
        order callers observe.
        """
        chunks: list[OrchestratorChunk] = []
        if response.content:
            chunks.append(ContentDelta(content=response.content))
        if not response.tool_calls:
            chunks.append(Completion(finish_reason="stop"))
            return chunks, "return"
        client_calls, internal_calls = _split_tool_calls(
            response.tool_calls, client_tool_names
        )
        if client_calls and internal_calls:
            # Mixed batch: Option C's one-kind-per-turn discipline is
            # violated. Reject all; the LLM retries with a pure batch
            # on the next iteration. See ``_record_mixed_batch_rejection``.
            _record_mixed_batch_rejection(messages, response)
            return chunks, "continue"
        if client_calls:
            # Option C: pure client-declared batch closes the turn.
            # DAG engine runs atomically (ADR-001/002); no mid-turn
            # callback. The client executes the tools and resumes the
            # Session on the next ``/v1/chat/completions`` request.
            chunks.append(_client_delegation_chunk(client_calls))
            return chunks, "return"
        messages.append(_assistant_message(response))
        async for chunk in self._dispatch_internal_calls(
            response.tool_calls, messages, session_id
        ):
            chunks.append(chunk)
        return chunks, "continue"

    def _budget_exhaustion_chunks(
        self, state: Any
    ) -> tuple[OrchestratorChunk, ...] | None:
        """Return the chunks for the budget-exhaustion path or ``None``.

        ``None`` means the Budget passed; the loop proceeds to the LLM
        call. A non-None tuple is the pair of chunks the caller yields
        before returning — the human-readable exhaustion message
        followed by ``Completion(finish_reason="length")``.
        """
        check = self._budget.check(
            turn_count=state.turn_count, token_spend=state.token_spend
        )
        if not isinstance(check, BudgetCheckExhausted):
            return None
        return (
            ContentDelta(content=_format_exhaustion_message(check)),
            Completion(finish_reason="length"),
        )

    def _record_phantom_if_detected(
        self,
        messages: list[dict[str, Any]],
        response: ToolCallingResponse,
        session_id: str,
    ) -> bool:
        """ADR-017 §Rejection — return True if a phantom claim was recorded.

        Delegates the structural validation to Tool Dispatch
        (``validate_response``). On detection, the rejected response's
        prose is NOT surfaced to the client (the phantom claim would
        mislead); the orchestrator's reasoning surface receives the
        structural feedback via an injected diagnostic and reformulates
        on the next iteration. Caller treats ``True`` as "continue the
        loop"; ``False`` means the response passed validation.
        """
        phantom_error = self._tool_dispatch.validate_response(
            response.content,
            tuple(tc.name for tc in response.tool_calls),
            session_id=session_id,
        )
        if phantom_error is None:
            return False
        _record_phantom_tool_call_rejection(messages, response, phantom_error)
        return True

    async def _dispatch_internal_calls(
        self,
        tool_calls: list[ToolCall],
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> AsyncIterator[OrchestratorChunk]:
        """Dispatch a pure-internal batch and yield observation chunks.

        Each call flows through Tool Dispatch, which interposes Autonomy
        Policy (WP-E), the Calibration Gate (WP-H), and the Result
        Summarizer Harness (WP-D) on the ``invoke_ensemble`` return
        path. The ``messages`` list is mutated in place: a ``role: tool``
        observation is appended per result so the next ReAct iteration's
        LLM call sees the outcome.

        ``session_id`` is the Session identity threaded through so
        dispatch-side per-session state (Calibration Gate records for
        composed ensembles under calibration) keys correctly.

        Extracting this loop out of :meth:`run` keeps that method's
        complexity under the project's cyclomatic ceiling while
        preserving the ReAct loop's readability at the outer level.
        """
        for tool_call in tool_calls:
            yield InternalToolCallInFlight(id=tool_call.id, name=tool_call.name)
            parsed = InternalToolCall(
                id=tool_call.id,
                name=tool_call.name,
                arguments=_safe_parse_arguments(tool_call.arguments_json),
            )
            result = await self._tool_dispatch.dispatch(parsed, session_id=session_id)
            for event in result.events:
                yield event
            yield InternalToolCallResult(
                id=result.id, summary=_tool_result_summary(result)
            )
            messages.append(_tool_result_message(result))


def _session_message_to_llm(message: ChatMessage) -> dict[str, Any]:
    """Translate a :class:`ChatMessage` to the dict shape LLM clients expect.

    Tool-round-trip fields (``tool_call_id``, ``tool_calls``) ride through
    when present so the orchestrator LLM sees its prior delegation and the
    client's ``role: tool`` result on the subsequent turn (Option C, Client
    Tool Surface Commitment). ``content`` is emitted as-is — OpenAI-compat
    providers accept ``None`` on assistant messages whose prior turn
    carried only tool calls.
    """
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        result["tool_calls"] = list(message.tool_calls)
    return result


def _client_tool_names(tools: list[dict[str, Any]]) -> frozenset[str]:
    """Extract the client-declared function names from the request's ``tools[]``.

    Tolerant of malformed entries: missing ``function`` key or missing
    ``name`` is skipped rather than raised so a hostile or truncated
    ``tools[]`` cannot break the Runtime. The Serving Layer already
    trusts client input at this stage; additional validation lives
    outside Runtime's concerns.
    """
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return frozenset(names)


def _split_tool_calls(
    tool_calls: list[ToolCall], client_tool_names: frozenset[str]
) -> tuple[list[ToolCall], list[ToolCall]]:
    """Partition a tool-calls batch into ``(client_calls, internal_calls)``.

    Membership in ``client_tool_names`` (derived from the request's
    ``tools[]``) is the classifier. Names in neither set — e.g., an LLM
    hallucination like ``create_script`` that matches no internal tool
    and no client-declared tool — route as internal and get rejected by
    Tool Dispatch's unknown-tool path, which keeps AS-6 closure
    structural.
    """
    client = [tc for tc in tool_calls if tc.name in client_tool_names]
    internal = [tc for tc in tool_calls if tc.name not in client_tool_names]
    return client, internal


def _record_mixed_batch_rejection(
    messages: list[dict[str, Any]], response: ToolCallingResponse
) -> None:
    """Append the LLM's mixed-batch response + per-call rejections to history.

    Records the assistant turn that violated the one-kind-per-turn
    discipline, then a ``role: tool`` error observation for every tool
    call in that turn so the LLM's next iteration sees a complete,
    correctly-correlated history and can retry with a pure batch.
    """
    messages.append(_assistant_message(response))
    for tool_call in response.tool_calls:
        messages.append(_mixed_batch_error_observation(tool_call))


def _record_phantom_tool_call_rejection(
    messages: list[dict[str, Any]],
    response: ToolCallingResponse,
    phantom_error: PhantomToolCallError,
) -> None:
    """Append the rejected assistant turn + a structural-feedback diagnostic.

    ADR-017 §Rejection: the orchestrator must take a different action —
    re-emit with actual tool-call structures, reformulate, or abstain.
    The diagnostic is appended as a ``role: user`` message because the
    rejected response had no tool-call structures to attach a
    ``role: tool`` observation to. The next iteration's LLM call sees
    the rejected turn followed by the rejection reason and can
    reformulate.
    """
    messages.append(_assistant_message(response))
    payload = {
        "error": "phantom_tool_call",
        "reason": (
            "Your previous response asserted that a tool call has been "
            "made or that a tool result is displayed above, but no "
            "tool-call structure was emitted in that response. The "
            "structural validation guard rejected the response. To "
            "continue, either emit the tool call as a structured "
            "tool_call rather than narrating it in prose, or revise "
            "the response so it does not assert that a tool call has "
            "occurred."
        ),
        "detected_prose_claim": phantom_error.dispatch_context["detected_prose_claim"],
        "recovery_action_required": phantom_error.recovery_action_required,
    }
    messages.append({"role": "user", "content": json.dumps(payload)})


def _client_delegation_chunk(client_calls: list[ToolCall]) -> ClientToolCall:
    """Build the :class:`ClientToolCall` chunk that closes the turn per Option C.

    The Serving Layer translates this into
    ``finish_reason: tool_calls`` on the wire (streaming or non-
    streaming). ``arguments`` stays in OpenAI's pre-encoded JSON-string
    form — :class:`ToolCallInvocation` carries it through unchanged.
    """
    return ClientToolCall(
        tool_calls=tuple(_client_invocation(tc) for tc in client_calls)
    )


def _mixed_batch_error_observation(tool_call: ToolCall) -> dict[str, Any]:
    """``role: tool`` error observation for a call in a rejected mixed batch.

    When the orchestrator LLM emits a batch that mixes internal
    (``TOOL_NAMES``) with client-declared tool calls, Option C's one-
    kind-per-turn discipline is violated. The Runtime feeds this
    observation back per call so the LLM can reason about how to retry
    without losing data — the ``mixed_batch`` error kind and the inline
    reason teach the model to re-emit either internal tools only or
    client-declared tools only on the next iteration.
    """
    payload = {
        "error": "mixed_batch",
        "reason": (
            "This turn mixed an internal ensemble tool "
            "(invoke_ensemble, compose_ensemble, list_ensembles, "
            "query_knowledge, record_outcome) with a client-declared "
            "tool. Retry with one kind per turn: internal tools run "
            "in-process; client-declared tools close the turn and are "
            "executed by the client."
        ),
    }
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(payload),
    }


def _client_invocation(tool_call: ToolCall) -> ToolCallInvocation:
    """Translate the LLM's client-declared ``ToolCall`` to a chunk invocation.

    ``arguments_json`` is the OpenAI-compatible pre-encoded string the LLM
    emitted; keeping it in wire form matches :class:`ToolCallInvocation`'s
    contract so the formatter avoids re-encoding.
    """
    return ToolCallInvocation(
        id=tool_call.id,
        name=tool_call.name,
        arguments=tool_call.arguments_json,
    )


def _safe_parse_arguments(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON-string arguments, defaulting to empty on bad JSON.

    Invalid JSON becomes an empty dict; the tool handler surfaces any
    argument-validation error as a tool observation (see Tool Dispatch
    ``invalid_arguments`` handling). The Runtime does not distinguish
    malformed JSON from missing fields — both become validation errors
    at the tool boundary.
    """
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_message(response: ToolCallingResponse) -> dict[str, Any]:
    """Assistant-turn message containing tool calls for the next iteration."""
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments_json},
            }
            for tc in response.tool_calls
        ],
    }


def _tool_result_summary(result: ToolCallResult) -> str:
    """Produce a short summary string for the ``InternalToolCallResult`` chunk.

    WP-C uses a trivial format (success/error + tool name). WP-D's
    Result Summarizer Harness produces the properly-summarized content
    for AS-7 compliance; this placeholder will be superseded.
    """
    if isinstance(result, ToolCallSuccess):
        return f"ok: {result.name}"
    return f"error: {result.name} ({result.kind})"


def _tool_result_message(result: ToolCallResult) -> dict[str, Any]:
    """``role: tool`` observation fed back to the LLM for the next iteration."""
    content = (
        json.dumps(result.content, default=str)
        if isinstance(result, ToolCallSuccess)
        else json.dumps({"error": result.kind, "reason": result.reason})
    )
    return {
        "role": "tool",
        "tool_call_id": result.id,
        "content": content,
    }


def _format_exhaustion_message(check: BudgetCheckExhausted) -> str:
    """Human-readable termination message surfaced to the client."""
    if check.reason == "turn_limit":
        return (
            f"[Session budget exhausted: turn limit reached "
            f"({check.turn_count}/{check.turn_limit})]"
        )
    return (
        f"[Session budget exhausted: token limit reached "
        f"({check.token_spend}/{check.token_limit})]"
    )
