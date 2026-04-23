"""Tests for the Orchestrator Tool Dispatch module.

Per `docs/agentic-serving/system-design.md` §Orchestrator Tool Dispatch
(L2) and §Integration Contracts (Orchestrator Runtime → Orchestrator
Tool Dispatch, Orchestrator Tool Dispatch → Autonomy Policy).

Covers scenarios:

* §Orchestrator tool surface is exactly the committed set (FC-5)
* §Invocation outside the tool set is rejected
* §Default Autonomy Level permits invocation, permits composition
* §Tool user without operator role observes composition events when configured
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from llm_orc.agentic.autonomy_policy import (
    BASELINE_LEVEL,
    PURE_TOOL_USER_VISIBLE_LEVEL,
    Allow,
    AutonomyDecision,
    AutonomyPolicy,
    Deny,
)
from llm_orc.agentic.composition_validator import (
    CompositionAccepted,
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
)
from llm_orc.agentic.orchestrator_chunk import VisibilityEvent
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    InternalToolCall,
    OrchestratorToolDispatch,
    ToolCallError,
    ToolCallSuccess,
)
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.core.config.ensemble_config import EnsembleConfig


class _RaisingOperations:
    """Defaults to raising on any call — vacuous-mock hazard prevention.

    Individual tests subclass or replace the specific method they
    exercise.
    """

    async def invoke(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError(
            f"operations.invoke should not be called in this test: {arguments!r}"
        )

    async def read_ensembles(self) -> list[dict[str, Any]]:  # pragma: no cover
        raise AssertionError(
            "operations.read_ensembles should not be called in this test"
        )


class _ScriptedOperations(_RaisingOperations):
    """Programmable ``EnsembleOperations`` double.

    Feeds canned ``invoke`` results and ensemble listings, records
    arguments for assertions, and optionally raises ``ValueError`` to
    simulate the "ensemble not found" path that the real
    ``ExecutionHandler.invoke`` surfaces.
    """

    def __init__(
        self,
        *,
        invoke_result: dict[str, Any] | None = None,
        invoke_raises: BaseException | None = None,
        ensembles: list[dict[str, Any]] | None = None,
    ) -> None:
        self._invoke_result = invoke_result
        self._invoke_raises = invoke_raises
        self._ensembles = list(ensembles or [])
        self.invoke_calls: list[dict[str, Any]] = []

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append(dict(arguments))
        if self._invoke_raises is not None:
            raise self._invoke_raises
        return self._invoke_result or {}

    async def read_ensembles(self) -> list[dict[str, Any]]:
        return list(self._ensembles)


class _StubSummarizerInvoker:
    """Handwritten double for ``SummarizerInvoker``.

    Default behavior is "summarizer returns a synthesis string so the
    Harness produces a ``SummarizationSuccess``". Tests that exercise
    the raw-output escape hatch or summarization failure override the
    returns/raises parameters.
    """

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


def _build_harness(
    *,
    returns: dict[str, Any] | None = None,
    raises: Exception | None = None,
    summarizer_name: str = "agentic-result-summarizer",
) -> ResultSummarizerHarness:
    """Construct a Harness with a stub invoker for dispatch-side tests."""
    invoker = _StubSummarizerInvoker(returns=returns, raises=raises)
    return ResultSummarizerHarness(invoker=invoker, summarizer_name=summarizer_name)


def _permissive_policy(level: str = BASELINE_LEVEL) -> AutonomyPolicy:
    """Real AutonomyPolicy at the given level — default baseline silent."""
    return AutonomyPolicy(level_provider=lambda: level)


class _UnusedValidator:
    """Validator stub that fails loudly if consulted.

    Used exclusively by the malformed-arguments test to prove the
    dispatch short-circuits before reaching the validator.
    """

    def validate(
        self, request: CompositionRequest
    ) -> CompositionOutcome:  # pragma: no cover
        raise AssertionError(
            f"composition_validator should not be called in this test: {request.name!r}"
        )


class _UnusedWriter:
    """Writer stub for tests that never accept a composition.

    Raises if the writer is consulted. Paired with the default
    rejecting validator in :func:`_build_dispatch` so tests that
    dispatch ``compose_ensemble`` without scripting a real acceptance
    never reach the write path.
    """

    def write(self, config: EnsembleConfig) -> str:  # pragma: no cover
        raise AssertionError(
            f"local_ensemble_writer should not be called in this test: {config.name!r}"
        )


def _rejecting_validator() -> _ScriptedValidator:
    """Default validator for tests that dispatch compose but do not test it.

    Rejects with a neutral reason; the dispatch produces a
    ``ToolCallError(kind="invocation_failed")`` that tests at
    autonomy/routing scope can ignore. Tests that assert on
    composition behavior pass a scripted validator explicitly.
    """
    return _ScriptedValidator(
        CompositionRejected(
            kind="missing_primitive",
            reason="default rejecting validator for non-compose tests",
        )
    )


def _build_dispatch(
    *,
    operations: Any,
    harness: ResultSummarizerHarness | None = None,
    autonomy_policy: Any = None,
    composition_validator: Any = None,
    local_ensemble_writer: Any = None,
) -> OrchestratorToolDispatch:
    """Construct a dispatch with sensible test defaults.

    Default validator rejects any request so tests that dispatch
    ``compose_ensemble`` incidentally (e.g., autonomy-gate tests)
    never reach the writer. Tests that exercise the accept path pass
    a scripted validator explicitly.
    """
    return OrchestratorToolDispatch(
        operations=operations,
        harness=harness if harness is not None else _build_harness(),
        autonomy_policy=(
            autonomy_policy if autonomy_policy is not None else _permissive_policy()
        ),
        composition_validator=(
            composition_validator
            if composition_validator is not None
            else _rejecting_validator()
        ),
        local_ensemble_writer=(
            local_ensemble_writer
            if local_ensemble_writer is not None
            else _UnusedWriter()
        ),
    )


class _ScriptedValidator:
    """Returns a scripted :class:`CompositionOutcome` and records calls."""

    def __init__(self, outcome: CompositionOutcome) -> None:
        self._outcome = outcome
        self.calls: list[CompositionRequest] = []

    def validate(self, request: CompositionRequest) -> CompositionOutcome:
        self.calls.append(request)
        return self._outcome


class _RecordingWriter:
    """Records the configs it receives and returns a canned path."""

    def __init__(self, path: str = "/tmp/composed.yaml") -> None:
        self._path = path
        self.written: list[EnsembleConfig] = []

    def write(self, config: EnsembleConfig) -> str:
        self.written.append(config)
        return self._path


class _RecordingPolicy:
    """AutonomyPolicy double that records calls and returns a canned decision.

    Satisfies the AutonomyGate Protocol structurally. Used by tests that
    need to assert the gate was consulted (with what arguments) or that
    need to script a Deny that the real Phase 1 policy never produces.
    """

    def __init__(self, decision: AutonomyDecision | None = None) -> None:
        self._decision: AutonomyDecision = decision if decision is not None else Allow()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def decide(self, *, tool_name: str, arguments: dict[str, Any]) -> AutonomyDecision:
        self.calls.append((tool_name, dict(arguments)))
        return self._decision


class TestDispatchRejectsUnknownTool:
    """Scenario: Invocation outside the tool set is rejected.

    Tool Dispatch is the structural enforcement point for ADR-003 —
    the closed tool set. A name outside the five committed tools
    returns a typed tool error the Runtime can surface to the
    orchestrator LLM as an observation; the ReAct loop continues.
    """

    @pytest.mark.asyncio
    async def test_dispatch_rejects_unknown_tool_name(self) -> None:
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_abc",
                name="hallucinated_tool",
                arguments={},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.id == "call_abc"
        assert result.name == "hallucinated_tool"
        assert result.kind == "unknown_tool"


class TestClosedToolSet:
    """Scenario: Orchestrator tool surface is exactly the committed set (FC-5).

    The closed-set property (ADR-003) is enforced structurally: the
    five tool names in ``TOOL_NAMES`` correspond to exactly five
    async methods on the dispatch class. A sixth public async tool
    method would mean the closed set is no longer closed.
    """

    def test_tool_dispatch_exposes_exactly_five_tool_methods(self) -> None:
        tool_methods = {
            name
            for name, member in inspect.getmembers(OrchestratorToolDispatch)
            if inspect.iscoroutinefunction(member) and name in TOOL_NAMES
        }

        assert tool_methods == {
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        }
        assert len(tool_methods) == 5

    def test_tool_names_set_matches_committed_five(self) -> None:
        """ADR-003's tool set is what the module advertises as ``TOOL_NAMES``."""
        assert TOOL_NAMES == frozenset(
            {
                "invoke_ensemble",
                "compose_ensemble",
                "list_ensembles",
                "query_knowledge",
                "record_outcome",
            }
        )


class TestNotYetWiredTools:
    """WP-G leaves only ``query_knowledge`` and ``record_outcome`` pending.

    Those two tools exist (to honor the closed-set property) but
    return a typed ``not_yet_wired`` tool error. The Runtime surfaces
    these to the orchestrator LLM as an observation — the LLM is free
    to plan around them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "landing_wp"),
        [
            ("query_knowledge", "WP-I"),
            ("record_outcome", "WP-I"),
        ],
    )
    async def test_not_yet_wired_tool_returns_typed_error(
        self, tool_name: str, landing_wp: str
    ) -> None:
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name=tool_name, arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.name == tool_name
        assert result.kind == "not_yet_wired"
        assert landing_wp in result.reason


class TestComposeEnsemble:
    """compose_ensemble delegates to CompositionValidator + writer (WP-G).

    Unit tests cover four branches: validator accept → writer writes
    and the ToolCallSuccess names the path; validator reject →
    invocation_failed surfaces the reason; writer failure →
    invocation_failed surfaces the write error; malformed arguments →
    invalid_arguments without touching the validator.
    """

    @pytest.mark.asyncio
    async def test_compose_writes_when_validator_accepts(self) -> None:
        accepted_config = EnsembleConfig(
            name="combo",
            description="x",
            agents=[],
        )
        validator = _ScriptedValidator(CompositionAccepted(config=accepted_config))
        writer = _RecordingWriter(path="/tmp/combo.yaml")
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            composition_validator=validator,
            local_ensemble_writer=writer,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c1",
                name="compose_ensemble",
                arguments={
                    "name": "combo",
                    "description": "x",
                    "agents": [],
                },
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == {"name": "combo", "path": "/tmp/combo.yaml"}
        assert writer.written == [accepted_config]
        assert [req.name for req in validator.calls] == ["combo"]

    @pytest.mark.asyncio
    async def test_compose_surfaces_validator_rejection_as_invocation_failed(
        self,
    ) -> None:
        rejection = CompositionRejected(
            kind="cross_ensemble_cycle",
            reason="cross-ensemble cycle detected: combo -> a -> combo",
        )
        validator = _ScriptedValidator(rejection)
        writer = _RecordingWriter()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            composition_validator=validator,
            local_ensemble_writer=writer,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c1",
                name="compose_ensemble",
                arguments={
                    "name": "combo",
                    "description": "x",
                    "agents": [{"name": "a", "ensemble": "combo"}],
                },
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert "cycle" in result.reason.lower()
        assert writer.written == [], "writer must not run on validator rejection"

    @pytest.mark.asyncio
    async def test_compose_surfaces_write_failure_as_invocation_failed(self) -> None:
        from llm_orc.agentic.composition_validator import EnsembleWriteError

        accepted_config = EnsembleConfig(name="combo", description="x", agents=[])
        validator = _ScriptedValidator(CompositionAccepted(config=accepted_config))

        class _FailingWriter:
            def write(self, config: EnsembleConfig) -> str:
                raise EnsembleWriteError(f"ensemble '{config.name}' already exists")

        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            composition_validator=validator,
            local_ensemble_writer=_FailingWriter(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c1",
                name="compose_ensemble",
                arguments={
                    "name": "combo",
                    "description": "x",
                    "agents": [],
                },
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert "already exists" in result.reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("arguments", "reason_contains"),
        [
            ({}, "name"),
            ({"name": "", "description": "x", "agents": []}, "name"),
            ({"name": "combo", "description": 42, "agents": []}, "description"),
            ({"name": "combo", "description": "x", "agents": "not-a-list"}, "agents"),
            ({"name": "combo", "description": "x", "agents": ["not-a-dict"]}, "agents"),
        ],
    )
    async def test_compose_rejects_malformed_arguments_without_calling_validator(
        self,
        arguments: dict[str, Any],
        reason_contains: str,
    ) -> None:
        validator = _UnusedValidator()
        writer = _UnusedWriter()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            composition_validator=validator,
            local_ensemble_writer=writer,
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="c1", name="compose_ensemble", arguments=arguments)
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invalid_arguments"
        assert reason_contains in result.reason


class TestListEnsembles:
    """list_ensembles delegates to ``EnsembleOperations.read_ensembles``."""

    @pytest.mark.asyncio
    async def test_list_ensembles_returns_library_entries(self) -> None:
        operations = _ScriptedOperations(
            ensembles=[
                {
                    "name": "analysis",
                    "description": "Analyzes code",
                    "source": "local",
                    "relative_path": "analysis.yaml",
                    "agent_count": 1,
                }
            ]
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_1", name="list_ensembles", arguments={})
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "list_ensembles"
        entries = result.content
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["name"] == "analysis"
        assert entries[0]["description"] == "Analyzes code"


class TestInvokeEnsemble:
    """invoke_ensemble delegates to ``EnsembleOperations.invoke``.

    ``OrchestraService.invoke`` returns a normalized
    ``{results, synthesis, status}`` shape; that dict flows through as
    the ``ToolCallSuccess.content``. Missing-ensemble errors surface
    as ``ValueError`` from the handler and translate to a
    ``ToolCallError(kind="invocation_failed")``.
    """

    @pytest.mark.asyncio
    async def test_invoke_ensemble_delegates_and_interposes_summarizer(self) -> None:
        """Scenario: Ensemble result is summarized before entering context.

        Per ADR-004 and system design Amendment #3, Tool Dispatch calls
        the Result Summarizer Harness on every successful invocation.
        The Runtime receives ``{"summary": ...}`` — never the raw
        result dict — unless the ensemble's ``raw_output`` flag is set.
        """
        normalized_result = {
            "results": {"analyst": {"response": "ok"}},
            "synthesis": "ensemble's own synthesis",
            "status": "success",
            "raw_output": False,
        }
        operations = _ScriptedOperations(invoke_result=normalized_result)
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(returns={"synthesis": "distilled summary"}),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_7",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "refactor the parser"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.id == "call_7"
        assert result.name == "invoke_ensemble"
        # The summary — produced by the Harness, not the raw ensemble result —
        # is what enters the orchestrator's context.
        assert result.content == {"summary": "distilled summary"}
        # Delegation still uses the handler's field name.
        assert operations.invoke_calls == [
            {"ensemble_name": "analysis", "input": "refactor the parser"}
        ]

    @pytest.mark.asyncio
    async def test_invoke_ensemble_passes_raw_when_ensemble_flagged(self) -> None:
        """Scenario: Raw-output escape hatch is explicit (ADR-004).

        An ensemble declaring ``raw_output: true`` bypasses the Harness
        — the raw result dict enters the orchestrator's context without
        passing through the summarizer.
        """
        raw_result = {
            "results": {"classifier": {"intent": "refactor"}},
            "synthesis": None,
            "status": "success",
            "raw_output": True,
        }
        operations = _ScriptedOperations(invoke_result=raw_result)
        harness = _build_harness(returns={"synthesis": "WOULD-SUMMARIZE-BUT-SHOULDNT"})
        dispatch = _build_dispatch(
            operations=operations,
            harness=harness,
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_raw",
                name="invoke_ensemble",
                arguments={"name": "intent_classifier", "input": "refactor this"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == raw_result, (
            "raw_output=True must pass the ensemble result through untouched"
        )

    @pytest.mark.asyncio
    async def test_invoke_ensemble_surfaces_summarization_failure_as_tool_error(
        self,
    ) -> None:
        """AS-7: the orchestrator never sees unsummarized results.

        If the summarizer ensemble fails, Tool Dispatch returns a typed
        ``ToolCallError(kind="summarization_failed")`` rather than
        exposing the raw dict as a fallback. The raw result still lives
        in the ensemble's execution artifact (Invariant 9) — only the
        orchestrator's context is gated.
        """
        normalized_result = {
            "results": {"a": "x"},
            "synthesis": "raw",
            "status": "success",
            "raw_output": False,
        }
        operations = _ScriptedOperations(invoke_result=normalized_result)
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(raises=ValueError("summarizer missing")),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_fail",
                name="invoke_ensemble",
                arguments={"name": "anything", "input": "x"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "summarization_failed"
        assert "summarizer missing" in result.reason

    @pytest.mark.asyncio
    async def test_invoke_ensemble_returns_error_when_name_not_in_library(
        self,
    ) -> None:
        """A hallucinated ensemble name becomes an observation, not a crash.

        The real handler raises ``ValueError("Ensemble does not exist: ...")``
        — Tool Dispatch converts that to ``ToolCallError``.
        """
        operations = _ScriptedOperations(
            invoke_raises=ValueError("Ensemble does not exist: does-not-exist")
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call_8",
                name="invoke_ensemble",
                arguments={"name": "does-not-exist", "input": "anything"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.name == "invoke_ensemble"
        assert result.kind == "invocation_failed"
        assert "does-not-exist" in result.reason

    @pytest.mark.asyncio
    async def test_invoke_ensemble_rejects_missing_name_argument(self) -> None:
        """Input validation — missing or empty ``name`` is a typed error
        surfaced without calling the handler."""
        operations = _ScriptedOperations()  # invoke would return {} if called
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="call_9", name="invoke_ensemble", arguments={})
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invalid_arguments"
        # Handler must not have been called — validation is local.
        assert operations.invoke_calls == []


class TestAutonomyGate:
    """Autonomy Policy gates every dispatch (FC-11).

    The gate sits between the unknown-tool filter and the tool method
    routing:

    * Unknown tool names short-circuit before the gate — the gate never
      sees names outside ``TOOL_NAMES`` and cannot be a source of AS-6
      leakage.
    * Allow → tool runs, decision events attach to the result.
    * Deny → tool does NOT run; a typed ``denied_by_autonomy`` error
      returns with the decision's reason.

    Covers scenarios §Default Autonomy Level permits invocation
    (unit-level) and §Tool user without operator role observes
    composition events (unit-level; full acceptance lands in Group 5).
    """

    @pytest.mark.asyncio
    async def test_gate_consulted_for_every_committed_tool(self) -> None:
        policy = _RecordingPolicy(decision=Allow())
        operations = _ScriptedOperations(
            invoke_result={
                "results": {"a": "x"},
                "synthesis": "ok",
                "status": "success",
                "raw_output": True,  # bypass summarizer so invoke path stays simple
            },
            ensembles=[],
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=policy,
        )

        for tool_name in [
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        ]:
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"call_{tool_name}",
                    name=tool_name,
                    arguments={"name": "anything", "input": "x"},
                )
            )

        seen_tool_names = [call[0] for call in policy.calls]
        assert seen_tool_names == [
            "invoke_ensemble",
            "compose_ensemble",
            "list_ensembles",
            "query_knowledge",
            "record_outcome",
        ]

    @pytest.mark.asyncio
    async def test_gate_not_consulted_for_unknown_tool(self) -> None:
        """AS-6 closure lives in ``TOOL_NAMES`` — unknown names never reach Autonomy."""
        policy = _RecordingPolicy()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=policy,
        )

        result = await dispatch.dispatch(
            InternalToolCall(id="x", name="hallucinated_tool", arguments={})
        )

        assert policy.calls == []
        assert isinstance(result, ToolCallError)
        assert result.kind == "unknown_tool"

    @pytest.mark.asyncio
    async def test_allow_with_events_attaches_events_to_success_result(self) -> None:
        composition_event = VisibilityEvent(
            kind="composition",
            payload={"tool": "compose_ensemble", "arguments": {}},
        )
        policy = _RecordingPolicy(decision=Allow(events=(composition_event,)))
        operations = _ScriptedOperations()
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=policy,
        )

        # list_ensembles returns a ToolCallSuccess so we can exercise the
        # success-result branch of event attachment.
        result = await dispatch.dispatch(
            InternalToolCall(id="c1", name="list_ensembles", arguments={})
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.events == (composition_event,)

    @pytest.mark.asyncio
    async def test_allow_with_events_attaches_events_to_error_result(self) -> None:
        """Events surface regardless of whether the tool succeeded or errored.

        A validator rejection on ``compose_ensemble`` still counts as a
        composition attempt from the user's perspective — the event
        fires, the tool-side error is what the orchestrator observes.
        """
        composition_event = VisibilityEvent(
            kind="composition",
            payload={"tool": "compose_ensemble", "arguments": {"name": "x"}},
        )
        policy = _RecordingPolicy(decision=Allow(events=(composition_event,)))
        rejecting_validator = _ScriptedValidator(
            CompositionRejected(
                kind="missing_primitive",
                reason="profile 'ghost' not in library",
            )
        )
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=policy,
            composition_validator=rejecting_validator,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c2",
                name="compose_ensemble",
                arguments={"name": "x", "description": "y", "agents": []},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert result.events == (composition_event,)

    @pytest.mark.asyncio
    async def test_deny_short_circuits_with_typed_error(self) -> None:
        policy = _RecordingPolicy(
            decision=Deny(reason="tightened level requires approval")
        )
        operations = _ScriptedOperations()
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=policy,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="d1",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "x"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "denied_by_autonomy"
        assert "tightened level requires approval" in result.reason
        # The tool method must not have been called.
        assert operations.invoke_calls == []

    @pytest.mark.asyncio
    async def test_pure_tool_user_visible_level_emits_composition_event(self) -> None:
        """End-to-end at the dispatch boundary with the real policy.

        The tightened level is ``pure-tool-user-visible``. Only
        ``compose_ensemble`` surfaces an event; other tools stay silent.
        """
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(level=PURE_TOOL_USER_VISIBLE_LEVEL),
        )

        compose_result = await dispatch.dispatch(
            InternalToolCall(
                id="c1", name="compose_ensemble", arguments={"name": "new"}
            )
        )

        assert len(compose_result.events) == 1
        event = compose_result.events[0]
        assert event.kind == "composition"
        assert event.payload == {
            "tool": "compose_ensemble",
            "arguments": {"name": "new"},
        }

    @pytest.mark.asyncio
    async def test_baseline_level_emits_no_events_on_compose(self) -> None:
        """Default level is silent on composition per ADR-008 baseline."""
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(level=BASELINE_LEVEL),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c1", name="compose_ensemble", arguments={"name": "new"}
            )
        )

        assert result.events == ()
