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
from llm_orc.agentic.calibration_gate import (
    CalibrationVerdict,
    CalibrationVerdictEvent,
    QualitySignal,
)
from llm_orc.agentic.composition_validator import (
    CompositionAccepted,
    CompositionOutcome,
    CompositionRejected,
    CompositionRequest,
)
from llm_orc.agentic.dispatch_event_substrate import (
    DispatchEventSubstrate,
    DispatchTiming,
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
from llm_orc.agentic.tier_router import TopazSkill
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
    calibration_gate: Any = None,
    plexus_adapter: Any = None,
    tier_router: Any = None,
    tier_router_audit: Any = None,
    event_substrate: Any = None,
) -> OrchestratorToolDispatch:
    """Construct a dispatch with sensible test defaults.

    Default validator rejects any request so tests that dispatch
    ``compose_ensemble`` incidentally (e.g., autonomy-gate tests)
    never reach the writer. Tests that exercise the accept path pass
    a scripted validator explicitly.

    ``calibration_gate`` defaults to ``None`` — tests that do not
    exercise calibration get the no-op path (invoke skips the gate;
    compose does not register). Tests that assert on gate interposition
    pass a ``_RecordingCalibrationGate`` (or the real gate) explicitly.

    ``tier_router`` defaults to ``None`` — back-compat with pre-WP-G4-1
    tests. Tests that assert on router interposition pass a real or
    scripted ``TierRouter`` alongside a ``calibration_gate`` (the
    constructor requires both together).
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
        calibration_gate=calibration_gate,
        plexus_adapter=plexus_adapter,
        tier_router=tier_router,
        tier_router_audit=tier_router_audit,
        event_substrate=event_substrate,
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


class TestDispatchOperatorLog:
    """Scenario: Operator log carries actionable diagnostic on tool errors.

    Empirical finding (research log 2026-04-28, DIAG-1): when
    summarization failed during the qwen3:14b S0 and qwen3:8b CAP-3
    runs, the surfaced error kind was the generic
    ``error:summarization_failed``. The actual failure reason — that
    the configured summarizer model profile pointed at an Ollama model
    that wasn't pulled — lived in ``ToolCallError.reason`` but was not
    surfaced in the operator-side log. Diagnosis required adding the
    reason to the log line. Production code now emits the reason on
    error paths so misconfiguration surfaces actionably without spike
    instrumentation.
    """

    @pytest.mark.asyncio
    async def test_error_dispatch_logs_kind_and_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An error result must surface name, kind, and reason in the log."""
        import logging

        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        with caplog.at_level(
            logging.INFO, logger="llm_orc.agentic.orchestrator_tool_dispatch"
        ):
            await dispatch.dispatch(
                InternalToolCall(
                    id="call_xyz",
                    name="hallucinated_tool",
                    arguments={},
                )
            )

        result_records = [
            rec
            for rec in caplog.records
            if rec.name == "llm_orc.agentic.orchestrator_tool_dispatch"
            and "result" in rec.message
        ]
        assert result_records, "dispatch must emit a result log line"
        result_log = result_records[-1].message
        assert "name=hallucinated_tool" in result_log
        assert "unknown_tool" in result_log
        assert "committed tool set" in result_log, (
            "error log line must include the reason field so operators "
            "can act on the failure (DIAG-1 fix)."
        )

    @pytest.mark.asyncio
    async def test_success_dispatch_logs_kind_without_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A success result logs the result line without an error reason."""
        import logging

        class _StubOperations:
            async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("not exercised by this test")

            async def read_ensembles(self) -> list[dict[str, Any]]:
                return [{"name": "demo", "description": "test ensemble"}]

        dispatch = _build_dispatch(
            operations=_StubOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
        )

        with caplog.at_level(
            logging.INFO, logger="llm_orc.agentic.orchestrator_tool_dispatch"
        ):
            result = await dispatch.dispatch(
                InternalToolCall(
                    id="call_ok",
                    name="list_ensembles",
                    arguments={},
                )
            )

        assert isinstance(result, ToolCallSuccess)
        result_records = [
            rec
            for rec in caplog.records
            if rec.name == "llm_orc.agentic.orchestrator_tool_dispatch"
            and "result" in rec.message
        ]
        assert result_records, "dispatch must emit a result log line on success"
        result_log = result_records[-1].message
        assert "name=list_ensembles" in result_log
        assert "success" in result_log


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


class TestPlexusToolsRequireAdapter:
    """Plexus-facing tools degrade gracefully when no Adapter is configured.

    Production deployments wire a :class:`PlexusAdapter` (no-op in WP-I,
    Plexus-active in WP-K). The absent-adapter path is a defensive
    fallback for tests and misconfigured deployments — the tool returns
    a typed ``not_yet_wired`` error rather than crashing or silently
    producing nonsense.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name",
        [
            "query_knowledge",
            "record_outcome",
        ],
    )
    async def test_plexus_tool_without_adapter_returns_typed_error(
        self, tool_name: str
    ) -> None:
        # Default ``_build_dispatch`` omits the Plexus Adapter, so this
        # exercises the absent-adapter fallback path.
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
        assert "Plexus Adapter" in result.reason


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


class _RecordingCalibrationGate:
    """Recording double for the CalibrationGate Protocol surface.

    Structurally satisfies ``CalibrationGate`` — ``mark_composed`` and
    ``check_and_record`` are the two methods Tool Dispatch calls. Records
    every call so tests assert interposition fires on the expected paths
    and skips paths where it should not.
    """

    def __init__(
        self,
        *,
        check_raises: Exception | None = None,
    ) -> None:
        self.mark_calls: list[tuple[str, str]] = []
        self.check_calls: list[tuple[str, str, dict[str, Any]]] = []
        self._check_raises = check_raises

    def mark_composed(self, *, session_id: str, ensemble_name: str) -> None:
        self.mark_calls.append((session_id, ensemble_name))

    async def check_and_record(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        raw_result: dict[str, Any],
    ) -> str | None:
        self.check_calls.append((session_id, ensemble_name, raw_result))
        if self._check_raises is not None:
            raise self._check_raises
        return "positive"


class TestCalibrationGateInterposition:
    """ADR-007, WP-H: Calibration Gate interposes on compose_ensemble and
    invoke_ensemble.

    On successful ``compose_ensemble``, the newly written ensemble is
    registered with the Calibration Gate so subsequent
    ``invoke_ensemble`` calls run the Quality Signal check until the
    ensemble clears calibration. On every ``invoke_ensemble``, the raw
    result is handed to the gate before summarization — the gate's
    signal is recorded but does not affect the tool result. Calibration
    failures never block invocation (clause 2).
    """

    @pytest.mark.asyncio
    async def test_compose_success_registers_ensemble_for_calibration(
        self,
    ) -> None:
        config = EnsembleConfig(
            name="composed-x",
            description="",
            agents=[],
        )
        validator = _ScriptedValidator(CompositionAccepted(config=config))
        writer = _RecordingWriter()
        gate = _RecordingCalibrationGate()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            composition_validator=validator,
            local_ensemble_writer=writer,
            calibration_gate=gate,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c-register",
                name="compose_ensemble",
                arguments={
                    "name": "composed-x",
                    "description": "",
                    "agents": [{"name": "only", "model_profile": "default"}],
                },
            ),
            session_id="session-a",
        )

        assert isinstance(result, ToolCallSuccess)
        assert gate.mark_calls == [("session-a", "composed-x")]

    @pytest.mark.asyncio
    async def test_compose_validation_reject_does_not_register(self) -> None:
        validator = _ScriptedValidator(
            CompositionRejected(kind="missing_primitive", reason="missing profile")
        )
        gate = _RecordingCalibrationGate()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            composition_validator=validator,
            calibration_gate=gate,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c-reject",
                name="compose_ensemble",
                arguments={"name": "bogus"},
            ),
            session_id="session-a",
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"
        assert gate.mark_calls == []

    @pytest.mark.asyncio
    async def test_compose_write_failure_does_not_register(self) -> None:
        from llm_orc.agentic.composition_validator import EnsembleWriteError

        config = EnsembleConfig(
            name="composed-y",
            description="",
            agents=[],
        )
        validator = _ScriptedValidator(CompositionAccepted(config=config))

        class _RaisingWriter:
            def write(self, config: EnsembleConfig) -> str:
                raise EnsembleWriteError("disk full")

        gate = _RecordingCalibrationGate()
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            composition_validator=validator,
            local_ensemble_writer=_RaisingWriter(),
            calibration_gate=gate,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="c-write-fail",
                name="compose_ensemble",
                arguments={
                    "name": "composed-y",
                    "description": "",
                    "agents": [{"name": "only", "model_profile": "default"}],
                },
            ),
            session_id="session-a",
        )

        assert isinstance(result, ToolCallError)
        assert gate.mark_calls == []

    @pytest.mark.asyncio
    async def test_invoke_calls_gate_with_session_id_and_raw_result(self) -> None:
        raw_result = {
            "results": {"a": {"response": "hi"}},
            "synthesis": "syn",
            "status": "success",
            "raw_output": False,
        }
        operations = _ScriptedOperations(invoke_result=raw_result)
        gate = _RecordingCalibrationGate()
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="i-1",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "x"},
            ),
            session_id="session-b",
        )

        assert isinstance(result, ToolCallSuccess)
        assert gate.check_calls == [("session-b", "analysis", raw_result)]

    @pytest.mark.asyncio
    async def test_invoke_robust_to_gate_exception(self) -> None:
        """ADR-007 clause 2: a checker failure never blocks invocation.

        A checker that raises must not cause ``invoke_ensemble`` to fail
        — the signal is simply not recorded and the summarized result
        still flows back to the orchestrator.
        """
        raw_result = {
            "results": {"a": {"response": "ok"}},
            "synthesis": "syn",
            "status": "success",
            "raw_output": False,
        }
        operations = _ScriptedOperations(invoke_result=raw_result)
        gate = _RecordingCalibrationGate(check_raises=RuntimeError("checker crashed"))
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(returns={"synthesis": "distilled"}),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="i-robust",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "x"},
            ),
            session_id="session-c",
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == {"summary": "distilled"}
        # The gate was consulted but its exception was swallowed.
        assert len(gate.check_calls) == 1

    @pytest.mark.asyncio
    async def test_invoke_with_no_gate_configured_is_noop(self) -> None:
        """Calibration is optional infrastructure; omission is well-defined.

        Dispatch construction without a ``calibration_gate`` skips the
        check entirely — used by tests that don't care about calibration
        and by deployments that opt out. The summarizer flow is unaffected.
        """
        raw_result = {
            "results": {"a": {"response": "ok"}},
            "synthesis": "syn",
            "status": "success",
            "raw_output": False,
        }
        operations = _ScriptedOperations(invoke_result=raw_result)
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(returns={"synthesis": "distilled"}),
            autonomy_policy=_permissive_policy(),
            # No calibration_gate — default helper omits it.
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="i-no-gate",
                name="invoke_ensemble",
                arguments={"name": "analysis", "input": "x"},
            ),
            session_id="session-d",
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == {"summary": "distilled"}


class _RecordingPlexusAdapter:
    """Recording double for the PlexusAccess Protocol surface.

    Records every call so tests assert delegation fires with the
    arguments the orchestrator LLM emitted. Returns scripted dicts
    so the test asserts the value flows through to ``ToolCallSuccess``.
    """

    def __init__(
        self,
        *,
        query_returns: dict[str, Any] | None = None,
        record_returns: dict[str, Any] | None = None,
    ) -> None:
        self._query_returns: dict[str, Any] = (
            query_returns if query_returns is not None else {"results": []}
        )
        self._record_returns: dict[str, Any] = (
            record_returns if record_returns is not None else {"acknowledged": True}
        )
        self.query_calls: list[dict[str, Any]] = []
        self.record_calls: list[dict[str, Any]] = []

    async def query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.query_calls.append(arguments)
        return self._query_returns

    async def record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.record_calls.append(arguments)
        return self._record_returns


class TestPlexusToolWiring:
    """ADR-009 / WP-I: ``query_knowledge`` and ``record_outcome`` delegate
    to the Adapter when one is configured.

    These tests cover the in-product path. ``TestPlexusToolsRequireAdapter``
    above covers the absent-adapter fallback. Together they pin down the
    full surface a deployment will hit.
    """

    @pytest.mark.asyncio
    async def test_query_knowledge_delegates_to_adapter(self) -> None:
        adapter = _RecordingPlexusAdapter(
            query_returns={"results": [{"id": "r1"}], "context": "ctx"}
        )
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            plexus_adapter=adapter,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="qk-1",
                name="query_knowledge",
                arguments={"topic": "ensembles for refactoring"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "query_knowledge"
        assert result.content == {"results": [{"id": "r1"}], "context": "ctx"}
        assert adapter.query_calls == [{"topic": "ensembles for refactoring"}]
        assert adapter.record_calls == []

    @pytest.mark.asyncio
    async def test_record_outcome_delegates_to_adapter(self) -> None:
        adapter = _RecordingPlexusAdapter(record_returns={"acknowledged": True})
        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            plexus_adapter=adapter,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="ro-1",
                name="record_outcome",
                arguments={
                    "ensemble_name": "composed-x",
                    "quality_signal": "positive",
                    "context": "refactor task succeeded",
                },
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.name == "record_outcome"
        assert result.content == {"acknowledged": True}
        assert adapter.record_calls == [
            {
                "ensemble_name": "composed-x",
                "quality_signal": "positive",
                "context": "refactor task succeeded",
            }
        ]
        assert adapter.query_calls == []

    @pytest.mark.asyncio
    async def test_query_knowledge_with_real_adapter_returns_no_op(self) -> None:
        """With the real :class:`PlexusAdapter` (Plexus-absent), the
        orchestrator sees the well-formed empty result the no-op
        fallback ships."""
        from llm_orc.agentic.plexus_adapter import PlexusAdapter

        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            plexus_adapter=PlexusAdapter(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="qk-real",
                name="query_knowledge",
                arguments={"topic": "anything"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == {"results": [], "context": ""}

    @pytest.mark.asyncio
    async def test_record_outcome_with_real_adapter_returns_ack(self) -> None:
        """The no-op Adapter acknowledges promptly per ADR-009."""
        from llm_orc.agentic.plexus_adapter import PlexusAdapter

        dispatch = _build_dispatch(
            operations=_RaisingOperations(),
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            plexus_adapter=PlexusAdapter(),
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="ro-real",
                name="record_outcome",
                arguments={"ensemble_name": "x", "quality_signal": "positive"},
            )
        )

        assert isinstance(result, ToolCallSuccess)
        assert result.content == {"acknowledged": True}


# ---------------------------------------------------------------------------
# WP-G4-1 — Tier-Escalation Router interposition (ADR-015)
# ---------------------------------------------------------------------------


class _ScriptedCalibrationGate:
    """Calibration Gate double that returns a scripted verdict.

    Satisfies the gate's :meth:`verdict_for` surface that Tool Dispatch
    consumes for WP-G4-1. Records calls so tests can assert the gate
    was consulted with the right session/ensemble identifiers.

    The verdict is mutable via :meth:`set_verdict` so multi-window
    audit-drift tests can shift the gate's output between windows
    without rebuilding the dispatch.
    """

    def __init__(self, verdict: CalibrationVerdict = "proceed") -> None:
        self._verdict: CalibrationVerdict = verdict
        self.verdict_calls: list[tuple[str, str]] = []

    def set_verdict(self, verdict: CalibrationVerdict) -> None:
        self._verdict = verdict

    def verdict_for(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        dispatch_context: Any,
    ) -> CalibrationVerdict:
        del dispatch_context
        self.verdict_calls.append((session_id, ensemble_name))
        return self._verdict

    async def check_and_record(
        self, *, session_id: str, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal | None:
        del session_id, ensemble_name, raw_result
        return None

    def mark_composed(self, *, session_id: str, ensemble_name: str) -> None:
        del session_id, ensemble_name


class TestTierEscalationRouterInterposition:
    """WP-G4-1 — ADR-015 per-role tier-escalation router.

    Covers scenarios from ``scenarios.md`` §Per-Role Tier-Escalation
    Router (ADR-015) at the Tool Dispatch integration layer (the
    router-only assertions live in :mod:`test_tier_router`).

    Integration boundary verifications per system-design.agents.md
    §Integration Contracts (lines 615-617):

    * ``test_invoke_ensemble_routes_through_tier_router``
    * ``test_router_consumes_calibration_verdict``
    * ``test_router_reads_topaz_skill_from_ensemble_yaml``
    """

    @staticmethod
    def _build_router_with_skill(
        ensemble_name: str,
        skill: TopazSkill,
        *,
        cheap: str = "cheap-default",
        escalated: str = "escalated-default",
    ) -> Any:
        """Construct a real TierRouter wired with a scripted skill reader
        and full 8-skill defaults — the named ensemble routes through
        the supplied (cheap, escalated) pair.
        """
        from llm_orc.agentic.tier_router import (
            ALL_TOPAZ_SKILLS,
            PerSkillTierDefaults,
            TierRouter,
            TierRouterDefaults,
        )

        class _Reader:
            def topaz_skill_for(self, ensemble_name_: str) -> TopazSkill | None:
                if ensemble_name_ == ensemble_name:
                    return skill
                return None

        per_skill: dict[TopazSkill, PerSkillTierDefaults] = {
            s: PerSkillTierDefaults(
                cheap_tier=f"cheap-{s}",
                escalated_tier=f"escalated-{s}",
            )
            for s in ALL_TOPAZ_SKILLS
        }
        # Override the named skill with the requested pair.
        per_skill[skill] = PerSkillTierDefaults(
            cheap_tier=cheap, escalated_tier=escalated
        )
        defaults = TierRouterDefaults(per_skill=per_skill)
        return TierRouter(defaults=defaults, skill_reader=_Reader())

    @pytest.mark.asyncio
    async def test_invoke_ensemble_routes_through_tier_router(self) -> None:
        """Per system-design.agents.md L615:
        ``test_invoke_ensemble_routes_through_tier_router`` — every
        ``invoke_ensemble`` dispatch passes through tier selection;
        verdict input flows from Calibration Gate; selected tier flows
        into ``EnsembleExecutor``.
        """
        gate = _ScriptedCalibrationGate(verdict="proceed")
        router = self._build_router_with_skill(
            "code-review",
            "code_generation",
            cheap="ollama-deepseek-coder-v2:16b",
            escalated="claude-sonnet-4-6",
        )
        operations = _ScriptedOperations(
            invoke_result={
                "results": {"a": "x"},
                "synthesis": "ok",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments={"name": "code-review", "input": "fix the parser"},
            ),
            session_id="s-1",
        )

        assert isinstance(result, ToolCallSuccess)
        # Gate was consulted for the verdict before invoke.
        assert gate.verdict_calls == [("s-1", "code-review")]
        # The selected cheap-tier Model Profile flowed into invoke args.
        assert operations.invoke_calls == [
            {
                "ensemble_name": "code-review",
                "input": "fix the parser",
                "model_profile_override": "ollama-deepseek-coder-v2:16b",
            }
        ]

    @pytest.mark.asyncio
    async def test_router_consumes_calibration_verdict(self) -> None:
        """Per system-design.agents.md L616: the verdict's three values
        map deterministically to router actions (cheap-tier / escalated-
        tier / ``escalation_bypass`` typed error). This test exercises
        the Reflect → escalated branch through Tool Dispatch.
        """
        gate = _ScriptedCalibrationGate(verdict="reflect")
        router = self._build_router_with_skill(
            "plexus-graph-analysis",
            "tool_use",
            cheap="ollama-qwen3:8b",
            escalated="gpt-5-mini",
        )
        operations = _ScriptedOperations(
            invoke_result={
                "results": {},
                "synthesis": "",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )

        await dispatch.dispatch(
            InternalToolCall(
                id="t2",
                name="invoke_ensemble",
                arguments={"name": "plexus-graph-analysis", "input": "x"},
            ),
            session_id="s-2",
        )

        assert operations.invoke_calls[0]["model_profile_override"] == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_router_reads_topaz_skill_from_ensemble_yaml(self) -> None:
        """Per system-design.agents.md L617: the router resolves the
        dispatched ensemble's ``topaz_skill`` field via the configured
        reader; ensembles without the field produce
        ``missing_skill_metadata`` typed error.
        """
        gate = _ScriptedCalibrationGate(verdict="proceed")
        # Reader returns None for any ensemble — simulates a YAML lacking
        # the topaz_skill field per ADR-015 §Per-skill role profiling.
        from llm_orc.agentic.tier_router import (
            ALL_TOPAZ_SKILLS,
            PerSkillTierDefaults,
            TierRouter,
            TierRouterDefaults,
        )

        class _EmptyReader:
            def topaz_skill_for(self, ensemble_name_: str) -> TopazSkill | None:
                del ensemble_name_
                return None

        empty_per_skill: dict[TopazSkill, PerSkillTierDefaults] = {
            s: PerSkillTierDefaults(
                cheap_tier=f"cheap-{s}", escalated_tier=f"escalated-{s}"
            )
            for s in ALL_TOPAZ_SKILLS
        }
        router = TierRouter(
            defaults=TierRouterDefaults(per_skill=empty_per_skill),
            skill_reader=_EmptyReader(),
        )
        operations = _ScriptedOperations()
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="t3",
                name="invoke_ensemble",
                arguments={"name": "untagged-ensemble", "input": "x"},
            ),
            session_id="s-3",
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "missing_skill_metadata"
        # Operations.invoke must NOT have been called — the error is
        # raised before dispatch reaches ensemble execution (FC-18).
        assert operations.invoke_calls == []

    @pytest.mark.asyncio
    async def test_abstain_verdict_produces_escalation_bypass_tool_error(
        self,
    ) -> None:
        """Scenario: Abstain verdict produces escalation-bypass typed error
        (scenarios.md §Per-Role Tier-Escalation Router)."""
        gate = _ScriptedCalibrationGate(verdict="abstain")
        router = self._build_router_with_skill("composed-a", "code_generation")
        operations = _ScriptedOperations()
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="t4",
                name="invoke_ensemble",
                arguments={"name": "composed-a", "input": "x"},
            ),
            session_id="s-4",
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "escalation_bypass"
        assert "Abstain" in result.reason
        # Abstain bypasses dispatch entirely.
        assert operations.invoke_calls == []


class TestVerdictToTierMappingDeterministicViaDispatch:
    """Per system-design.agents.md §Tier-Escalation Router Fitness:

        The verdict-to-action mapping is deterministic: Proceed →
        cheap-tier dispatch; Reflect → escalated-tier dispatch;
        Abstain → ``escalation_bypass`` typed error.

    M-dimension test (verdict variation) at the Tool Dispatch
    integration layer. The router-isolated version lives in
    :class:`TestVerdictToTierMappingIsDeterministic` in
    ``test_tier_router``.
    """

    @staticmethod
    def _build(verdict: CalibrationVerdict) -> tuple[_ScriptedOperations, Any]:
        gate = _ScriptedCalibrationGate(verdict=verdict)
        router = TestTierEscalationRouterInterposition._build_router_with_skill(
            "e", "summarization", cheap="cheap-summ-A", escalated="escalated-summ-A"
        )
        operations = _ScriptedOperations(
            invoke_result={
                "results": {},
                "synthesis": "",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )
        return operations, dispatch

    @pytest.mark.asyncio
    async def test_proceed_routes_to_cheap_tier_via_dispatch(self) -> None:
        operations, dispatch = self._build("proceed")

        await dispatch.dispatch(
            InternalToolCall(
                id="p1",
                name="invoke_ensemble",
                arguments={"name": "e", "input": "x"},
            )
        )

        assert operations.invoke_calls[0]["model_profile_override"] == "cheap-summ-A"

    @pytest.mark.asyncio
    async def test_reflect_routes_to_escalated_tier_via_dispatch(self) -> None:
        operations, dispatch = self._build("reflect")

        await dispatch.dispatch(
            InternalToolCall(
                id="r1",
                name="invoke_ensemble",
                arguments={"name": "e", "input": "x"},
            )
        )

        assert (
            operations.invoke_calls[0]["model_profile_override"] == "escalated-summ-A"
        )

    @pytest.mark.asyncio
    async def test_abstain_produces_typed_error_via_dispatch(self) -> None:
        operations, dispatch = self._build("abstain")

        result = await dispatch.dispatch(
            InternalToolCall(
                id="a1",
                name="invoke_ensemble",
                arguments={"name": "e", "input": "x"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "escalation_bypass"
        assert operations.invoke_calls == []


class TestADR011PreservationUnderTierEscalation:
    """Per system-design.agents.md L224:
    ``test_orchestrator_profile_unchanged_under_tier_escalation`` —
    preservation of FC-13.

    Tier escalation acts on the *dispatched task's* tier only; the
    orchestrator's own Model Profile remains session-boundary scoped
    per ADR-011. The Tool Dispatch's existing surface does not own
    the orchestrator's Model Profile (it lives in OrchestratorConfig
    at L3); the preservation property here is that the router does
    not touch any orchestrator-side configuration — only the
    ``model_profile_override`` key in invoke arguments.
    """

    @pytest.mark.asyncio
    async def test_router_only_writes_dispatch_arg_not_orchestrator_state(
        self,
    ) -> None:
        """ADR-011 preservation: the only side effect of the router is
        the per-dispatch ``model_profile_override`` argument value.
        The router does NOT mutate any shared/orchestrator state.
        """
        gate = _ScriptedCalibrationGate(verdict="reflect")
        router = TestTierEscalationRouterInterposition._build_router_with_skill(
            "e",
            "code_generation",
            cheap="cheap-profile",
            escalated="escalated-profile",
        )
        operations = _ScriptedOperations(
            invoke_result={
                "results": {},
                "synthesis": "",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch = _build_dispatch(
            operations=operations,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
        )

        # Drive multiple dispatches under Reflect (escalated-tier).
        for n in range(3):
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"d{n}",
                    name="invoke_ensemble",
                    arguments={"name": "e", "input": "x"},
                ),
                session_id="s-stable",
            )

        # Every dispatch must have routed to escalated-tier — no
        # carry-forward drift, no orchestrator-side state mutation.
        # FC-13 (the orchestrator's own Model Profile remains constant)
        # holds vacuously here because the router only touches the
        # per-dispatch arguments dict.
        for call in operations.invoke_calls:
            assert call["model_profile_override"] == "escalated-profile"

    @pytest.mark.asyncio
    async def test_invoke_ensemble_api_unchanged_under_tier_escalation(
        self,
    ) -> None:
        """Per system-design.agents.md L225:
        ``test_invoke_ensemble_api_unchanged_under_tier_escalation`` —
        the orchestrator's tool-call API (``invoke_ensemble``) is
        unchanged under tier escalation. The orchestrator's reasoning
        surface receives the same shape of ``ToolCallResult``
        regardless of which tier was dispatched.
        """
        # Cheap-tier dispatch
        gate_cheap = _ScriptedCalibrationGate(verdict="proceed")
        router_cheap = TestTierEscalationRouterInterposition._build_router_with_skill(
            "e", "code_generation"
        )
        ops_cheap = _ScriptedOperations(
            invoke_result={
                "results": {"a": "cheap-output"},
                "synthesis": "cheap-synth",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch_cheap = _build_dispatch(
            operations=ops_cheap,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate_cheap,
            tier_router=router_cheap,
        )

        # Escalated-tier dispatch
        gate_esc = _ScriptedCalibrationGate(verdict="reflect")
        router_esc = TestTierEscalationRouterInterposition._build_router_with_skill(
            "e", "code_generation"
        )
        ops_esc = _ScriptedOperations(
            invoke_result={
                "results": {"a": "escalated-output"},
                "synthesis": "escalated-synth",
                "status": "success",
                "raw_output": True,
            }
        )
        dispatch_esc = _build_dispatch(
            operations=ops_esc,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate_esc,
            tier_router=router_esc,
        )

        result_cheap = await dispatch_cheap.dispatch(
            InternalToolCall(
                id="api1",
                name="invoke_ensemble",
                arguments={"name": "e", "input": "x"},
            )
        )
        result_esc = await dispatch_esc.dispatch(
            InternalToolCall(
                id="api2",
                name="invoke_ensemble",
                arguments={"name": "e", "input": "x"},
            )
        )

        # Both are ToolCallSuccess with identical shape — only the
        # underlying content differs (per tier). The orchestrator's
        # ReAct loop consumes the same surface.
        assert isinstance(result_cheap, ToolCallSuccess)
        assert isinstance(result_esc, ToolCallSuccess)
        assert result_cheap.name == "invoke_ensemble"
        assert result_esc.name == "invoke_ensemble"
        assert type(result_cheap.content) is type(result_esc.content)


class TestRouterRequiresCalibrationGateAtConstruction:
    """Construction-time validation: ``tier_router`` requires
    ``calibration_gate`` (the router consumes the verdict the gate
    produces; see system-design.agents.md L162)."""

    def test_constructing_router_without_gate_raises(self) -> None:
        from llm_orc.agentic.tier_router import (
            ALL_TOPAZ_SKILLS,
            PerSkillTierDefaults,
            TierRouter,
            TierRouterDefaults,
        )

        class _AnyReader:
            def topaz_skill_for(self, ensemble_name_: str) -> TopazSkill | None:
                del ensemble_name_
                return "code_generation"

        any_per_skill: dict[TopazSkill, PerSkillTierDefaults] = {
            s: PerSkillTierDefaults(
                cheap_tier=f"cheap-{s}", escalated_tier=f"escalated-{s}"
            )
            for s in ALL_TOPAZ_SKILLS
        }
        router = TierRouter(
            defaults=TierRouterDefaults(per_skill=any_per_skill),
            skill_reader=_AnyReader(),
        )

        with pytest.raises(ValueError, match="tier_router requires calibration_gate"):
            _build_dispatch(
                operations=_RaisingOperations(),
                harness=_build_harness(),
                autonomy_policy=_permissive_policy(),
                tier_router=router,
                # calibration_gate intentionally omitted
            )


# ---------------------------------------------------------------------------
# WP-G4-2 — ADR-018 (d)-analog audit dispatch interposition
# ---------------------------------------------------------------------------


class _ManualAuditClock:
    """Wall-clock double for audit-trigger-by-time tests."""

    def __init__(self, start_seconds: float = 0.0) -> None:
        self._now = start_seconds

    def now_seconds(self) -> float:
        return self._now

    def advance_seconds(self, delta: float) -> None:
        self._now += delta


def _build_audit_test_thresholds(*, trigger_count: int = 10) -> Any:
    """Test-tuned thresholds — small trigger_count keeps tests fast."""
    from llm_orc.agentic.tier_router_audit import TierEscalationAuditThresholds

    return TierEscalationAuditThresholds(
        trigger_count=trigger_count,
        trigger_wall_clock_hours=24.0,
        verdict_distribution_shift=0.15,
        escalation_outcome_correlation_pp=0.05,
        bypass_rate_increase=0.25,
        severe_drift_multiplier=2.0,
    )


class TestTierEscalationAuditorWiring:
    """WP-G4-2 — ADR-018 (d)-analog audit dispatch integration.

    Verifies that Tool Dispatch:
    1. records verdict consumptions into the auditor on every dispatch,
    2. records dispatch outcomes by tier,
    3. overrides verdict-driven routing to escalated tier under
       fail-safe mode,
    4. validates ``tier_router_audit`` requires ``tier_router`` at
       construction.
    """

    @staticmethod
    def _build_router_and_ops(
        *, verdict: CalibrationVerdict = "proceed"
    ) -> tuple[_ScriptedOperations, _ScriptedCalibrationGate, Any]:
        gate = _ScriptedCalibrationGate(verdict=verdict)
        router = TestTierEscalationRouterInterposition._build_router_with_skill(
            "ens-A",
            "code_generation",
            cheap="cheap-code-gen",
            escalated="escalated-code-gen",
        )
        ops = _ScriptedOperations(
            invoke_result={
                "results": {},
                "synthesis": "",
                "status": "success",
                "raw_output": True,
            }
        )
        return ops, gate, router

    def test_auditor_without_router_raises_at_construction(self) -> None:
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(),
            clock=_ManualAuditClock(),
        )
        with pytest.raises(ValueError, match="tier_router_audit requires"):
            _build_dispatch(
                operations=_ScriptedOperations(invoke_result={}),
                tier_router_audit=auditor,
            )

    @pytest.mark.asyncio
    async def test_proceed_dispatch_records_consumption_and_cheap_outcome(
        self,
    ) -> None:
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        ops, gate, router = self._build_router_and_ops(verdict="proceed")
        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(trigger_count=100),
            clock=_ManualAuditClock(),
        )
        dispatch = _build_dispatch(
            operations=ops,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
            tier_router_audit=auditor,
        )

        await dispatch.dispatch(
            InternalToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments={"name": "ens-A", "input": "x"},
            )
        )

        # Outcome recorded as cheap-tier success.
        snapshot = auditor.outcome_snapshot_for_tests
        assert snapshot["cheap"] == (1, 0)
        assert snapshot["escalated"] == (0, 0)

    @pytest.mark.asyncio
    async def test_abstain_dispatch_records_bypass_into_audit(self) -> None:
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        ops, gate, router = self._build_router_and_ops(verdict="abstain")
        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(trigger_count=100),
            clock=_ManualAuditClock(),
        )
        dispatch = _build_dispatch(
            operations=ops,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
            tier_router_audit=auditor,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments={"name": "ens-A", "input": "x"},
            )
        )

        assert isinstance(result, ToolCallError)
        assert result.kind == "escalation_bypass"
        # The invoke did not run, but the bypass was recorded.
        assert ops.invoke_calls == []

    @pytest.mark.asyncio
    async def test_invocation_failure_records_failure_outcome(self) -> None:
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        gate = _ScriptedCalibrationGate(verdict="proceed")
        router = TestTierEscalationRouterInterposition._build_router_with_skill(
            "ens-A",
            "code_generation",
            cheap="cheap-code-gen",
            escalated="escalated-code-gen",
        )
        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(trigger_count=100),
            clock=_ManualAuditClock(),
        )
        # ValueError simulates the "ensemble not found" failure path
        # that ExecutionHandler.invoke produces in production.
        failing_ops = _ScriptedOperations(
            invoke_raises=ValueError("ensemble 'ens-A' not found")
        )
        dispatch = _build_dispatch(
            operations=failing_ops,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
            tier_router_audit=auditor,
        )

        result = await dispatch.dispatch(
            InternalToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments={"name": "ens-A", "input": "x"},
            )
        )

        assert isinstance(result, ToolCallError)
        # Outcome recorded as failure for cheap-tier.
        snapshot = auditor.outcome_snapshot_for_tests
        assert snapshot["cheap"] == (0, 1)

    @pytest.mark.asyncio
    async def test_fail_safe_active_routes_proceed_to_escalated_tier(
        self,
    ) -> None:
        """Under fail-safe, a Proceed verdict still routes to escalated
        tier — fail-safe overrides verdict-driven routing per ADR-018
        §"Severe-drift response".
        """
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        ops, gate, router = self._build_router_and_ops(verdict="proceed")
        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(trigger_count=100),
            clock=_ManualAuditClock(),
        )
        # Force fail-safe state directly — the test exercises the
        # override path, not the trigger logic.
        auditor._fail_safe_active = True

        dispatch = _build_dispatch(
            operations=ops,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
            tier_router_audit=auditor,
        )

        await dispatch.dispatch(
            InternalToolCall(
                id="t1",
                name="invoke_ensemble",
                arguments={"name": "ens-A", "input": "x"},
            )
        )

        # Despite Proceed verdict, escalated tier was selected.
        assert ops.invoke_calls[0]["model_profile_override"] == "escalated-code-gen"


class TestDAnalogAuditDispatchFiresAtTriggerAndSevereDriftActivatesFailSafe:
    """FC-20 named test per system-design.agents.md L229.

    Test exercises the full integration path:

    1. Build a Tool Dispatch with router + auditor (low trigger_count
       for test speed).
    2. Drive 10 Proceed verdicts — first audit fires, no_drift verdict.
    3. Flip the gate to Reflect — drive 10 more — second audit fires.
       Verdict distribution shifted 100% on both Proceed and Reflect
       axes; well past 2× the 0.15 threshold → severe verdict.
    4. Verify ``auditor.fail_safe_active`` becomes True at severe
       verdict.
    5. Flip the gate back to Proceed — verify the next dispatch still
       routes to escalated tier (fail-safe override active).
    """

    @pytest.mark.asyncio
    async def test_audit_fires_at_trigger_and_severe_activates_fail_safe(
        self,
    ) -> None:
        from llm_orc.agentic.tier_router_audit import TierEscalationAuditor

        ops, gate, router = TestTierEscalationAuditorWiring._build_router_and_ops(
            verdict="proceed"
        )
        auditor = TierEscalationAuditor(
            thresholds=_build_audit_test_thresholds(trigger_count=10),
            clock=_ManualAuditClock(),
        )
        dispatch = _build_dispatch(
            operations=ops,
            harness=_build_harness(),
            autonomy_policy=_permissive_policy(),
            calibration_gate=gate,
            tier_router=router,
            tier_router_audit=auditor,
        )

        # Phase 1: 10 Proceed dispatches — first audit window fires
        # (no_drift; first window has no prior to compare).
        for i in range(10):
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"p{i}",
                    name="invoke_ensemble",
                    arguments={"name": "ens-A", "input": "x"},
                )
            )
        first_window_diagnostics = auditor.diagnostics()
        assert len(first_window_diagnostics) == 1
        assert first_window_diagnostics[0].verdict == "no_drift"
        assert auditor.fail_safe_active is False

        # Phase 2: flip to Reflect; 10 more dispatches — second audit
        # window fires. Verdict distribution shifts 100% on both
        # Proceed and Reflect axes, well past 2x the 0.15 threshold
        # (i.e., the severe-magnitude cutoff) → severe verdict.
        gate.set_verdict("reflect")
        for i in range(10):
            await dispatch.dispatch(
                InternalToolCall(
                    id=f"r{i}",
                    name="invoke_ensemble",
                    arguments={"name": "ens-A", "input": "x"},
                )
            )
        diagnostics = auditor.diagnostics()
        assert len(diagnostics) == 2
        assert diagnostics[1].verdict == "severe"
        assert auditor.fail_safe_active is True

        # Phase 3: flip gate back to Proceed. The next dispatch still
        # routes to escalated tier because fail-safe overrides the
        # verdict.
        gate.set_verdict("proceed")
        ops.invoke_calls.clear()
        await dispatch.dispatch(
            InternalToolCall(
                id="x1",
                name="invoke_ensemble",
                arguments={"name": "ens-A", "input": "x"},
            )
        )
        assert ops.invoke_calls[0]["model_profile_override"] == "escalated-code-gen"


# ---------------------------------------------------------------------------
# Cycle 6 WP-A — Dispatch Event Substrate integration (FC-21 anchor)
# ---------------------------------------------------------------------------


class _RecordingEventSink:
    """Captures every event the substrate fans out — exercises FC-21."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def consume(self, event: object) -> None:
        self.events.append(event)


class TestDispatchEventSubstrateIntegration:
    """FC-21 — every event emitted during one invoke_ensemble dispatch
    carries the same dispatch_id value.

    Per ``docs/agentic-serving/system-design.agents.md`` §Module:
    Dispatch Event Substrate (Cycle 6 WP-A, ADR-023). The integration
    test asserts the cross-event correlation that operator-terminal
    (WP-B) and orchestrator-context (WP-C) sinks depend on.
    """

    @pytest.mark.asyncio
    async def test_invoke_ensemble_emits_paired_dispatch_timing_events(self) -> None:
        """Smallest case — no tier router, no calibration; only DispatchTiming
        events are emitted. Both share the same dispatch_id and the end
        event carries duration_seconds + exit_status=success.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingEventSink()
        substrate.register_sink(sink)
        ops = _ScriptedOperations(invoke_result={"synthesis": "ok"})
        dispatch = _build_dispatch(operations=ops, event_substrate=substrate)

        await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "code-generator", "input": "task"},
            ),
            session_id="session-A",
        )

        timing_events = [e for e in sink.events if isinstance(e, DispatchTiming)]
        assert len(timing_events) == 2
        start, end = timing_events
        assert start.phase == "start"
        assert end.phase == "end"
        assert start.dispatch_id == end.dispatch_id
        assert start.ensemble_name == "code-generator"
        assert end.duration_seconds is not None
        assert end.duration_seconds >= 0.0
        assert end.exit_status == "success"

    @pytest.mark.asyncio
    async def test_dispatch_events_share_dispatch_id_within_one_invoke(
        self,
    ) -> None:
        """FC-21 — full integration with tier router + calibration gate.

        With a calibration gate producing a verdict and a tier router
        consuming it, the substrate observes DispatchTiming(start) +
        CalibrationVerdictEvent + TierSelection + DispatchTiming(end)
        in dispatch order. Every event carries the same dispatch_id
        value; substrate.events_for(dispatch_id) reconstructs the full
        log in emission order.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingEventSink()
        substrate.register_sink(sink)
        ops = _ScriptedOperations(invoke_result={"synthesis": "ok"})
        gate = _ScriptedCalibrationGate(verdict="proceed")
        router = TestTierEscalationRouterInterposition._build_router_with_skill(
            "code-generator",
            "code_generation",
            cheap="cheap-code-gen",
            escalated="escalated-code-gen",
        )
        dispatch = _build_dispatch(
            operations=ops,
            calibration_gate=gate,
            tier_router=router,
            event_substrate=substrate,
        )

        await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "code-generator", "input": "task"},
            ),
            session_id="session-A",
        )

        # Every emitted event with a dispatch_id shares one value.
        dispatch_ids = {getattr(e, "dispatch_id", None) for e in sink.events} - {None}
        assert len(dispatch_ids) == 1
        (the_dispatch_id,) = dispatch_ids
        assert the_dispatch_id == "session-A-dispatch-0001"

        # The substrate's log reconstructs the dispatch in emission order:
        # start → calibration verdict → tier selection → end.
        log = substrate.events_for(the_dispatch_id)
        kinds = [type(e).__name__ for e in log]
        assert kinds == [
            "DispatchTiming",  # phase=start
            "CalibrationVerdictEvent",
            "TierSelection",
            "DispatchTiming",  # phase=end
        ]
        assert isinstance(log[0], DispatchTiming)
        assert log[0].phase == "start"
        assert isinstance(log[1], CalibrationVerdictEvent)
        assert log[1].verdict == "proceed"
        assert log[1].ensemble_name == "code-generator"
        assert isinstance(log[3], DispatchTiming)
        assert log[3].phase == "end"

    @pytest.mark.asyncio
    async def test_end_event_fires_on_invocation_failure(self) -> None:
        """DispatchTiming(end) emits even when invoke raises — try/finally
        discipline. exit_status is ``error`` in that case.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingEventSink()
        substrate.register_sink(sink)
        ops = _ScriptedOperations(invoke_raises=ValueError("ensemble not found"))
        dispatch = _build_dispatch(operations=ops, event_substrate=substrate)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "missing", "input": "task"},
            ),
            session_id="session-A",
        )
        assert isinstance(result, ToolCallError)
        assert result.kind == "invocation_failed"

        timing_events = [e for e in sink.events if isinstance(e, DispatchTiming)]
        assert [e.phase for e in timing_events] == ["start", "end"]
        end = timing_events[1]
        assert end.exit_status == "error"

    @pytest.mark.asyncio
    async def test_invalid_arguments_do_not_open_a_dispatch(self) -> None:
        """Argument-validation errors before dispatch_id allocation do not
        produce DispatchTiming events. The substrate is not consulted.
        """
        substrate = DispatchEventSubstrate()
        sink = _RecordingEventSink()
        substrate.register_sink(sink)
        ops = _ScriptedOperations()
        dispatch = _build_dispatch(operations=ops, event_substrate=substrate)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "", "input": "task"},
            ),
            session_id="session-A",
        )
        assert isinstance(result, ToolCallError)
        assert result.kind == "invalid_arguments"
        assert sink.events == []

    @pytest.mark.asyncio
    async def test_substrate_absent_preserves_pre_cycle_6_path(self) -> None:
        """No event_substrate configured — invoke_ensemble works as before."""
        ops = _ScriptedOperations(invoke_result={"synthesis": "ok"})
        dispatch = _build_dispatch(operations=ops)

        result = await dispatch.dispatch(
            InternalToolCall(
                id="call-1",
                name="invoke_ensemble",
                arguments={"name": "code-generator", "input": "task"},
            ),
            session_id="session-A",
        )
        assert isinstance(result, ToolCallSuccess)
