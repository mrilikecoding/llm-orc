"""Serving Layer ``/v1/chat/completions`` endpoint.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3),
§Integration Contracts, and roadmap WP-C. The endpoint:

1. Parses the OpenAI-compatible request.
2. Resolves a Session via :class:`SessionRegistry`.
3. Runs :func:`resolve_session_start_context` exactly once per session
   (FC-9); Phase 1 returns an empty list.
4. Constructs an :class:`OrchestratorRuntime` from the resolved
   Model Profile's LLM, a :class:`BudgetController` sized from the
   session's configured Budget, and a shared Tool Dispatch wrapping
   :class:`OrchestraService`.
5. Drives the Runtime to produce either a non-streaming
   ``chat.completion`` body or an SSE stream.

The Runtime is constructed per request. The Tool Dispatch facade
and the LLM loader are process-scoped factories module-level tests
override via ``monkeypatch.setattr``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_orc.agentic.autonomy_policy import AutonomyPolicy
from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.calibration_gate import (
    CalibrationGate,
    EnsembleBackedChecker,
)
from llm_orc.agentic.composition_validator import (
    CompositionValidator,
    ConfigManagerEnsembleWriter,
    ConfigManagerPrimitiveRegistry,
)
from llm_orc.agentic.conversation_compaction import ConversationCompaction
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.inference_wait_heartbeat import (
    InferenceWaitHeartbeatScheduler,
)
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    VisibilityEvent,
)
from llm_orc.agentic.orchestrator_context_event_sink import (
    OrchestratorContextEventSink,
)
from llm_orc.agentic.orchestrator_runtime import (
    OrchestratorLLM,
    OrchestratorRuntime,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    OrchestratorToolDispatch,
)
from llm_orc.agentic.plexus_adapter import PlexusAdapter
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.agentic.session_registry import SessionRegistry
from llm_orc.agentic.session_start import ChatMessage, SessionContext, SessionStartCache
from llm_orc.agentic.tier_router import (
    EnsembleConfigTopazSkillReader,
    TierRouter,
    TierRouterDefaults,
)
from llm_orc.agentic.tier_router_audit import TierEscalationAuditor
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.models.base import ToolCallingNotSupportedError
from llm_orc.web.api import get_orchestra_service
from llm_orc.web.api.sse_format import (
    OpenAiSseFormatter,
    encode_tool_call_for_message,
    render_visibility_narration,
)
from llm_orc.web.api.v1_models import get_orchestrator_config_resolver

router = APIRouter(prefix="/v1", tags=["openai-compat"])


_SHARED_REGISTRY = SessionRegistry()
_SHARED_SESSION_START_CACHE = SessionStartCache()
_SHARED_TOOL_DISPATCH: OrchestratorToolDispatch | None = None
_SHARED_CONVERSATION_COMPACTION: ConversationCompaction | None = None
_SHARED_DISPATCH_EVENT_SUBSTRATE: DispatchEventSubstrate | None = None
_SHARED_OPERATOR_TERMINAL_SINK: OperatorTerminalEventSink | None = None


def get_session_registry() -> SessionRegistry:
    """Return the process-scoped Session Registry.

    Tests override this factory to inject an isolated registry.
    """
    return _SHARED_REGISTRY


def get_session_start_cache() -> SessionStartCache:
    """Return the process-scoped session-start cache."""
    return _SHARED_SESSION_START_CACHE


def get_dispatch_event_substrate() -> DispatchEventSubstrate:
    """Return the process-scoped Dispatch Event Substrate (Cycle 6 WP-A).

    Tool Dispatch emits ``DispatchTiming``, ``TierSelection``,
    ``CalibrationVerdictEvent``, and ``CalibrationSignal`` events
    through this substrate. WP-B's operator-terminal sink is registered
    via :func:`get_operator_terminal_event_sink`; WP-C's orchestrator-
    context sink and per-request inference-wait heartbeat schedulers
    (Cycle 6 WP-B piece 5) register on top of the same shared instance.
    """
    global _SHARED_DISPATCH_EVENT_SUBSTRATE
    if _SHARED_DISPATCH_EVENT_SUBSTRATE is None:
        _SHARED_DISPATCH_EVENT_SUBSTRATE = DispatchEventSubstrate()
    return _SHARED_DISPATCH_EVENT_SUBSTRATE


def get_operator_terminal_event_sink() -> OperatorTerminalEventSink:
    """Return the process-scoped Operator-Terminal Event Sink (Cycle 6 WP-B).

    On first construction the sink registers with the shared dispatch
    event substrate, primes the shared :class:`EnsembleLoader` for each
    operator-configured ensemble directory, and emits one ``WARN`` line
    per validation failure via
    :meth:`OperatorTerminalEventSink.report_validation_results` —
    closing the validate-once-at-load scenario per ADR-023 §"Noise-floor
    remediation". Subsequent ``list_ensembles`` calls on the shared
    loader return the cached validated subset without re-emitting
    warnings.
    """
    global _SHARED_OPERATOR_TERMINAL_SINK
    if _SHARED_OPERATOR_TERMINAL_SINK is None:
        sink = OperatorTerminalEventSink()
        sink.register_with(get_dispatch_event_substrate())
        service = get_orchestra_service()
        for dir_path in service.config_manager.get_ensembles_dirs():
            service.ensemble_loader.prime(str(dir_path))
        sink.report_validation_results(service.ensemble_loader.validation_results())
        _SHARED_OPERATOR_TERMINAL_SINK = sink
    return _SHARED_OPERATOR_TERMINAL_SINK


def get_orchestrator_tool_dispatch() -> OrchestratorToolDispatch:
    """Return the process-scoped Tool Dispatch.

    Wraps :class:`OrchestraService` so the orchestrator's tool surface
    is a thin adapter on the existing ensemble-operations facade. The
    Result Summarizer Harness is constructed with the configured
    summarizer ensemble name and the same ``OrchestraService`` (which
    satisfies the ``SummarizerInvoker`` Protocol structurally). The
    AutonomyPolicy reads the operator-configured default Autonomy Level
    on every decision, so a ``config.yaml`` edit takes effect on the
    next request without a server restart. Tests override this factory
    to inject a stub dispatcher.
    """
    global _SHARED_TOOL_DISPATCH
    if _SHARED_TOOL_DISPATCH is None:
        service = get_orchestra_service()
        resolver = get_orchestrator_config_resolver()
        config = resolver.resolve()
        harness = ResultSummarizerHarness(
            invoker=service, summarizer_name=config.summarizer_ensemble
        )
        autonomy_policy = AutonomyPolicy(
            level_provider=lambda: resolver.resolve().autonomy_level
        )
        primitive_registry = ConfigManagerPrimitiveRegistry(service.config_manager)
        composition_validator = CompositionValidator(primitives=primitive_registry)
        local_ensemble_writer = ConfigManagerEnsembleWriter(service.config_manager)
        calibration_checker = EnsembleBackedChecker(
            invoker=service,
            checker_ensemble_name=config.calibration.checker_ensemble,
        )
        calibration_gate = CalibrationGate(
            default_n=config.calibration.default_n,
            checker=calibration_checker,
        )
        plexus_adapter = PlexusAdapter()
        tier_router, tier_router_audit = _build_tier_router_and_audit(
            service=service, config=config
        )
        substrate = get_dispatch_event_substrate()
        sink = get_operator_terminal_event_sink()
        _SHARED_TOOL_DISPATCH = OrchestratorToolDispatch(
            operations=service,
            harness=harness,
            autonomy_policy=autonomy_policy,
            composition_validator=composition_validator,
            local_ensemble_writer=local_ensemble_writer,
            calibration_gate=calibration_gate,
            plexus_adapter=plexus_adapter,
            tool_call_validation_patterns=config.tool_call_validation_patterns,
            tier_router=tier_router,
            tier_router_audit=tier_router_audit,
            event_substrate=substrate,
            # WP-B feed-forward advisory 2: in production wiring this
            # logger slot is the bare operator-terminal sink (not the
            # heartbeat scheduler). The scheduler observes dispatch
            # activity via the substrate's DispatchTiming fan-out
            # filtered by session_id prefix — NOT via this emit-logger
            # slot. The scheduler's ``emit_tool_call_log`` method
            # exists for testability and as a future composition
            # surface; it is dead in this wiring path.
            tool_call_emit_logger=sink,
        )
    return _SHARED_TOOL_DISPATCH


def _build_tier_router_and_audit(
    *,
    service: Any,
    config: Any,
) -> tuple[TierRouter | None, TierEscalationAuditor | None]:
    """Construct the Tier-Escalation Router and audit per operator config.

    Returns ``(None, None)`` when the operator has not configured
    ``per_skill_tier_defaults`` — Tool Dispatch then runs without tier
    escalation (pre-WP-G4-1 behavior). When configured, builds:

    * an :class:`EnsembleConfigTopazSkillReader` bridged to the
      service's ensemble loader (the L0 metadata source);
    * a :class:`TierRouter` with the operator-configured 8-skill
      defaults (WP-G4-1, ADR-015);
    * a :class:`TierEscalationAuditor` with operator-tunable
      thresholds (WP-G4-2, ADR-018).

    Per-skill tier defaults flow from L3 → L2 at construction time
    (no L3 import inside the router or auditor). The audit's
    SystemAuditClock default is used; tests inject a controllable
    clock via the factory override.
    """
    if config.per_skill_tier_defaults is None:
        return None, None
    skill_reader = EnsembleConfigTopazSkillReader(
        find_ensemble=service.find_ensemble_by_name
    )
    router = TierRouter(
        defaults=TierRouterDefaults(per_skill=dict(config.per_skill_tier_defaults)),
        skill_reader=skill_reader,
    )
    auditor = TierEscalationAuditor(thresholds=config.tier_router_audit)
    return router, auditor


def get_conversation_compaction() -> ConversationCompaction:
    """Return the process-scoped Conversation Compaction module.

    Per ADR-012 and system-design.agents.md §"Orchestrator Runtime →
    Conversation Compaction" — invoked at every orchestrator turn
    boundary; per-session state (circuit-breaker, tool first-seen
    timestamps, nine-section session notes) is keyed by session_id
    inside the instance so a singleton serves all in-flight sessions.

    The persistence root is rooted under the operator's global
    configuration directory (``~/.config/llm-orc/compaction-artifacts/``
    on a typical XDG setup); created lazily on first Layer 0 persist.

    The Layer 4 summarizer is left unconfigured by default —
    operators who want LLM-based semantic summary opt in by setting
    ``orchestrator.compaction.summarizer_ensemble`` in config.yaml
    and wiring a summarizer adapter. WP-E4 ships Layers 0–3 as the
    primary value; Layer 4 wiring lands when an operator-facing
    summarizer-ensemble convention exists.

    Tests override this factory to inject a stub or a tmp-rooted
    compaction.
    """
    global _SHARED_CONVERSATION_COMPACTION
    if _SHARED_CONVERSATION_COMPACTION is None:
        service = get_orchestra_service()
        resolver = get_orchestrator_config_resolver()
        config = resolver.resolve()
        persistence_root = (
            service.config_manager.global_config_dir / "compaction-artifacts"
        )
        _SHARED_CONVERSATION_COMPACTION = ConversationCompaction(
            defaults=config.compaction,
            persistence_root=persistence_root,
            summarizer=None,
        )
    return _SHARED_CONVERSATION_COMPACTION


async def _default_orchestrator_llm_loader(model_profile: str) -> OrchestratorLLM:
    """Load the orchestrator LLM from a Model Profile via existing ModelFactory.

    Uses the project's multi-provider machinery so any provider whose
    ``ModelInterface`` implementation overrides ``generate_with_tools``
    (currently :class:`OpenAICompatibleModel`; others land in
    follow-up WPs) works as the orchestrator. Fails fast if the
    resolved model does not support tool calling so misconfiguration
    surfaces at session start rather than mid-loop.
    """
    service = get_orchestra_service()
    credential_storage = CredentialStorage(service.config_manager)
    factory = ModelFactory(service.config_manager, credential_storage)
    model = await factory.load_model_from_agent_config({"model_profile": model_profile})
    if not model.supports_tool_calling:
        raise ToolCallingNotSupportedError(
            f"Orchestrator model_profile '{model_profile}' resolves to "
            f"'{model.name}' which does not support tool calling. Configure "
            "a profile with an OpenAI-compatible provider (Ollama, OpenAI, "
            "OpenRouter, LM Studio, vLLM, etc.)."
        )
    return model


def get_orchestrator_llm_loader() -> Callable[[str], Awaitable[OrchestratorLLM]]:
    """Return the LLM loader callable.

    Tests override this factory via ``monkeypatch.setattr`` to inject a
    scripted :class:`OrchestratorLLM` double.
    """
    return _default_orchestrator_llm_loader


class _ChatCompletionMessage(BaseModel):
    """One message in the OpenAI chat-completions request.

    ``tool_call_id`` and ``tool_calls`` carry the OpenAI tool-round-trip
    shape: ``role: assistant`` messages whose prior turn closed with
    ``finish_reason: tool_calls`` echo their ``tool_calls[]`` back, and
    ``role: tool`` messages carry a ``tool_call_id`` plus the tool
    result as ``content``. Both are optional so the common case
    (``role: user``, ``role: system``) parses without change.
    ``content`` is nullable because OpenAI accepts ``content: null`` on
    an assistant message whose turn carried only tool calls.
    """

    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class _ChatCompletionsRequest(BaseModel):
    """Minimal subset of the OpenAI ``/v1/chat/completions`` request body."""

    model: str
    messages: list[_ChatCompletionMessage]
    stream: bool = False
    tools: list[dict[str, Any]] = Field(default_factory=list)
    user: str | None = None


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: _ChatCompletionsRequest,
) -> dict[str, Any] | StreamingResponse:
    """Resolve session, build Runtime, then stream or return a body.

    Per Cycle 6 WP-B piece 5 a per-request
    :class:`InferenceWaitHeartbeatScheduler` is registered with the
    shared dispatch event substrate at request open and unregistered
    after the response stream (streaming) or body (non-streaming)
    completes. The scheduler emits one
    ``INFO: inference wait: elapsed=<sec> session_id=<id>`` line per
    ``heartbeat_interval_seconds`` (default 30s) of cloud-LLM
    inference inactivity, giving operators liveness signal during
    long-inference waits.
    """
    context = _resolve_context(request)
    session_id = context.state.identity.value
    context_sink = _build_context_sink(session_id=session_id)
    runtime = await _build_runtime(context_sink=context_sink)
    scheduler = _build_heartbeat_scheduler(session_id=session_id)

    if request.stream:
        return StreamingResponse(
            _stream_completion_with_heartbeat(
                context,
                runtime,
                model=request.model,
                scheduler=scheduler,
                context_sink=context_sink,
            ),
            media_type="text/event-stream",
        )

    return await _build_completion_body_with_heartbeat(
        context,
        runtime,
        model=request.model,
        scheduler=scheduler,
        context_sink=context_sink,
    )


def _build_heartbeat_scheduler(*, session_id: str) -> InferenceWaitHeartbeatScheduler:
    """Construct a per-request heartbeat scheduler.

    The scheduler uses ``observability.heartbeat_interval_seconds`` from
    the operator config (default 30s) and the shared operator-terminal
    sink as its emission target. Reads via ``resolve_validated`` mirrors
    :func:`_build_runtime` — the model-profile validation that runs at
    session start has already succeeded by the time chat_completions
    constructs the scheduler.
    """
    resolver = get_orchestrator_config_resolver()
    config = resolver.resolve_validated()
    sink = get_operator_terminal_event_sink()
    return InferenceWaitHeartbeatScheduler(
        sink=sink,
        session_id=session_id,
        interval_seconds=config.observability.heartbeat_interval_seconds,
    )


def _build_context_sink(*, session_id: str) -> OrchestratorContextEventSink:
    """Construct a per-request Orchestrator-Context Event Sink (Cycle 6 WP-C).

    Per ADR-023 §Destination 2: each request has its own sink so the
    session-prefix filter cleanly isolates dispatches from cross-session
    traffic — same pattern as :func:`_build_heartbeat_scheduler` per
    WP-B feed-forward advisory 3. Reads
    ``observability.orchestrator_context_routes_calibration_signal``
    (default ``False``) to decide whether to include
    :class:`CalibrationSignal` events in the end-of-session
    ``dispatch_log`` summary.
    """
    resolver = get_orchestrator_config_resolver()
    config = resolver.resolve_validated()
    return OrchestratorContextEventSink(
        session_id=session_id,
        routes_calibration_signal=(
            config.observability.orchestrator_context_routes_calibration_signal
        ),
    )


def _reject_reserved_tool_names(tools: list[dict[str, Any]]) -> None:
    """Reject client-declared tools that shadow llm-orc's internal surface.

    The five ``TOOL_NAMES`` (ADR-003) are the orchestrator's closed action
    set. A request that declares a client tool sharing one of those names
    would silently misroute: the Runtime classifies by ``context.tools``
    membership, so a shadowed internal call would be emitted as a
    :class:`ClientToolCall` delegation instead of running in-process.
    Rejecting at the Serving Layer with HTTP 400 surfaces the conflict
    immediately rather than as a mysterious stream of
    ``finish_reason: tool_calls``.
    """
    collisions: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name in TOOL_NAMES:
            collisions.append(name)
    if collisions:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "reserved_tool_name",
                "message": (
                    "Client-declared tool names collide with the "
                    "orchestrator's reserved internal tool surface "
                    f"(ADR-003): {sorted(set(collisions))}. Rename or "
                    "remove these tools from the request's 'tools' array."
                ),
                "reserved_names": sorted(TOOL_NAMES),
            },
        )


def _resolve_context(request: _ChatCompletionsRequest) -> SessionContext:
    """Run the pre-handoff work shared by streaming and non-streaming paths."""
    _reject_reserved_tool_names(request.tools)
    messages = [
        ChatMessage(
            role=message.role,
            content=message.content,
            tool_call_id=message.tool_call_id,
            tool_calls=tuple(message.tool_calls) if message.tool_calls else (),
        )
        for message in request.messages
    ]

    registry = get_session_registry()
    identity = registry.resolve_identity(
        messages=messages,
        user_field=request.user,
    )
    state = registry.get_or_create_state(identity)
    context = SessionContext(
        messages=messages,
        tools=list(request.tools),
        state=state,
    )

    cache = get_session_start_cache()
    cache.resolve(context)
    return context


async def _build_runtime(
    *, context_sink: OrchestratorContextEventSink | None = None
) -> OrchestratorRuntime:
    """Construct a per-session Runtime from the resolved orchestrator config.

    ``resolve_validated`` raises if the operator-configured Model
    Profile is absent from the library, so session start fails loudly
    rather than booting with a profile that cannot be loaded.

    ``context_sink`` (Cycle 6 WP-C, ADR-023 §Destination 2) is the
    per-request orchestrator-context sink, when supplied. Runtime
    queries it at each turn boundary after an ``invoke_ensemble``
    dispatch and prepends the structured observation to the next
    LLM call's messages.
    """
    resolver = get_orchestrator_config_resolver()
    config = resolver.resolve_validated()
    loader = get_orchestrator_llm_loader()
    llm = await loader(config.model_profile)
    budget = BudgetController(
        turn_limit=config.budget.turn_limit,
        token_limit=config.budget.token_limit,
    )
    return OrchestratorRuntime(
        llm=llm,
        budget=budget,
        tool_dispatch=get_orchestrator_tool_dispatch(),
        system_prompt=config.orchestrator_system_prompt,
        compaction=get_conversation_compaction(),
        context_sink=context_sink,
    )


@dataclass(frozen=True)
class _NonStreamingResult:
    """Collected chunk output shaped for the non-streaming response body.

    ``tool_calls`` is non-None when the Runtime closed the turn with a
    :class:`ClientToolCall` (Option C) — the client-declared tool
    delegations are carried on the response's ``message.tool_calls``
    field, and ``finish_reason`` is ``"tool_calls"`` per OpenAI shape.
    """

    content: str
    finish_reason: str
    tool_calls: list[dict[str, Any]] | None = field(default=None)


async def _collect_non_streaming(
    context: SessionContext, runtime: OrchestratorRuntime
) -> _NonStreamingResult:
    """Drive the Runtime and flatten its chunks to the non-streaming shape.

    VisibilityEvent chunks render as inline narration using the same
    helper the SSE formatter uses, so non-streaming response bodies
    carry the same Autonomy Policy visibility the streaming path emits —
    tool-user observability does not depend on transport.

    A :class:`ClientToolCall` chunk terminates the Runtime's generator
    and is shaped into ``message.tool_calls`` per OpenAI's chat-
    completion schema (Option C, Client Tool Surface Commitment).
    """
    content_parts: list[str] = []
    finish_reason = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    async for chunk in runtime.run(context):
        if isinstance(chunk, ContentDelta):
            content_parts.append(chunk.content)
        elif isinstance(chunk, VisibilityEvent):
            content_parts.append(render_visibility_narration(chunk.kind, chunk.payload))
        elif isinstance(chunk, ClientToolCall):
            tool_calls = [encode_tool_call_for_message(tc) for tc in chunk.tool_calls]
            finish_reason = "tool_calls"
        elif isinstance(chunk, Completion):
            finish_reason = chunk.finish_reason
    return _NonStreamingResult(
        content="".join(content_parts),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


async def _stream_completion(
    context: SessionContext, runtime: OrchestratorRuntime, *, model: str
) -> AsyncIterator[bytes]:
    """Drive the Runtime's chunk stream through the OpenAI SSE formatter."""
    formatter = OpenAiSseFormatter(
        stream_id=f"chatcmpl-{uuid.uuid4().hex}",
        model=model,
        created=int(time.time()),
    )
    yield formatter.start_assistant_turn()
    async for chunk in runtime.run(context):
        framed = formatter.format(chunk)
        if framed:
            yield framed
    yield formatter.done()


async def _stream_completion_with_heartbeat(
    context: SessionContext,
    runtime: OrchestratorRuntime,
    *,
    model: str,
    scheduler: InferenceWaitHeartbeatScheduler,
    context_sink: OrchestratorContextEventSink,
) -> AsyncIterator[bytes]:
    """Wrap :func:`_stream_completion` with heartbeat + context-sink lifecycle.

    The scheduler and the orchestrator-context sink both register with
    the shared substrate at request open; the scheduler starts its
    async heartbeat loop; the response stream runs; on completion (or
    exception, or client disconnect) the heartbeat task is cancelled,
    the context sink writes its end-of-session ``dispatch_log`` to
    the per-session path, and both register/unregister handshakes
    reverse. FastAPI's streaming response runtime invokes the cleanup
    branch when the underlying async generator's ``aclose`` runs.
    """
    substrate = get_dispatch_event_substrate()
    scheduler.register_with(substrate)
    context_sink.register_with(substrate)
    heartbeat_task = asyncio.create_task(scheduler.run())
    try:
        async for chunk in _stream_completion(context, runtime, model=model):
            yield chunk
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        _write_dispatch_log_safe(context_sink, session_id=context.state.identity.value)
        context_sink.unregister_with(substrate)
        scheduler.unregister_with(substrate)


async def _build_completion_body_with_heartbeat(
    context: SessionContext,
    runtime: OrchestratorRuntime,
    *,
    model: str,
    scheduler: InferenceWaitHeartbeatScheduler,
    context_sink: OrchestratorContextEventSink,
) -> dict[str, Any]:
    """Wrap :func:`_build_completion_body` with heartbeat + context-sink lifecycle.

    Mirrors :func:`_stream_completion_with_heartbeat` for non-streaming
    requests — both lifecycles wrap the body-construction call.
    """
    substrate = get_dispatch_event_substrate()
    scheduler.register_with(substrate)
    context_sink.register_with(substrate)
    heartbeat_task = asyncio.create_task(scheduler.run())
    try:
        return await _build_completion_body(context, runtime, model=model)
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        _write_dispatch_log_safe(context_sink, session_id=context.state.identity.value)
        context_sink.unregister_with(substrate)
        scheduler.unregister_with(substrate)


def _write_dispatch_log_safe(
    context_sink: OrchestratorContextEventSink, *, session_id: str
) -> None:
    """Write the orchestrator-context sink's dispatch_log to per-session path.

    Per ADR-023 §"end-of-session summary" — the path is
    ``<agentic_sessions_root>/<session_id>/dispatch_log.json`` (default
    root ``.llm-orc/agentic-sessions/`` per config). WP-E lands the
    broader agentic-sessions tree (per-dispatch directories per
    ADR-025); WP-C writes a standalone dispatch_log.json under the
    session-scoped subdirectory so the integration composes when WP-E
    arrives.

    Exceptions during the write are caught and logged at WARN — a
    filesystem failure at request close must not propagate as a
    response-time error to the client. The operator-terminal sink
    surfaces the failure separately.
    """
    from pathlib import Path

    try:
        resolver = get_orchestrator_config_resolver()
        config = resolver.resolve_validated()
        root = Path(config.observability.agentic_sessions_root)
        path = root / session_id / "dispatch_log.json"
        context_sink.write_dispatch_log(path)
    except Exception:  # noqa: BLE001 — close-time IO failure must not propagate
        import logging

        logging.getLogger("llm_orc.agentic.orchestrator_context").warning(
            "dispatch_log write failed for session_id=%s", session_id, exc_info=True
        )


async def _build_completion_body(
    context: SessionContext, runtime: OrchestratorRuntime, *, model: str
) -> dict[str, Any]:
    """Shape the non-streaming response body from the Runtime's chunks.

    ``usage`` carries the Session's cumulative accounting ---
    ``SessionState.token_spend`` is the sum over every LLM call in the
    session, and ``turn_count`` is the number of ReAct iterations. Per-
    request delta accounting can land in a follow-up when clients need
    it; the cumulative shape is what agentic coding tools display
    today.
    """
    pre_token_spend = context.state.token_spend
    result = await _collect_non_streaming(context, runtime)
    turn_tokens = max(0, context.state.token_spend - pre_token_spend)
    message: dict[str, Any] = {"role": "assistant"}
    if result.tool_calls is not None:
        # OpenAI convention: content is null on a pure tool_calls turn.
        message["content"] = None
        message["tool_calls"] = result.tool_calls
    else:
        message["content"] = result.content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": turn_tokens,
            "total_tokens": turn_tokens,
        },
    }
