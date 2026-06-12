"""Serving Layer ``/v1/chat/completions`` endpoint.

Per ``docs/agentic-serving/system-design.md`` §Serving Layer (L3) and
ADR-027 (Framework-Driven Dispatch Pipeline). Cycle 7 WP-A replaced the
``OrchestratorRuntime`` ReAct loop on this surface with the framework-
driven :class:`DispatchPipeline`. The endpoint:

1. Parses the OpenAI-compatible request.
2. Resolves a Session via :class:`SessionRegistry`.
3. Runs :func:`resolve_session_start_context` exactly once per session
   (FC-9); Phase 1 returns an empty list.
4. Constructs a :class:`DispatchPipeline` over a routing-planner
   ensemble (Stage 1), the shared Tool Dispatch wrapping
   :class:`OrchestraService` (Stage 2), and a response-synthesizer
   ensemble (Stage 3). The orchestrator-LLM is not in the routing-
   decision or post-dispatch-synthesis surface (AS-9 + ADR-027).
5. Drives the pipeline to produce either a non-streaming
   ``chat.completion`` body or an SSE stream.

The pipeline is constructed per request. The Tool Dispatch facade and
the pipeline factory are process-scoped factories module-level tests
override via ``monkeypatch.setattr``. ``OrchestratorRuntime`` remains in
the codebase per ADR-027 §"OrchestratorRuntime status" disposition (a)
(preserved as architecture-for-future-surfaces) but has no caller here.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_orc.agentic.artifact_bridge import ArtifactBridge, parse_check_form_gate
from llm_orc.agentic.autonomy_policy import AutonomyPolicy
from llm_orc.agentic.budget_controller import BudgetController
from llm_orc.agentic.calibration_gate import (
    CalibrationGate,
    EnsembleBackedChecker,
)
from llm_orc.agentic.client_tool_action_terminal import ClientToolActionTerminal
from llm_orc.agentic.composition_validator import (
    CompositionValidator,
    ConfigManagerEnsembleWriter,
    ConfigManagerPrimitiveRegistry,
)
from llm_orc.agentic.dispatch_event_substrate import DispatchEventSubstrate
from llm_orc.agentic.dispatch_pipeline import DispatchPipeline
from llm_orc.agentic.ensemble_backed_roles import (
    EnsembleResponseSynthesizer,
    EnsembleRoutingPlanner,
)
from llm_orc.agentic.inference_wait_heartbeat import (
    InferenceWaitHeartbeatScheduler,
)
from llm_orc.agentic.loop_driver import LoopDriver
from llm_orc.agentic.operator_terminal_event_sink import OperatorTerminalEventSink
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
)
from llm_orc.agentic.orchestrator_context_event_sink import (
    OrchestratorContextEventSink,
)
from llm_orc.agentic.orchestrator_tool_dispatch import (
    TOOL_NAMES,
    EnsembleConfigOutputSchemaReader,
    EnsembleConfigSubstrateReader,
    OrchestratorToolDispatch,
)
from llm_orc.agentic.plexus_adapter import PlexusAdapter
from llm_orc.agentic.result_summarizer_harness import ResultSummarizerHarness
from llm_orc.agentic.session_action_record import SessionActionRecord
from llm_orc.agentic.session_artifact_store import SessionArtifactStore
from llm_orc.agentic.session_registry import SessionRegistry
from llm_orc.agentic.session_start import ChatMessage, SessionContext, SessionStartCache
from llm_orc.agentic.single_step_enforcer import SingleStepEnforcer
from llm_orc.agentic.tier_router import (
    EnsembleConfigTopazSkillReader,
    TierRouter,
    TierRouterDefaults,
)
from llm_orc.agentic.tier_router_audit import TierEscalationAuditor
from llm_orc.core.auth.authentication import CredentialStorage
from llm_orc.core.models.model_factory import ModelFactory
from llm_orc.models.base import ModelInterface
from llm_orc.web.api import get_orchestra_service
from llm_orc.web.api.sse_format import OpenAiSseFormatter, encode_tool_call_for_message
from llm_orc.web.api.v1_models import get_orchestrator_config_resolver

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# System-ensemble names for the WP-A dispatch pipeline (ADR-028 / ADR-029).
# WP-B and WP-C ship the production ensembles under these names; WP-D's
# Capability List Builder formalizes the capability source. Held as module
# constants until that work makes them operator-configurable.
ROUTING_PLANNER_ENSEMBLE = "agentic-routing-planner"
RESPONSE_SYNTHESIZER_ENSEMBLE = "agentic-response-synthesizer"


_SHARED_REGISTRY = SessionRegistry()
_SHARED_SESSION_START_CACHE = SessionStartCache()
_SHARED_TOOL_DISPATCH: OrchestratorToolDispatch | None = None
_SHARED_DISPATCH_EVENT_SUBSTRATE: DispatchEventSubstrate | None = None
_SHARED_OPERATOR_TERMINAL_SINK: OperatorTerminalEventSink | None = None
_SHARED_SESSION_ARTIFACT_STORE: SessionArtifactStore | None = None
_SHARED_SESSION_ACTION_RECORD: SessionActionRecord | None = None


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


def get_session_artifact_store() -> SessionArtifactStore:
    """Return the process-scoped Session Artifact Store (Cycle 6 WP-E).

    Per ADR-025 §"Session-dir location" the store owns the
    ``.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/<deliverable>.<ext>``
    layout. Construction reads the operator-configured
    ``observability.agentic_sessions_root`` (default
    ``.llm-orc/agentic-sessions/``) and resolves it to an absolute
    :class:`~pathlib.Path` rooted at the current working directory so
    operator overrides flow through.

    On first construction the store registers its
    :meth:`~llm_orc.agentic.session_artifact_store.SessionArtifactStore.cleanup_session`
    bound method as a close callback on the shared Session Registry
    per ADR-025 §"Cleanup" + ``system-design.agents.md`` §"Session
    Registry → Session Artifact Store" — session close drives
    cleanup of ``retention: session`` artifacts.
    """
    global _SHARED_SESSION_ARTIFACT_STORE
    if _SHARED_SESSION_ARTIFACT_STORE is None:
        resolver = get_orchestrator_config_resolver()
        config = resolver.resolve()
        root = Path(config.observability.agentic_sessions_root)
        store = SessionArtifactStore(agentic_sessions_root=root)
        registry = get_session_registry()
        registry.register_close_callback(
            lambda identity: store.cleanup_session(identity.value)
        )
        _SHARED_SESSION_ARTIFACT_STORE = store
    return _SHARED_SESSION_ARTIFACT_STORE


def get_session_action_record() -> SessionActionRecord:
    """Return the process-scoped Session Action Record (Cycle 7 LB#5, ADR-037).

    The framework-owned digest's home: the Loop Driver records each
    emitted client-tool action at decision time and joins the client's
    per-call ``role: tool`` result on the next request (FC-64 — records
    derive from the framework's own emissions, never client-serialized
    reconstruction). On first construction the store registers its
    ``cleanup_session`` as a close callback on the shared Session
    Registry — lifecycle rides session scope, the Session Artifact Store
    retention pattern.
    """
    global _SHARED_SESSION_ACTION_RECORD
    if _SHARED_SESSION_ACTION_RECORD is None:
        store = SessionActionRecord()
        registry = get_session_registry()
        registry.register_close_callback(
            lambda identity: store.cleanup_session(identity.value)
        )
        _SHARED_SESSION_ACTION_RECORD = store
    return _SHARED_SESSION_ACTION_RECORD


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
        # Cycle 6 WP-E (ADR-025) production wiring: SessionArtifactStore
        # at the substrate slot; EnsembleConfigSubstrateReader bridges
        # to OrchestraService.find_ensemble_by_name for per-ensemble
        # output_substrate / output_retention / calibration_substrate_access
        # lookup; EnsembleConfigOutputSchemaReader activates the ADR-024
        # advisory JSON-parse path the WP-D envelope construction reads.
        artifact_store = get_session_artifact_store()
        substrate_reader = EnsembleConfigSubstrateReader(
            find_ensemble=service.find_ensemble_by_name
        )
        output_schema_reader = EnsembleConfigOutputSchemaReader(
            find_ensemble=service.find_ensemble_by_name
        )
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
            output_schema_reader=output_schema_reader,
            ensemble_substrate_reader=substrate_reader,
            session_artifact_store=artifact_store,
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


async def _build_capability_names() -> frozenset[str]:
    """Build the registered capability-ensemble name set (interim — WP-A).

    The Dispatch Pipeline validates ``plan.ensemble`` against this set
    before dispatching (the Spike ν E1/E3 backstop). A capability
    ensemble is one that carries ADR-015's ``topaz_skill`` marker;
    system ensembles (planner, synthesizer, summarizer, checker) carry
    no ``topaz_skill`` and are excluded.

    ``read_ensembles`` returns name/source/path metadata without
    ``topaz_skill``, so the marker is read per name via
    ``find_ensemble_by_name`` — the same EnsembleConfig surface the
    tier-router's skill reader uses. WP-D's Capability List Builder
    formalizes this as the single source of truth.
    """
    service = get_orchestra_service()
    entries = await service.read_ensembles()
    names: set[str] = set()
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        config = service.find_ensemble_by_name(name)
        if config is not None and getattr(config, "topaz_skill", None):
            names.add(name)
    return frozenset(names)


async def get_dispatch_pipeline() -> DispatchPipeline:
    """Construct a per-request framework-driven Dispatch Pipeline (ADR-027).

    Stage 1 (Routing Planner) and Stage 3 (Response Synthesizer) are
    ensemble-backed adapters over ``OrchestraService.invoke``; Stage 2
    reuses the shared Tool Dispatch chokepoint so the calibration gate,
    tier router, and autonomy interpositions fire unchanged (per the
    "OrchestratorToolDispatch.dispatch() contract is unchanged"
    preservation scenario). Capability names gate plan validation.

    Tests override this factory via ``monkeypatch.setattr`` to inject a
    stub pipeline (shape/streaming tests) or a real pipeline wired to
    stub ports (acceptance tests).
    """
    service = get_orchestra_service()
    return DispatchPipeline(
        planner=EnsembleRoutingPlanner(
            invoker=service, ensemble_name=ROUTING_PLANNER_ENSEMBLE
        ),
        synthesizer=EnsembleResponseSynthesizer(
            invoker=service, ensemble_name=RESPONSE_SYNTHESIZER_ENSEMBLE
        ),
        tool_dispatch=get_orchestrator_tool_dispatch(),
        capability_names=await _build_capability_names(),
        event_substrate=get_dispatch_event_substrate(),
    )


async def _resolve_seat_filler() -> ModelInterface:
    """Resolve the loop-driver's seat-filler LLM from the Model Profile.

    The seat-filler fills the client's "model" seat (ADR-033 §Decision 5):
    the Serving Layer resolves the orchestrator's configured Model Profile
    (ADR-011) to a tool-calling ``ModelInterface``, mirroring how the
    Orchestrator Runtime is handed its orchestrator-LLM. Swapping the driver
    model (cheap-tier ↔ frontier-tier — the named axis-2 fallback) is a
    config edit to the profile, never a change to the Loop Driver (FC-46).

    The returned ``ModelInterface`` satisfies both of the driver's model
    ports: ``SeatFiller`` (``generate_with_tools`` — the action calls) and
    ``JudgmentSeat`` (``generate_response`` — ADR-037's bare judgment
    call). The judgment seat defaults to this same resolved model (FC-68:
    shared profile = one re-validation covers both instruments).

    The model must support tool calling — the loop-driver decides each turn
    via ``generate_with_tools``. A profile resolving to a non-tool-calling
    model is an operator misconfiguration surfaced as a clear error rather
    than a silent failure at the first turn. Tests override this seam to
    inject a seat-filler double without standing up model resolution.
    """
    service = get_orchestra_service()
    config_manager = service.config_manager
    config = get_orchestrator_config_resolver().resolve()
    profile = config_manager.get_model_profiles().get(config.model_profile)
    if profile is None or not profile.get("model"):
        raise RuntimeError(
            f"Seat-filler Model Profile '{config.model_profile}' is not "
            "configured with a model in ConfigurationManager.get_model_profiles()"
        )
    credential_storage = CredentialStorage(config_manager)
    model_factory = ModelFactory(config_manager, credential_storage)
    model = await model_factory.load_model(
        profile["model"],
        profile.get("provider"),
        base_url=profile.get("base_url"),
    )
    if not model.supports_tool_calling:
        raise RuntimeError(
            f"Seat-filler Model Profile '{config.model_profile}' resolves to a "
            "model that does not support tool calling; the loop-driver decides "
            "each turn via generate_with_tools"
        )
    return model


async def get_loop_driver() -> LoopDriver:
    """Construct the real layer-A Loop Driver (ADR-033).

    The seat-filler resolved from the Model Profile, the Single-Step Enforcer
    (batch-truncation grounding guarantee), the shared Tool Dispatch chokepoint
    for per-turn callee generation, and the shared Dispatch Event Substrate for
    ``TurnDecision`` diagnostics. The driver decides each turn; the
    Client-Tool-Action Terminal (composed by
    :func:`get_client_tool_action_terminal`) emits the wire response. Tests
    override this factory (or :func:`_resolve_seat_filler`) via
    ``monkeypatch.setattr``.
    """
    seat_filler_model = await _resolve_seat_filler()
    budget = get_orchestrator_config_resolver().resolve().budget
    return LoopDriver(
        seat_filler=seat_filler_model,
        enforcer=SingleStepEnforcer(),
        tool_dispatch=get_orchestrator_tool_dispatch(),
        action_record=get_session_action_record(),
        # FC-68: the judgment seat defaults to the seat-filler's profile —
        # one re-validation event covers both instruments. A split
        # judgment-seat profile is a config choice (LB-8), not built here.
        judgment_seat=seat_filler_model,
        # FC-69: AS-3's turn cap is the absolute ceiling beneath the
        # termination mechanism — wired onto the loop-driver path so the
        # zombie-revision loop ADR-037 fixes has a deterministic backstop.
        budget=BudgetController(
            turn_limit=budget.turn_limit, token_limit=budget.token_limit
        ),
        capabilities=await _build_capability_names(),
        event_substrate=get_dispatch_event_substrate(),
        # ADR-041 §Decision 3 — the store is the server-side form-recovery
        # dependency: it resolves the substrate-routed deliverable content the
        # parse-check re-dispatches on.
        artifact_store=get_session_artifact_store(),
    )


async def get_client_tool_action_terminal() -> _ChatCompletionsCaller:
    """Construct the tool-driven-surface caller (ADR-034).

    Parallel to :func:`get_dispatch_pipeline`. The Client-Tool-Action Terminal
    composes the real Loop Driver (:func:`get_loop_driver`) and owns tool-call
    emission + multi-turn loop participation, marshalling substrate-routed
    deliverable content via the Artifact Bridge (ADR-034 §Decision 3) over the
    shared Session Artifact Store. The surface-mode discriminator engages this
    caller when a request carries client ``tools[]``. Tests override this
    factory via ``monkeypatch.setattr`` to inject a stub caller.
    """
    return ClientToolActionTerminal(
        loop_driver=await get_loop_driver(),
        # ADR-041 §Decision 1 — the deterministic destination-validity gate
        # installs at the FormGate seam (FC-57, zero-Terminal-edits): a
        # deliverable that does not parse as what its destination path claims
        # is refused before it reaches the client.
        bridge=ArtifactBridge(
            get_session_artifact_store(), form_gate=parse_check_form_gate
        ),
        # ADR-039 V-04: the Terminal captures each resolved deliverable's
        # content onto the same process-scoped record the driver writes its
        # actions to — the content anchor's source for a later turn's callee.
        action_record=get_session_action_record(),
    )


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


def _is_tool_driven(request: _ChatCompletionsRequest) -> bool:
    """Surface-mode discriminator (ADR-033 §Decision 1; D1).

    A request that carries client ``tools[]`` is a tool-driven client
    prepared to execute tool calls (e.g. OpenCode's build agent declaring
    ``write``/``edit``/``bash``/``read``), so it engages the layer-A
    loop-driver. A request with no client tools is a non-agentic
    answer-a-question request and continues through ADR-027's single-turn
    ``plan → dispatch → synthesize`` pipeline.

    Validate-not-assume (loop-back ARCHITECT advisory #2): whether
    ``tools[]`` presence is the right discriminator is a drafting-time
    design choice grounded in Spike π Phase 0, not a measured result. A
    tool-capable client could send ``tools[]`` for bookkeeping while
    wanting a plain answer on a given turn; that edge case is safe
    because the loop-driver can finish with text. Treat the first
    unexpected ``tools[]`` pattern from production traffic as a named
    validation event, not a defect.
    """
    return len(request.tools) > 0


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: _ChatCompletionsRequest,
) -> dict[str, Any] | StreamingResponse:
    """Resolve session, select the surface-mode caller, then stream or return a body.

    The surface-mode discriminator (:func:`_is_tool_driven`, ADR-033 D1)
    selects the caller: a tool-driven request (client ``tools[]`` present)
    engages the Client-Tool-Action Terminal (ADR-034, composing the layer-A
    loop-driver); a non-tool request continues through ADR-027's single-turn
    Dispatch Pipeline. Both satisfy :class:`_ChatCompletionsCaller`, so the
    heartbeat + context-sink lifecycle below wraps either caller unchanged.

    Per Cycle 6 WP-B piece 5 a per-request
    :class:`InferenceWaitHeartbeatScheduler` is registered with the
    shared dispatch event substrate at request open and unregistered
    after the response stream (streaming) or body (non-streaming)
    completes. The scheduler emits one
    ``INFO: inference wait: elapsed=<sec> session_id=<id>`` line per
    ``heartbeat_interval_seconds`` (default 30s) of cloud-LLM
    inference inactivity, giving operators liveness signal during
    long-inference waits.

    The per-request context sink (Cycle 6 WP-C) keeps its substrate-
    registration + end-of-session ``dispatch_log`` role under ADR-027;
    its turn-boundary observation injection is moot because the
    pipeline is single-turn (no orchestrator-LLM loop to inject into).
    """
    context = _resolve_context(request)
    session_id = context.state.identity.value
    context_sink = _build_context_sink(session_id=session_id)
    caller: _ChatCompletionsCaller
    if _is_tool_driven(request):
        caller = await get_client_tool_action_terminal()
    else:
        caller = await get_dispatch_pipeline()
    scheduler = _build_heartbeat_scheduler(session_id=session_id)

    if request.stream:
        return StreamingResponse(
            _stream_completion_with_heartbeat(
                context,
                caller,
                model=request.model,
                scheduler=scheduler,
                context_sink=context_sink,
            ),
            media_type="text/event-stream",
        )

    return await _build_completion_body_with_heartbeat(
        context,
        caller,
        model=request.model,
        scheduler=scheduler,
        context_sink=context_sink,
    )


def _build_heartbeat_scheduler(*, session_id: str) -> InferenceWaitHeartbeatScheduler:
    """Construct a per-request heartbeat scheduler.

    The scheduler uses ``observability.heartbeat_interval_seconds`` from
    the operator config (default 30s) and the shared operator-terminal
    sink as its emission target. The config has already resolved
    successfully by the time chat_completions constructs the scheduler.
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

    The five ``TOOL_NAMES`` (ADR-003) are llm-orc's closed internal
    action set. Rejecting a client tool that reuses one of those names
    preserves the Cycle 1 reserved-name commitment ("reserved TOOL_NAMES
    enforced") and surfaces the collision at the Serving Layer with
    HTTP 400 rather than letting a confusingly-named tool through. Under
    ADR-027 the pipeline does not route by ``tools`` membership, so the
    guard is a defensive input check rather than a misrouting backstop.
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


class _ChatCompletionsCaller(Protocol):
    """The chat-completions request driver.

    Both :class:`DispatchPipeline` (the ADR-027 caller on this surface)
    and :class:`OrchestratorRuntime` (preserved for future surfaces per
    ADR-027 disposition (a)) satisfy this structurally — ``run`` yields
    the shared :class:`OrchestratorChunk` vocabulary the SSE formatter
    and non-streaming collector consume.
    """

    def run(self, context: SessionContext) -> AsyncIterator[OrchestratorChunk]: ...


@dataclass(frozen=True)
class _NonStreamingResult:
    """Collected chunk output shaped for the non-streaming response body.

    ``tool_calls`` is non-``None`` when the caller closed the turn with a
    :class:`ClientToolCall` (ADR-034's tool-driven terminal) — the client-tool
    delegations are carried on the response's ``message.tool_calls`` field and
    ``finish_reason`` is ``"tool_calls"`` per the OpenAI shape.
    """

    content: str
    finish_reason: str
    tool_calls: list[dict[str, Any]] | None = None


async def _collect_non_streaming(
    context: SessionContext, caller: _ChatCompletionsCaller
) -> _NonStreamingResult:
    """Drive the caller and flatten its chunks to the non-streaming shape.

    The single-turn (non-tool) surface — the Dispatch Pipeline (ADR-027) —
    yields ``ContentDelta`` text and a terminal ``Completion``; the composed
    response is the concatenation of the content deltas. The tool-driven
    multi-turn surface (ADR-033 / ADR-034) closes a turn with a
    ``ClientToolCall``, which shapes into ``message.tool_calls`` +
    ``finish_reason: "tool_calls"`` per the OpenAI non-streaming tool-call
    shape (the streaming path renders the same chunk via the SSE formatter).
    """
    content_parts: list[str] = []
    finish_reason = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    async for chunk in caller.run(context):
        if isinstance(chunk, ContentDelta):
            content_parts.append(chunk.content)
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
    context: SessionContext, caller: _ChatCompletionsCaller, *, model: str
) -> AsyncIterator[bytes]:
    """Drive the caller's chunk stream through the OpenAI SSE formatter."""
    formatter = OpenAiSseFormatter(
        stream_id=f"chatcmpl-{uuid.uuid4().hex}",
        model=model,
        created=int(time.time()),
    )
    yield formatter.start_assistant_turn()
    async for chunk in caller.run(context):
        framed = formatter.format(chunk)
        if framed:
            yield framed
    yield formatter.done()


async def _stream_completion_with_heartbeat(
    context: SessionContext,
    caller: _ChatCompletionsCaller,
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
        async for chunk in _stream_completion(context, caller, model=model):
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
    caller: _ChatCompletionsCaller,
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
        return await _build_completion_body(context, caller, model=model)
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
    context: SessionContext, caller: _ChatCompletionsCaller, *, model: str
) -> dict[str, Any]:
    """Shape the non-streaming response body from the caller's chunks.

    ``usage`` carries the Session's cumulative accounting ---
    ``SessionState.token_spend`` is the sum over every LLM call in the
    session. Per-request delta accounting can land in a follow-up when
    clients need it; the cumulative shape is what agentic coding tools
    display today. The pipeline's per-dispatch token accounting is WP-C
    work, so ``token_spend`` is unchanged by the pipeline in WP-A.
    """
    pre_token_spend = context.state.token_spend
    result = await _collect_non_streaming(context, caller)
    turn_tokens = max(0, context.state.token_spend - pre_token_spend)
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls is not None:
        # OpenAI convention: content is null on a pure tool_calls turn.
        message["content"] = None
        message["tool_calls"] = result.tool_calls
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
