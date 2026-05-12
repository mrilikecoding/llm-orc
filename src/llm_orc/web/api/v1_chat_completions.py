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
from llm_orc.agentic.orchestrator_chunk import (
    ClientToolCall,
    Completion,
    ContentDelta,
    VisibilityEvent,
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


def get_session_registry() -> SessionRegistry:
    """Return the process-scoped Session Registry.

    Tests override this factory to inject an isolated registry.
    """
    return _SHARED_REGISTRY


def get_session_start_cache() -> SessionStartCache:
    """Return the process-scoped session-start cache."""
    return _SHARED_SESSION_START_CACHE


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
        _SHARED_TOOL_DISPATCH = OrchestratorToolDispatch(
            operations=service,
            harness=harness,
            autonomy_policy=autonomy_policy,
            composition_validator=composition_validator,
            local_ensemble_writer=local_ensemble_writer,
            calibration_gate=calibration_gate,
            plexus_adapter=plexus_adapter,
            tool_call_validation_patterns=config.tool_call_validation_patterns,
        )
    return _SHARED_TOOL_DISPATCH


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
    """Resolve session, build Runtime, then stream or return a body."""
    context = _resolve_context(request)
    runtime = await _build_runtime()

    if request.stream:
        return StreamingResponse(
            _stream_completion(context, runtime, model=request.model),
            media_type="text/event-stream",
        )

    return await _build_completion_body(context, runtime, model=request.model)


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


async def _build_runtime() -> OrchestratorRuntime:
    """Construct a per-session Runtime from the resolved orchestrator config.

    ``resolve_validated`` raises if the operator-configured Model
    Profile is absent from the library, so session start fails loudly
    rather than booting with a profile that cannot be loaded.
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
