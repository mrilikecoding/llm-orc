"""Conversation Compaction five-layer pipeline (WP-E4, ADR-012).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Conversation Compaction (L2; new in Cycle 4).

The pipeline runs at every orchestrator turn boundary and applies five
layers in strict cheapest-first order; Layer N+1 fires only if Layers
0..N together cannot bring context below the configured trigger
threshold. Layer ordering is the load-bearing design property per
ADR-012 §Decision and FC-14 — verified by static AST inspection of the
``compact()`` method and by the multi-turn integration fixture.

Layer responsibilities:

* **Layer 0** — Persist tool-call results larger than the configured
  character threshold to disk; replace in-context with a small preview
  plus the persistent artifact path.
* **Layer 1** — Cache-edit. ADR-012 prescribes deleting old cache
  entries without invalidating the conversation prefix. llm-orc does
  not yet have a provider-cache abstraction (zero ``cache_control``
  references in ``src/``); WP-E4 ships Layer 1 as a structural
  placeholder that preserves the prefix-invalidation invariant by
  doing nothing observable to the messages. Real cache integration
  is Cycle 5+ work.
* **Layer 2** — Idle-expiry. Tool-call results not touched for the
  configured idle window are cleared from active context; their
  persistent artifacts (if Layer 0 persisted them) remain reclaimable
  by path.
* **Layer 3** — Free summary via the nine-section session-notes
  template. Updated each turn by deterministic logic with zero LLM
  cost; bounded by the operator-configured token cap. This is
  bookkeeping in WP-E4 — the template populates the data surface but
  does not transform the in-context messages directly. Layer 4
  (or future Layer 3 elaboration) consumes the notes as input.
* **Layer 4** — LLM semantic summary via a configured summarizer
  ensemble (distinct from AS-7's Result Summarizer Harness, which
  operates on ensemble outputs rather than conversation history per
  system-design.agents.md L202). Per-session circuit-breaker:
  three consecutive Layer 4 failures suspend the layer for the
  session and raise a typed :class:`CompactionLayer4FailureError`
  with ``recovery_action_required="operator_intervention_required"``.
  Suspension auto-resets at session start.

Per FC-17 the typed error derives from :class:`LlmOrcStructuralError`
with the four common fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol
from uuid import uuid4

from llm_orc.models.structural_errors import LlmOrcStructuralError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREVIEW_BYTES: Final[int] = 2_048
"""Layer 0 preview size per ADR-012 §Decision."""

_SECONDS_PER_MINUTE: Final[float] = 60.0

_CHARS_PER_TOKEN: Final[int] = 4
"""Token-estimation heuristic — standard char/4 approximation used by
operator-readable size diagnostics. Not provider-specific; the
trigger-token-count threshold is operator-tunable to absorb the
heuristic's imprecision."""

NINE_SECTIONS: Final[tuple[str, ...]] = (
    "current_state",
    "tasks",
    "files",
    "workflow",
    "errors",
    "learnings",
    "worklog",
    "reserved_a",
    "reserved_b",
)
"""The nine-section template per ADR-012 §Decision §Layer 3.

Seven named sections (current state, tasks, files, workflow, errors,
learnings, worklog) plus two reserved for deployment customization."""


# ---------------------------------------------------------------------------
# Defaults and configuration shape (consumed by Orchestrator Configuration L3)
# ---------------------------------------------------------------------------

DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS: Final[int] = 50_000
"""Layer 0 trigger per ADR-012 §Decision (Anthropic-published value)."""

DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES: Final[int] = 60
"""Layer 2 idle window per ADR-012 §Decision."""

DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP: Final[int] = 12_288
"""Layer 3 nine-section template token cap per ADR-012 §Decision."""

DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 3
"""Layer 4 circuit-breaker trip threshold per ADR-012 §Decision."""

DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT: Final[int] = 100_000
"""WP-E4 addition — aggregate token threshold above which the
pipeline runs. ADR-012 specifies "until the context budget is
satisfied" without naming the budget; this default keeps llm-orc
under Claude Code's 50K rot-onset evidence (Chroma 2025) with
headroom for the unwound chars/4 heuristic."""

DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE: str | None = None
"""WP-E4 addition — Layer 4 ensemble name. Distinct from AS-7's
``summarizer_ensemble`` (Result Summarizer Harness operates on
ensemble outputs per system-design.agents.md L202). ``None`` means
Layer 4 is unconfigured and short-circuits without dispatch."""


@dataclass(frozen=True)
class CompactionDefaults:
    """Conversation Compaction configuration (ADR-012, WP-E4).

    The four named thresholds (``persist_threshold_chars``,
    ``idle_window_minutes``, ``session_notes_token_cap``,
    ``layer_4_circuit_breaker_threshold``) are Anthropic's published
    operational values per ADR-012's defaults-provenance note.
    ``trigger_token_count`` and ``summarizer_ensemble`` are WP-E4
    additions per the build-time disposition recorded in
    cycle-status.md.
    """

    persist_threshold_chars: int
    idle_window_minutes: int
    session_notes_token_cap: int
    layer_4_circuit_breaker_threshold: int
    trigger_token_count: int
    summarizer_ensemble: str | None


# ---------------------------------------------------------------------------
# Protocols and value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummarizerInvocation:
    """Input to the Layer 4 summarizer.

    The Compaction module is layered above Ensemble Engine but does not
    import it directly — the integration site (Serving Layer
    construction; see Runtime wiring in the WP-E4 integration commit)
    adapts an ``EnsembleExecutor.execute`` call into this duck-typed
    shape. Keeping Compaction free of L0 imports preserves FC-2.
    """

    ensemble: str
    messages: tuple[dict[str, Any], ...]
    session_id: str


class Summarizer(Protocol):
    """Layer 4 summarizer adapter.

    The integration site supplies an object satisfying this protocol;
    the protocol is intentionally narrow so production wiring and
    test doubles agree on shape.
    """

    def summarize(self, invocation: SummarizerInvocation) -> str: ...


class Clock(Protocol):
    """Monotonic clock surface for idle-expiry timestamps.

    Tests pass a controllable clock; production wiring uses
    :class:`_MonotonicClock`.
    """

    def now_seconds(self) -> float: ...


class _MonotonicClock:
    def now_seconds(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class CompactedContext:
    """Result of a compaction pass at a turn boundary.

    * ``messages`` — the (possibly reduced) messages array the Runtime
      passes to the next LLM call.
    * ``layers_applied`` — the layers (0-4) that did observable work
      this turn, ascending. Layer 0 records when at least one
      oversized tool result was persisted; Layer 2 when at least one
      idle tool result was cleared; Layer 4 when a summarizer
      dispatch was attempted. Layer 1 (structural placeholder) and
      Layer 3 (session-notes bookkeeping) are recorded whenever the
      pipeline reaches them.
    * ``triggered`` — whether the pipeline ran at all this turn.
      ``False`` means the messages were below the trigger threshold
      and returned unchanged.
    """

    messages: list[dict[str, Any]]
    layers_applied: tuple[int, ...]
    triggered: bool


@dataclass
class SessionNotes:
    """The nine-section deterministically-updated session notes.

    ``estimated_token_count`` is the heuristic char/4 estimate over
    all section contents — bounded by the operator-configured cap.
    """

    sections: dict[str, str] = field(
        default_factory=lambda: dict.fromkeys(NINE_SECTIONS, "")
    )
    estimated_token_count: int = 0


# ---------------------------------------------------------------------------
# Typed error
# ---------------------------------------------------------------------------


class CompactionLayer4FailureError(LlmOrcStructuralError):
    """Raised on the Layer 4 circuit-breaker trip per ADR-012 §Decision.

    Fourth concrete subclass of :class:`LlmOrcStructuralError` per FC-17
    (after ``ToolCallingNotSupportedError``, ``PhantomToolCallError``,
    and ``WriteGateRejectionError``). The ``error_kind`` is fixed by
    construction; ``recovery_action_required`` is always
    ``operator_intervention_required`` because the orchestrator cannot
    reformulate its way out of a Layer 4 outage — the summarizer
    ensemble itself is the structural condition needing operator
    attention.
    """

    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        consecutive_failures: int,
        underlying_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            message,
            error_kind="compaction_layer_4_failure",
            recovery_action_required="operator_intervention_required",
            dispatch_context={
                "session_id": session_id,
                "consecutive_failures": consecutive_failures,
                "underlying_errors": list(underlying_errors),
            },
            operator_diagnostic=message,
        )


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


@dataclass
class _SessionCompactionState:
    """Per-session state carried across turns; reset by
    :meth:`ConversationCompaction.reset_session`."""

    layer_4_consecutive_failures: int = 0
    layer_4_suspended: bool = False
    layer_4_recent_errors: list[str] = field(default_factory=list)
    tool_first_seen: dict[str, float] = field(default_factory=dict)
    session_notes: SessionNotes = field(default_factory=SessionNotes)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConversationCompaction:
    """Five-layer cheapest-first compaction pipeline (ADR-012)."""

    def __init__(
        self,
        *,
        defaults: CompactionDefaults,
        persistence_root: Path,
        summarizer: Summarizer | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._defaults = defaults
        self._persistence_root = persistence_root
        self._summarizer = summarizer
        self._clock: Clock = clock if clock is not None else _MonotonicClock()
        self._per_session: dict[str, _SessionCompactionState] = {}

    # -- Public API ---------------------------------------------------------

    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
    ) -> CompactedContext:
        """Run the five-layer pipeline at a turn boundary.

        Layers fire cheapest-first; short-circuits on the first layer
        that brings context below the trigger threshold per FC-14.
        """
        state = self._state_for(session_id)
        # Fresh shallow copies — caller's list and per-dict shape stay
        # untouched (the contract observed by Runtime).
        current = [dict(m) for m in messages]
        self._record_tool_first_seen(current, state)

        if not self._exceeds_trigger(current):
            return CompactedContext(
                messages=current, layers_applied=(), triggered=False
            )

        applied: list[int] = []

        # Layer 0 — persist large tool results
        current, layer_0_fired = self._layer_0_persist(current, session_id)
        if layer_0_fired:
            applied.append(0)
            if not self._exceeds_trigger(current):
                return self._final(current, applied)

        # Layer 1 — cache-edit (structural placeholder)
        # Recorded whenever the pipeline reaches it; no-op in WP-E4.
        applied.append(1)
        if not self._exceeds_trigger(current):
            return self._final(current, applied)

        # Layer 2 — idle-expiry
        current, layer_2_fired = self._layer_2_idle_expiry(current, state)
        if layer_2_fired:
            applied.append(2)
            if not self._exceeds_trigger(current):
                return self._final(current, applied)

        # Layer 3 — session notes update (zero LLM cost)
        self._layer_3_update_session_notes(messages, state)
        applied.append(3)
        if not self._exceeds_trigger(current):
            return self._final(current, applied)

        # Layer 4 — LLM semantic summary (last resort)
        if self._layer_4_available(state):
            current = self._layer_4_summarize(current, state, session_id)
            applied.append(4)

        return self._final(current, applied)

    def reset_session(self, session_id: str) -> None:
        """Clear per-session compaction state — circuit-breaker, tool
        first-seen timestamps, session notes — per ADR-012's auto-
        reset-at-session-start property (argument-audit P3.1)."""
        self._per_session.pop(session_id, None)

    def session_notes_for(self, session_id: str) -> SessionNotes:
        """Inspect a session's nine-section notes (read-only view).

        Tests and operator-facing diagnostic surfaces use this; the
        Runtime does not consume notes directly in WP-E4 — Layer 3 is
        bookkeeping that future cycles can elaborate into a prompt
        contribution.
        """
        return self._state_for(session_id).session_notes

    def circuit_breaker_failures_for(self, session_id: str) -> int:
        return self._state_for(session_id).layer_4_consecutive_failures

    def circuit_breaker_suspended_for(self, session_id: str) -> bool:
        return self._state_for(session_id).layer_4_suspended

    # -- Layer implementations ---------------------------------------------

    def _layer_0_persist(
        self, messages: list[dict[str, Any]], session_id: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Persist oversized tool-result content to disk; replace
        in-context with preview-plus-path envelope."""
        threshold = self._defaults.persist_threshold_chars
        fired = False
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or len(content) <= threshold:
                continue
            artifact_path = self._persist_artifact(content, session_id)
            preview = content[:_PREVIEW_BYTES]
            message["content"] = (
                f"[Tool result persisted to {artifact_path} "
                f"(full size {len(content)} chars). "
                f"Preview ({len(preview)} chars):\n{preview}]"
            )
            fired = True
        return messages, fired

    def _layer_2_idle_expiry(
        self,
        messages: list[dict[str, Any]],
        state: _SessionCompactionState,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Drop tool-result messages whose first-seen timestamp is past
        the configured idle window."""
        window_seconds = self._defaults.idle_window_minutes * _SECONDS_PER_MINUTE
        now = self._clock.now_seconds()
        kept: list[dict[str, Any]] = []
        fired = False
        for message in messages:
            tool_call_id = message.get("tool_call_id")
            if (
                message.get("role") == "tool"
                and isinstance(tool_call_id, str)
                and tool_call_id in state.tool_first_seen
                and (now - state.tool_first_seen[tool_call_id]) >= window_seconds
            ):
                fired = True
                # Clear the first-seen entry so a re-seen tool result
                # gets a fresh window if the orchestrator re-introduces
                # it (e.g., re-invocation).
                state.tool_first_seen.pop(tool_call_id, None)
                continue
            kept.append(message)
        return kept, fired

    def _layer_3_update_session_notes(
        self,
        original_messages: list[dict[str, Any]],
        state: _SessionCompactionState,
    ) -> None:
        """Deterministically update the nine-section template from the
        most recent turn — zero LLM cost; bounded by the configured
        token cap.

        The update rule for WP-E4 is intentionally minimal: each turn's
        content appends a one-line entry to the ``worklog`` section
        with a deterministic prefix; the other sections remain
        operator-managed surfaces for future cycles to elaborate.
        """
        notes = state.session_notes
        # Build the worklog entry from the latest turn (last user or
        # assistant message). Deterministic — no LLM call.
        latest = self._latest_speaker_message(original_messages)
        if latest is not None:
            role = latest.get("role", "")
            content = latest.get("content", "")
            if isinstance(content, str):
                preview = content[:120].replace("\n", " ")
                entry = f"- [{role}] {preview}\n"
                notes.sections["worklog"] = notes.sections["worklog"] + entry

        self._enforce_session_notes_cap(notes)

    def _layer_4_available(self, state: _SessionCompactionState) -> bool:
        """Layer 4 dispatches only if a summarizer is configured AND
        the circuit-breaker is not tripped AND an ensemble name is
        set in operator config."""
        return (
            self._summarizer is not None
            and self._defaults.summarizer_ensemble is not None
            and not state.layer_4_suspended
        )

    def _layer_4_summarize(
        self,
        messages: list[dict[str, Any]],
        state: _SessionCompactionState,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Dispatch the configured summarizer ensemble; on failure
        advance the circuit-breaker; on the third consecutive failure
        suspend Layer 4 and raise."""
        assert self._summarizer is not None  # narrowed by _layer_4_available
        ensemble_name = self._defaults.summarizer_ensemble
        assert ensemble_name is not None  # narrowed by _layer_4_available
        invocation = SummarizerInvocation(
            ensemble=ensemble_name,
            messages=tuple(messages),
            session_id=session_id,
        )
        try:
            summary = self._summarizer.summarize(invocation)
        except Exception as failure:  # noqa: BLE001 — adapter surface
            self._record_layer_4_failure(state, session_id, failure)
            return messages

        # Reset the consecutive-failures counter on a successful
        # summary — "three consecutive failures" is the trip semantic.
        state.layer_4_consecutive_failures = 0
        state.layer_4_recent_errors.clear()
        # Replace the messages with a single system-level summary
        # entry; the structural reduction is the load-bearing property.
        return [
            {
                "role": "system",
                "content": f"[Conversation summary (Layer 4): {summary}]",
            }
        ]

    # -- Helpers -----------------------------------------------------------

    def _record_layer_4_failure(
        self,
        state: _SessionCompactionState,
        session_id: str,
        failure: BaseException,
    ) -> None:
        state.layer_4_consecutive_failures += 1
        state.layer_4_recent_errors.append(f"{type(failure).__name__}: {failure}")
        if (
            state.layer_4_consecutive_failures
            >= self._defaults.layer_4_circuit_breaker_threshold
        ):
            state.layer_4_suspended = True
            raise CompactionLayer4FailureError(
                "Layer 4 circuit-breaker tripped after "
                f"{state.layer_4_consecutive_failures} consecutive "
                f"summarizer failures; Layer 4 is suspended for the "
                f"remainder of session {session_id!r}. Operator "
                "intervention required (inspect summarizer ensemble; "
                "circuit-breaker auto-resets at session start).",
                session_id=session_id,
                consecutive_failures=state.layer_4_consecutive_failures,
                underlying_errors=tuple(state.layer_4_recent_errors),
            )

    def _persist_artifact(self, content: str, session_id: str) -> Path:
        """Write ``content`` under ``persistence_root/<session_id>/``."""
        session_dir = self._persistence_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        artifact = session_dir / f"tool-result-{uuid4().hex}.txt"
        artifact.write_text(content, encoding="utf-8")
        return artifact

    def _record_tool_first_seen(
        self,
        messages: list[dict[str, Any]],
        state: _SessionCompactionState,
    ) -> None:
        """Populate ``tool_first_seen`` for any tool result not yet
        timestamped — driven by tool_call_id so re-invocations of the
        same tool produce a fresh window when re-introduced."""
        now = self._clock.now_seconds()
        for message in messages:
            tool_call_id = message.get("tool_call_id")
            if message.get("role") == "tool" and isinstance(tool_call_id, str):
                state.tool_first_seen.setdefault(tool_call_id, now)

    def _latest_speaker_message(
        self, messages: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        for message in reversed(messages):
            if message.get("role") in ("user", "assistant"):
                return message
        return None

    def _enforce_session_notes_cap(self, notes: SessionNotes) -> None:
        """Bound notes by the configured token cap.

        Truncates the worklog (the only growing section in WP-E4) until
        the total estimated token count is at or below the cap. Future
        sections that grow can extend this logic.
        """
        cap_chars = self._defaults.session_notes_token_cap * _CHARS_PER_TOKEN
        current_chars = sum(len(text) for text in notes.sections.values())
        if current_chars <= cap_chars:
            notes.estimated_token_count = current_chars // _CHARS_PER_TOKEN
            return

        # Truncate the worklog (oldest entries) — keep the most recent
        # entries within the budget.
        worklog = notes.sections["worklog"]
        other_chars = current_chars - len(worklog)
        budget_for_worklog = max(0, cap_chars - other_chars)
        if len(worklog) > budget_for_worklog:
            # Keep the tail (most recent entries) within budget; clip
            # to a line boundary for readability.
            tail = worklog[-budget_for_worklog:]
            newline_at = tail.find("\n")
            if newline_at >= 0:
                tail = tail[newline_at + 1 :]
            notes.sections["worklog"] = tail
        notes.estimated_token_count = (
            sum(len(text) for text in notes.sections.values()) // _CHARS_PER_TOKEN
        )

    def _exceeds_trigger(self, messages: list[dict[str, Any]]) -> bool:
        estimated = self._estimate_total_tokens(messages)
        return estimated > self._defaults.trigger_token_count

    def _estimate_total_tokens(self, messages: list[dict[str, Any]]) -> int:
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            # Other message components (tool_calls list, etc.) contribute
            # less; the chars-of-content heuristic is intentionally an
            # under-estimate that the operator-tunable trigger absorbs.
        return total_chars // _CHARS_PER_TOKEN

    def _final(
        self, messages: list[dict[str, Any]], applied: list[int]
    ) -> CompactedContext:
        return CompactedContext(
            messages=messages,
            layers_applied=tuple(applied),
            triggered=True,
        )

    def _state_for(self, session_id: str) -> _SessionCompactionState:
        if session_id not in self._per_session:
            self._per_session[session_id] = _SessionCompactionState()
        return self._per_session[session_id]
