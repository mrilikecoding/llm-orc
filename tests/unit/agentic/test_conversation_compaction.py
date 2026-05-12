"""Tests for Conversation Compaction five-layer pipeline (WP-E4, ADR-012).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Conversation Compaction.

The pipeline applies five layers in strict cheapest-first order at every
turn boundary; Layer N+1 fires only if Layers 0..N together cannot bring
context below the configured trigger threshold. Layer 4 (LLM-summary)
carries a per-session circuit-breaker that suspends Layer 4 after three
consecutive failures and raises a typed
:class:`CompactionLayer4FailureError` with
``recovery_action_required="operator_intervention_required"`` per
ADR-012 §Decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_orc.agentic.conversation_compaction import (
    CompactedContext,
    CompactionDefaults,
    CompactionLayer4FailureError,
    ConversationCompaction,
    SummarizerInvocation,
)
from llm_orc.models.structural_errors import LlmOrcStructuralError

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _defaults(
    *,
    persist_threshold_chars: int = 50_000,
    idle_window_minutes: int = 60,
    session_notes_token_cap: int = 12_288,
    layer_4_circuit_breaker_threshold: int = 3,
    trigger_token_count: int = 100_000,
    summarizer_ensemble: str | None = None,
) -> CompactionDefaults:
    return CompactionDefaults(
        persist_threshold_chars=persist_threshold_chars,
        idle_window_minutes=idle_window_minutes,
        session_notes_token_cap=session_notes_token_cap,
        layer_4_circuit_breaker_threshold=layer_4_circuit_breaker_threshold,
        trigger_token_count=trigger_token_count,
        summarizer_ensemble=summarizer_ensemble,
    )


def _message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    """Build a minimal message dict matching the runtime's wire shape."""
    msg: dict[str, Any] = {"role": role, "content": content}
    msg.update(extra)
    return msg


def _tool_result(tool_call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


class _FakeClock:
    """Monotonic stub for idle-expiry timestamps."""

    def __init__(self, *, start_seconds: float = 0.0) -> None:
        self._now = start_seconds

    def now_seconds(self) -> float:
        return self._now

    def advance_minutes(self, minutes: float) -> None:
        self._now += minutes * 60.0


class _ScriptedSummarizer:
    """Per-call scripted summarizer for Layer 4 tests.

    Each entry in ``script`` is either:

    * a string — the next call returns this as the summary
    * an exception instance — the next call raises this

    Tracks invocations so tests can verify Layer 4 fired exactly once
    per ``compact()`` call.
    """

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.invocations: list[SummarizerInvocation] = []

    def summarize(self, invocation: SummarizerInvocation) -> str:
        self.invocations.append(invocation)
        if not self._script:
            raise AssertionError("scripted summarizer exhausted")
        next_value = self._script.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value


# ---------------------------------------------------------------------------
# G1: Pipeline scaffold + FC-14 layer ordering
# ---------------------------------------------------------------------------


class TestCompactionPipelineScaffold:
    """The compact() method returns a CompactedContext; below-threshold
    inputs short-circuit (no layers fire)."""

    def test_below_threshold_input_returns_untouched(self, tmp_path: Path) -> None:
        """Compaction does not trigger when context is below the
        configured trigger token count — no layers fire and the
        messages are returned untouched."""
        compaction = ConversationCompaction(
            defaults=_defaults(trigger_token_count=1_000_000),
            persistence_root=tmp_path,
        )
        messages = [
            _message("user", "hello"),
            _message("assistant", "hi"),
        ]

        result = compaction.compact(messages, session_id="s1")

        assert isinstance(result, CompactedContext)
        assert result.triggered is False
        assert result.layers_applied == ()
        assert result.messages == messages

    def test_compaction_does_not_mutate_input_messages(self, tmp_path: Path) -> None:
        """The caller's messages list is not mutated. The returned
        list is a fresh list (possibly with fresh dicts when a layer
        rewrites entries)."""
        compaction = ConversationCompaction(
            defaults=_defaults(trigger_token_count=10),
            persistence_root=tmp_path,
        )
        messages = [_message("user", "x" * 1000)]
        snapshot = [dict(m) for m in messages]

        compaction.compact(messages, session_id="s1")

        assert messages == snapshot


class TestLayerOrdering:
    """FC-14: layers fire in 0 → 1 → 2 → 3 → 4 order; Layer N+1 fires
    only if Layers 0..N together cannot bring context below threshold."""

    def test_layer_0_alone_reduces_short_circuits_pipeline(
        self, tmp_path: Path
    ) -> None:
        """When Layer 0 alone brings context below threshold, Layers
        1–4 do not fire.

        The trigger budget is sized large enough to absorb Layer 0's
        2 KB preview envelope but smaller than the original tool
        result — so the post-Layer-0 message array fits and the
        pipeline short-circuits before reaching Layer 1.
        """
        compaction = ConversationCompaction(
            defaults=_defaults(
                # Layer 0 persists tool results above 100 chars.
                persist_threshold_chars=100,
                # 2,800 chars (700 tokens) — accommodates the 2 KB
                # preview envelope (~2,200 chars) without leaking
                # into Layer 1.
                trigger_token_count=700,
                summarizer_ensemble="never-invoked",
            ),
            persistence_root=tmp_path,
            summarizer=_ScriptedSummarizer(["should not run"]),
        )
        # One large tool result that Layer 0 will persist.
        big_payload = "x" * 50_000
        messages = [
            _message("user", "do thing"),
            _tool_result("tc1", big_payload),
        ]

        result = compaction.compact(messages, session_id="s1")

        assert result.triggered is True
        assert result.layers_applied == (0,)

    def test_layer_4_fires_only_after_layers_0_3_attempted(
        self, tmp_path: Path
    ) -> None:
        """When Layers 0–3 together cannot bring context below
        threshold, Layer 4 dispatches the summarizer. Layers 0–3 all
        show as applied in the trace."""
        summarizer = _ScriptedSummarizer(["compact summary"])
        compaction = ConversationCompaction(
            defaults=_defaults(
                # Force Layers 0-3 to be unable to reduce enough.
                persist_threshold_chars=10_000_000,  # nothing triggers Layer 0
                trigger_token_count=10,
                summarizer_ensemble="conversation-summarizer",
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        # Many large messages so the aggregate exceeds threshold and
        # no single tool result exceeds the persist threshold.
        messages = [
            _message("user", "hello " * 1_000),
            _message("assistant", "world " * 1_000),
            _message("user", "more " * 1_000),
        ]

        result = compaction.compact(messages, session_id="s1")

        assert result.triggered is True
        # Layers 0–3 are attempted; Layer 4 ultimately reduces.
        assert 4 in result.layers_applied
        # Ordering invariant: 0 ≤ 1 ≤ 2 ≤ 3 ≤ 4 in the trace.
        assert list(result.layers_applied) == sorted(result.layers_applied)
        assert len(summarizer.invocations) == 1


# ---------------------------------------------------------------------------
# G2: Layer 0 — persist-large-tool-results (>50K chars to disk)
# ---------------------------------------------------------------------------


class TestLayer0Persist:
    """Tool results larger than 50K characters persist to disk; the
    orchestrator context receives a 2 KB preview plus the persistent
    path."""

    def test_oversized_tool_result_persists_to_disk(self, tmp_path: Path) -> None:
        """The full payload writes to ``persistence_root`` and the
        in-context message is replaced with a preview-plus-path
        envelope."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=1_000,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        full_payload = "PAYLOAD-CONTENT-" + ("x" * 5_000)
        messages = [
            _message("user", "do thing"),
            _tool_result("tc1", full_payload),
        ]

        result = compaction.compact(messages, session_id="s1")

        # In-context message replaced with envelope.
        envelope = next(m for m in result.messages if m.get("tool_call_id") == "tc1")
        assert envelope["content"] != full_payload
        assert len(envelope["content"]) <= 2_500  # 2 KB preview + small envelope

        # The persisted artifact exists and contains the full payload.
        persisted_files = list(tmp_path.rglob("*"))
        persisted = [p for p in persisted_files if p.is_file()]
        assert persisted, "Layer 0 should persist at least one artifact"
        assert any(full_payload in p.read_text() for p in persisted)

    def test_envelope_records_persistent_path(self, tmp_path: Path) -> None:
        """The envelope MUST surface the persistent path so the
        orchestrator (or operator) can retrieve the full content
        later via existing query channels."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=100,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        full_payload = "z" * 5_000
        messages = [_tool_result("tc1", full_payload)]

        result = compaction.compact(messages, session_id="s1")

        envelope = result.messages[0]
        envelope_content = envelope["content"]
        # The envelope content references a path under persistence_root.
        # Path format is module-private; the contract is "operator-readable
        # reference to the persisted artifact".
        persisted = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert persisted, "expected a persisted artifact"
        assert any(str(p) in envelope_content for p in persisted) or any(
            p.name in envelope_content for p in persisted
        )

    def test_below_threshold_tool_result_is_not_persisted(self, tmp_path: Path) -> None:
        """Tool results at or below the threshold are not persisted —
        they remain in-context unchanged."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=1_000,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        small_payload = "small payload"
        messages = [_tool_result("tc1", small_payload)]

        result = compaction.compact(messages, session_id="s1")

        # No persisted file should appear (or, if Layer 0 created the
        # directory but didn't persist, no .json/.txt artifact contains
        # the payload).
        persisted = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert not any(small_payload in p.read_text() for p in persisted)
        # And Layer 0 should not appear in the layers_applied trace.
        assert 0 not in result.layers_applied


# ---------------------------------------------------------------------------
# G3: Layer 1 — cache-edit (structural placeholder)
# ---------------------------------------------------------------------------


class TestLayer1CacheEdit:
    """Layer 1 cache-edit is a structural placeholder: no underlying
    cache abstraction exists in llm-orc yet. The layer is invocable
    and never invalidates the prefix; in the no-cache substrate it
    reports "did not reduce" so subsequent layers fire."""

    def test_layer_1_preserves_task_anchor_through_pipeline(
        self, tmp_path: Path
    ) -> None:
        """The no-prefix-invalidation property is structurally
        guaranteed for Layer 1 in the no-cache substrate (its body
        does not mutate messages). Layer 3 may legitimately replace
        older history with a notes summary per ADR-012 §Layer 3's
        "free summary" design — the property to preserve is that the
        original user message (the task anchor) survives the pipeline
        so the orchestrator can continue reasoning from the
        load-bearing context."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        task_anchor = _message("user", "stable task description")
        messages = [
            task_anchor,
            _message("assistant", "stable prefix reply"),
            _message("user", "more " * 1_000),
        ]

        result = compaction.compact(messages, session_id="s1")

        # The task anchor — the original user message — survives.
        # Layer 3's summary replaces the middle history but never
        # the anchor.
        assert result.messages[0]["content"] == task_anchor["content"]

    def test_layer_1_is_invocable_and_traced(self, tmp_path: Path) -> None:
        """The layer participates in the pipeline trace when it
        executes — even though the structural no-op cannot reduce the
        context, the pipeline records the attempt so downstream debug
        + cycle-acceptance verification can observe layer flow."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        messages = [_message("user", "x" * 10_000)]

        result = compaction.compact(messages, session_id="s1")

        assert 1 in result.layers_applied


# ---------------------------------------------------------------------------
# G4: Layer 2 — idle-expiry (tool results inactive > 60 min)
# ---------------------------------------------------------------------------


class TestLayer2IdleExpiry:
    """Tool results not touched for the configured idle window expire
    from active context. Recent tool results survive."""

    def test_idle_tool_result_is_cleared(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
            clock=clock,
        )
        # First turn — old_result is seen.
        messages_t0 = [_tool_result("old", "x" * 5_000)]
        compaction.compact(messages_t0, session_id="s1")

        # Advance past the idle window.
        clock.advance_minutes(61)

        # Second turn — old_result is still in messages but recent
        # tool result is fresh.
        messages_t1 = [
            _tool_result("old", "x" * 5_000),
            _tool_result("recent", "y" * 5_000),
        ]
        result = compaction.compact(messages_t1, session_id="s1")

        # "old" tool result is cleared from the active context.
        active_tool_ids = {
            m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"
        }
        assert "recent" in active_tool_ids
        assert "old" not in active_tool_ids
        assert 2 in result.layers_applied

    def test_no_idle_results_means_layer_2_is_a_no_reduction(
        self, tmp_path: Path
    ) -> None:
        """When no tool results have exceeded the idle window, Layer
        2 cannot reduce — Layer 3 must fire next."""
        clock = _FakeClock()
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
            clock=clock,
        )
        messages = [_tool_result("fresh", "z" * 5_000)]

        result = compaction.compact(messages, session_id="s1")

        # Layer 2 doesn't reduce; Layer 3 is reached.
        assert 3 in result.layers_applied


# ---------------------------------------------------------------------------
# G5: Layer 3 — nine-section session notes (zero LLM cost)
# ---------------------------------------------------------------------------


class TestLayer3SessionNotes:
    """The nine-section template updates each turn at zero LLM cost
    and stays bounded by the configured session-notes token cap."""

    def test_session_notes_template_has_nine_sections(self, tmp_path: Path) -> None:
        """Per ADR-012 the template has nine sections (seven named
        plus two reserved for deployment customization)."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        messages = [_message("user", "test " * 2_000)]

        compaction.compact(messages, session_id="s1")
        notes = compaction.session_notes_for("s1")

        assert len(notes.sections) == 9

    def test_layer_3_does_not_dispatch_llm(self, tmp_path: Path) -> None:
        """Layer 3 must update at zero LLM cost. If the summarizer
        were dispatched here it would record an invocation; assert it
        does not."""
        summarizer = _ScriptedSummarizer([])  # would raise if called
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                # Force pipeline to reach Layer 3 but stop before Layer 4
                # by sizing the trigger so Layer 3's update brings it
                # below threshold. Choose a tiny trigger and rely on
                # Layer 3's bookkeeping to reduce.
                session_notes_token_cap=12_288,
                trigger_token_count=10,
                summarizer_ensemble=None,  # no Layer 4 available
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        messages = [_message("user", "test " * 200)]

        compaction.compact(messages, session_id="s1")

        assert summarizer.invocations == []

    def test_session_notes_respect_token_cap(self, tmp_path: Path) -> None:
        """Per ADR-012 the template's token budget caps at the
        configured ``session_notes_token_cap``."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                session_notes_token_cap=256,
                trigger_token_count=10,
            ),
            persistence_root=tmp_path,
        )
        # Lots of input to force the notes to grow.
        for turn in range(20):
            messages = [
                _message("user", f"turn {turn}: " + "noise " * 200),
                _message("assistant", "ok " * 200),
            ]
            compaction.compact(messages, session_id="s1")

        notes = compaction.session_notes_for("s1")
        assert notes.estimated_token_count <= 256


# ---------------------------------------------------------------------------
# G6: Layer 4 — LLM semantic summary, circuit-breaker, session reset
# ---------------------------------------------------------------------------


class TestLayer4LlmSummary:
    """Layer 4 dispatches the summarizer ensemble only when Layers 0–3
    cannot reduce context below threshold. Failures count toward a
    per-session circuit-breaker; the third consecutive failure
    suspends Layer 4 for the session."""

    def test_layer_4_dispatches_summarizer_on_overflow(self, tmp_path: Path) -> None:
        summarizer = _ScriptedSummarizer(["short summary"])
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
                summarizer_ensemble="conversation-summarizer",
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        messages = [_message("user", "lots " * 2_000)]

        result = compaction.compact(messages, session_id="s1")

        assert len(summarizer.invocations) == 1
        assert 4 in result.layers_applied

    def test_layer_4_unconfigured_short_circuits_quietly(self, tmp_path: Path) -> None:
        """When ``summarizer_ensemble`` is ``None``, Layer 4 reports
        "not configured" rather than raising. The circuit-breaker is
        not advanced — operator simply hasn't enabled Layer 4."""
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
                summarizer_ensemble=None,
            ),
            persistence_root=tmp_path,
        )
        messages = [_message("user", "lots " * 2_000)]

        result = compaction.compact(messages, session_id="s1")

        # No raise. Layer 4 may or may not appear in the trace; what
        # matters is that the layers-applied list is well-formed and
        # there's no circuit-breaker trip.
        assert compaction.circuit_breaker_failures_for("s1") == 0
        assert result.triggered is True

    def test_layer_4_failure_increments_circuit_breaker(self, tmp_path: Path) -> None:
        summarizer = _ScriptedSummarizer(
            [RuntimeError("ensemble failed"), "fallback summary"]
        )
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
                summarizer_ensemble="conversation-summarizer",
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        messages = [_message("user", "lots " * 2_000)]

        # First call: Layer 4 fails → counter increments to 1; no raise.
        compaction.compact(messages, session_id="s1")
        assert compaction.circuit_breaker_failures_for("s1") == 1

        # Second call: Layer 4 succeeds → counter resets to 0 (per
        # the "consecutive failures" semantic).
        compaction.compact(messages, session_id="s1")
        assert compaction.circuit_breaker_failures_for("s1") == 0

    def test_third_consecutive_failure_raises_typed_error_and_suspends_layer_4(
        self, tmp_path: Path
    ) -> None:
        """After three consecutive Layer 4 failures the circuit-breaker
        trips: subsequent calls do not re-dispatch the summarizer, and
        the trip itself raises :class:`CompactionLayer4FailureError`
        with ``recovery_action_required="operator_intervention_required"``
        per FC-17 and ADR-012 §Layer 4."""
        summarizer = _ScriptedSummarizer(
            [
                RuntimeError("fail 1"),
                RuntimeError("fail 2"),
                RuntimeError("fail 3"),
                "must-not-be-invoked-after-trip",
            ]
        )
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
                layer_4_circuit_breaker_threshold=3,
                summarizer_ensemble="conversation-summarizer",
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        messages = [_message("user", "lots " * 2_000)]

        # Failures 1 and 2 silently advance the counter (no raise).
        compaction.compact(messages, session_id="s1")
        compaction.compact(messages, session_id="s1")
        assert compaction.circuit_breaker_failures_for("s1") == 2

        # The third consecutive failure trips the breaker: a typed
        # error is raised at the trip moment.
        with pytest.raises(CompactionLayer4FailureError) as exc:
            compaction.compact(messages, session_id="s1")

        error = exc.value
        assert isinstance(error, LlmOrcStructuralError)
        assert error.error_kind == "compaction_layer_4_failure"
        assert error.recovery_action_required == "operator_intervention_required"
        assert compaction.circuit_breaker_suspended_for("s1") is True

        # Subsequent compact() calls do NOT re-dispatch the summarizer
        # while the breaker is tripped.
        before = len(summarizer.invocations)
        try:
            compaction.compact(messages, session_id="s1")
        except CompactionLayer4FailureError:
            pass  # may or may not raise on subsequent calls — what matters
            # is that the summarizer is not re-invoked.
        assert len(summarizer.invocations) == before

    def test_circuit_breaker_resets_at_session_start(self, tmp_path: Path) -> None:
        """``reset_session`` clears the per-session circuit-breaker
        state without operator intervention — the auto-reset
        property argument-audit P3.1 added to ADR-012."""
        summarizer = _ScriptedSummarizer(
            [
                RuntimeError("fail 1"),
                RuntimeError("fail 2"),
                RuntimeError("fail 3"),
            ]
        )
        compaction = ConversationCompaction(
            defaults=_defaults(
                persist_threshold_chars=10_000_000,
                idle_window_minutes=60,
                trigger_token_count=10,
                layer_4_circuit_breaker_threshold=3,
                summarizer_ensemble="conversation-summarizer",
            ),
            persistence_root=tmp_path,
            summarizer=summarizer,
        )
        messages = [_message("user", "lots " * 2_000)]

        compaction.compact(messages, session_id="s1")
        compaction.compact(messages, session_id="s1")
        with pytest.raises(CompactionLayer4FailureError):
            compaction.compact(messages, session_id="s1")
        assert compaction.circuit_breaker_suspended_for("s1") is True

        compaction.reset_session("s1")

        assert compaction.circuit_breaker_failures_for("s1") == 0
        assert compaction.circuit_breaker_suspended_for("s1") is False


# ---------------------------------------------------------------------------
# G6 preservation: AS-7 is independent of Conversation Compaction
# ---------------------------------------------------------------------------


class TestAS7Preservation:
    """ADR-012 §Preservation: Conversation Compaction operates on the
    orchestrator's conversation history; AS-7 Result Summarizer Harness
    operates on ensemble outputs. They compose without one subsuming
    the other.

    The structural shape: Conversation Compaction must not touch the
    Tool Dispatch return-path summarization surface. The structural
    enforcement is the FC-2 layer map and the FC-4 import-surface
    test — at this level the test just verifies the module does NOT
    import the Result Summarizer Harness."""

    def test_module_does_not_import_result_summarizer_harness(self) -> None:
        """Conversation Compaction's import graph must not include the
        AS-7 Harness — they are independent modules per ADR-012
        §Preservation and system-design.agents.md L202."""
        import ast

        import llm_orc.agentic.conversation_compaction as module

        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text())
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)

        assert "llm_orc.agentic.result_summarizer_harness" not in imports
