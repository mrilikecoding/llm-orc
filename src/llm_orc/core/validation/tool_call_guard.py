"""Tool-call structural validation guard (ADR-017, WP-C4).

Per ``docs/serving.md`` §Orchestrator Tool
Dispatch (extended in Cycle 4 per ADR-017). The guard is interposed
between the orchestrator's response and the dispatch path: it scans
response text for tool-call assertion patterns and rejects responses
that claim a tool call occurred without emitting a corresponding
tool-call structure.

The default pattern set is **minimal rather than calibrated** (per
ADR-017 §"Minimal default pattern set with operator-extension
surface"); operators extend the set as deployment evidence accumulates.
The conservative false-positive discipline (ADR-017 rejected
alternative (f)) excludes future-intent patterns ("I will call X")
because pre-call narration may legitimately precede the structure on
the next turn.
"""

from __future__ import annotations

import re
from typing import Any

from llm_orc.models.structural_errors import LlmOrcStructuralError


class PhantomToolCallError(LlmOrcStructuralError):
    """Raised when prose claims a tool call but no structure was emitted.

    Fourth concrete subclass of :class:`LlmOrcStructuralError` per
    ADR-017 §"Shared typed-error base class" and FC-17. The
    discriminator ``error_kind="phantom_tool_call"`` and the
    disposition ``recovery_action_required="reformulate"`` are fixed
    by construction.

    The constructor's ``detected_prose_claim`` and
    ``emitted_tool_call_names`` arguments populate ``dispatch_context``
    so the orchestrator's reformulated response can incorporate the
    structural feedback per ADR-017 §Rejection.
    """

    def __init__(
        self,
        message: str,
        *,
        detected_prose_claim: str,
        emitted_tool_call_names: tuple[str, ...],
        dispatch_context: dict[str, Any] | None = None,
        operator_diagnostic: str | None = None,
    ) -> None:
        merged_context: dict[str, Any] = {
            "detected_prose_claim": detected_prose_claim,
            "emitted_tool_call_names": emitted_tool_call_names,
        }
        if dispatch_context:
            merged_context.update(dispatch_context)
        super().__init__(
            message,
            error_kind="phantom_tool_call",
            recovery_action_required="reformulate",
            dispatch_context=merged_context,
            operator_diagnostic=operator_diagnostic,
        )


DEFAULT_ASSERTION_PATTERNS: tuple[str, ...] = (
    r"\bthe tool returned\b",
    r"\bthe tool call returned\b",
    r"\bthe tool call has been made\b",
    r"\bI called\b[^.]*\band the result was\b",
    r"\bthe result of\b[^.]*\bis displayed above\b",
    r"\bthe result is displayed above\b",
    r"\bas the observation above shows\b",
    r"\bafter running\b",
    r"\bafter invoking\b",
)
"""Default assertion-pattern set per ADR-017 §Detection.

Each pattern is a case-insensitive regex flagging prose that asserts a
tool call has already occurred. Future-intent phrasings (*"I will call
X"*, *"I am going to invoke X"*, *"Let me call X"*) are deliberately
absent — they describe upcoming action and may legitimately precede
the structure on the next turn (rejected alternative (f)).

The set is intentionally narrow: ADR-017's spike evidence (essay 005
Wave 3.A Trial 3) was confounded by adversarial prompt design and does
not support calibration of a richer default; operator extension is the
operational refinement surface.
"""


def scan_response_for_phantom_claims(
    response_text: str,
    emitted_tool_call_names: tuple[str, ...],
    *,
    dispatch_context: dict[str, Any] | None = None,
    extra_patterns: tuple[str, ...] = (),
) -> PhantomToolCallError | None:
    """Scan an orchestrator response for phantom tool-call claims.

    The structural correspondence check (ADR-017 §Validation): if any
    tool-call structure was emitted in the same response, the response
    passes regardless of prose content — at least one structural
    anchor exists. Otherwise, scan the prose against the union of
    :data:`DEFAULT_ASSERTION_PATTERNS` and ``extra_patterns``; the
    first matching assertion produces a :class:`PhantomToolCallError`
    naming the matched substring and the (empty) emitted set.

    ``extra_patterns`` is the operator-extension surface (per ADR-017
    §"Minimal default pattern set with operator-extension surface");
    deployment configuration routes operator additions here via
    :class:`OrchestratorConfig` (L3).

    ``dispatch_context`` carries caller-supplied identifiers (session
    id, turn) that flow through to the produced error so operators
    can trace rejections in operator-readable diagnostics.
    """
    if emitted_tool_call_names:
        return None
    all_patterns = DEFAULT_ASSERTION_PATTERNS + tuple(extra_patterns)
    for pattern in all_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return PhantomToolCallError(
                "orchestrator response asserted a tool call without "
                "emitting a corresponding tool-call structure",
                detected_prose_claim=match.group(0),
                emitted_tool_call_names=emitted_tool_call_names,
                dispatch_context=dispatch_context,
            )
    return None
