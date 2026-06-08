"""Delegation Rate Meter (Cycle 7 loop-back #3, ADR-036 §Decision 3) — L2.

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Delegation
Rate Meter. The regression-visibility mechanism for the stack-scoped
delegation win: the win is a (composition × qwen3:14b × OpenCode 1.15.5)
property that Spike ψ′ Arm D showed does not transfer across models, so the
meter is the safety net that makes a regression observable rather than silent.

Two surfaces, both pure:

* :func:`classify_turn` — the deterministic turn-shape rule graduated from
  Spike ψ.4a (``scratch/spike-psi-delegation-rate/psi4a_prefilter.py``). It
  classifies a turn from raw inputs (the driving message + values observed
  earlier in the conversation + the registered capability domains), with no
  LLM call. Generation-shaped turns are the delegation-rate denominator;
  boundary turns (repair-shaped, or generation-shaped in an uncovered
  domain) are excluded from the denominator and never guessed into it.
* :func:`delegation_rate` — the rate over a stream of turn events, read
  through a narrow structural protocol (``turn_shape`` + ``delegated_ensemble``)
  with no access to request content, logs, or replay (FC-59 "events alone").

The module imports nothing from the Loop Driver or the sinks — ``classify_turn``
is a pure function over raw inputs and :func:`delegation_rate` consumes events
structurally. This is what keeps the Loop Driver → Meter edge cycle-free.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

__all__ = [
    "TurnShape",
    "RateReading",
    "classify_turn",
    "delegation_rate",
    "CAPABILITY_DOMAINS",
    "domains_for",
]

TurnShape = Literal["generation", "carry", "boundary_excluded"]
"""A turn's delegation-rate shape (FC-59).

``generation`` — a fresh content-producing turn in a covered capability
domain; the delegation-rate denominator. ``carry`` — a read, command,
literal, or observed-value passthrough; not a generation opportunity.
``boundary_excluded`` — a repair-shaped turn or a generation-shaped turn in
an uncovered domain; excluded from the denominator, the exclusion share
itself observable (the denominator-degradation signal).
"""


# --- The Spike ψ.4a rule, graduated verbatim (the validated patterns) -------
_GENERATION_VERBS = (
    r"\b(write|create|implement|build|generate|add|refactor|compose|draft)\b"
)
_CONTENT_OBJECTS = (
    r"\b(module|file|function|class|script|test|tests|code|doc|docs|"
    r"documentation|readme|notes|analysis|report|essay)\b"
)
_OBSERVED_CARRY_MARKERS = (
    r"\b(exactly this|the following content|the output above|the observed|"
    r"that you just|from the previous|the result above|verbatim)\b"
)
_READ_ONLY = r"^\s*(read|show|list|cat|display|what is in|open)\b"
_COMMAND_SHAPED = r"^\s*(run|execute|invoke)\b|\b(run the tests|run pytest|npm test)\b"
# Boundary refinement beyond the spike's binary decide(): repair-shaped turns
# are read-then-generate, multi-step — excluded from the clean-generation
# denominator rather than counted as a delegation opportunity. "debug" is
# deliberately absent: it collides with the literal ``DEBUG`` constant that
# appears in legitimate carry tasks (spike clear case c07), and "fix" carries
# the repair shape in both the spike's labeled set and the scenarios.
_REPAIR_SHAPED = r"\b(fix|repair|troubleshoot)\b"

CAPABILITY_DOMAINS: dict[str, str] = {
    # ``code-generator`` is the shipped substrate-routed capability ensemble;
    # the domain pattern mirrors its declared content capability.
    "code-generator": (
        r"\b(python|code|module|function|class|script|test|tests|refactor)\b"
    ),
}
"""Registered capability ensembles → their content-domain match patterns.

Each registration widens the capability-domain term; a turn whose domain no
registered capability covers is ``boundary_excluded`` (the honest
denominator-degradation behavior). Maintained alongside the ensemble
registry; the meter knows only the capabilities listed here.
"""


def domains_for(capabilities: Iterable[str]) -> list[str]:
    """The content-domain patterns for the registered capabilities the meter knows.

    Unknown capability names contribute no pattern — a generation-shaped turn
    the meter cannot match against a known domain classifies as
    ``boundary_excluded``, the observable denominator-degradation signal,
    never a guessed denominator entry.
    """
    return [
        CAPABILITY_DOMAINS[name] for name in capabilities if name in CAPABILITY_DOMAINS
    ]


def classify_turn(
    message: str,
    observed_values: list[str],
    capability_domains: Iterable[str],
) -> TurnShape:
    """Classify a turn's delegation shape from raw inputs (FC-59 denominator).

    ``message`` is the turn's driving instruction; ``observed_values`` are
    values seen earlier in the conversation (prior tool results), feeding the
    observed-carry exclusion; ``capability_domains`` are the content-domain
    patterns of the registered capabilities (see :func:`domains_for`).

    Deterministic — no RNG, no LLM, no clock. The rule:

    1. repair-shaped → ``boundary_excluded`` (multi-step read-then-generate)
    2. read-only / command-shaped without generation → ``carry``
    3. not generation-shaped (no generation verb × content object) → ``carry``
    4. generation-shaped but the content is already observed/literal → ``carry``
    5. generation-shaped and fresh: covered domain → ``generation``;
       uncovered domain → ``boundary_excluded``
    """
    msg = message.lower()

    if re.search(_REPAIR_SHAPED, msg):
        return "boundary_excluded"

    if re.search(_READ_ONLY, msg) and not re.search(_GENERATION_VERBS, msg):
        return "carry"
    if re.search(_COMMAND_SHAPED, msg) and not re.search(
        _GENERATION_VERBS + r".*" + _CONTENT_OBJECTS, msg
    ):
        return "carry"

    generation_shaped = bool(
        re.search(_GENERATION_VERBS, msg) and re.search(_CONTENT_OBJECTS, msg)
    )
    if not generation_shaped:
        return "carry"

    if re.search(_OBSERVED_CARRY_MARKERS, msg):
        return "carry"
    # A literal payload embedded in the message (fenced or long quoted block).
    if "```" in message or re.search(r"['\"].{120,}['\"]", message, re.S):
        return "carry"
    # The exact content to write was observed earlier in the conversation.
    if any(value and value.lower() in msg for value in observed_values):
        return "carry"

    covered = any(re.search(pattern, msg) for pattern in capability_domains)
    return "generation" if covered else "boundary_excluded"


@dataclass(frozen=True)
class RateReading:
    """A delegation-rate reading computed from a turn-event stream (FC-59).

    ``rate`` is ``None`` when no generation-shaped turns have been observed
    yet (an undefined ratio, not zero). ``boundary_excluded`` is part of the
    reading — the denominator-degradation signal is observable, not internal.
    """

    rate: float | None
    delegated: int
    generation_turns: int
    boundary_excluded: int
    considered: int


class _TurnShapeEvent(Protocol):
    """The narrow structural protocol :func:`delegation_rate` reads.

    Any event carrying ``turn_shape`` and ``delegated_ensemble`` satisfies it
    — the meter never imports ``TurnDecision``, keeping the edge cycle-free.
    """

    @property
    def turn_shape(self) -> TurnShape | None: ...

    @property
    def delegated_ensemble(self) -> str | None: ...


def delegation_rate(events: Iterable[_TurnShapeEvent]) -> RateReading:
    """The delegation rate over a turn-event stream — events alone (FC-59).

    Denominator = generation-shaped turns; numerator = generation-shaped
    turns that delegated to a capability ensemble. Boundary-excluded turns
    are counted separately (the degradation signal), never folded into the
    denominator. Windowing is the caller's concern — pass the windowed slice.
    """
    delegated = 0
    generation_turns = 0
    boundary_excluded = 0
    considered = 0
    for event in events:
        considered += 1
        if event.turn_shape == "generation":
            generation_turns += 1
            if event.delegated_ensemble is not None:
                delegated += 1
        elif event.turn_shape == "boundary_excluded":
            boundary_excluded += 1
    rate = (delegated / generation_turns) if generation_turns else None
    return RateReading(
        rate=rate,
        delegated=delegated,
        generation_turns=generation_turns,
        boundary_excluded=boundary_excluded,
        considered=considered,
    )
