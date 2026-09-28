"""Engine-owned node outcome (fail-closed-composition addendum 2026-09-23).

One authoritative field, ``outcome``, describes what happened to a node:
did it run and succeed, run and fail, get skipped because nothing
upstream of it survived, get skipped by an intentional ``when:`` guard,
or run only because ``on_dependency_failure: run`` waived rule 1 and
produced a deliverable anyway? Three independent review rounds each
found a call site inferring this from some overlapping combination of
``status``, whether ``reason`` was set, ``handled_failure``, and a
merged script payload. This module is the one place that inference
happens now; every other reader (``GuardEvaluator``,
``terminal_failure_summary``, ``DependencyResolver``,
``FanOutCoordinator``, the phase/skip-record builders in
``EnsembleExecutor``, ...) asks ``is_ok``/``is_blocking``/``is_neutral``
instead of re-deriving it from raw fields.

The engine stamps ``outcome`` explicitly onto every node result it
builds (``stamp_outcome``). A result that predates stamping, or one
hand-built by a test/caller without an explicit ``outcome`` key, is
still classified correctly by ``compute_outcome`` from its legacy
``status``/``reason``/``handled_failure`` fields — the fallback exists
so existing callers and fixtures do not all need an ``outcome`` key on
day one, not as a second source of truth: once a record carries an
explicit ``outcome``, that value is authoritative and wins over
whatever its other fields might otherwise imply.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    """The six things that can happen to a node (addendum table)."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED_BY_FAILURE = "skipped_by_failure"
    SKIPPED_BY_GUARD = "skipped_by_guard"
    HANDLED_FAILURE = "handled_failure"


#: ok = succeeded | partial (a gathered fan-out with >=1 instance ok).
OK_OUTCOMES = frozenset({Outcome.SUCCEEDED, Outcome.PARTIAL})
#: blocking = failed | skipped_by_failure | handled_failure.
BLOCKING_OUTCOMES = frozenset(
    {Outcome.FAILED, Outcome.SKIPPED_BY_FAILURE, Outcome.HANDLED_FAILURE}
)
#: neutral = skipped_by_guard.
NEUTRAL_OUTCOMES = frozenset({Outcome.SKIPPED_BY_GUARD})


def _attr(result: Any, key: str) -> Any:
    """Read ``key`` from a plain dict or a result-like object (some
    readers, e.g. ``GuardEvaluator``, see ``AgentResult``-shaped objects
    as well as dicts)."""
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def compute_outcome(result: Any) -> Outcome | None:
    """Classify a node result from its legacy ``status``/``reason``/
    ``handled_failure`` fields.

    ``None`` for a shape the engine never produces (not a dict/result-
    like object, or a ``status`` outside the recognized set) — every
    predicate below treats ``None`` as neither ok, blocking, nor
    neutral, matching how an absent/garbage result already read before
    this module existed.
    """
    if _attr(result, "handled_failure"):
        return Outcome.HANDLED_FAILURE
    status = _attr(result, "status")
    if status == "success":
        return Outcome.SUCCEEDED
    if status == "partial":
        return Outcome.PARTIAL
    if status == "failed":
        return Outcome.FAILED
    if status == "skipped":
        reason = _attr(result, "reason")
        return (
            Outcome.SKIPPED_BY_FAILURE
            if reason is not None
            else Outcome.SKIPPED_BY_GUARD
        )
    return None


def outcome_of(result: Any) -> Outcome | None:
    """The node's outcome: the stamped value when present and
    recognized, else derived from legacy fields (``compute_outcome``)
    for a hand-built or pre-stamping result."""
    stamped = _attr(result, "outcome")
    if isinstance(stamped, Outcome):
        return stamped
    if isinstance(stamped, str):
        try:
            return Outcome(stamped)
        except ValueError:
            pass
    return compute_outcome(result)


def is_ok(result: Any) -> bool:
    """True when the node counts as a succeeded terminal/dependency."""
    return outcome_of(result) in OK_OUTCOMES


def is_blocking(result: Any) -> bool:
    """True when the node's lack of forward progress should read as a
    real failure to a cascade check, a terminal-success rule, or
    ``has_errors``."""
    return outcome_of(result) in BLOCKING_OUTCOMES


def is_neutral(result: Any) -> bool:
    """True for an intentional ``when:``-false skip: neither ok nor
    blocking."""
    return outcome_of(result) in NEUTRAL_OUTCOMES


def stamp_outcome(record: dict[str, Any]) -> dict[str, Any]:
    """Attach ``outcome`` and ``has_errors`` to ``record`` in place, and
    return it.

    The one function every stamping site (LLM/script/ensemble/dispatch/
    loop dispatch, fan-out gather, guard skip, cascade skip, input_key
    contract failure) funnels through, so ``outcome`` means the same
    thing everywhere it is set. ``has_errors`` here reflects only the
    node's OWN outcome; a child-execution node's inner subtree errors
    are folded in separately (``child_has_errors``) once its response is
    known, since a ``succeeded`` ``ensemble:``/``dispatch:``/``loop:``
    node can still wrap a subtree that had an intermediate failure.
    """
    outcome = compute_outcome(record)
    record["outcome"] = outcome.value if outcome is not None else None
    record["has_errors"] = outcome in BLOCKING_OUTCOMES
    return record


def _parsed(response: Any) -> Any:
    """``response`` parsed from JSON when it is a string, else returned
    as-is."""
    if not isinstance(response, str):
        return response
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None


def child_has_errors(response: Any) -> bool:
    """Whether a child execution's own subtree had a blocking outcome
    anywhere in it — read from a raw ``ensemble:``/``dispatch:`` child
    result (``EnsembleExecutor.execute``'s own dict, str or already
    parsed) or a ``loop:`` node's wrapped ``{output, iterations,
    terminated, has_errors}`` response.

    Prefers the child's own ``has_errors`` field; falls back to its
    ``status`` (``completed_with_errors``) for a child result recorded
    before this addendum added the field explicitly. ``False`` for
    anything that doesn't parse to a dict.
    """
    parsed = _parsed(response)
    if not isinstance(parsed, dict):
        return False
    if "has_errors" in parsed:
        return bool(parsed["has_errors"])
    return parsed.get("status") == "completed_with_errors"
