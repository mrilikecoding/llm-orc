"""Calibration Gate — result-checking of composed ensembles (ADR-007, WP-H).

Per ``docs/agentic-serving/system-design.md`` §Calibration Gate (L1
Domain Policy) and §Integration Contracts (Orchestrator Tool Dispatch
→ Calibration Gate). ADR-007 commits every composed ensemble to a
calibration period: the first N invocations always run a check that
attaches a Quality Signal to the invocation. Accumulated signals
decide whether the ensemble transitions to ``trusted``. AS-5 forbids
frequency-only transitions.

The gate is L1 — it takes a plain ``session_id`` string on each call so
the module stays free of imports from Session Registry (L3). Per-session
records are held internally, indexed by ``(session_id, ensemble_name)``.
When Plexus lands (WP-I), persistence is added by composing a Plexus-
backed store behind the same surface; stateless mode continues to work
exactly as today (AS-8).

Key invariant for Phase 1: calibration check does **not** prevent
invocation (ADR-007 clause 2). Tool Dispatch calls ``check_and_record``
alongside ``invoke_ensemble``'s return; the signal is recorded but the
result still flows through the Summarizer Harness and back to the
orchestrator. A failing check produces a ``negative`` or ``absent``
signal; it does not raise.

**Cycle 4 extension (WP-F4, ADR-014):** the gate also produces a
dispatch-time calibration verdict trichotomy (Proceed / Reflect /
Abstain) via :meth:`CalibrationGate.verdict_for`. The verdict composes
AUQ verbalized-confidence (System 2 binary threshold; default 0.85
within literature-supported 0.8–1.0 range), HTC trajectory features
(token-level entropy comparison against time-decay-windowed running
mean), and the existing ADR-007 post-hoc result-check signal. The
verdict surface is additive — ADR-007's first-N mechanism is
unchanged, and the two layers compose: post-hoc tracks *whether an
ensemble can be trusted*; in-process tracks *whether a specific
dispatch should proceed right now*.

Per ADR-014 §"Feature-extraction location", trajectory features are
extracted at L0 when ADR-016's read-only signal channel is active and
propagated upward; when ADR-016 is rejected or not yet landed, the
features are extracted in-layer at L1 from the L1-internal trajectory
data the consumer supplies through :class:`DispatchContext`. WP-F4
ships the verdict-producer API; the L0 extraction infrastructure
lands with WP-H4.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

if TYPE_CHECKING:
    from llm_orc.agentic.calibration_signal_channel import (
        CalibrationSignalChannel,
    )

from llm_orc.models.structural_errors import LlmOrcStructuralError

# WP-H4 / ADR-016 conditional-acceptance type-only reference. Two
# Python idioms compose here:
#
# * ``from __future__ import annotations`` (PEP-563) makes ALL
#   annotations in this file deferred — they are stored as strings
#   at module-import time and resolved lazily by ``typing.get_type_
#   hints()`` or by tooling that explicitly inspects annotations.
# * The ``TYPE_CHECKING`` block (above) prevents
#   :class:`CalibrationSignalChannel` from being imported at runtime;
#   mypy sees the import and can resolve the name, but Python's
#   import machinery does not execute the line.
#
# Why both patterns together: the channel module imports
# :data:`CalibrationVerdict` from this file at runtime, which would
# create a circular import if this file imported the channel back
# at runtime. PEP-563 alone would suffice for the constructor's
# annotation, but mypy-strict requires the name to resolve in some
# scope — TYPE_CHECKING is the canonical idiom for that. PEP-563 +
# TYPE_CHECKING is the standard mypy-strict pattern for type-only
# circular references; FC-2's layering check explicitly excludes
# imports nested under ``if TYPE_CHECKING:`` for the same reason
# (no runtime edge, no upward-edge concern).

DEFAULT_CALIBRATION_CHECKER_ENSEMBLE = "agentic-calibration-checker"
"""Default checker ensemble bundled with the library.

Operators override via ``agentic_serving.orchestrator.calibration.\
checker_ensemble`` in ``config.yaml`` to point at a domain-specific
checker. The ensemble's agent reads the composed ensemble's output and
emits ``signal: positive|negative|absent``.
"""

DEFAULT_CALIBRATION_N = 3
"""Default number of invocations required to earn trust.

A WP-H build decision per ADR-007 clause 5. Three balances check cost
against single-invocation noise tolerance — one unlucky reading does
not prevent transition indefinitely; two unlucky readings extend the
calibration window, which is the intent. Operators tune via
``agentic_serving.orchestrator.calibration.default_n``.
"""


QualitySignal = Literal["positive", "negative", "absent"]
"""Per system design §Integration Contracts (Orchestrator Tool Dispatch
→ Calibration Gate). ``positive`` = plausible on-task output;
``negative`` = hallucinated or off-task; ``absent`` = too short, empty,
or not evaluable."""

CalibrationStatus = Literal["in_calibration", "trusted"]


@dataclass
class CalibrationRecord:
    """Per-ensemble calibration state within one Session.

    Mutable so the gate can update it in place — mirrors
    :class:`~llm_orc.agentic.session_registry.SessionState` which is
    mutable for cumulative turn/token accounting. ``signals`` stays an
    immutable tuple so readers cannot inadvertently corrupt the record.
    """

    ensemble_name: str
    status: CalibrationStatus = "in_calibration"
    signals: tuple[QualitySignal, ...] = ()
    invocations_seen: int = 0


class CalibrationChecker(Protocol):
    """Produces a :data:`QualitySignal` for a raw ensemble result.

    The default implementation (:class:`EnsembleBackedChecker`) invokes
    a configured checker ensemble; tests pass scripted doubles. Kept
    narrow — the checker is the one variable the operator tunes when
    calibration quality needs attention.
    """

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal: ...


@dataclass
class _SessionRecords:
    """Per-session bundle of per-ensemble records.

    Exists so the gate's internal store is ``dict[str, _SessionRecords]``
    rather than ``dict[str, dict[str, CalibrationRecord]]`` — the former
    reads as "one bundle per session" at the call site, the latter as
    "two levels of key indirection."
    """

    records: dict[str, CalibrationRecord] = field(default_factory=dict)
    trajectory_window: list[_WindowedDispatchSample] = field(default_factory=list)
    """Time-decay-windowed history of dispatch trajectory samples per
    ADR-014 §"Time-decay windowing on trajectory features". The list is
    pruned at every :meth:`CalibrationGate.verdict_for` call so signals
    outside the dual-bound do not influence current verdicts.
    """


# ---------------------------------------------------------------------------
# Cycle 4 extension (WP-F4, ADR-014) — verdict trichotomy + windowing
# ---------------------------------------------------------------------------


CalibrationVerdict = Literal["proceed", "reflect", "abstain"]
"""Per ADR-014 §Decision: the three-value verdict the Calibration Gate
publishes for every dispatch decision.

* ``proceed`` — confidence is above the System 2 threshold, no
  trajectory anomaly, post-hoc signal is not a hard failure.
* ``reflect`` — confidence is below the System 2 threshold but no
  trajectory anomaly. The consumer (Tier-Escalation Router per ADR-015)
  may route the dispatch to an escalated-tier Model Profile.
* ``abstain`` — a severe-anomaly criterion fires. The consumer raises
  :class:`CalibrationAbstainError` (or transforms to
  ``escalation_bypass`` at the router edge per ADR-015).
"""

DriftLevel = Literal["no_drift", "drift_detected", "severe_drift"]
"""Per ADR-016 §"Mechanism (d)" the periodic out-of-band audit's
verdict trichotomy. ``None`` in :class:`DispatchContext` means the
ADR-016 read-only signal channel is not active for this dispatch
(WP-F4 lands before WP-H4)."""

AbstainCriterion = Literal[
    "entropy_collapse",
    "post_hoc_hard_failure",
    "severe_drift",
]
"""The three Abstain criteria per ADR-014 §Decision and system-design
.agents.md §Module: Calibration Gate."""


DEFAULT_AUQ_CONFIDENCE_THRESHOLD: Final[float] = 0.85
"""AUQ System 2 binary-gate threshold per ADR-014 §Decision.

The literature-supported range is 0.8–1.0 (arXiv:2601.15703). The
default of 0.85 is drafting-time synthesis within the range per ADR-014
§Provenance check; Chuang et al. (arXiv:2502.04428) establish that
uncertainty-quantification *method* choice dominates threshold choice
as a design lever, so this default is operational tuning territory
rather than a load-bearing architectural decision.
"""

DEFAULT_TRAJECTORY_WINDOW_MINUTES: Final[float] = 60.0
"""Time-decay windowing primary bound per ADR-014 §Decision (60-minute
/ 100-signal dual-bound, whichever shorter)."""

DEFAULT_TRAJECTORY_WINDOW_DISPATCHES: Final[int] = 100
"""Time-decay windowing secondary bound per ADR-014 §Decision."""

DEFAULT_ENTROPY_COLLAPSE_SIGMA: Final[float] = 1.5
"""Entropy-collapse Abstain criterion — token-level entropy in the most
recent N tokens drops > 1.5σ below the trajectory's running mean per
ADR-014 §Decision and scenarios.md §"Abstain verdict blocks dispatch"."""

_SECONDS_PER_MINUTE: Final[float] = 60.0


@dataclass(frozen=True)
class CalibrationVerdictEvent:
    """Routed event carrying one :data:`CalibrationVerdict` per ADR-023.

    The :data:`CalibrationVerdict` literal carries only the verdict value;
    routing it to either ADR-023 destination requires call-site composition
    (per ADR-023 §"`CalibrationVerdict` call-site composition"). This event
    wraps the verdict with the ``dispatch_id`` correlation identifier and
    the call-site context (ensemble name, timestamp) so operator-terminal
    and orchestrator-context sinks can format and structure the verdict
    without a cross-stream join.

    The verdict :data:`CalibrationVerdict` literal is unchanged — code that
    consumes verdicts as values continues to use the literal. This event is
    the *emission* form on the Dispatch Event Substrate.
    """

    verdict: CalibrationVerdict
    ensemble_name: str
    timestamp_seconds: float
    dispatch_id: str | None = None
    """ADR-023 correlation identifier (Cycle 6 WP-A — additive). ``None``
    during the progressive conversion when the gate's caller does not yet
    pass the substrate's allocated identifier."""


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Per-dispatch HTC-derived trajectory features per ADR-014.

    WP-F4 ships the verdict-producer API with this single load-bearing
    feature; the L0 extraction infrastructure (token probabilities,
    attention-weight access) lands with WP-H4 when ADR-016's signal
    channel becomes operative. Until then, the consumer (Tool Dispatch
    or Tier-Escalation Router) supplies whatever entropy estimate is
    available; ``None`` is honest about feature unavailability and
    causes the entropy-collapse Abstain criterion to skip rather than
    misfire.
    """

    recent_token_entropy: float | None = None
    """Token-level entropy of the dispatched ensemble's most recent N
    tokens (HTC feature). Compared against the time-decay-windowed
    running mean of trajectory entropy by the entropy-collapse Abstain
    criterion."""


@dataclass(frozen=True)
class DispatchContext:
    """Inputs to :meth:`CalibrationGate.verdict_for` per ADR-014.

    The consumer assembles this at dispatch time. ``None`` values
    degrade gracefully — a missing AUQ confidence does not produce
    Reflect (the consumer has no evidence to escalate on), and a
    missing entropy reading does not trigger Abstain. The verdict
    trichotomy continues to function under partial-feature
    availability per the system-design fitness criterion
    ``test_verdict_computation_works_without_signal_channel``.
    """

    auq_confidence: float | None = None
    """Verbalized confidence emitted by the dispatched ensemble's
    component agents (AUQ; arXiv:2601.15703). Range 0.0–1.0. ``None``
    when the ensemble does not emit verbalized confidence — verdict
    defaults to Proceed in that case (no Reflect without evidence)."""

    trajectory_features: TrajectoryFeatures = field(default_factory=TrajectoryFeatures)
    """HTC trajectory features per ADR-014 §"Quality signal composition"."""

    post_hoc_hard_failure: bool = False
    """Set when the consumer determines ADR-007's first-N result-check
    produced a non-recoverable-error outcome per ADR-014 §"Calibration
    verdict" Abstain criterion 2. Distinct from a routine ``negative``
    :data:`QualitySignal` — hard failure is the severe class."""

    drift_verdict: DriftLevel | None = None
    """ADR-016 mechanism (d) periodic audit verdict. ``None`` means the
    cross-layer signal channel is not active (WP-F4 lands before
    WP-H4); ``severe_drift`` triggers the third Abstain criterion."""

    dispatch_timestamp_seconds: float | None = None
    """Wall-clock dispatch timestamp for time-decay windowing. ``None``
    defaults to ``time.time()`` at call time — tests pass an explicit
    value to drive deterministic windowing assertions."""


@dataclass(frozen=True)
class _WindowedDispatchSample:
    """Per-dispatch entry retained in the time-decay window."""

    timestamp_seconds: float
    recent_token_entropy: float | None


class CalibrationAbstainError(LlmOrcStructuralError):
    """Raised when a consumer observes an Abstain verdict per ADR-014.

    Fifth concrete subclass of :class:`LlmOrcStructuralError` per FC-17
    (after ``ToolCallingNotSupportedError``, ``PhantomToolCallError``,
    ``WriteGateRejectionError``, and ``CompactionLayer4FailureError``).
    ``error_kind`` is fixed by construction; ``recovery_action_required``
    is always ``abstain`` because the orchestrator must take a
    different action (reformulate, dispatch elsewhere, or abstain
    entirely) — per ADR-014 §Decision and the four-value Literal in
    :mod:`llm_orc.models.structural_errors`.

    The Tier-Escalation Router (WP-G4-1, ADR-015) transforms Abstain
    verdicts to ``escalation_bypass`` at the router edge; this class
    is the in-layer surface for consumers that bypass the router.
    """

    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        ensemble_name: str,
        criterion: AbstainCriterion,
    ) -> None:
        super().__init__(
            message,
            error_kind="calibration_abstain",
            recovery_action_required="abstain",
            dispatch_context={
                "session_id": session_id,
                "ensemble_name": ensemble_name,
                "criterion": criterion,
            },
            operator_diagnostic=message,
        )


class WallClock(Protocol):
    """Wall-clock surface for time-decay windowing timestamps.

    Tests pass a controllable clock; production wiring uses
    :class:`_SystemWallClock`. Per ADR-014's time-decay windowing the
    timestamps are wall-clock seconds (not monotonic) so the 60-minute
    window aligns with operator-readable session duration.
    """

    def now_seconds(self) -> float: ...


class _SystemWallClock:
    def now_seconds(self) -> float:
        return time.time()


class CalibrationGate:
    """Tracks Calibration state and runs Quality-Signal checks.

    Construction takes ``default_n`` (the calibration window) and a
    :class:`CalibrationChecker`. The gate owns per-session state
    internally; the Session Registry stays agnostic. WP-I layers a
    Plexus-backed store behind the same surface for cross-session trust.

    **Cycle 4 extension (WP-F4, ADR-014):** ``auq_confidence_threshold``,
    ``trajectory_window_minutes``, ``trajectory_window_dispatches``, and
    ``entropy_collapse_sigma`` are operationally-tunable parameters for
    the verdict-producer surface. Defaults match ADR-014 §Decision.
    """

    def __init__(
        self,
        *,
        default_n: int = DEFAULT_CALIBRATION_N,
        checker: CalibrationChecker,
        auq_confidence_threshold: float = DEFAULT_AUQ_CONFIDENCE_THRESHOLD,
        trajectory_window_minutes: float = DEFAULT_TRAJECTORY_WINDOW_MINUTES,
        trajectory_window_dispatches: int = DEFAULT_TRAJECTORY_WINDOW_DISPATCHES,
        entropy_collapse_sigma: float = DEFAULT_ENTROPY_COLLAPSE_SIGMA,
        clock: WallClock | None = None,
        signal_channel: CalibrationSignalChannel | None = None,
    ) -> None:
        if default_n < 1:
            raise ValueError(
                f"Calibration Gate default_n must be >= 1, got {default_n}"
            )
        if not 0.0 <= auq_confidence_threshold <= 1.0:
            raise ValueError(
                "Calibration Gate auq_confidence_threshold must be in "
                f"[0.0, 1.0], got {auq_confidence_threshold}"
            )
        if trajectory_window_minutes <= 0.0:
            raise ValueError(
                "Calibration Gate trajectory_window_minutes must be > 0, "
                f"got {trajectory_window_minutes}"
            )
        if trajectory_window_dispatches < 1:
            raise ValueError(
                "Calibration Gate trajectory_window_dispatches must be >= 1, "
                f"got {trajectory_window_dispatches}"
            )
        if entropy_collapse_sigma <= 0.0:
            raise ValueError(
                "Calibration Gate entropy_collapse_sigma must be > 0, "
                f"got {entropy_collapse_sigma}"
            )
        self._default_n = default_n
        self._checker = checker
        self._auq_confidence_threshold = auq_confidence_threshold
        self._trajectory_window_minutes = trajectory_window_minutes
        self._trajectory_window_dispatches = trajectory_window_dispatches
        self._entropy_collapse_sigma = entropy_collapse_sigma
        self._clock: WallClock = clock if clock is not None else _SystemWallClock()
        self._sessions: dict[str, _SessionRecords] = {}
        # WP-H4 / ADR-016: optional cross-layer calibration channel.
        # When present, the gate consumes the channel's windowed
        # features at verdict time and reports verdict outcomes back
        # to the channel's audit (mechanism (d)). When None (default —
        # the inactive-ADR-016 case), the gate operates on L1-internal
        # trajectory data per the scenarios.md §"ADR-016 not active"
        # scenario.
        self._signal_channel = signal_channel

    def mark_composed(self, *, session_id: str, ensemble_name: str) -> None:
        """Register a newly composed ensemble for calibration.

        Called by Tool Dispatch after a successful ``compose_ensemble``
        write. Idempotent: if the ensemble is already tracked, no state
        is reset — accidental double-calls from a retry path do not
        discard accumulated signals.
        """
        bundle = self._sessions.setdefault(session_id, _SessionRecords())
        bundle.records.setdefault(
            ensemble_name, CalibrationRecord(ensemble_name=ensemble_name)
        )

    def status(self, *, session_id: str, ensemble_name: str) -> CalibrationStatus:
        """Return the calibration status for ``ensemble_name`` in the session.

        Untracked ensembles (library entries, or any ensemble the
        orchestrator did not compose within this session) return
        ``trusted`` — calibration applies only to composed ensembles.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return "trusted"
        record = bundle.records.get(ensemble_name)
        if record is None:
            return "trusted"
        return record.status

    def record_for(
        self, *, session_id: str, ensemble_name: str
    ) -> CalibrationRecord | None:
        """Return the underlying record; ``None`` if the ensemble is untracked.

        Tests read this; production paths should not need to — the gate's
        public API (``status``, ``check_and_record``) is the intended
        surface for Tool Dispatch.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return None
        return bundle.records.get(ensemble_name)

    async def check_and_record(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        raw_result: dict[str, Any],
    ) -> QualitySignal | None:
        """Run the check on in-calibration ensembles; record the signal.

        Returns the Quality Signal produced for this invocation, or
        ``None`` if the ensemble is ``trusted`` (untracked or previously
        cleared). Per ADR-007 clause 2 the gate never prevents
        invocation — a negative signal is recorded alongside the return,
        it does not raise.

        After recording, the record's status is recomputed from the
        last-N signals. A negative or absent signal in the last-N window
        keeps the ensemble in calibration even after more than N
        invocations — the "calibration period extends" semantic from
        scenario §Calibration fails to clear.
        """
        bundle = self._sessions.get(session_id)
        if bundle is None:
            return None
        record = bundle.records.get(ensemble_name)
        if record is None or record.status == "trusted":
            return None

        signal = await self._checker.check(
            ensemble_name=ensemble_name, raw_result=raw_result
        )
        record.signals = record.signals + (signal,)
        record.invocations_seen += 1
        record.status = self._compute_status(record.signals)
        return signal

    def _compute_status(self, signals: tuple[QualitySignal, ...]) -> CalibrationStatus:
        """Decide status from accumulated signals (AS-5).

        Transition to ``trusted`` requires two conditions:

        1. At least ``default_n`` signals have accumulated (not frequency
           alone — AS-5 — but enough evidence to judge).
        2. The most-recent ``default_n`` signals are all ``positive``.

        A negative or absent signal in the last-N window keeps the
        ensemble ``in_calibration`` — subsequent invocations keep
        checking until either a clean run of ``default_n`` positives
        accumulates, or the operator intervenes.
        """
        if len(signals) < self._default_n:
            return "in_calibration"
        last_n = signals[-self._default_n :]
        if all(s == "positive" for s in last_n):
            return "trusted"
        return "in_calibration"

    # -- WP-F4 / ADR-014: verdict-producer surface ------------------------------

    def verdict_for(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        dispatch_context: DispatchContext,
    ) -> CalibrationVerdict:
        """Produce the calibration verdict for this dispatch per ADR-014.

        The verdict is one of three values (``proceed`` / ``reflect`` /
        ``abstain``). The trichotomy is structurally exhaustive given
        the criterion specifications — every dispatch produces exactly
        one verdict (FC-19, per system-design.agents.md §Calibration
        Gate Fitness).

        Decision tree (per ADR-014 §"Calibration verdict"):

        1. If any Abstain criterion fires → ``abstain``.
        2. Else if AUQ confidence is below the System 2 threshold
           (and an entropy value is available — Reflect requires
           evidence of low confidence) → ``reflect``.
        3. Else → ``proceed``.

        Side effect: the dispatch's trajectory features are recorded
        into the per-session time-decay window before the verdict is
        computed. This is the in-layer instance of ADR-014's
        time-decay windowing (the mechanism (b) analog per essay 005's
        bounding-mechanism framing).
        """
        bundle = self._sessions.setdefault(session_id, _SessionRecords())
        timestamp = (
            dispatch_context.dispatch_timestamp_seconds
            if dispatch_context.dispatch_timestamp_seconds is not None
            else self._clock.now_seconds()
        )

        # Prune window before reading running stats so the entropy-
        # collapse comparison sees only in-window history (ADR-014
        # §"Time-decay windowing").
        self._prune_trajectory_window(bundle, now_seconds=timestamp)
        running = self._windowed_running_entropy(bundle)

        # Record this dispatch's sample after computing the running
        # stats — the current dispatch's own entropy must not anchor
        # the criterion it is being compared against (otherwise
        # entropy-collapse trivially never fires).
        bundle.trajectory_window.append(
            _WindowedDispatchSample(
                timestamp_seconds=timestamp,
                recent_token_entropy=(
                    dispatch_context.trajectory_features.recent_token_entropy
                ),
            )
        )
        # Keep the window bounded after the append; the dispatch-count
        # bound prunes oldest entries even if they sit inside the time
        # window.
        self._prune_trajectory_window(bundle, now_seconds=timestamp)

        criterion = self._abstain_criterion(dispatch_context, running)
        if criterion is not None:
            verdict: CalibrationVerdict = "abstain"
        elif self._signal_channel is not None and self._signal_channel.fail_safe_active:
            # ADR-016 §"Mechanism (d)" — severe-drift fail-safe defaults
            # calibration verdicts to Reflect-or-Abstain. The channel's
            # audit declared its own state degraded; the gate's verdict
            # producer honors that synchronously.
            verdict = "reflect"
        elif (
            dispatch_context.auq_confidence is not None
            and dispatch_context.auq_confidence < self._auq_confidence_threshold
        ):
            verdict = "reflect"
        else:
            verdict = "proceed"

        # Mechanism (d) audit feedback. The channel records the
        # verdict-and-feature-snapshot pair into its audit window;
        # diagnostics accumulate for asynchronous operator review.
        if self._signal_channel is not None:
            channel_features = self._signal_channel.windowed_features(
                now_seconds=timestamp,
                ensemble_name=ensemble_name,
            )
            self._signal_channel.record_verdict_outcome(
                verdict=verdict,
                ensemble_name=ensemble_name,
                signal_features=channel_features,
            )

        return verdict

    def abstain_criterion_for(
        self,
        *,
        session_id: str,
        ensemble_name: str,
        dispatch_context: DispatchContext,
    ) -> AbstainCriterion | None:
        """Return the criterion that *would* trigger Abstain for the
        same inputs as :meth:`verdict_for`, or ``None`` if no criterion
        fires.

        Provided for consumers that need to raise
        :class:`CalibrationAbstainError` with the specific criterion
        attached to ``dispatch_context``. The method is side-effect-
        free: it does *not* record the dispatch into the trajectory
        window. Use ``verdict_for`` to drive the dispatch decision and
        this method afterwards to extract the criterion for the typed
        error.
        """
        bundle = self._sessions.get(session_id)
        running = (
            self._windowed_running_entropy(bundle)
            if bundle is not None
            else _RunningEntropy.empty()
        )
        return self._abstain_criterion(dispatch_context, running)

    def _abstain_criterion(
        self,
        dispatch_context: DispatchContext,
        running: _RunningEntropy,
    ) -> AbstainCriterion | None:
        """Evaluate the three Abstain criteria per ADR-014 §"Calibration
        verdict".

        Criterion ordering reflects evaluation cost: drift verdict is a
        single-field check; post-hoc hard failure is a single-field
        check; entropy collapse requires the windowed running mean and
        stdev.
        """
        if dispatch_context.drift_verdict == "severe_drift":
            return "severe_drift"
        if dispatch_context.post_hoc_hard_failure:
            return "post_hoc_hard_failure"
        if self._entropy_collapse_triggers(
            dispatch_context.trajectory_features, running
        ):
            return "entropy_collapse"
        return None

    def _entropy_collapse_triggers(
        self,
        features: TrajectoryFeatures,
        running: _RunningEntropy,
    ) -> bool:
        """Return whether the recent-token-entropy reading is more than
        ``entropy_collapse_sigma`` below the windowed running mean.

        Returns ``False`` when the recent reading is missing or the
        window is empty / degenerate — the criterion needs a
        statistical basis to fire.
        """
        recent = features.recent_token_entropy
        if recent is None:
            return False
        if not running.has_basis:
            return False
        threshold = running.mean - self._entropy_collapse_sigma * running.stdev
        return recent < threshold

    def _prune_trajectory_window(
        self,
        bundle: _SessionRecords,
        *,
        now_seconds: float,
    ) -> None:
        """Drop samples outside the time bound; cap at the dispatch
        bound (whichever bound is tighter wins per ADR-014 §"Time-decay
        windowing")."""
        window_seconds = self._trajectory_window_minutes * _SECONDS_PER_MINUTE
        cutoff = now_seconds - window_seconds
        bundle.trajectory_window = [
            sample
            for sample in bundle.trajectory_window
            if sample.timestamp_seconds >= cutoff
        ]
        max_count = self._trajectory_window_dispatches
        if len(bundle.trajectory_window) > max_count:
            bundle.trajectory_window = bundle.trajectory_window[-max_count:]

    def _windowed_running_entropy(
        self,
        bundle: _SessionRecords | None,
    ) -> _RunningEntropy:
        """Compute linear-decay-weighted mean and stdev of in-window
        entropy samples.

        Weights run from 1.0 (most recent in the window) to 0.0 (window
        edge) per ADR-014 §"Time-decay windowing" and scenarios.md
        §"Time-decay windowing limits trajectory features to dual-bound
        recent window". Samples with ``None`` entropy are skipped. With
        fewer than two valid samples the running stats have no basis
        and :attr:`_RunningEntropy.has_basis` is ``False``.
        """
        if bundle is None or not bundle.trajectory_window:
            return _RunningEntropy.empty()

        samples = [
            sample
            for sample in bundle.trajectory_window
            if sample.recent_token_entropy is not None
        ]
        if len(samples) < 2:
            return _RunningEntropy.empty()

        n = len(samples)
        # Linear-decay weights: most recent sample (index n-1) gets 1.0;
        # window edge (index 0) gets 0.0. With n >= 2 the denominator
        # n - 1 is safe.
        weights = [i / (n - 1) for i in range(n)]
        total_weight = sum(weights)
        if total_weight == 0.0:
            return _RunningEntropy.empty()

        weighted_sum = 0.0
        for sample, weight in zip(samples, weights, strict=True):
            entropy = sample.recent_token_entropy
            assert entropy is not None  # filtered above
            weighted_sum += weight * entropy
        mean = weighted_sum / total_weight

        weighted_sq = 0.0
        for sample, weight in zip(samples, weights, strict=True):
            entropy = sample.recent_token_entropy
            assert entropy is not None  # filtered above
            weighted_sq += weight * (entropy - mean) ** 2
        variance = weighted_sq / total_weight
        stdev = math.sqrt(variance)

        # Zero-stdev windows (all-identical samples) provide no
        # statistical basis for the 1.5σ entropy-collapse test —
        # "more than 1.5 standard deviations below the mean" is
        # undefined when σ=0. Honest about lack of basis rather than
        # collapsing the criterion to a strict mean comparison.
        return _RunningEntropy(mean=mean, stdev=stdev, has_basis=stdev > 0.0)


@dataclass(frozen=True)
class _RunningEntropy:
    """Weighted running mean and stdev over the windowed trajectory.

    ``has_basis`` is ``False`` when fewer than two valid samples exist;
    callers must check this before using ``mean`` / ``stdev``.
    """

    mean: float
    stdev: float
    has_basis: bool

    @classmethod
    def empty(cls) -> _RunningEntropy:
        return cls(mean=0.0, stdev=0.0, has_basis=False)


class CheckerInvoker(Protocol):
    """Narrow facade for invoking the checker ensemble.

    ``OrchestraService`` satisfies this structurally; tests pass a
    handwritten double. Named for intent — the checker itself is an
    ensemble (composed, configured — not coded), invoked with the
    target ensemble's name and a JSON-serialized raw result.
    """

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


_SIGNAL_PATTERN = re.compile(
    r"signal\s*[:=]\s*(positive|negative|absent)\b", re.IGNORECASE
)


class EnsembleBackedChecker:
    """Production :class:`CalibrationChecker` — invokes a checker ensemble.

    The configured ensemble is invoked with ``{"target_ensemble": ...,
    "output": ...}`` as JSON input. The agent's response is parsed for
    ``signal: positive|negative|absent``. An unparseable response yields
    ``absent`` — honest about evaluability rather than silently assuming
    a signal. Invoker exceptions also map to ``absent``; ADR-007
    clause 2's "does not prevent invocation" means a crash here must
    not become a ``ToolCallError``.
    """

    def __init__(
        self,
        *,
        invoker: CheckerInvoker,
        checker_ensemble_name: str = DEFAULT_CALIBRATION_CHECKER_ENSEMBLE,
    ) -> None:
        self._invoker = invoker
        self._checker_ensemble_name = checker_ensemble_name

    async def check(
        self, *, ensemble_name: str, raw_result: dict[str, Any]
    ) -> QualitySignal:
        payload = json.dumps(
            {"target_ensemble": ensemble_name, "output": raw_result},
            default=str,
        )
        try:
            response = await self._invoker.invoke(
                {"ensemble_name": self._checker_ensemble_name, "input": payload}
            )
        except Exception:  # noqa: BLE001 — ADR-007 clause 2: degrade, never fail
            return "absent"
        text = _extract_checker_text(response)
        if text is None:
            return "absent"
        return _parse_signal(text)


def _extract_checker_text(response: dict[str, Any]) -> str | None:
    """Pull the response text from the checker ensemble's output.

    Accepts either a populated ``synthesis`` string or a single-agent
    ``results[agent_name]["response"]`` — the same two-shape tolerance
    :class:`ResultSummarizerHarness` uses, for the same reason:
    operators shape summarizer-style ensembles naturally and the
    dependency-free single-agent case leaves ``synthesis`` unpopulated.
    """
    synthesis = response.get("synthesis")
    if isinstance(synthesis, str) and synthesis:
        return synthesis

    results = response.get("results")
    if isinstance(results, dict) and len(results) == 1:
        only = next(iter(results.values()))
        if isinstance(only, dict):
            agent_response = only.get("response")
            if isinstance(agent_response, str) and agent_response:
                return agent_response
    return None


def _parse_signal(text: str) -> QualitySignal:
    """Extract a :data:`QualitySignal` from free-form checker output.

    Tolerates surrounding prose — the checker LLM may preface the
    signal with reasoning. A match must appear somewhere in the text
    with the canonical ``signal: <value>`` shape. An unrecognised
    response yields ``absent``.
    """
    match = _SIGNAL_PATTERN.search(text)
    if match is None:
        return "absent"
    value = match.group(1).lower()
    if value == "positive":
        return "positive"
    if value == "negative":
        return "negative"
    return "absent"
