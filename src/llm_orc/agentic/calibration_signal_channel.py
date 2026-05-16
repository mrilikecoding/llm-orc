"""Calibration Signal Channel — read-only L0→L1 cross-layer channel (WP-H4, ADR-016).

Per ``docs/agentic-serving/system-design.agents.md`` §Module: Calibration
Signal Channel (L1; CONDITIONAL ACCEPTANCE — first-deployment evidence
pending). The channel is the *single narrow exception* to ADR-002's
"edges never upward" layering rule: read-only, calibration-data-only,
structurally typed at the boundary.

Five bounding mechanisms operationalize within the channel module per
ADR-016 §Decision:

* **(a) Fresh-context isolation in the consumer.** The L1 reader
  receives an aggregated :class:`WindowedSignalFeatures` value from
  :meth:`CalibrationSignalChannel.windowed_features`; no signal data
  is carried forward through the consumer's context. Influence on the
  next verdict is only through the time-decay-windowed feature
  aggregation specified by mechanism (b).
* **(b) Time-decay windowing.** Signals contribute to feature
  aggregation with linear weights from 1.0 (most recent) to 0.0
  (window edge). The window is dual-bound by the lesser of
  ``window_minutes`` (default 60.0) or ``window_signals`` (default
  100); whichever bound is tighter wins.
* **(c) Categorical anchors via deterministic-tool-output.** When a
  signal carries a non-``None`` ``deterministic_anchor``, the
  aggregated view exposes the anchor presence and counts; the
  categorical signal anchors the feedback loop against probabilistic
  drift. LLM-only ensembles (no script-model slot) emit
  ``deterministic_anchor=None`` — mechanisms (a), (b), (d), (e)
  remain load-bearing without (c).
* **(d) Periodic out-of-band audit dispatch.** Every ``trigger_count``
  verdict consumptions or ``trigger_wall_clock_hours`` of wall-clock
  time, the audit fires in a fresh context, evaluates three drift
  criteria, and emits a verdict trichotomy (no_drift / advisory /
  severe). Severe drift activates :attr:`fail_safe_active`; the L1
  consumer defaults to Reflect-or-Abstain while fail-safe is active.
* **(e) Read-only structural validation at the consumer.** Incoming
  signals are validated against the typed schema at the channel
  boundary. Malformed signals raise :class:`MalformedSignalError`
  (the eighth FC-17 typed-error surface). Per ADR-016 §"Mechanism
  (e)" the error is *internal* — L0 callers catch it and drop the
  signal; the channel never propagates malformed-signal errors to
  the orchestrator's reasoning surface.

**FC-2 layering.** The channel is L1. The Ensemble Engine (L0) imports
the channel module to call :meth:`record_signal` — the single allowed
upward edge ADR-002 amends to permit, pre-declared in FC-2's
``_ALLOWED_UPWARD_EDGES``. All other upward signal attempts continue
to be rejected by the static layering check.

**Falsification trigger (load-bearing carry-forward from ADR-016).**
If BUILD or first-deployment evidence finds that mechanism (b) windowing
or mechanism (d) audit dispatch cannot be operationalized within ADR-
002's L0-L3 structure (e.g., they require a top-level module orthogonal
to the four-layer architecture), the elaboration-by-evidence framing
commitment is invalidated, the reorganization branch re-opens, and
ADR-016 is re-deliberated. Both mechanisms are implemented inside the
L1 module here; the falsification trigger has not fired in BUILD.

**Module-decomposition note (sibling-vs-monolithic, per WP-H4
post-build susceptibility snapshot Advisory 1).** WP-G4-2's
:class:`~llm_orc.agentic.tier_router_audit.TierEscalationAuditor` is
a public sibling module (separate file from `tier_router.py`) — a
directly analogous precedent that could have been applied here by
splitting :class:`_ChannelAuditWindow` (and the audit-firing logic)
into a separate file. The choice to keep mechanism (d)'s audit
state private inside this module rather than as a public sibling is
deliberate: the channel *owns* the audit data (it is channel-internal
state that no other module reads), and exposing the auditor as a
sibling would require widening the channel's public surface to include
audit-window state for the sibling to consume. The TierEscalationAuditor
split made sense because the router's :meth:`select_tier` is a
stateless pure function (FC-19) — the auditor's state had to live
somewhere outside the router. Here, the channel is already stateful
(it holds the signal buffer), so the audit state composes naturally
with the existing state, and the sibling pattern would add a coupling
surface without removing one.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from llm_orc.agentic.calibration_gate import CalibrationVerdict
from llm_orc.models.structural_errors import LlmOrcStructuralError

__all__ = [
    "CalibrationChannelAuditDiagnostic",
    "CalibrationChannelAuditThresholds",
    "CalibrationChannelAuditVerdict",
    "CalibrationChannelCriterionFinding",
    "CalibrationSignal",
    "CalibrationSignalChannel",
    "ChannelClock",
    "MalformedSignalError",
    "SystemChannelClock",
    "WindowedSignalFeatures",
]


CalibrationChannelAuditVerdict = Literal["no_drift", "advisory", "severe"]
"""Per ADR-016 §"Mechanism (d)" — three-value audit verdict trichotomy.

Parallel-by-construction to ADR-018's tier-router audit verdict (also
the susceptibility-snapshot pattern's clean / advisory / Grounding-
Reframe-trigger trichotomy from RDD methodology). The shape is
*structural transfer* from RDD methodology to architectural calibration
drift detection per Spike (d) (research log ``005f-spike-adr016-d-
structural-transfer-audit.md``)."""


DEFAULT_WINDOW_MINUTES: Final[float] = 60.0
"""Time-decay window primary bound (mechanism (b)). Default 60 minutes
per ADR-016 §"Mechanism (b)"; operationally tunable. Spike (b) suggests
smaller defaults track better — deployment may tune downward."""

DEFAULT_WINDOW_SIGNALS: Final[int] = 100
"""Time-decay window secondary bound (mechanism (b)). Default 100
signals per ADR-016 §"Mechanism (b)"; operationally tunable."""

DEFAULT_AUDIT_TRIGGER_COUNT: Final[int] = 100
"""Mechanism (d) audit trigger — every 100 verdicts per ADR-016
§"Mechanism (d)"; operationally tunable."""

DEFAULT_AUDIT_TRIGGER_HOURS: Final[float] = 24.0
"""Mechanism (d) audit trigger — every 24 wall-clock hours per ADR-016
§"Mechanism (d)"; operationally tunable."""

DEFAULT_VERDICT_DISTRIBUTION_SHIFT: Final[float] = 0.15
"""Mechanism (d) drift criterion: verdict-skew threshold. Default ±15%
per verdict class per ADR-016 §"Mechanism (d)" §"Verdict skew"."""

DEFAULT_OUTCOME_DIVERGENCE_PP: Final[float] = 0.10
"""Mechanism (d) drift criterion: predictive-accuracy decline threshold.
Default 10 percentage points per ADR-016 §"Mechanism (d)" §"Outcome
divergence"."""

DEFAULT_SIGNAL_VERDICT_CORRELATION_DRIFT: Final[float] = 0.20
"""Mechanism (d) drift criterion: correlation-coefficient change
threshold. Default ±0.20 per ADR-016 §"Mechanism (d)" §"Signal-to-
verdict correlation drift"."""

DEFAULT_SEVERE_DRIFT_MULTIPLIER: Final[float] = 2.0
"""Severe-magnitude single-criterion cutoff: a criterion at >= 2x its
advisory threshold reaches severe magnitude on its own per ADR-016
§"Mechanism (d)" §"Audit verdict shape"."""

_SECONDS_PER_MINUTE: Final[float] = 60.0
_SECONDS_PER_HOUR: Final[float] = 3600.0


# ----------------------------------------------------------------------------
# Signal schema (mechanism (e) — the typed boundary)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSignal:
    """Read-only typed signal data carried L0 → L1 per ADR-016 §"The signal channel".

    The channel boundary enforces this shape. Signals that do not match
    are rejected at :meth:`CalibrationSignalChannel.record_signal` with
    :class:`MalformedSignalError` per mechanism (e).

    Per ADR-016 §"The signal channel", the data shape covers:

    * Trajectory features (HTC per ADR-014's specification —
      ``recent_token_entropy``)
    * Dispatch outcomes (``dispatch_success`` — structural signal, not
      LLM-judgment summary)
    * Deterministic-tool-output anchors (mechanism (c); only present
      when the ensemble has a script-model slot — ``None`` otherwise)

    The schema is generous per the practitioner's no-token-limit-pre-
    optimization guidance (ADR-016 §"The signal channel" 2026-05-05
    Tranche-A close).
    """

    timestamp_seconds: float
    """Wall-clock dispatch completion time. Used by mechanism (b)
    windowing to determine in-window vs. out-of-window status."""

    ensemble_name: str
    """The ensemble whose dispatch produced this signal."""

    dispatch_success: bool
    """Did the dispatch complete successfully? Structural signal —
    derived from L0's dispatch outcome, not from LLM-judgment
    summary."""

    recent_token_entropy: float | None = None
    """HTC trajectory feature per ADR-014 §"Quality signal composition".
    Token-level entropy of the dispatched ensemble's most recent tokens.
    ``None`` when the L0 surface cannot extract entropy (e.g.,
    deterministic script-model dispatch)."""

    deterministic_anchor: bool | None = None
    """Mechanism (c) — categorical anchor presence from deterministic-
    tool-output. ``True`` when the deterministic output is positive
    (e.g., a CrossHair-style counterexample-free pass);  ``False`` when
    deterministic output is negative; ``None`` for LLM-only ensemble
    configurations per ADR-016 §"Mechanism (c)" §"Ensemble-composition-
    conditional applicability"."""

    dispatch_id: str | None = None
    """ADR-023 correlation identifier (Cycle 6 WP-A — additive). Joins
    this signal to the ``DispatchTiming`` start/end events and other
    per-dispatch events emitted through the Dispatch Event Substrate.
    ``None`` during the progressive conversion when the L0 emission
    site does not yet pass the substrate's allocated identifier. The
    cross-layer channel's bounding mechanisms (a)–(e) and read-only
    scope are unchanged by this additive field (preservation scenario
    in scenarios.md §Observability Event Routing)."""


# ----------------------------------------------------------------------------
# Mechanism (e) — typed-error surface (FC-17 8 of 8)
# ----------------------------------------------------------------------------


class MalformedSignalError(LlmOrcStructuralError):
    """Eighth and final FC-17 typed-error surface per ADR-016 mechanism (e).

    Concrete subclass of :class:`LlmOrcStructuralError` brings FC-17
    coverage to 8 of 8 (after ``ToolCallingNotSupportedError``,
    ``PhantomToolCallError``, ``WriteGateRejectionError``,
    ``CompactionLayer4FailureError``, ``CalibrationAbstainError``,
    ``EscalationBypassError``, ``MissingSkillMetadataError``).

    Per ADR-016 §"Mechanism (e)" the error is *internal* — produced at
    the channel boundary so the schema validation pattern is uniform
    with the rest of FC-17, but L0 callers catch and drop the signal
    rather than propagating to the orchestrator's reasoning surface.
    ``recovery_action_required`` is ``reformulate`` to express the
    same family-shape as the other typed errors; in practice, L0 never
    propagates this error to the orchestrator and the recovery action
    is exercised only by tests.
    """

    def __init__(
        self,
        message: str,
        *,
        ensemble_name: str | None,
        field_name: str | None = None,
    ) -> None:
        context: dict[str, Any] = {}
        if ensemble_name is not None:
            context["ensemble_name"] = ensemble_name
        if field_name is not None:
            context["field_name"] = field_name
        super().__init__(
            message,
            error_kind="malformed_signal",
            recovery_action_required="reformulate",
            dispatch_context=context,
            operator_diagnostic=message,
        )


# ----------------------------------------------------------------------------
# Mechanism (a) — fresh-context view of windowed features
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowedSignalFeatures:
    """Aggregated view of in-window signals — mechanism (a)'s fresh-context envelope.

    Returned by :meth:`CalibrationSignalChannel.windowed_features`. The
    L1 consumer reads this aggregated value; no underlying signal data
    is exposed by reference. This is the structural enforcement of
    mechanism (a): the consumer cannot accumulate signal history
    through the channel's API because the channel only exposes
    aggregated features per call.

    All fields are ``None`` when the aggregation has no statistical
    basis (zero in-window signals, or zero in-window signals with the
    relevant feature). Callers must check for ``None`` before consuming
    — the entropy-collapse criterion at ADR-014 §"Calibration verdict"
    skips when basis is absent rather than misfiring.
    """

    in_window_count: int
    """Number of signals inside the dual-bound window."""

    running_entropy_mean: float | None
    """Linear-decay-weighted mean of ``recent_token_entropy`` across
    in-window signals (excluding signals with ``None`` entropy). ``None``
    when fewer than two valid entropy samples are in window."""

    running_entropy_stdev: float | None
    """Linear-decay-weighted stdev of ``recent_token_entropy`` across
    in-window signals. ``None`` when basis is absent or window is
    degenerate (all-identical samples)."""

    has_entropy_basis: bool
    """Whether ``running_entropy_mean``/``stdev`` have a statistical
    basis. ``True`` requires at least two valid entropy samples and
    non-zero stdev."""

    deterministic_anchor_count: int
    """Number of in-window signals carrying a non-``None``
    ``deterministic_anchor`` (mechanism (c)). Zero on LLM-only
    ensemble configurations."""

    deterministic_anchor_positive_fraction: float | None
    """Fraction of deterministic-anchor signals that are positive
    (anchor == True). ``None`` when ``deterministic_anchor_count == 0``."""

    aggregated_success_rate: float | None
    """Fraction of in-window signals with ``dispatch_success == True``.
    ``None`` when ``in_window_count == 0``."""

    @classmethod
    def empty(cls) -> WindowedSignalFeatures:
        return cls(
            in_window_count=0,
            running_entropy_mean=None,
            running_entropy_stdev=None,
            has_entropy_basis=False,
            deterministic_anchor_count=0,
            deterministic_anchor_positive_fraction=None,
            aggregated_success_rate=None,
        )


# ----------------------------------------------------------------------------
# Mechanism (d) — audit configuration + verdict shape
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationChannelAuditThresholds:
    """ADR-016 §"Mechanism (d)" audit thresholds.

    All thresholds operator-tunable per ADR-016 §"Audit scope"; defaults
    match ADR-016's stated defaults.
    """

    trigger_count: int = DEFAULT_AUDIT_TRIGGER_COUNT
    trigger_wall_clock_hours: float = DEFAULT_AUDIT_TRIGGER_HOURS
    verdict_distribution_shift: float = DEFAULT_VERDICT_DISTRIBUTION_SHIFT
    outcome_divergence_pp: float = DEFAULT_OUTCOME_DIVERGENCE_PP
    signal_verdict_correlation_drift: float = DEFAULT_SIGNAL_VERDICT_CORRELATION_DRIFT
    severe_drift_multiplier: float = DEFAULT_SEVERE_DRIFT_MULTIPLIER

    def __post_init__(self) -> None:
        if self.trigger_count <= 0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.trigger_count must be > 0 "
                "per ADR-016 §Mechanism (d) §Trigger"
            )
        if self.trigger_wall_clock_hours <= 0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.trigger_wall_clock_hours "
                "must be > 0 per ADR-016 §Mechanism (d) §Trigger"
            )
        if self.verdict_distribution_shift <= 0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.verdict_distribution_shift "
                "must be > 0 per ADR-016 §Mechanism (d) §Verdict skew"
            )
        if self.outcome_divergence_pp <= 0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.outcome_divergence_pp "
                "must be > 0 per ADR-016 §Mechanism (d) §Outcome divergence"
            )
        if self.signal_verdict_correlation_drift <= 0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.signal_verdict_correlation_"
                "drift must be > 0 per ADR-016 §Mechanism (d)"
            )
        if self.severe_drift_multiplier < 1.0:
            raise ValueError(
                "CalibrationChannelAuditThresholds.severe_drift_multiplier "
                "must be >= 1.0 (severe magnitude cannot be below advisory)"
            )


@dataclass(frozen=True)
class CalibrationChannelCriterionFinding:
    """One drift criterion's evaluation in an audit window."""

    name: str
    value: float
    threshold: float
    exceeds: bool
    severe: bool


@dataclass(frozen=True)
class CalibrationChannelAuditDiagnostic:
    """One audit-window verdict, surfaced for operator review.

    Per ADR-016 §"Mechanism (d)" §"Asynchronous-operator-review dynamic"
    advisory diagnostics accumulate; the operator reviews at their
    cadence. Severe diagnostics also activate fail-safe mode
    synchronously.
    """

    window_id: int
    verdict: CalibrationChannelAuditVerdict
    timestamp_seconds: float
    consumption_count: int
    criteria_findings: tuple[CalibrationChannelCriterionFinding, ...]


# ----------------------------------------------------------------------------
# Clock surface (parallel to TierEscalationAuditor's AuditClock)
# ----------------------------------------------------------------------------


class ChannelClock(Protocol):
    """Wall-clock surface for the channel's audit triggers and windowing."""

    def now_seconds(self) -> float: ...


class SystemChannelClock:
    """Default :class:`ChannelClock` backed by :func:`time.time`."""

    def now_seconds(self) -> float:
        return time.time()


# ----------------------------------------------------------------------------
# Internal audit accumulator (mechanism (d) state)
# ----------------------------------------------------------------------------


@dataclass
class _ChannelAuditWindow:
    """Per-window accumulator state — internal to the channel's audit."""

    window_started_seconds: float
    consumption_count: int = 0
    proceed_count: int = 0
    reflect_count: int = 0
    abstain_count: int = 0
    proceed_outcomes: list[bool] = field(default_factory=list)
    """Outcome (success/failure) booleans for Proceed-verdict dispatches.
    Used to compute the predictive-accuracy criterion: Proceed verdicts
    should correlate with successful dispatches."""

    paired_entropy_verdict: list[tuple[float, float]] = field(default_factory=list)
    """``(entropy, verdict_numeric)`` pairs used for signal-to-verdict
    correlation. ``verdict_numeric`` encodes proceed=1.0, reflect=0.5,
    abstain=0.0 — a monotonic encoding so Pearson correlation is
    well-defined."""

    def fraction(self, count: int) -> float:
        if self.consumption_count == 0:
            return 0.0
        return count / self.consumption_count

    def proceed_fraction(self) -> float:
        return self.fraction(self.proceed_count)

    def reflect_fraction(self) -> float:
        return self.fraction(self.reflect_count)

    def abstain_fraction(self) -> float:
        return self.fraction(self.abstain_count)

    def proceed_success_rate(self) -> float | None:
        if not self.proceed_outcomes:
            return None
        return sum(1 for s in self.proceed_outcomes if s) / len(self.proceed_outcomes)

    def signal_verdict_correlation(self) -> float | None:
        """Pearson correlation between in-window entropy and verdict
        encoding. ``None`` when insufficient or degenerate."""
        if len(self.paired_entropy_verdict) < 2:
            return None
        n = len(self.paired_entropy_verdict)
        sum_x = sum(x for x, _ in self.paired_entropy_verdict)
        sum_y = sum(y for _, y in self.paired_entropy_verdict)
        mean_x = sum_x / n
        mean_y = sum_y / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in self.paired_entropy_verdict)
        var_x = sum((x - mean_x) ** 2 for x, _ in self.paired_entropy_verdict)
        var_y = sum((y - mean_y) ** 2 for _, y in self.paired_entropy_verdict)
        if var_x == 0.0 or var_y == 0.0:
            return None
        return num / math.sqrt(var_x * var_y)


_VERDICT_NUMERIC: Final[Mapping[CalibrationVerdict, float]] = {
    "proceed": 1.0,
    "reflect": 0.5,
    "abstain": 0.0,
}


# ----------------------------------------------------------------------------
# Time-decay-windowed signal storage (mechanism (b))
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _StoredSignal:
    """A signal retained in the channel's time-decay window."""

    signal: CalibrationSignal


# ----------------------------------------------------------------------------
# Channel — composes mechanisms (a), (b), (c), (d), (e)
# ----------------------------------------------------------------------------


class CalibrationSignalChannel:
    """The read-only L0→L1 calibration signal channel per ADR-016.

    Constructed with operationally-tunable parameters; defaults match
    ADR-016. The channel exposes:

    * :meth:`record_signal` — L0's signal-emission surface (the upward
      edge data path). Validates against the typed schema; raises
      :class:`MalformedSignalError` on schema mismatch (mechanism (e)).
    * :meth:`windowed_features` — L1's read-only aggregated-features
      surface (mechanism (a) fresh-context envelope; mechanism (b)
      windowing; mechanism (c) anchor exposure). The L1 consumer reads
      this view to compute its verdict.
    * :meth:`record_verdict_outcome` — L1's verdict-and-outcome record
      for the audit (mechanism (d)). Returns a diagnostic when the
      audit trigger fires; ``None`` otherwise.
    * :attr:`fail_safe_active` — severe-drift indicator. While ``True``,
      the L1 consumer should default to Reflect-or-Abstain.

    **What the channel intentionally does NOT expose:**

    * No L1 → L0 write path. The channel API has no method for
      writing data downward. The structural absence of a write API is
      the enforcement of the read-only constraint (scenario "Upward
      write attempt through channel is rejected").
    * No raw-signal read API. The L1 consumer reads aggregated features,
      not signal-by-signal data — the structural enforcement of
      mechanism (a)'s fresh-context isolation.
    """

    def __init__(
        self,
        *,
        window_minutes: float = DEFAULT_WINDOW_MINUTES,
        window_signals: int = DEFAULT_WINDOW_SIGNALS,
        audit_thresholds: CalibrationChannelAuditThresholds | None = None,
        clock: ChannelClock | None = None,
    ) -> None:
        if window_minutes <= 0.0:
            raise ValueError(
                f"CalibrationSignalChannel.window_minutes must be > 0, "
                f"got {window_minutes}"
            )
        if window_signals < 1:
            raise ValueError(
                f"CalibrationSignalChannel.window_signals must be >= 1, "
                f"got {window_signals}"
            )
        self._window_minutes = window_minutes
        self._window_signals = window_signals
        self._audit_thresholds: CalibrationChannelAuditThresholds = (
            audit_thresholds
            if audit_thresholds is not None
            else CalibrationChannelAuditThresholds()
        )
        self._clock: ChannelClock = clock if clock is not None else SystemChannelClock()
        self._signals: list[_StoredSignal] = []
        self._malformed_count = 0
        self._window = _ChannelAuditWindow(
            window_started_seconds=self._clock.now_seconds()
        )
        self._prior_window: _ChannelAuditWindow | None = None
        self._diagnostics: list[CalibrationChannelAuditDiagnostic] = []
        self._fail_safe_active = False
        self._window_id = 0

    # ----- L0 → L1 signal emission (mechanism (e) validation) -----

    def record_signal(self, signal: CalibrationSignal | Mapping[str, Any]) -> None:
        """Accept a calibration signal from L0 (the upward edge).

        Per ADR-016 §"Mechanism (e)" the signal is validated against
        the typed schema before storage. Schema mismatch raises
        :class:`MalformedSignalError`; the L0 caller catches and drops
        the signal (the error is internal — not propagated to the
        orchestrator).

        Accepts either a :class:`CalibrationSignal` instance (the
        structurally-typed path; no validation can fail) or a
        ``Mapping`` (the dict path used when L0 builds the signal
        ad-hoc; validation runs to coerce to :class:`CalibrationSignal`).
        """
        coerced = self._validate_and_coerce(signal)
        self._signals.append(_StoredSignal(signal=coerced))
        self._prune_window(now_seconds=self._clock.now_seconds())

    def malformed_signal_count(self) -> int:
        """How many malformed signals have been rejected at the boundary.

        Useful for tests and for operator dashboards that surface
        signal-quality issues at the channel level. The count is also
        the operator-visible footprint of mechanism (e) — non-zero
        values flag schema-mismatch issues upstream of the channel.
        """
        return self._malformed_count

    # ----- L1 consumer read (mechanisms (a), (b), (c)) -----

    def windowed_features(
        self,
        *,
        now_seconds: float | None = None,
        ensemble_name: str | None = None,
    ) -> WindowedSignalFeatures:
        """Return the aggregated view of in-window signals for L1 consumers.

        Mechanism (a): the returned :class:`WindowedSignalFeatures` is
        a fresh value — the channel's internal signal list is not
        exposed by reference. Mechanism (b): only in-window signals
        contribute, weighted linearly (1.0 most recent → 0.0 window
        edge). Mechanism (c): anchor presence and positive-fraction
        are surfaced when any in-window signal carries a non-``None``
        anchor.

        ``ensemble_name`` filters to per-ensemble features; ``None``
        aggregates across all ensembles. The filter matches the
        ensemble-composition-conditional intent of mechanism (c) —
        an LLM-only ensemble's per-ensemble view will simply have
        ``deterministic_anchor_count == 0`` and downstream verdicts
        proceed under (a), (b), (d), (e) only.
        """
        ts = self._clock.now_seconds() if now_seconds is None else now_seconds
        self._prune_window(now_seconds=ts)
        if not self._signals:
            return WindowedSignalFeatures.empty()

        in_window = [
            stored
            for stored in self._signals
            if (ensemble_name is None or stored.signal.ensemble_name == ensemble_name)
        ]
        if not in_window:
            return WindowedSignalFeatures.empty()

        anchor_signals = [
            stored
            for stored in in_window
            if stored.signal.deterministic_anchor is not None
        ]
        anchor_count = len(anchor_signals)
        anchor_positive_fraction: float | None
        if anchor_count == 0:
            anchor_positive_fraction = None
        else:
            anchor_positive_fraction = (
                sum(1 for s in anchor_signals if s.signal.deterministic_anchor)
                / anchor_count
            )

        success_rate = sum(1 for s in in_window if s.signal.dispatch_success) / len(
            in_window
        )

        mean, stdev, has_basis = self._weighted_entropy_stats(in_window)

        return WindowedSignalFeatures(
            in_window_count=len(in_window),
            running_entropy_mean=mean if has_basis else None,
            running_entropy_stdev=stdev if has_basis else None,
            has_entropy_basis=has_basis,
            deterministic_anchor_count=anchor_count,
            deterministic_anchor_positive_fraction=anchor_positive_fraction,
            aggregated_success_rate=success_rate,
        )

    # ----- L1 audit feedback (mechanism (d)) -----

    def record_verdict_outcome(
        self,
        *,
        verdict: CalibrationVerdict,
        ensemble_name: str,
        signal_features: WindowedSignalFeatures,
        proceed_outcome: bool | None = None,
    ) -> CalibrationChannelAuditDiagnostic | None:
        """Record the verdict produced for a dispatch — feeds the audit.

        ``signal_features`` is the aggregated view the L1 consumer
        used to compute its verdict (the same value
        :meth:`windowed_features` returned). The signal-to-verdict
        correlation criterion uses ``signal_features.running_entropy_
        mean`` paired with the verdict's numeric encoding.

        ``proceed_outcome`` is the post-hoc dispatch outcome (success/
        failure) for a Proceed verdict. ``None`` when the verdict was
        not Proceed (Reflect/Abstain dispatches are routed elsewhere
        and their outcomes do not bear on the predictive-accuracy
        criterion).

        Returns the :class:`CalibrationChannelAuditDiagnostic` if this
        call crossed the audit trigger boundary; ``None`` otherwise.
        Diagnostics accumulate in :meth:`audit_diagnostics` regardless.
        """
        del ensemble_name  # per-ensemble accounting is a Cycle 5+ surface
        self._window.consumption_count += 1
        if verdict == "proceed":
            self._window.proceed_count += 1
            if proceed_outcome is not None:
                self._window.proceed_outcomes.append(proceed_outcome)
        elif verdict == "reflect":
            self._window.reflect_count += 1
        else:  # verdict == "abstain"
            self._window.abstain_count += 1

        if signal_features.running_entropy_mean is not None:
            self._window.paired_entropy_verdict.append(
                (signal_features.running_entropy_mean, _VERDICT_NUMERIC[verdict])
            )

        if self._trigger_fired():
            return self._fire_audit()
        return None

    @property
    def fail_safe_active(self) -> bool:
        """Severe-drift fail-safe indicator per ADR-016 §"Mechanism (d)".

        While ``True``, L1 consumers should default to Reflect-or-
        Abstain — the calibration system has flagged its own drift as
        severe; the verdict-producer should not optimistically Proceed.
        Operator clears via :meth:`clear_fail_safe`."""
        return self._fail_safe_active

    def audit_diagnostics(self) -> tuple[CalibrationChannelAuditDiagnostic, ...]:
        """Accumulated audit diagnostics for asynchronous operator review.

        Per ADR-016 §"Mechanism (d)" §"Asynchronous-operator-review
        dynamic" — diagnostics surface advisory drift findings;
        operator reviews at their cadence. Severe diagnostics also
        activate fail-safe synchronously.
        """
        return tuple(self._diagnostics)

    def clear_fail_safe(self) -> None:
        """Operator action — clear fail-safe after review.

        Per ADR-016 §"Mechanism (d)" the operator must explicitly clear
        fail-safe. The channel does not self-clear even after subsequent
        no-drift audits — sustained fail-safe persists until operator
        review.
        """
        self._fail_safe_active = False

    # ------------------------------------------------------------------
    # Internal: validation, windowing, audit
    # ------------------------------------------------------------------

    def _validate_and_coerce(
        self, signal: CalibrationSignal | Mapping[str, Any]
    ) -> CalibrationSignal:
        """Validate mechanism (e); raise :class:`MalformedSignalError` on mismatch.

        The structurally-typed :class:`CalibrationSignal` path is always
        accepted (Python's type system already validated). The mapping
        path runs field-by-field validation — operator-extensibility
        territory but kept conservative for the cycle's BUILD scope.
        """
        if isinstance(signal, CalibrationSignal):
            return signal
        if not isinstance(signal, Mapping):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal must be a CalibrationSignal or Mapping; "
                f"got {type(signal).__name__}",
                ensemble_name=None,
            )

        ensemble_name = signal.get("ensemble_name")
        if not isinstance(ensemble_name, str):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal missing required 'ensemble_name' string field",
                ensemble_name=(
                    ensemble_name if isinstance(ensemble_name, str) else None
                ),
                field_name="ensemble_name",
            )

        timestamp = signal.get("timestamp_seconds")
        if not isinstance(timestamp, int | float):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal missing required 'timestamp_seconds' numeric field",
                ensemble_name=ensemble_name,
                field_name="timestamp_seconds",
            )

        dispatch_success = signal.get("dispatch_success")
        if not isinstance(dispatch_success, bool):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal missing required 'dispatch_success' bool field",
                ensemble_name=ensemble_name,
                field_name="dispatch_success",
            )

        recent_token_entropy = signal.get("recent_token_entropy")
        if recent_token_entropy is not None and not isinstance(
            recent_token_entropy, int | float
        ):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal field 'recent_token_entropy' must be "
                "numeric or None",
                ensemble_name=ensemble_name,
                field_name="recent_token_entropy",
            )

        deterministic_anchor = signal.get("deterministic_anchor")
        if deterministic_anchor is not None and not isinstance(
            deterministic_anchor, bool
        ):
            self._malformed_count += 1
            raise MalformedSignalError(
                "calibration signal field 'deterministic_anchor' must be bool or None",
                ensemble_name=ensemble_name,
                field_name="deterministic_anchor",
            )

        return CalibrationSignal(
            timestamp_seconds=float(timestamp),
            ensemble_name=ensemble_name,
            dispatch_success=dispatch_success,
            recent_token_entropy=(
                float(recent_token_entropy)
                if recent_token_entropy is not None
                else None
            ),
            deterministic_anchor=deterministic_anchor,
        )

    def _prune_window(self, *, now_seconds: float) -> None:
        """Drop signals outside the dual-bound window per mechanism (b).

        The shorter bound wins: signals older than ``window_minutes`` are
        dropped first; then if the count still exceeds ``window_signals``
        the oldest are trimmed.
        """
        cutoff = now_seconds - self._window_minutes * _SECONDS_PER_MINUTE
        self._signals = [
            stored
            for stored in self._signals
            if stored.signal.timestamp_seconds >= cutoff
        ]
        if len(self._signals) > self._window_signals:
            self._signals = self._signals[-self._window_signals :]

    def _weighted_entropy_stats(
        self, in_window: list[_StoredSignal]
    ) -> tuple[float, float, bool]:
        """Return (mean, stdev, has_basis) for in-window entropy samples.

        Linear-decay weighting from 1.0 (most recent) to 0.0 (oldest in
        window). Mirrors :class:`CalibrationGate`'s in-layer trajectory
        windowing — the same construction at the cross-layer surface
        per ADR-016 §"Mechanism (b)".
        """
        valid = [
            stored.signal.recent_token_entropy
            for stored in in_window
            if stored.signal.recent_token_entropy is not None
        ]
        if len(valid) < 2:
            return 0.0, 0.0, False

        n = len(valid)
        weights = [i / (n - 1) for i in range(n)]
        total_weight = sum(weights)
        if total_weight == 0.0:
            return 0.0, 0.0, False
        mean = sum(w * e for w, e in zip(weights, valid, strict=True)) / total_weight
        variance = (
            sum(w * (e - mean) ** 2 for w, e in zip(weights, valid, strict=True))
            / total_weight
        )
        stdev = math.sqrt(variance)
        return mean, stdev, stdev > 0.0

    def _trigger_fired(self) -> bool:
        if self._window.consumption_count >= self._audit_thresholds.trigger_count:
            return True
        elapsed = self._clock.now_seconds() - self._window.window_started_seconds
        return elapsed >= (
            self._audit_thresholds.trigger_wall_clock_hours * _SECONDS_PER_HOUR
        )

    def _fire_audit(self) -> CalibrationChannelAuditDiagnostic:
        self._window_id += 1
        findings = self._evaluate_criteria()
        verdict = self._verdict_from_findings(findings)
        diagnostic = CalibrationChannelAuditDiagnostic(
            window_id=self._window_id,
            verdict=verdict,
            timestamp_seconds=self._clock.now_seconds(),
            consumption_count=self._window.consumption_count,
            criteria_findings=findings,
        )
        self._diagnostics.append(diagnostic)
        if verdict == "severe":
            self._fail_safe_active = True
        self._prior_window = self._window
        self._window = _ChannelAuditWindow(
            window_started_seconds=self._clock.now_seconds()
        )
        return diagnostic

    def _evaluate_criteria(
        self,
    ) -> tuple[CalibrationChannelCriterionFinding, ...]:
        return (
            self._verdict_skew_finding(),
            self._outcome_divergence_finding(),
            self._signal_verdict_correlation_finding(),
        )

    def _verdict_skew_finding(self) -> CalibrationChannelCriterionFinding:
        threshold = self._audit_thresholds.verdict_distribution_shift
        severe_threshold = threshold * self._audit_thresholds.severe_drift_multiplier
        if self._prior_window is None:
            return CalibrationChannelCriterionFinding(
                name="verdict_distribution_shift",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        max_axis_shift = max(
            abs(
                self._window.proceed_fraction() - self._prior_window.proceed_fraction()
            ),
            abs(
                self._window.reflect_fraction() - self._prior_window.reflect_fraction()
            ),
            abs(
                self._window.abstain_fraction() - self._prior_window.abstain_fraction()
            ),
        )
        return CalibrationChannelCriterionFinding(
            name="verdict_distribution_shift",
            value=max_axis_shift,
            threshold=threshold,
            exceeds=max_axis_shift > threshold,
            severe=max_axis_shift >= severe_threshold,
        )

    def _outcome_divergence_finding(self) -> CalibrationChannelCriterionFinding:
        threshold = self._audit_thresholds.outcome_divergence_pp
        severe_threshold = threshold * self._audit_thresholds.severe_drift_multiplier
        current_rate = self._window.proceed_success_rate()
        prior_rate = (
            self._prior_window.proceed_success_rate()
            if self._prior_window is not None
            else None
        )
        if current_rate is None or prior_rate is None:
            return CalibrationChannelCriterionFinding(
                name="outcome_divergence",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        decline = prior_rate - current_rate
        return CalibrationChannelCriterionFinding(
            name="outcome_divergence",
            value=decline,
            threshold=threshold,
            exceeds=decline > threshold,
            severe=decline >= severe_threshold,
        )

    def _signal_verdict_correlation_finding(
        self,
    ) -> CalibrationChannelCriterionFinding:
        threshold = self._audit_thresholds.signal_verdict_correlation_drift
        severe_threshold = threshold * self._audit_thresholds.severe_drift_multiplier
        current_corr = self._window.signal_verdict_correlation()
        prior_corr = (
            self._prior_window.signal_verdict_correlation()
            if self._prior_window is not None
            else None
        )
        if current_corr is None or prior_corr is None:
            return CalibrationChannelCriterionFinding(
                name="signal_verdict_correlation_drift",
                value=0.0,
                threshold=threshold,
                exceeds=False,
                severe=False,
            )
        delta = abs(current_corr - prior_corr)
        return CalibrationChannelCriterionFinding(
            name="signal_verdict_correlation_drift",
            value=delta,
            threshold=threshold,
            exceeds=delta > threshold,
            severe=delta >= severe_threshold,
        )

    @staticmethod
    def _verdict_from_findings(
        findings: tuple[CalibrationChannelCriterionFinding, ...],
    ) -> CalibrationChannelAuditVerdict:
        exceeding = [f for f in findings if f.exceeds]
        if not exceeding:
            return "no_drift"
        if any(f.severe for f in exceeding) or len(exceeding) >= 2:
            return "severe"
        return "advisory"
