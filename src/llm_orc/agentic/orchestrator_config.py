"""Orchestrator Configuration — per-session config resolution.

Per `docs/agentic-serving/system-design.md` §Orchestrator Configuration
(L3). Resolves the orchestrator Model Profile (ADR-011), Budget
defaults (ADR-005), Autonomy default (ADR-008), Plexus enablement flag
(ADR-009), and operator-set bounds on per-request overrides from the
existing ConfigurationManager.

Per-request override *application* (clamping or rejecting overrides
against the configured bounds) is not implemented in Phase 1; the
operator-side bounds surface is in place so a future scenario group
can wire in the override path without revisiting this module's shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_orc.agentic.calibration_gate import (
    DEFAULT_CALIBRATION_CHECKER_ENSEMBLE,
    DEFAULT_CALIBRATION_N,
)
from llm_orc.agentic.conversation_compaction import (
    DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES,
    DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS,
    DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP,
    DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE,
    DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT,
    CompactionDefaults,
)
from llm_orc.agentic.tier_router import (
    ALL_TOPAZ_SKILLS,
    PerSkillTierDefaults,
    TopazSkill,
)
from llm_orc.agentic.tier_router_audit import TierEscalationAuditThresholds
from llm_orc.core.config.config_manager import ConfigurationManager


class ModelProfileNotFoundError(LookupError):
    """The configured orchestrator Model Profile is not in the library.

    Raised by :meth:`OrchestratorConfigResolver.resolve_validated` so
    session start fails loudly instead of the orchestrator booting with a
    name that no Ensemble Engine profile resolution will accept (FF #40).
    """


DEFAULT_MODEL_PROFILE = "default"
DEFAULT_TURN_LIMIT = 500
DEFAULT_TOKEN_LIMIT = 10_000_000
DEFAULT_AUTONOMY_LEVEL = "operator-as-tool-user"
DEFAULT_PLEXUS_ENABLED = False
DEFAULT_ALLOW_BUDGET_OVERRIDE = True
DEFAULT_MAX_TURN_LIMIT = 1_000
DEFAULT_MAX_TOKEN_LIMIT = 50_000_000
DEFAULT_SUMMARIZER_ENSEMBLE = "agentic-result-summarizer"

DEFAULT_TIER_AUDIT_TRIGGER_COUNT = 100
DEFAULT_TIER_AUDIT_TRIGGER_WALL_CLOCK_HOURS = 24.0
DEFAULT_TIER_AUDIT_VERDICT_DISTRIBUTION_SHIFT = 0.15
DEFAULT_TIER_AUDIT_ESCALATION_OUTCOME_CORRELATION_PP = 0.05
DEFAULT_TIER_AUDIT_BYPASS_RATE_INCREASE = 0.25
DEFAULT_TIER_AUDIT_SEVERE_DRIFT_MULTIPLIER = 2.0
"""ADR-018 §"Drift criteria" defaults for the Tier-Escalation Router
(d)-analog audit dispatch (WP-G4-2). All operationally tunable via
``orchestrator.tier_router_audit`` in ``config.yaml``."""

# Conversation Compaction defaults are owned by the L2 module
# (``llm_orc.agentic.conversation_compaction``) and imported above —
# the L3 config module composes them but does not own them, per the
# layering rule (L3 → L2 → L1 → L0).

DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the llm-orc orchestrator. You route tool-user tasks by invoking \
ensembles, composing new ones from library primitives, and (when Plexus is \
active) querying or recording against a knowledge graph.

Your internal tool surface is exactly five functions. They are \
server-side calls — fast, free, and authoritative for any question \
about what this llm-orc instance can do:
- list_ensembles() — list ensembles in the library with descriptions. \
**Call this FIRST whenever the user asks about ensembles, what is \
available in this instance, what the system can do, what can be \
composed, or any other question about llm-orc capabilities.** It is \
the canonical answer to capability queries.
- invoke_ensemble(name, input) — run a library ensemble on a task \
input. Use after list_ensembles when you know which ensemble fits.
- compose_ensemble(...) — create a new ensemble from existing \
primitives when no library ensemble fits.
- query_knowledge(...) — query the knowledge graph (Plexus, when active).
- record_outcome(...) — record a routing decision or outcome.

Every tool call the tool user's client declares (e.g. file_read, bash, \
file_edit, skill, glob) is a *client-declared* tool. They touch the \
user's filesystem or run code on their machine — they are slower and \
operate on user-side state, not llm-orc state. Use them when you need \
file contents, directory structure, code execution, or edits to the \
user's files.

**Do not pick a client-declared tool for questions about llm-orc's own \
state.** Client tool names that sound related to capability queries \
(such as `skill`, `command`, or similar) do not answer "what ensembles \
are available" or "what can this orchestration system do" — those are \
always answered by list_ensembles. When in doubt for a capability \
query, choose the internal tool.

When you need a client-declared tool, emit it alone in a single \
assistant turn — the turn will close and the client will execute it; \
you resume on the next request with the result as a role:tool \
observation. Never mix internal tools and client-declared tools in the \
same assistant turn — emit one kind per turn.

Retry convention for composed ensembles. A composed ensemble's agent may \
report that it cannot proceed without a client-side tool result by \
returning a JSON object shaped like \
{"needs_client_tool": {"tool": "<name>", "args": {...}}}. When you see \
that signal in an invoke_ensemble summary, emit the named client-declared \
tool with those args; on the next request, re-invoke the same ensemble \
with the client-tool result folded into your input string. The Ensemble \
Engine never suspends mid-execution — re-invocation with augmented input \
is the retry path.
"""
"""Default orchestrator system prompt (Phase 1, WP-F Group 3).

Teaches the orchestrator LLM:

* the closed five-tool internal surface (ADR-003);
* Option C's one-kind-per-turn discipline — internal tools dispatch
  in-process; client-declared tools close the turn (system-design
  §Client Tool Surface Commitment);
* the ``needs_client_tool`` retry convention for composed ensembles
  whose mid-execution dependencies were not predicted at invoke-time
  (roadmap Open Decision Point #8, mechanism i).

Operators override via ``agentic_serving.orchestrator.system_prompt``
in ``config.yaml`` when a deployment wants tighter or more specific
guidance. Always prepended ahead of any client-supplied system message
so the orchestrator discipline survives competing instructions.
"""


@dataclass(frozen=True)
class BudgetDefaults:
    """Per-session Budget defaults (ADR-005)."""

    turn_limit: int
    token_limit: int


@dataclass(frozen=True)
class OverrideBounds:
    """Operator-set bounds on per-request overrides.

    Expressible in Phase 1; enforced when the request-override mechanism
    is specified in a later scenario group.
    """

    allow_budget_override: bool
    max_turn_limit: int
    max_token_limit: int


@dataclass(frozen=True)
class CalibrationDefaults:
    """Calibration Gate configuration (ADR-007, WP-H)."""

    default_n: int
    checker_ensemble: str


@dataclass(frozen=True)
class ObservabilityDefaults:
    """ADR-023 observability surface defaults (Cycle 6 WP-B + WP-C).

    The Serving Layer reads ``heartbeat_interval_seconds`` to size the
    per-request inference-wait heartbeat scheduler (default 30s),
    ``orchestrator_context_routes_calibration_signal`` to decide
    whether the orchestrator-context sink includes
    :class:`CalibrationSignal` events in its end-of-session summary
    (default ``False``), and ``agentic_sessions_root`` for the per-
    session dispatch_log filesystem destination (default
    ``.llm-orc/agentic-sessions/``).

    Operators override via ``agentic_serving.observability.*``.
    """

    heartbeat_interval_seconds: float
    orchestrator_context_routes_calibration_signal: bool = False
    agentic_sessions_root: str = ".llm-orc/agentic-sessions/"


DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0


@dataclass(frozen=True)
class OrchestratorConfig:
    """Immutable per-session configuration surface.

    Immutability matches ADR-011's session-boundary discipline: changes
    to operator config take effect on new Sessions; active Sessions
    continue under their existing configuration.
    """

    model_profile: str
    budget: BudgetDefaults
    autonomy_level: str
    plexus_enabled: bool
    override_bounds: OverrideBounds
    allowed_profiles: tuple[str, ...]
    summarizer_ensemble: str
    orchestrator_system_prompt: str
    calibration: CalibrationDefaults
    tool_call_validation_patterns: tuple[str, ...] = ()
    """Operator-extension regexes for ADR-017's structural validation guard.

    The default pattern set lives in the guard module
    (``DEFAULT_ASSERTION_PATTERNS``); these values are appended at scan
    time. Empty tuple = guard scans defaults only.
    """
    compaction: CompactionDefaults = field(
        default_factory=lambda: CompactionDefaults(
            persist_threshold_chars=DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS,
            idle_window_minutes=DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES,
            session_notes_token_cap=DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP,
            layer_4_circuit_breaker_threshold=(
                DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD
            ),
            trigger_token_count=DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT,
            summarizer_ensemble=DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE,
        )
    )
    """ADR-012 Conversation Compaction defaults (WP-E4).

    Operator-tunable thresholds and Layer 4 summarizer ensemble. The
    default factory keeps existing OrchestratorConfig construction
    sites working without specifying compaction settings.
    """
    per_skill_tier_defaults: Mapping[TopazSkill, PerSkillTierDefaults] | None = None
    """Per-skill tier defaults for the Tier-Escalation Router (ADR-015).

    ``None`` when the operator has not configured any tier defaults —
    in that case the Serving Layer does not construct a Router and
    dispatches run without tier escalation (pre-WP-G4-1 behavior).
    When present, the mapping must cover all 8 Topaz skills; partial
    configurations raise at session start.
    """
    tier_router_audit: TierEscalationAuditThresholds = field(
        default_factory=lambda: TierEscalationAuditThresholds(
            trigger_count=DEFAULT_TIER_AUDIT_TRIGGER_COUNT,
            trigger_wall_clock_hours=DEFAULT_TIER_AUDIT_TRIGGER_WALL_CLOCK_HOURS,
            verdict_distribution_shift=(DEFAULT_TIER_AUDIT_VERDICT_DISTRIBUTION_SHIFT),
            escalation_outcome_correlation_pp=(
                DEFAULT_TIER_AUDIT_ESCALATION_OUTCOME_CORRELATION_PP
            ),
            bypass_rate_increase=DEFAULT_TIER_AUDIT_BYPASS_RATE_INCREASE,
            severe_drift_multiplier=DEFAULT_TIER_AUDIT_SEVERE_DRIFT_MULTIPLIER,
        )
    )
    """ADR-018 (d)-analog audit dispatch thresholds (WP-G4-2).

    Operator-tunable via ``orchestrator.tier_router_audit``. The audit
    fires only when the Serving Layer constructs a Router (i.e.,
    ``per_skill_tier_defaults`` is configured); the defaults are
    still composed so consumers can read them unconditionally.
    """
    observability: ObservabilityDefaults = field(
        default_factory=lambda: ObservabilityDefaults(
            heartbeat_interval_seconds=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        )
    )
    """ADR-023 observability defaults (Cycle 6 WP-B piece 5).

    Operator-tunable via ``agentic_serving.observability``. The Serving
    Layer reads ``heartbeat_interval_seconds`` per request to size the
    inference-wait heartbeat scheduler. Default 30s.
    """


class OrchestratorConfigResolver:
    """Reads operator configuration and produces typed OrchestratorConfig."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self._config_manager = config_manager

    def resolve(self) -> OrchestratorConfig:
        raw = self._config_manager.load_agentic_serving_config()

        orchestrator = _as_mapping(raw.get("orchestrator"))
        budget = _as_mapping(raw.get("budget"))
        autonomy = _as_mapping(raw.get("autonomy"))
        plexus = _as_mapping(raw.get("plexus"))
        overrides = _as_mapping(raw.get("overrides"))
        summarizer = _as_mapping(raw.get("summarizer"))
        calibration = _as_mapping(orchestrator.get("calibration"))
        compaction = _as_mapping(orchestrator.get("compaction"))
        tier_audit = _as_mapping(orchestrator.get("tier_router_audit"))
        observability = _as_mapping(raw.get("observability"))
        per_skill_raw = orchestrator.get("per_skill_tier_defaults")

        model_profile = str(orchestrator.get("model_profile", DEFAULT_MODEL_PROFILE))
        allowed_profiles = _resolve_allowed_profiles(
            orchestrator.get("allowed_profiles"), model_profile
        )

        return OrchestratorConfig(
            model_profile=model_profile,
            budget=BudgetDefaults(
                turn_limit=int(budget.get("turn_limit", DEFAULT_TURN_LIMIT)),
                token_limit=int(budget.get("token_limit", DEFAULT_TOKEN_LIMIT)),
            ),
            autonomy_level=str(autonomy.get("default_level", DEFAULT_AUTONOMY_LEVEL)),
            plexus_enabled=bool(plexus.get("enabled", DEFAULT_PLEXUS_ENABLED)),
            override_bounds=OverrideBounds(
                allow_budget_override=bool(
                    overrides.get(
                        "allow_budget_override", DEFAULT_ALLOW_BUDGET_OVERRIDE
                    )
                ),
                max_turn_limit=int(
                    overrides.get("max_turn_limit", DEFAULT_MAX_TURN_LIMIT)
                ),
                max_token_limit=int(
                    overrides.get("max_token_limit", DEFAULT_MAX_TOKEN_LIMIT)
                ),
            ),
            allowed_profiles=allowed_profiles,
            summarizer_ensemble=str(
                summarizer.get("ensemble", DEFAULT_SUMMARIZER_ENSEMBLE)
            ),
            orchestrator_system_prompt=str(
                orchestrator.get("system_prompt", DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT)
            ),
            calibration=CalibrationDefaults(
                default_n=_positive_int(
                    calibration.get("default_n"), DEFAULT_CALIBRATION_N
                ),
                checker_ensemble=str(
                    calibration.get(
                        "checker_ensemble", DEFAULT_CALIBRATION_CHECKER_ENSEMBLE
                    )
                ),
            ),
            tool_call_validation_patterns=_resolve_pattern_tuple(
                orchestrator.get("tool_call_validation_patterns")
            ),
            compaction=CompactionDefaults(
                persist_threshold_chars=_positive_int(
                    compaction.get("persist_threshold_chars"),
                    DEFAULT_COMPACTION_PERSIST_THRESHOLD_CHARS,
                ),
                idle_window_minutes=_positive_int(
                    compaction.get("idle_window_minutes"),
                    DEFAULT_COMPACTION_IDLE_WINDOW_MINUTES,
                ),
                session_notes_token_cap=_positive_int(
                    compaction.get("session_notes_token_cap"),
                    DEFAULT_COMPACTION_SESSION_NOTES_TOKEN_CAP,
                ),
                layer_4_circuit_breaker_threshold=_positive_int(
                    compaction.get("layer_4_circuit_breaker_threshold"),
                    DEFAULT_COMPACTION_LAYER_4_CIRCUIT_BREAKER_THRESHOLD,
                ),
                trigger_token_count=_positive_int(
                    compaction.get("trigger_token_count"),
                    DEFAULT_COMPACTION_TRIGGER_TOKEN_COUNT,
                ),
                summarizer_ensemble=_resolve_optional_str(
                    compaction.get("summarizer_ensemble"),
                    DEFAULT_COMPACTION_SUMMARIZER_ENSEMBLE,
                ),
            ),
            per_skill_tier_defaults=_resolve_per_skill_tier_defaults(per_skill_raw),
            tier_router_audit=TierEscalationAuditThresholds(
                trigger_count=_positive_int(
                    tier_audit.get("trigger_count"),
                    DEFAULT_TIER_AUDIT_TRIGGER_COUNT,
                ),
                trigger_wall_clock_hours=_positive_float(
                    tier_audit.get("trigger_wall_clock_hours"),
                    DEFAULT_TIER_AUDIT_TRIGGER_WALL_CLOCK_HOURS,
                ),
                verdict_distribution_shift=_positive_float(
                    tier_audit.get("verdict_distribution_shift"),
                    DEFAULT_TIER_AUDIT_VERDICT_DISTRIBUTION_SHIFT,
                ),
                escalation_outcome_correlation_pp=_positive_float(
                    tier_audit.get("escalation_outcome_correlation_pp"),
                    DEFAULT_TIER_AUDIT_ESCALATION_OUTCOME_CORRELATION_PP,
                ),
                bypass_rate_increase=_positive_float(
                    tier_audit.get("bypass_rate_increase"),
                    DEFAULT_TIER_AUDIT_BYPASS_RATE_INCREASE,
                ),
                severe_drift_multiplier=_severe_multiplier(
                    tier_audit.get("severe_drift_multiplier"),
                    DEFAULT_TIER_AUDIT_SEVERE_DRIFT_MULTIPLIER,
                ),
            ),
            observability=ObservabilityDefaults(
                heartbeat_interval_seconds=_positive_float(
                    observability.get("heartbeat_interval_seconds"),
                    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                ),
                orchestrator_context_routes_calibration_signal=bool(
                    observability.get(
                        "orchestrator_context_routes_calibration_signal", False
                    )
                ),
                agentic_sessions_root=str(
                    observability.get(
                        "agentic_sessions_root", ".llm-orc/agentic-sessions/"
                    )
                ),
            ),
        )

    def resolve_validated(self) -> OrchestratorConfig:
        """Resolve and validate the orchestrator Model Profile exists.

        Session start (WP-C) calls this so it fails fast when an operator
        has misconfigured ``orchestrator.model_profile``. ``resolve`` stays
        pure translation for the ``/v1/models`` shop-window path.
        """
        config = self.resolve()
        library = self._config_manager.get_model_profiles()
        if config.model_profile not in library:
            raise ModelProfileNotFoundError(
                f"Orchestrator model_profile '{config.model_profile}' is not "
                "configured in ConfigurationManager.get_model_profiles()"
            )
        return config

    def list_allowed_model_profile_ids(self) -> tuple[str, ...]:
        """Return the allowlist intersected with the Model Profile library.

        Consumer: the Serving Layer's ``/v1/models`` endpoint. Preserves the
        operator-configured ordering; drops names that do not correspond to
        a profile in ``ConfigurationManager.get_model_profiles()``. The
        endpoint enumerates what is actually resolvable; proactive
        validation (raising) is the responsibility of ``resolve_validated``.
        """
        config = self.resolve()
        library = self._config_manager.get_model_profiles()
        return tuple(name for name in config.allowed_profiles if name in library)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a config section to a dict, tolerating missing / malformed sections."""
    if isinstance(value, dict):
        return value
    return {}


def _positive_int(raw: Any, fallback: int) -> int:
    """Coerce ``raw`` to an int >= 1, falling back to ``fallback`` otherwise.

    ADR-007 requires ``default_n >= 1``; an operator typo that produces
    ``0`` or a string should not crash session start — fall back to the
    shipped default so calibration still runs.
    """
    try:
        candidate = int(raw)
    except (TypeError, ValueError):
        return fallback
    return candidate if candidate >= 1 else fallback


def _resolve_allowed_profiles(raw: Any, model_profile: str) -> tuple[str, ...]:
    """Normalize the operator-configured allowlist.

    An absent or malformed allowlist falls back to ``(model_profile,)`` so a
    single-profile deployment works without additional configuration.
    """
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw)
    return (model_profile,)


def _resolve_pattern_tuple(raw: Any) -> tuple[str, ...]:
    """Normalize an operator-supplied list of regex pattern strings.

    Per ADR-017 §"Minimal default pattern set with operator-extension
    surface". A missing or malformed value (typo: string instead of list)
    falls back to the empty tuple so session start does not crash on a
    bad config entry; the guard then scans defaults only.
    """
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    return ()


def _resolve_optional_str(raw: Any, fallback: str | None) -> str | None:
    """Normalize an operator-supplied optional string config value.

    ``None`` or missing key returns ``fallback``. Any other value is
    coerced to ``str``. Used by the Compaction summarizer-ensemble
    setting (ADR-012 Layer 4): unconfigured by default; when an
    operator supplies an ensemble name, Layer 4 dispatches it.
    """
    if raw is None:
        return fallback
    return str(raw)


def _positive_float(raw: Any, fallback: float) -> float:
    """Coerce ``raw`` to a float > 0, falling back on malformed values.

    Mirrors :func:`_positive_int` semantics for ADR-018 audit thresholds
    that take fractional values (windowing hours, percentage-point
    tolerances expressed as fractions).
    """
    try:
        candidate = float(raw)
    except (TypeError, ValueError):
        return fallback
    return candidate if candidate > 0 else fallback


def _severe_multiplier(raw: Any, fallback: float) -> float:
    """Coerce ``raw`` to a float >= 1.0 (severe must be at or above advisory).

    Per ADR-018: the severe-magnitude cutoff cannot sit below the
    advisory threshold. Operator-supplied values below 1.0 fall back
    to the shipped default so misconfiguration does not collapse
    severity into a tautology.
    """
    try:
        candidate = float(raw)
    except (TypeError, ValueError):
        return fallback
    return candidate if candidate >= 1.0 else fallback


def _resolve_per_skill_tier_defaults(
    raw: Any,
) -> Mapping[TopazSkill, PerSkillTierDefaults] | None:
    """Parse the operator's per-skill tier defaults.

    Returns ``None`` when the section is absent or empty — the Serving
    Layer then skips Router construction and dispatches run without
    tier escalation. When present, all 8 Topaz skills must be supplied
    and each must declare ``cheap_tier`` and ``escalated_tier`` Model
    Profile names; partial or malformed entries raise so operator
    misconfiguration surfaces at session start (per ADR-015's
    "explicit error" stance from rejected alternative §(c)).
    """
    if not isinstance(raw, dict) or not raw:
        return None
    per_skill: dict[TopazSkill, PerSkillTierDefaults] = {}
    missing: list[str] = []
    for skill in ALL_TOPAZ_SKILLS:
        entry = raw.get(skill)
        if not isinstance(entry, dict):
            missing.append(skill)
            continue
        cheap_tier = entry.get("cheap_tier")
        escalated_tier = entry.get("escalated_tier")
        if not isinstance(cheap_tier, str) or not isinstance(escalated_tier, str):
            raise ValueError(
                "OrchestratorConfig per_skill_tier_defaults: "
                f"skill {skill!r} must declare 'cheap_tier' and "
                "'escalated_tier' as string Model Profile names"
            )
        per_skill[skill] = PerSkillTierDefaults(
            cheap_tier=cheap_tier, escalated_tier=escalated_tier
        )
    if missing:
        raise ValueError(
            "OrchestratorConfig per_skill_tier_defaults must cover all 8 "
            f"Topaz skills per ADR-015 §Decision; missing: {missing!r}"
        )
    return per_skill
