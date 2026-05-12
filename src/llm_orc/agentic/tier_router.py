"""Tier-Escalation Router — per-dispatch Model Profile selection (WP-G4-1).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Tier-Escalation Router (L2 — new in Cycle 4 per ADR-015; extended in
Cycle 4 architect-gate close per ADR-018). The router selects a
per-dispatch Model Profile (cheap-tier or escalated-tier) for
``invoke_ensemble`` based on the dispatched ensemble's Topaz skill
metadata and the Calibration Gate's verdict.

Verdict → tier mapping (ADR-015 §Router logic):

* ``proceed`` → per-skill cheap-tier Model Profile
* ``reflect`` → per-skill escalated-tier Model Profile
* ``abstain`` → :class:`EscalationBypassError` typed error

Missing ``topaz_skill`` metadata → :class:`MissingSkillMetadataError`.

The router operates inside Tool Dispatch (L2 interposition); the
orchestrator's tool-call surface (``invoke_ensemble``) is unchanged
per ADR-015 §"ADR-011 compatibility". The orchestrator's own Model
Profile remains session-boundary scoped (ADR-011 preserved).

**Stateless pure function property (FC-19, per ADR-018 inherited
mechanism (a)).** :meth:`TierRouter.select_tier` reads no module-level
state and mutates no instance state. Identical inputs produce
identical outputs regardless of call history. Spike β analytical
transfer audit (research log ``005h-``, 2026-05-11) confirms this
satisfies ADR-016 mechanism (a) by construction at the L1→L2 verdict→
router edge.

**WP-G4-1 scope:** the core router (verdict consumption, per-skill
mapping, typed errors). The ADR-018 (d)-analog audit dispatch
(periodic out-of-band routing-vs-tier-config drift detection) is
WP-G4-2 territory and lands separately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, get_args

from llm_orc.agentic.calibration_gate import CalibrationVerdict
from llm_orc.models.structural_errors import LlmOrcStructuralError

if TYPE_CHECKING:
    from llm_orc.core.config.ensemble_config import EnsembleConfig

__all__ = [
    "ALL_TOPAZ_SKILLS",
    "EnsembleConfigTopazSkillReader",
    "EscalationBypassError",
    "MissingSkillMetadataError",
    "PerSkillTierDefaults",
    "Tier",
    "TierRouter",
    "TierRouterDefaults",
    "TierSelection",
    "TopazSkill",
    "TopazSkillReader",
]


TopazSkill = Literal[
    "code_generation",
    "tool_use",
    "mathematical_reasoning",
    "logical_reasoning",
    "factual_knowledge",
    "writing_quality",
    "instruction_following",
    "summarization",
]
"""The eight Topaz skills per ADR-015 §Per-skill role profiling.

Direct adoption of the Topaz paper's role-profiling vocabulary. The
full 8-skill taxonomy is load-bearing per practitioner friction-trades-
for-discovery guidance (ADR-015 rejected alternative §(d)) and
confirmed by Spike α (research log ``005g-``, 2026-05-11) — 21 of 21
production-style ensembles satisfy the clean-primary criterion.
"""


ALL_TOPAZ_SKILLS: Final[tuple[TopazSkill, ...]] = (
    "code_generation",
    "tool_use",
    "mathematical_reasoning",
    "logical_reasoning",
    "factual_knowledge",
    "writing_quality",
    "instruction_following",
    "summarization",
)
"""Exhaustive enumeration of Topaz skills for runtime validation.

Mirrors the :data:`TopazSkill` Literal at the value level so the
constructor can verify per-skill defaults cover the full taxonomy.
"""


Tier = Literal["cheap", "escalated"]
"""Per ADR-015 §Per-skill tier defaults. Each Topaz skill has a
cheap-tier Model Profile and an escalated-tier Model Profile."""


@dataclass(frozen=True)
class PerSkillTierDefaults:
    """Operator-configured pair of Model Profiles for one Topaz skill.

    Frozen because configuration is session-boundary scoped per
    ADR-011 — changes take effect on new sessions, active sessions
    continue under their existing tier defaults.
    """

    cheap_tier: str
    """Cheap-tier Model Profile name. Operator-local (when local
    capability meets the floor for the skill) or cheap-cloud (when
    local falls below the floor) per ADR-015 §Per-skill tier defaults.
    """

    escalated_tier: str
    """Escalated-tier Model Profile name. Cheap-cloud or
    operator-cloud (frontier) per ADR-015 §Per-skill tier defaults."""

    def __post_init__(self) -> None:
        if not self.cheap_tier:
            raise ValueError(
                "PerSkillTierDefaults.cheap_tier must be a non-empty Model Profile name"
            )
        if not self.escalated_tier:
            raise ValueError(
                "PerSkillTierDefaults.escalated_tier must be a non-empty "
                "Model Profile name"
            )


@dataclass(frozen=True)
class TierRouterDefaults:
    """The 8 skills × 2 tiers = 16 Model Profile slots per ADR-015.

    Per ADR-015 §Per-skill tier defaults: defaults may share Model
    Profiles across skills (e.g., a single local-7B profile may serve
    as cheap-tier for ``code_generation`` and ``tool_use``). The 16-
    slot configuration surface is friction-trades-for-discovery per
    practitioner gate-conversation guidance.
    """

    per_skill: Mapping[TopazSkill, PerSkillTierDefaults]


@dataclass(frozen=True)
class TierSelection:
    """Output of :meth:`TierRouter.select_tier` per dispatch.

    Frozen because the selection is a value type that flows downward
    into ensemble execution; no consumer mutates it. Tests assert on
    the ``model_profile`` field per scenarios.md §Per-Role Tier-
    Escalation Router.
    """

    model_profile: str
    """Selected Model Profile name for this dispatch — flows into
    ``_operations.invoke({..., "model_profile_override": ...})``."""

    tier: Tier
    """Whether the cheap-tier or escalated-tier was selected. Carried
    alongside ``model_profile`` so observability and (d)-analog audit
    inputs (WP-G4-2) do not need to re-derive the tier from the
    Model Profile name."""

    topaz_skill: TopazSkill
    """The ensemble's declared Topaz skill that drove this selection.
    Recorded for observability and so the (d)-analog audit (WP-G4-2)
    can correlate escalation rate against skill."""


def _coerce_topaz_skill(raw: object) -> TopazSkill | None:
    """Return ``raw`` if it is a recognized Topaz skill, else ``None``.

    Operator-authored YAML values that are not in the closed taxonomy
    surface as :class:`MissingSkillMetadataError` at the router (the
    reader returns ``None`` and the router raises). Unrecognized values
    are *not* tolerated silently — that would defeat ADR-015's per-
    skill role profiling contract.
    """
    return raw if raw in get_args(TopazSkill) else None  # type: ignore[return-value]


class TopazSkillReader(Protocol):
    """Reads the Topaz skill metadata for a dispatched ensemble.

    Per system-design.agents.md §Tier-Escalation Router → Ensemble
    Engine integration contract: the router resolves the dispatched
    ensemble's ``topaz_skill`` field via the existing config manager.
    Production wiring queries :class:`EnsembleConfig`; tests pass a
    scripted double.

    The Protocol returns ``TopazSkill | None`` — ``None`` signals the
    ensemble's YAML lacks the metadata field, which the router
    surfaces as :class:`MissingSkillMetadataError` (verdict-
    independent except under Abstain, which short-circuits).
    """

    def topaz_skill_for(self, ensemble_name: str) -> TopazSkill | None: ...


class EscalationBypassError(LlmOrcStructuralError):
    """Raised when the router observes an Abstain verdict per ADR-015.

    Sixth concrete subclass of :class:`LlmOrcStructuralError` per
    FC-17 (after :class:`ToolCallingNotSupportedError`,
    :class:`PhantomToolCallError`, :class:`WriteGateRejectionError`,
    :class:`CompactionLayer4FailureError`,
    :class:`CalibrationAbstainError`).

    Per ADR-015 §Router logic: Abstain verdicts bypass routing
    entirely. Escalation does not happen on Abstain — the orchestrator
    must reformulate, dispatch to a different ensemble, or abandon the
    task. ``recovery_action_required="reformulate"`` flags this for
    the orchestrator's ReAct loop.

    Distinct from :class:`CalibrationAbstainError`: that error fires
    when a consumer reads an Abstain verdict directly from the gate;
    this error fires at the router edge per the ADR-015 verdict-to-
    action mapping. The two compose for FC-17 coverage but represent
    different consumer paths.
    """

    def __init__(
        self,
        message: str,
        *,
        ensemble_name: str,
        session_id: str = "",
    ) -> None:
        super().__init__(
            message,
            error_kind="escalation_bypass",
            recovery_action_required="reformulate",
            dispatch_context={
                "session_id": session_id,
                "ensemble_name": ensemble_name,
            },
            operator_diagnostic=message,
        )


class MissingSkillMetadataError(LlmOrcStructuralError):
    """Raised when a dispatched ensemble lacks ``topaz_skill`` metadata.

    Seventh concrete subclass of :class:`LlmOrcStructuralError` per
    FC-17 (after the six listed in :class:`EscalationBypassError`'s
    docstring).

    Per ADR-015 §Per-skill role profiling: every ensemble in the
    library must declare its primary Topaz skill in YAML metadata.
    Dispatch of an ensemble lacking the field surfaces this error
    before tier selection runs (FC-18). The operator diagnostic lists
    the valid skill values so misconfiguration can be corrected
    without consulting the ADR.

    ``recovery_action_required="reformulate"`` — the orchestrator must
    dispatch to a different ensemble (one with the metadata field) or
    abandon the task; operator-side correction (adding the field to
    the ensemble's YAML) is the durable fix.
    """

    def __init__(
        self,
        message: str,
        *,
        ensemble_name: str,
        session_id: str = "",
    ) -> None:
        super().__init__(
            message,
            error_kind="missing_skill_metadata",
            recovery_action_required="reformulate",
            dispatch_context={
                "session_id": session_id,
                "ensemble_name": ensemble_name,
            },
            operator_diagnostic=message,
        )


class EnsembleConfigTopazSkillReader:
    """Default :class:`TopazSkillReader` backed by EnsembleConfig.

    Production wiring at the Serving Layer (L3) constructs this with a
    ``find_ensemble`` callable that resolves a name to an EnsembleConfig
    (e.g., via ``EnsembleLoader.find_ensemble`` or
    ``OrchestraService.find_ensemble_by_name``). The reader honors
    ADR-015's per-skill role profiling contract — operator-authored
    YAML strings outside the closed Topaz taxonomy return ``None``,
    causing the router to raise :class:`MissingSkillMetadataError`.

    The dependency on EnsembleConfig is L0 (downward from L2), inside
    the layering rule. The TYPE_CHECKING import keeps the runtime
    coupling narrow — production wiring imports the loader directly.
    """

    def __init__(
        self,
        find_ensemble: Callable[[str], EnsembleConfig | None],
    ) -> None:
        self._find_ensemble = find_ensemble

    def topaz_skill_for(self, ensemble_name: str) -> TopazSkill | None:
        config = self._find_ensemble(ensemble_name)
        if config is None:
            return None
        return _coerce_topaz_skill(config.topaz_skill)


class TierRouter:
    """Stateless per-dispatch Model Profile selector (ADR-015).

    Construction binds operator-configured per-skill tier defaults and
    a :class:`TopazSkillReader`. :meth:`select_tier` is a pure
    function of its inputs (FC-19): identical inputs produce identical
    outputs regardless of call history. No instance state is mutated;
    no module-level state is read.
    """

    def __init__(
        self,
        *,
        defaults: TierRouterDefaults,
        skill_reader: TopazSkillReader,
    ) -> None:
        missing = [
            skill for skill in ALL_TOPAZ_SKILLS if skill not in defaults.per_skill
        ]
        if missing:
            raise ValueError(
                "TierRouterDefaults must cover all eight Topaz skills per "
                f"ADR-015 §Decision; missing: {missing!r}"
            )
        self._defaults = defaults
        self._skill_reader = skill_reader

    def select_tier(
        self,
        *,
        ensemble_name: str,
        verdict: CalibrationVerdict,
        session_id: str = "",
    ) -> TierSelection:
        """Select the per-dispatch Model Profile per ADR-015 §Router logic.

        Decision tree (deterministic, exhaustive over the verdict
        trichotomy):

        1. ``verdict == "abstain"`` → :class:`EscalationBypassError`.
           Short-circuits before skill lookup — Abstain handling is
           verdict-driven and metadata-independent.
        2. Skill lookup. ``None`` → :class:`MissingSkillMetadataError`.
        3. ``verdict == "proceed"`` → cheap-tier Model Profile.
        4. ``verdict == "reflect"`` → escalated-tier Model Profile.

        Per ADR-018 inherited bounding mechanism (a): this method
        reads no module-level state and mutates no instance state.
        The stateless property is what makes the L1→L2 verdict→router
        edge fresh-context isolated by construction.
        """
        if verdict == "abstain":
            raise EscalationBypassError(
                f"Calibration verdict is Abstain for ensemble {ensemble_name!r}; "
                "escalation bypassed per ADR-015 — orchestrator must "
                "reformulate or dispatch to a different ensemble",
                ensemble_name=ensemble_name,
                session_id=session_id,
            )

        skill = self._skill_reader.topaz_skill_for(ensemble_name)
        if skill is None:
            valid = ", ".join(ALL_TOPAZ_SKILLS)
            raise MissingSkillMetadataError(
                f"Ensemble {ensemble_name!r} does not declare a Topaz skill in "
                "its YAML metadata; the Tier-Escalation Router cannot select a "
                "per-skill Model Profile without it. Per ADR-015 §Per-skill "
                "role profiling, every ensemble must declare 'topaz_skill' as "
                f"one of: {valid}",
                ensemble_name=ensemble_name,
                session_id=session_id,
            )

        per_skill = self._defaults.per_skill[skill]
        if verdict == "proceed":
            return TierSelection(
                model_profile=per_skill.cheap_tier,
                tier="cheap",
                topaz_skill=skill,
            )
        # Exhaustive over CalibrationVerdict literal — only "reflect"
        # remains after the Abstain short-circuit above.
        return TierSelection(
            model_profile=per_skill.escalated_tier,
            tier="escalated",
            topaz_skill=skill,
        )
