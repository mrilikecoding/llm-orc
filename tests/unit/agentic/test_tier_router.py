"""Tests for the Tier-Escalation Router (WP-G4-1, ADR-015 + ADR-018).

Per ``docs/agentic-serving/system-design.agents.md`` §Module:
Tier-Escalation Router (L2 — new in Cycle 4). The router selects a
per-dispatch Model Profile (cheap-tier or escalated-tier) for
``invoke_ensemble`` based on the dispatched ensemble's Topaz skill
metadata and the Calibration Gate's verdict.

Verdict → tier mapping (ADR-015 §Router logic):

* ``proceed`` → cheap-tier Model Profile for the ensemble's skill
* ``reflect`` → escalated-tier Model Profile for the ensemble's skill
* ``abstain`` → :class:`EscalationBypassError` typed error

Missing ``topaz_skill`` metadata → :class:`MissingSkillMetadataError`.

WP-G4-1's scope is the core router (verdict consumption, per-skill
mapping, typed errors). The ADR-018 (d)-analog audit dispatch is
WP-G4-2 territory.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from llm_orc.agentic.tier_router import (
    EnsembleConfigTopazSkillReader,
    EscalationBypassError,
    MissingSkillMetadataError,
    PerSkillTierDefaults,
    TierRouter,
    TierRouterDefaults,
    TierSelection,
    TopazSkill,
    TopazSkillReader,
)
from llm_orc.core.config.ensemble_config import EnsembleConfig

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedSkillReader:
    """Scripted :class:`TopazSkillReader` for tests.

    Maps ensemble names to declared Topaz skills. Names not in the
    mapping return ``None`` — the missing-metadata path.
    """

    def __init__(self, mapping: dict[str, TopazSkill | None]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    def topaz_skill_for(self, ensemble_name: str) -> TopazSkill | None:
        self.calls.append(ensemble_name)
        return self._mapping.get(ensemble_name)


def _all_skills_defaults() -> TierRouterDefaults:
    """Build a TierRouterDefaults covering all 8 Topaz skills.

    Used by tests that exercise non-skill-specific behavior (Proceed
    routes to cheap, Abstain raises, stateless property). Each skill
    gets a distinct pair so a misrouting bug surfaces as a wrong-skill
    Model Profile rather than a coincidental match.
    """
    return TierRouterDefaults(
        per_skill={
            "code_generation": PerSkillTierDefaults(
                cheap_tier="cheap-code-gen",
                escalated_tier="escalated-code-gen",
            ),
            "tool_use": PerSkillTierDefaults(
                cheap_tier="cheap-tool-use",
                escalated_tier="escalated-tool-use",
            ),
            "mathematical_reasoning": PerSkillTierDefaults(
                cheap_tier="cheap-math",
                escalated_tier="escalated-math",
            ),
            "logical_reasoning": PerSkillTierDefaults(
                cheap_tier="cheap-logic",
                escalated_tier="escalated-logic",
            ),
            "factual_knowledge": PerSkillTierDefaults(
                cheap_tier="cheap-fact",
                escalated_tier="escalated-fact",
            ),
            "writing_quality": PerSkillTierDefaults(
                cheap_tier="cheap-write",
                escalated_tier="escalated-write",
            ),
            "instruction_following": PerSkillTierDefaults(
                cheap_tier="cheap-instr",
                escalated_tier="escalated-instr",
            ),
            "summarization": PerSkillTierDefaults(
                cheap_tier="cheap-summ",
                escalated_tier="escalated-summ",
            ),
        }
    )


def _defaults_with_override(
    skill: TopazSkill, override: PerSkillTierDefaults
) -> TierRouterDefaults:
    """Build full 8-skill defaults with ``skill`` swapped to ``override``.

    Per ADR-015 §Decision, the constructor requires all 8 Topaz skills.
    Tests that exercise a specific (skill, Model Profile) pair compose
    the override onto the full default set so the constructor accepts
    the configuration.
    """
    per_skill = dict(_all_skills_defaults().per_skill)
    per_skill[skill] = override
    return TierRouterDefaults(per_skill=per_skill)


# ---------------------------------------------------------------------------
# ADR-015 scenarios — verdict-to-tier mapping (FC-19)
# ---------------------------------------------------------------------------


class TestCodeGenerationCheapTierOnProceed:
    """Scenario: Code-generation skill routes to per-skill cheap-tier on Proceed.

    Per scenarios.md §Per-Role Tier-Escalation Router (ADR-015):

        Given an ensemble whose YAML metadata declares Topaz skill
        ``code_generation`` and operator-configured tier defaults
        specify ``cheap-tier: ollama-deepseek-coder-v2:16b`` for that
        skill, When the orchestrator dispatches the ensemble via
        ``invoke_ensemble`` with calibration verdict *Proceed*, Then
        the Tool Dispatch router selects ``ollama-deepseek-coder-v2:16b``
        as the dispatch's Model Profile.
    """

    def test_proceed_verdict_routes_code_generation_ensemble_to_cheap_tier(
        self,
    ) -> None:
        reader = _ScriptedSkillReader({"code-review-A": "code_generation"})
        defaults = _defaults_with_override(
            "code_generation",
            PerSkillTierDefaults(
                cheap_tier="ollama-deepseek-coder-v2:16b",
                escalated_tier="claude-sonnet-4-6",
            ),
        )
        router = TierRouter(defaults=defaults, skill_reader=reader)

        selection = router.select_tier(ensemble_name="code-review-A", verdict="proceed")

        assert selection == TierSelection(
            model_profile="ollama-deepseek-coder-v2:16b",
            tier="cheap",
            topaz_skill="code_generation",
        )


class TestReflectVerdictRoutesToEscalatedTier:
    """Scenario: Reflect verdict routes to per-skill escalated-tier.

    Per scenarios.md §Per-Role Tier-Escalation Router (ADR-015):

        Given an ensemble with Topaz skill ``tool_use`` and operator-
        configured tier defaults specifying ``escalated-tier:
        gpt-5-mini`` for that skill, When the calibration verdict for
        the dispatch is *Reflect*, Then the router selects
        ``gpt-5-mini`` as the dispatch's Model Profile.
    """

    def test_reflect_verdict_routes_tool_use_ensemble_to_escalated_tier(
        self,
    ) -> None:
        reader = _ScriptedSkillReader({"plexus-graph-analysis": "tool_use"})
        defaults = _defaults_with_override(
            "tool_use",
            PerSkillTierDefaults(
                cheap_tier="ollama-qwen3:8b",
                escalated_tier="gpt-5-mini",
            ),
        )
        router = TierRouter(defaults=defaults, skill_reader=reader)

        selection = router.select_tier(
            ensemble_name="plexus-graph-analysis", verdict="reflect"
        )

        assert selection.model_profile == "gpt-5-mini"
        assert selection.tier == "escalated"


class TestAbstainVerdictProducesEscalationBypass:
    """Scenario: Abstain verdict produces escalation-bypass typed error.

    Per scenarios.md §Per-Role Tier-Escalation Router (ADR-015):

        Given any dispatch with calibration verdict *Abstain*, When the
        per-role tier-escalation router processes the dispatch, Then
        the router does not perform tier escalation, instead produces
        a typed ``escalation_bypass`` error to the orchestrator, and
        the orchestrator must reformulate or take a different action.
    """

    def test_abstain_verdict_raises_escalation_bypass_error(self) -> None:
        reader = _ScriptedSkillReader({"composed-a": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(EscalationBypassError) as exc_info:
            router.select_tier(ensemble_name="composed-a", verdict="abstain")

        error = exc_info.value
        assert error.error_kind == "escalation_bypass"
        assert error.recovery_action_required == "reformulate"
        assert error.dispatch_context["ensemble_name"] == "composed-a"

    def test_abstain_error_carries_session_id_when_supplied(self) -> None:
        reader = _ScriptedSkillReader({"composed-a": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(EscalationBypassError) as exc_info:
            router.select_tier(
                ensemble_name="composed-a",
                verdict="abstain",
                session_id="s-42",
            )

        assert exc_info.value.dispatch_context["session_id"] == "s-42"

    def test_abstain_short_circuits_before_skill_lookup(self) -> None:
        """ADR-015 §Router logic: Abstain bypasses routing entirely.

        Skill lookup is not required to produce the typed error — the
        verdict is the load-bearing input. This keeps Abstain handling
        independent of metadata-completeness concerns.
        """
        reader = _ScriptedSkillReader({"composed-a": None})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(EscalationBypassError):
            router.select_tier(ensemble_name="composed-a", verdict="abstain")


class TestMissingTopazSkillMetadata:
    """Scenario: Ensemble lacking Topaz skill metadata fails dispatch with error.

    Per scenarios.md §Per-Role Tier-Escalation Router (ADR-015):

        Given an ensemble in the library whose YAML configuration does
        not declare a Topaz skill metadata field, When the orchestrator
        dispatches the ensemble, Then the Tool Dispatch router rejects
        the dispatch with a typed ``missing_skill_metadata`` error
        explaining that all ensembles must declare their primary Topaz
        skill, and the rejection includes a list of valid skill values.
    """

    def test_missing_skill_raises_typed_error_on_proceed_verdict(self) -> None:
        reader = _ScriptedSkillReader({"untagged-ensemble": None})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(MissingSkillMetadataError) as exc_info:
            router.select_tier(ensemble_name="untagged-ensemble", verdict="proceed")

        error = exc_info.value
        assert error.error_kind == "missing_skill_metadata"
        assert error.recovery_action_required == "reformulate"
        assert error.dispatch_context["ensemble_name"] == "untagged-ensemble"

    def test_missing_skill_error_message_lists_valid_topaz_skills(self) -> None:
        """Per scenarios.md: the rejection includes a list of valid skill values."""
        reader = _ScriptedSkillReader({"untagged-ensemble": None})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(MissingSkillMetadataError) as exc_info:
            router.select_tier(ensemble_name="untagged-ensemble", verdict="proceed")

        # All 8 Topaz skills per ADR-015 §Per-skill role profiling must
        # appear in the operator-facing error so a misconfigured ensemble
        # can be corrected without consulting the ADR.
        message = str(exc_info.value)
        for skill in (
            "code_generation",
            "tool_use",
            "mathematical_reasoning",
            "logical_reasoning",
            "factual_knowledge",
            "writing_quality",
            "instruction_following",
            "summarization",
        ):
            assert skill in message

    def test_missing_skill_raises_typed_error_on_reflect_verdict(self) -> None:
        """Reflect verdict also requires skill metadata — same error class."""
        reader = _ScriptedSkillReader({"untagged-ensemble": None})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(MissingSkillMetadataError):
            router.select_tier(ensemble_name="untagged-ensemble", verdict="reflect")


class TestPerSkillNotPerEnsembleDefaults:
    """Scenario: Per-skill (not per-ensemble) tier defaults.

    Per scenarios.md §Per-Role Tier-Escalation Router (ADR-015):

        Given two ensembles ``code-review-pair-A`` and ``code-review-
        pair-B`` both declaring Topaz skill ``code_generation``, When
        the operator's tier-default configuration specifies cheap-tier
        and escalated-tier Model Profiles for ``code_generation``,
        Then both ensembles use the same cheap-tier and escalated-tier
        Model Profiles when dispatched — the configuration is per-
        skill, not per-ensemble.

    Per ADR-015 rejected alternative (b) — per-ensemble tier
    alternatives — confirmed by Spike α (research log ``005g-``,
    2026-05-11) as well-founded for the actual library (21/21 clean
    primary skill).
    """

    def test_two_ensembles_with_same_skill_share_cheap_tier_default(self) -> None:
        reader = _ScriptedSkillReader(
            {
                "code-review-pair-A": "code_generation",
                "code-review-pair-B": "code_generation",
            }
        )
        defaults = _defaults_with_override(
            "code_generation",
            PerSkillTierDefaults(
                cheap_tier="shared-cheap-code-gen",
                escalated_tier="shared-escalated-code-gen",
            ),
        )
        router = TierRouter(defaults=defaults, skill_reader=reader)

        selection_a = router.select_tier(
            ensemble_name="code-review-pair-A", verdict="proceed"
        )
        selection_b = router.select_tier(
            ensemble_name="code-review-pair-B", verdict="proceed"
        )

        assert selection_a.model_profile == "shared-cheap-code-gen"
        assert selection_b.model_profile == "shared-cheap-code-gen"

    def test_two_ensembles_with_same_skill_share_escalated_tier_default(
        self,
    ) -> None:
        reader = _ScriptedSkillReader(
            {
                "code-review-pair-A": "code_generation",
                "code-review-pair-B": "code_generation",
            }
        )
        defaults = _defaults_with_override(
            "code_generation",
            PerSkillTierDefaults(
                cheap_tier="shared-cheap-code-gen",
                escalated_tier="shared-escalated-code-gen",
            ),
        )
        router = TierRouter(defaults=defaults, skill_reader=reader)

        selection_a = router.select_tier(
            ensemble_name="code-review-pair-A", verdict="reflect"
        )
        selection_b = router.select_tier(
            ensemble_name="code-review-pair-B", verdict="reflect"
        )

        assert selection_a.model_profile == "shared-escalated-code-gen"
        assert selection_b.model_profile == "shared-escalated-code-gen"


# ---------------------------------------------------------------------------
# FC-19 — stateless pure function (ADR-018 inherited mechanism (a))
# ---------------------------------------------------------------------------


class TestSelectTierIsStatelessPureFunction:
    """FC-19: ``select_tier`` is a stateless pure function — each verdict
    consumption runs in fresh context with no prior verdict state carried
    forward through the call. Per system-design.agents.md §Module:
    Tier-Escalation Router Fitness, this satisfies ADR-016 mechanism (a)
    by construction (ADR-018 inherited bounding-mechanism property)."""

    def test_sequential_calls_produce_identical_output_for_identical_inputs(
        self,
    ) -> None:
        reader = _ScriptedSkillReader({"e": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        first = router.select_tier(ensemble_name="e", verdict="proceed")
        second = router.select_tier(ensemble_name="e", verdict="proceed")
        third = router.select_tier(ensemble_name="e", verdict="proceed")

        assert first == second == third

    def test_call_history_does_not_influence_subsequent_verdicts(self) -> None:
        """A Reflect call followed by Proceed must produce a pristine
        Proceed verdict — no carry-forward of escalation state."""
        reader = _ScriptedSkillReader({"e": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        # Drive a Reflect → escalated selection first
        router.select_tier(ensemble_name="e", verdict="reflect")
        # Then a Proceed must still return cheap-tier
        selection = router.select_tier(ensemble_name="e", verdict="proceed")

        assert selection.tier == "cheap"
        assert selection.model_profile == "cheap-code-gen"

    def test_router_holds_no_mutable_state_after_construction(self) -> None:
        """No public attributes mutate across calls (inspectable
        property of the stateless design)."""
        reader = _ScriptedSkillReader({"e": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        public_attrs_before = {
            name: getattr(router, name)
            for name in dir(router)
            if not name.startswith("_") and not callable(getattr(router, name))
        }

        for _ in range(5):
            router.select_tier(ensemble_name="e", verdict="proceed")

        public_attrs_after = {
            name: getattr(router, name)
            for name in dir(router)
            if not name.startswith("_") and not callable(getattr(router, name))
        }

        assert public_attrs_before == public_attrs_after


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestTierRouterConstructorValidation:
    """The constructor rejects malformed defaults loudly so misconfiguration
    surfaces at session start rather than at first dispatch."""

    def test_defaults_must_cover_all_eight_topaz_skills(self) -> None:
        """ADR-015 §Decision: full 8-skill taxonomy is load-bearing per
        practitioner friction-trades-for-discovery guidance. A defaults
        config missing one or more skills is misconfiguration."""
        partial = TierRouterDefaults(
            per_skill={
                "code_generation": PerSkillTierDefaults(
                    cheap_tier="cheap", escalated_tier="escalated"
                ),
            }
        )
        reader = _ScriptedSkillReader({})

        with pytest.raises(ValueError, match="all eight Topaz skills"):
            TierRouter(defaults=partial, skill_reader=reader)

    def test_per_skill_tier_defaults_reject_empty_cheap_tier(self) -> None:
        """Empty Model Profile names are misconfiguration — caught at
        construction so they don't surface as ensemble-not-found errors."""
        with pytest.raises(ValueError, match="cheap_tier"):
            PerSkillTierDefaults(cheap_tier="", escalated_tier="profile-A")

    def test_per_skill_tier_defaults_reject_empty_escalated_tier(self) -> None:
        with pytest.raises(ValueError, match="escalated_tier"):
            PerSkillTierDefaults(cheap_tier="profile-A", escalated_tier="")


# ---------------------------------------------------------------------------
# All 8 Topaz skills — N-dimension coverage (per skill, Proceed routes cheap)
# ---------------------------------------------------------------------------


_ALL_TOPAZ_SKILLS: tuple[TopazSkill, ...] = (
    "code_generation",
    "tool_use",
    "mathematical_reasoning",
    "logical_reasoning",
    "factual_knowledge",
    "writing_quality",
    "instruction_following",
    "summarization",
)


class TestAllEightSkillsRouteOnProceed:
    """N-dimension test (one per skill, holding verdict=Proceed constant)
    per the build skill's COMPOSABLE TESTS section. The M-dimension test
    (verdict variation, holding skill constant) lives in
    :class:`TestVerdictToTierMappingIsDeterministic`."""

    @pytest.mark.parametrize("skill", _ALL_TOPAZ_SKILLS)
    def test_proceed_routes_to_skill_specific_cheap_tier(
        self, skill: TopazSkill
    ) -> None:
        reader = _ScriptedSkillReader({"e": skill})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        selection = router.select_tier(ensemble_name="e", verdict="proceed")

        assert selection.topaz_skill == skill
        assert selection.tier == "cheap"


class TestVerdictToTierMappingIsDeterministic:
    """Per system-design.agents.md §Module: Tier-Escalation Router Fitness:

        The verdict-to-action mapping is deterministic: Proceed →
        cheap-tier dispatch; Reflect → escalated-tier dispatch; Abstain →
        ``escalation_bypass`` typed error.

    M-dimension test (verdict variation, holding skill constant).
    """

    def test_proceed_maps_to_cheap_tier(self) -> None:
        reader = _ScriptedSkillReader({"e": "summarization"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        assert router.select_tier(ensemble_name="e", verdict="proceed").tier == "cheap"

    def test_reflect_maps_to_escalated_tier(self) -> None:
        reader = _ScriptedSkillReader({"e": "summarization"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        assert (
            router.select_tier(ensemble_name="e", verdict="reflect").tier == "escalated"
        )

    def test_abstain_maps_to_escalation_bypass(self) -> None:
        reader = _ScriptedSkillReader({"e": "summarization"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        with pytest.raises(EscalationBypassError):
            router.select_tier(ensemble_name="e", verdict="abstain")


# ---------------------------------------------------------------------------
# Protocol compliance — TopazSkillReader
# ---------------------------------------------------------------------------


class TestRouterUsesSkillReader:
    """The router consults the configured reader for every dispatch — no
    caching, no name-based heuristics. This keeps the metadata-read
    contract centralized in the reader (system-design §Tier-Escalation
    Router → Ensemble Engine integration contract)."""

    def test_skill_reader_is_consulted_for_each_dispatch(self) -> None:
        reader = _ScriptedSkillReader({"e": "code_generation"})
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        router.select_tier(ensemble_name="e", verdict="proceed")
        router.select_tier(ensemble_name="e", verdict="proceed")
        router.select_tier(ensemble_name="e", verdict="proceed")

        assert reader.calls == ["e", "e", "e"]


class TestTopazSkillReaderProtocolStructure:
    """Anything with the right ``topaz_skill_for`` shape satisfies the
    :class:`TopazSkillReader` Protocol — the production reader is
    decoupled from the router via structural typing."""

    def test_structural_subtype_is_accepted(self) -> None:
        class _AdHocReader:
            def topaz_skill_for(self, ensemble_name: str) -> TopazSkill | None:
                del ensemble_name
                return "code_generation"

        reader: TopazSkillReader = _AdHocReader()
        router = TierRouter(defaults=_all_skills_defaults(), skill_reader=reader)

        selection = router.select_tier(ensemble_name="anything", verdict="proceed")
        assert selection.topaz_skill == "code_generation"


# ---------------------------------------------------------------------------
# Production reader — EnsembleConfigTopazSkillReader
# ---------------------------------------------------------------------------


class TestEnsembleConfigTopazSkillReader:
    """The default reader bridges EnsembleConfig to the router Protocol.

    Production wiring constructs this with a ``find_ensemble`` callable
    that resolves a name to an EnsembleConfig. Tests pass a scripted
    callable. The reader honors ADR-015's per-skill role profiling
    contract — operator-authored values outside the Topaz taxonomy
    return None, so the router raises MissingSkillMetadataError.
    """

    @staticmethod
    def _config_with_skill(name: str, skill: str | None) -> EnsembleConfig:
        return EnsembleConfig(
            name=name,
            description="",
            agents=[],
            topaz_skill=skill,
        )

    def test_returns_declared_skill_when_within_taxonomy(self) -> None:
        config = self._config_with_skill("code-review", "code_generation")
        reader = EnsembleConfigTopazSkillReader(
            find_ensemble=lambda name: config if name == "code-review" else None
        )

        assert reader.topaz_skill_for("code-review") == "code_generation"

    def test_returns_none_when_ensemble_not_found(self) -> None:
        reader = EnsembleConfigTopazSkillReader(find_ensemble=lambda _: None)

        assert reader.topaz_skill_for("missing") is None

    def test_returns_none_when_topaz_skill_field_absent(self) -> None:
        config = self._config_with_skill("legacy-ensemble", None)
        reader = EnsembleConfigTopazSkillReader(find_ensemble=lambda _: config)

        assert reader.topaz_skill_for("legacy-ensemble") is None

    def test_returns_none_when_value_outside_topaz_taxonomy(self) -> None:
        """An operator typo (``code-gen`` instead of ``code_generation``)
        surfaces as MissingSkillMetadataError at the router rather than
        silently mapping to a default — the closed taxonomy is load-
        bearing per ADR-015 §Per-skill role profiling."""
        config = self._config_with_skill("typo-ensemble", "code-gen")
        reader = EnsembleConfigTopazSkillReader(find_ensemble=lambda _: config)

        assert reader.topaz_skill_for("typo-ensemble") is None

    def test_supports_all_eight_topaz_skills(self) -> None:
        """Sanity: every skill in the closed Topaz taxonomy round-trips
        through the reader without truncation."""

        def _finder_for(
            config: EnsembleConfig,
        ) -> Callable[[str], EnsembleConfig | None]:
            def _find(_name: str) -> EnsembleConfig | None:
                return config

            return _find

        for skill in _ALL_TOPAZ_SKILLS:
            config = self._config_with_skill("e", skill)
            reader = EnsembleConfigTopazSkillReader(find_ensemble=_finder_for(config))
            assert reader.topaz_skill_for("e") == skill


class TestEnsembleConfigTopazSkillFieldOnLoad:
    """The EnsembleConfig YAML loader carries the ``topaz_skill`` field
    through to the runtime config so the production reader can find
    it (ADR-015)."""

    def test_loader_parses_topaz_skill_field_from_yaml(
        self, tmp_path: pathlib.Path
    ) -> None:
        from llm_orc.core.config.ensemble_config import EnsembleLoader

        yaml_path = tmp_path / "test-ensemble.yaml"
        yaml_path.write_text(
            "name: test-ensemble\n"
            "description: example\n"
            "topaz_skill: code_generation\n"
            "agents:\n"
            "  - name: a\n"
            "    model_profile: micro-local\n"
        )

        config = EnsembleLoader().load_from_file(str(yaml_path))

        assert config.topaz_skill == "code_generation"

    def test_loader_treats_missing_topaz_skill_as_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        from llm_orc.core.config.ensemble_config import EnsembleLoader

        yaml_path = tmp_path / "no-skill.yaml"
        yaml_path.write_text(
            "name: no-skill\n"
            "description: example\n"
            "agents:\n"
            "  - name: a\n"
            "    model_profile: micro-local\n"
        )

        config = EnsembleLoader().load_from_file(str(yaml_path))

        assert config.topaz_skill is None
