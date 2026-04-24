"""Tests for the Orchestrator Configuration module.

Per `docs/agentic-serving/system-design.md` §Orchestrator Configuration (L3).
Resolves the orchestrator Model Profile (ADR-011), Budget defaults (ADR-005),
Autonomy default (ADR-008), Plexus enablement flag (ADR-009), and
operator-set bounds on per-request overrides from the existing
ConfigurationManager. Values follow defaults → global → local precedence.
"""

from pathlib import Path

import pytest
import yaml

from llm_orc.agentic.orchestrator_config import (
    DEFAULT_AUTONOMY_LEVEL,
    DEFAULT_MAX_TOKEN_LIMIT,
    DEFAULT_MAX_TURN_LIMIT,
    DEFAULT_MODEL_PROFILE,
    DEFAULT_PLEXUS_ENABLED,
    DEFAULT_TOKEN_LIMIT,
    DEFAULT_TURN_LIMIT,
    ModelProfileNotFoundError,
    OrchestratorConfigResolver,
)
from llm_orc.core.config.config_manager import ConfigurationManager


def _seed_model_profiles(global_yaml: dict[str, object], names: list[str]) -> None:
    """Merge a model_profiles section with the given names into a global config dict."""
    profiles = {name: {"model": "dummy-model", "provider": "dummy"} for name in names}
    existing = global_yaml.get("model_profiles")
    if isinstance(existing, dict):
        existing.update(profiles)
    else:
        global_yaml["model_profiles"] = profiles


def _make_config_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    global_yaml: dict[str, object] | None = None,
    local_yaml: dict[str, object] | None = None,
) -> ConfigurationManager:
    """Construct a ConfigurationManager with an isolated global/local root.

    Uses XDG_CONFIG_HOME to redirect the global config root into tmp_path.
    """
    global_root = tmp_path / "xdg"
    global_root.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(global_root))

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    if global_yaml is not None:
        global_llm_orc = global_root / "llm-orc"
        global_llm_orc.mkdir(parents=True, exist_ok=True)
        (global_llm_orc / "config.yaml").write_text(yaml.safe_dump(global_yaml))

    if local_yaml is not None:
        local_llm_orc = project_dir / ".llm-orc"
        local_llm_orc.mkdir()
        (local_llm_orc / "config.yaml").write_text(yaml.safe_dump(local_yaml))

    return ConfigurationManager(project_dir=project_dir, provision=False)


class TestOrchestratorConfigResolver:
    """resolve() merges defaults → global → local per established precedence."""

    def test_returns_defaults_when_no_config_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.model_profile == DEFAULT_MODEL_PROFILE
        assert config.budget.turn_limit == DEFAULT_TURN_LIMIT
        assert config.budget.token_limit == DEFAULT_TOKEN_LIMIT
        assert config.autonomy_level == DEFAULT_AUTONOMY_LEVEL
        assert config.plexus_enabled is DEFAULT_PLEXUS_ENABLED
        assert config.override_bounds.max_turn_limit == DEFAULT_MAX_TURN_LIMIT
        assert config.override_bounds.max_token_limit == DEFAULT_MAX_TOKEN_LIMIT
        # WP-H: Calibration defaults come from the Calibration Gate module.
        assert config.calibration.default_n == 3
        assert config.calibration.checker_ensemble == "agentic-calibration-checker"

    def test_global_config_overlays_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            global_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "global-profile"},
                    "budget": {"turn_limit": 111},
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.model_profile == "global-profile"
        assert config.budget.turn_limit == 111
        # unset fields still come from defaults
        assert config.budget.token_limit == DEFAULT_TOKEN_LIMIT
        assert config.autonomy_level == DEFAULT_AUTONOMY_LEVEL

    def test_local_config_overlays_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            global_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "global-profile"},
                    "budget": {"turn_limit": 111, "token_limit": 222},
                }
            },
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "local-profile"},
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.model_profile == "local-profile"
        # fields only set globally survive local overlay
        assert config.budget.turn_limit == 111
        assert config.budget.token_limit == 222

    def test_all_fields_resolvable_from_full_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "gpt-4o-orchestrator"},
                    "budget": {"turn_limit": 300, "token_limit": 800_000},
                    "autonomy": {"default_level": "fully-autonomous"},
                    "plexus": {"enabled": True},
                    "overrides": {
                        "allow_budget_override": False,
                        "max_turn_limit": 1000,
                        "max_token_limit": 5_000_000,
                    },
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.model_profile == "gpt-4o-orchestrator"
        assert config.budget.turn_limit == 300
        assert config.budget.token_limit == 800_000
        assert config.autonomy_level == "fully-autonomous"
        assert config.plexus_enabled is True
        assert config.override_bounds.allow_budget_override is False
        assert config.override_bounds.max_turn_limit == 1000
        assert config.override_bounds.max_token_limit == 5_000_000

    def test_calibration_overrides_land_under_orchestrator_calibration_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operators can tighten N and swap the checker via config.yaml."""
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {
                        "calibration": {
                            "default_n": 7,
                            "checker_ensemble": "strict-checker",
                        }
                    },
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.calibration.default_n == 7
        assert config.calibration.checker_ensemble == "strict-checker"

    def test_invalid_calibration_default_n_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero, negative, or non-integer values fall back to the shipped default."""
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"calibration": {"default_n": 0}},
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.calibration.default_n == 3

    def test_default_orchestrator_system_prompt_teaches_retry_convention(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default system prompt mentions all load-bearing disciplines.

        Phase 1 orchestrators operate behind a prompt that teaches (a) the
        five internal tools, (b) Option C's one-kind-per-turn discipline,
        and (c) the ``needs_client_tool`` retry convention for composed
        ensembles (roadmap ODP #8 mechanism i). The exact wording is an
        operator-tunable default; this test verifies all three topics
        appear in the default so an operator override that misses a
        topic is a visible regression rather than a silent one.
        """
        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        prompt = config.orchestrator_system_prompt
        assert "invoke_ensemble" in prompt
        assert "compose_ensemble" in prompt
        assert "list_ensembles" in prompt
        assert "query_knowledge" in prompt
        assert "record_outcome" in prompt
        assert "client-declared" in prompt or "client tool" in prompt.lower()
        assert "needs_client_tool" in prompt

    def test_operator_override_replaces_default_system_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``orchestrator.system_prompt`` in config.yaml wins over the default."""
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {
                        "system_prompt": "Operator override.",
                    }
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.orchestrator_system_prompt == "Operator override."

    def test_resolved_config_is_frozen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolved config is immutable for the session's duration (ADR-011)."""
        import dataclasses

        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert dataclasses.is_dataclass(config)
        # frozen dataclass rejects attribute assignment
        try:
            config.model_profile = "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("OrchestratorConfig should be frozen")


class TestOrchestratorAllowedProfiles:
    """allowed_profiles shapes what /v1/models will expose.

    When unset, it defaults to [model_profile] so single-profile deployments
    work without extra configuration. Operators extend the allowlist to
    expose additional orchestrator profiles.
    """

    def test_allowed_profiles_defaults_to_configured_model_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.allowed_profiles == (DEFAULT_MODEL_PROFILE,)

    def test_allowed_profiles_defaults_follow_overridden_model_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "custom-orchestrator"},
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        # Unspecified allowlist falls back to the configured model_profile only.
        assert config.allowed_profiles == ("custom-orchestrator",)

    def test_allowed_profiles_reads_operator_configured_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {
                        "model_profile": "primary",
                        "allowed_profiles": ["primary", "fast", "deep"],
                    }
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.allowed_profiles == ("primary", "fast", "deep")

    def test_local_allowed_profiles_override_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            global_yaml={
                "agentic_serving": {
                    "orchestrator": {
                        "model_profile": "g",
                        "allowed_profiles": ["g", "g-alt"],
                    }
                }
            },
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"allowed_profiles": ["only-local"]}
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve()

        assert config.allowed_profiles == ("only-local",)
        # model_profile is inherited from global; overlay did not replace it.
        assert config.model_profile == "g"


class TestListAllowedModelProfileIds:
    """list_allowed_model_profile_ids intersects the allowlist with the library.

    Consumer: the Serving Layer's /v1/models endpoint. Returns only profile
    IDs that are both configured as allowed AND present in
    ConfigurationManager.get_model_profiles() — absent names silently drop
    out of the list (the endpoint is a shop window, not a validator).
    """

    def test_returns_intersection_with_configured_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_yaml: dict[str, object] = {
            "agentic_serving": {
                "orchestrator": {
                    "model_profile": "primary",
                    "allowed_profiles": ["primary", "fast", "missing"],
                }
            }
        }
        _seed_model_profiles(global_yaml, ["primary", "fast", "unrelated"])
        cm = _make_config_manager(tmp_path, monkeypatch, global_yaml=global_yaml)
        resolver = OrchestratorConfigResolver(cm)

        ids = resolver.list_allowed_model_profile_ids()

        assert ids == ("primary", "fast")

    def test_preserves_allowlist_ordering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_yaml: dict[str, object] = {
            "agentic_serving": {
                "orchestrator": {
                    "model_profile": "a",
                    "allowed_profiles": ["c", "a", "b"],
                }
            }
        }
        _seed_model_profiles(global_yaml, ["a", "b", "c"])
        cm = _make_config_manager(tmp_path, monkeypatch, global_yaml=global_yaml)
        resolver = OrchestratorConfigResolver(cm)

        assert resolver.list_allowed_model_profile_ids() == ("c", "a", "b")

    def test_returns_empty_when_no_library_profiles_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        assert resolver.list_allowed_model_profile_ids() == ()

    def test_default_allowlist_resolves_against_library(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Allowlist defaults to [model_profile]; when that profile exists,
        the listing returns it.
        """
        global_yaml: dict[str, object] = {
            "agentic_serving": {"orchestrator": {"model_profile": "resolved-default"}}
        }
        _seed_model_profiles(global_yaml, ["resolved-default", "extra"])
        cm = _make_config_manager(tmp_path, monkeypatch, global_yaml=global_yaml)
        resolver = OrchestratorConfigResolver(cm)

        # Default allowlist is (model_profile,); "extra" is in the library
        # but not allowed, so it is excluded.
        assert resolver.list_allowed_model_profile_ids() == ("resolved-default",)


class TestResolveValidated:
    """resolve_validated raises when model_profile is absent from the library.

    This is the session-start seam (FF #40): sessions cannot begin on a
    profile that does not exist. ``resolve`` stays pure translation so
    ``/v1/models`` can enumerate whatever is available without raising.
    """

    def test_raises_when_configured_model_profile_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(tmp_path, monkeypatch)
        resolver = OrchestratorConfigResolver(cm)

        with pytest.raises(ModelProfileNotFoundError):
            resolver.resolve_validated()

    def test_raises_names_missing_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cm = _make_config_manager(
            tmp_path,
            monkeypatch,
            local_yaml={
                "agentic_serving": {
                    "orchestrator": {"model_profile": "no-such-profile"}
                }
            },
        )
        resolver = OrchestratorConfigResolver(cm)

        with pytest.raises(ModelProfileNotFoundError, match="no-such-profile"):
            resolver.resolve_validated()

    def test_returns_config_when_model_profile_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_yaml: dict[str, object] = {
            "agentic_serving": {"orchestrator": {"model_profile": "configured"}}
        }
        _seed_model_profiles(global_yaml, ["configured"])
        cm = _make_config_manager(tmp_path, monkeypatch, global_yaml=global_yaml)
        resolver = OrchestratorConfigResolver(cm)

        config = resolver.resolve_validated()

        assert config.model_profile == "configured"
        # resolve_validated returns the same shape as resolve, just checked.
        assert config == resolver.resolve()
