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
    OrchestratorConfigResolver,
)
from llm_orc.core.config.config_manager import ConfigurationManager


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
