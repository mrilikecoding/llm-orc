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

from dataclasses import dataclass
from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager

DEFAULT_MODEL_PROFILE = "default"
DEFAULT_TURN_LIMIT = 500
DEFAULT_TOKEN_LIMIT = 10_000_000
DEFAULT_AUTONOMY_LEVEL = "operator-as-tool-user"
DEFAULT_PLEXUS_ENABLED = False
DEFAULT_ALLOW_BUDGET_OVERRIDE = True
DEFAULT_MAX_TURN_LIMIT = 1_000
DEFAULT_MAX_TOKEN_LIMIT = 50_000_000


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

        return OrchestratorConfig(
            model_profile=str(orchestrator.get("model_profile", DEFAULT_MODEL_PROFILE)),
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
        )


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a config section to a dict, tolerating missing / malformed sections."""
    if isinstance(value, dict):
        return value
    return {}
