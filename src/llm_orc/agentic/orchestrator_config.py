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
    allowed_profiles: tuple[str, ...]


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


def _resolve_allowed_profiles(raw: Any, model_profile: str) -> tuple[str, ...]:
    """Normalize the operator-configured allowlist.

    An absent or malformed allowlist falls back to ``(model_profile,)`` so a
    single-profile deployment works without additional configuration.
    """
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw)
    return (model_profile,)
