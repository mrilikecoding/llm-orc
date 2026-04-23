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
DEFAULT_SUMMARIZER_ENSEMBLE = "agentic-result-summarizer"

DEFAULT_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the llm-orc orchestrator. You route tool-user tasks by invoking \
ensembles, composing new ones from library primitives, and (when Plexus is \
active) querying or recording against a knowledge graph.

Your internal tool surface is exactly five functions:
- invoke_ensemble(name, input) — run a library ensemble on a task input.
- compose_ensemble(...) — create a new ensemble from existing primitives.
- list_ensembles() — list ensembles in the library with descriptions.
- query_knowledge(...) — query the knowledge graph (Plexus, when active).
- record_outcome(...) — record a routing decision or outcome.

Every tool call the tool user's client declares (e.g. file_read, bash, \
file_edit) is a *client-declared* tool. When you need a client-declared \
tool, emit it alone in a single assistant turn — the turn will close and \
the client will execute it; you resume on the next request with the result \
as a role:tool observation. Never mix internal tools and client-declared \
tools in the same assistant turn — emit one kind per turn.

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
