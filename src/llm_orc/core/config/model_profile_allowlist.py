"""Model-profile allowlist for the ``/v1/models`` endpoint.

The Serving Layer's ``/v1/models`` endpoint advertises the model-profile
IDs an operator has exposed (OpenAI-compatible), so agentic coding tools
can populate their model picker. The allowlist is the operator-configured
``agentic_serving.orchestrator.allowed_profiles`` intersected with the
Model Profile library.

Relocated out of the dissolved ``agentic/orchestrator_config.py`` at
Cycle-8 WP-F8 (ADR-046 orchestrator-actor dissolution): only the allowlist
resolution survives. The orchestrator-actor config surface it used to sit
beside (budget, autonomy, calibration, tier defaults, compaction,
observability) dissolved with the actor.
"""

from __future__ import annotations

from typing import Any

from llm_orc.core.config.config_manager import ConfigurationManager

DEFAULT_MODEL_PROFILE = "default"


class ModelProfileAllowlist:
    """Resolves the operator-exposed model-profile allowlist for /v1/models."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self._config_manager = config_manager

    def list_allowed_model_profile_ids(self) -> tuple[str, ...]:
        """Return the allowlist intersected with the Model Profile library.

        Preserves the operator-configured ordering; drops names that do not
        correspond to a profile in
        ``ConfigurationManager.get_model_profiles()``. The endpoint
        enumerates what is actually resolvable.
        """
        raw = self._config_manager.load_agentic_serving_config()
        orchestrator = raw.get("orchestrator")
        if not isinstance(orchestrator, dict):
            orchestrator = {}
        model_profile = str(orchestrator.get("model_profile", DEFAULT_MODEL_PROFILE))
        allowed = _resolve_allowed_profiles(
            orchestrator.get("allowed_profiles"), model_profile
        )
        library = self._config_manager.get_model_profiles()
        return tuple(name for name in allowed if name in library)


def _resolve_allowed_profiles(raw: Any, model_profile: str) -> tuple[str, ...]:
    """Normalize the operator-configured allowlist.

    An absent or malformed allowlist falls back to ``(model_profile,)`` so a
    single-profile deployment works without additional configuration.
    """
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw)
    return (model_profile,)
