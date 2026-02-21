"""Validation handler for MCP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    assert_no_cycles,
)
from llm_orc.mcp.utils import get_agent_attr as _get_agent_attr


def _is_script_agent(agent: Any) -> bool:
    """Check if agent is a script-based agent."""
    return (
        _get_agent_attr(agent, "type") == "script"
        or _get_agent_attr(agent, "script") is not None
    )


class ValidationHandler:
    """Handles ensemble validation operations."""

    def __init__(
        self,
        config_manager: Any,
        find_ensemble: Callable[[str], EnsembleConfig | None],
        get_all_profiles_fn: Callable[[], dict[str, dict[str, Any]]],
    ) -> None:
        self._config_manager = config_manager
        self._find_ensemble = find_ensemble
        self._get_all_profiles_fn = get_all_profiles_fn

    async def validate_ensemble(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate an ensemble configuration."""
        ensemble_name = arguments.get("ensemble_name")

        if not ensemble_name:
            raise ValueError("ensemble_name is required")

        config = self._find_ensemble(ensemble_name)
        if not config:
            raise ValueError(f"Ensemble not found: {ensemble_name}")

        validation_errors = self._collect_validation_errors(config)

        return {
            "valid": len(validation_errors) == 0,
            "details": {
                "errors": validation_errors,
                "agent_count": len(config.agents),
            },
        }

    def _collect_validation_errors(self, config: Any) -> list[str]:
        """Collect validation errors for an ensemble."""
        validation_errors: list[str] = []

        try:
            assert_no_cycles(config.agents)
        except ValueError as e:
            validation_errors.append(str(e))

        validation_errors.extend(self._validate_agent_references(config))
        validation_errors.extend(self._validate_model_profiles(config))

        return validation_errors

    def _validate_agent_references(self, config: Any) -> list[str]:
        """Validate agent dependency references."""
        errors: list[str] = []
        agent_names = {_get_agent_attr(agent, "name") for agent in config.agents}

        for agent in config.agents:
            depends_on = _get_agent_attr(agent, "depends_on") or []
            for dep in depends_on:
                if dep not in agent_names:
                    agent_name = _get_agent_attr(agent, "name")
                    errors.append(
                        f"Agent '{agent_name}' depends on unknown agent '{dep}'"
                    )

        return errors

    def _validate_model_profiles(self, config: Any) -> list[str]:
        """Validate model profiles exist and are configured."""
        errors: list[str] = []
        available_profiles = self._get_all_profiles_fn()

        for agent in config.agents:
            agent_name = _get_agent_attr(agent, "name")

            if _is_script_agent(agent):
                continue

            model_profile = _get_agent_attr(agent, "model_profile")
            if not model_profile:
                errors.append(f"Agent '{agent_name}' has no model_profile configured")
                continue

            if model_profile not in available_profiles:
                errors.append(
                    f"Agent '{agent_name}' uses unknown profile '{model_profile}'"
                )
                continue

            profile_config = available_profiles[model_profile]
            provider = profile_config.get("provider")
            if not provider:
                errors.append(
                    f"Profile '{model_profile}' missing 'provider' configuration"
                )
            else:
                from llm_orc.providers.registry import (
                    provider_registry,
                )

                if not provider_registry.provider_exists(provider):
                    errors.append(
                        f"Profile '{model_profile}' uses unknown provider '{provider}'"
                    )

            if not profile_config.get("model"):
                errors.append(
                    f"Profile '{model_profile}' missing 'model' configuration"
                )

        return errors
