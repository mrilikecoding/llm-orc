"""Unit tests for ValidationHandler."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig
from llm_orc.services.handlers.validation_handler import ValidationHandler


def _llm_agent(
    name: str,
    model_profile: str = "some-profile",
    depends_on: list[str | dict[str, Any]] | None = None,
) -> LlmAgentConfig:
    return LlmAgentConfig(
        name=name,
        model_profile=model_profile,
        depends_on=depends_on or [],
    )


def _script_agent(name: str) -> ScriptAgentConfig:
    return ScriptAgentConfig(name=name, script="echo hi")


def _make_config(agents: list[Any]) -> Any:
    config = MagicMock()
    config.agents = agents
    return config


def _make_handler(
    find_ensemble_result: Any = None,
    available_profiles: dict[str, dict[str, Any]] | None = None,
) -> ValidationHandler:
    config_manager = MagicMock()
    find_ensemble = MagicMock(return_value=find_ensemble_result)
    profiles = available_profiles if available_profiles is not None else {}
    get_all_profiles_fn: Any = MagicMock(return_value=profiles)
    return ValidationHandler(config_manager, find_ensemble, get_all_profiles_fn)


# ---------------------------------------------------------------------------
# validate_ensemble — ensemble not found
# ---------------------------------------------------------------------------


class TestValidateEnsembleNotFound:
    """validate_ensemble raises when ensemble is missing (line 50)."""

    async def test_raises_when_ensemble_not_found(self) -> None:
        handler = _make_handler(find_ensemble_result=None)

        with pytest.raises(ValueError, match="Ensemble not found: missing"):
            await handler.validate_ensemble({"ensemble_name": "missing"})


# ---------------------------------------------------------------------------
# _collect_validation_errors — cycle detection
# ---------------------------------------------------------------------------


class TestCollectValidationErrorsCycles:
    """_collect_validation_errors catches cycle errors (lines 68-69)."""

    async def test_cycle_error_is_collected(self) -> None:
        """A ValueError from assert_no_cycles is appended to the error list."""
        agent = _llm_agent("a")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "some-profile": {"provider": "ollama", "model": "llama3"}
            },
        )

        with patch(
            "llm_orc.services.handlers.validation_handler.assert_no_cycles",
            side_effect=ValueError("Cycle detected: a -> b -> a"),
        ):
            result = await handler.validate_ensemble({"ensemble_name": "my-ensemble"})

        assert result["valid"] is False
        assert any("Cycle" in e for e in result["details"]["errors"])


# ---------------------------------------------------------------------------
# _validate_agent_references
# ---------------------------------------------------------------------------


class TestValidateAgentReferences:
    """_validate_agent_references detects unknown dependency (lines 84-86)."""

    async def test_unknown_dependency_is_reported(self) -> None:
        """Agent depending on a nonexistent agent produces an error."""
        agent = _llm_agent("a", depends_on=["nonexistent"])
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "some-profile": {"provider": "ollama", "model": "llama3"}
            },
        )

        with patch(
            "llm_orc.providers.registry.provider_registry.provider_exists",
            return_value=True,
        ):
            result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        errors = result["details"]["errors"]
        assert any("nonexistent" in e for e in errors)

    async def test_known_dependency_passes(self) -> None:
        """Agent depending on another agent in the same config is valid."""
        agent_a = _llm_agent("a", depends_on=["b"])
        agent_b = _llm_agent("b")
        config = _make_config([agent_a, agent_b])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "some-profile": {"provider": "ollama", "model": "llama3"}
            },
        )

        with patch(
            "llm_orc.providers.registry.provider_registry.provider_exists",
            return_value=True,
        ):
            result = await handler.validate_ensemble({"ensemble_name": "x"})

        errors = result["details"]["errors"]
        assert not any("depends on unknown" in e for e in errors)


# ---------------------------------------------------------------------------
# _validate_model_profiles
# ---------------------------------------------------------------------------


class TestValidateModelProfiles:
    """_validate_model_profiles covers lines 101, 105-106, 109-112, 117, 126, 131."""

    async def test_script_agent_is_skipped(self) -> None:
        """Script agents bypass profile validation (line 101)."""
        agent = _script_agent("s")
        config = _make_config([agent])

        handler = _make_handler(find_ensemble_result=config, available_profiles={})

        result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is True

    async def test_missing_model_profile_is_reported(self) -> None:
        """Non-script agent without model_profile gets an error (lines 105-106).

        We bypass LlmAgentConfig validation (which also requires model_profile)
        by supplying a plain mock agent object.
        """
        agent = MagicMock()
        agent.name = "a"
        # type != "script" and script is None so _is_script_agent returns False
        del agent.type  # avoid hasattr returning True for .type
        agent.depends_on = []
        # Patch _get_agent_attr so model_profile returns None
        config = _make_config([agent])

        handler = _make_handler(find_ensemble_result=config, available_profiles={})

        with patch(
            "llm_orc.services.handlers.validation_handler._get_agent_attr",
        ) as mock_get:

            def side_effect(a: Any, attr: str, default: Any = None) -> Any:
                mapping: dict[str, Any] = {
                    "name": "a",
                    "type": None,
                    "script": None,
                    "depends_on": [],
                    "model_profile": None,
                }
                return mapping.get(attr, default)

            mock_get.side_effect = side_effect

            result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        assert any("no model_profile" in e for e in result["details"]["errors"])

    async def test_unknown_profile_is_reported(self) -> None:
        """Agent referencing a nonexistent profile gets an error (lines 109-112)."""
        agent = _llm_agent("a", model_profile="ghost-profile")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={},
        )

        result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        assert any("ghost-profile" in e for e in result["details"]["errors"])

    async def test_profile_missing_provider_is_reported(self) -> None:
        """Profile without 'provider' key gets an error (line 117)."""
        agent = _llm_agent("a", model_profile="no-provider-profile")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "no-provider-profile": {"model": "llama3"},
            },
        )

        result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        assert any("missing 'provider'" in e for e in result["details"]["errors"])

    async def test_unknown_provider_is_reported(self) -> None:
        """Profile with unregistered provider gets an error (line 126)."""
        agent = _llm_agent("a", model_profile="bad-provider-profile")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "bad-provider-profile": {
                    "provider": "nonexistent-provider",
                    "model": "llama3",
                },
            },
        )

        with patch(
            "llm_orc.providers.registry.provider_registry.provider_exists",
            return_value=False,
        ):
            result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        assert any("unknown provider" in e for e in result["details"]["errors"])

    async def test_profile_missing_model_is_reported(self) -> None:
        """Profile without 'model' key gets an error (line 131)."""
        agent = _llm_agent("a", model_profile="no-model-profile")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "no-model-profile": {
                    "provider": "ollama",
                },
            },
        )

        with patch(
            "llm_orc.providers.registry.provider_registry.provider_exists",
            return_value=True,
        ):
            result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is False
        assert any("missing 'model'" in e for e in result["details"]["errors"])

    async def test_valid_profile_produces_no_errors(self) -> None:
        """A fully valid agent+profile passes validation."""
        agent = _llm_agent("a", model_profile="good-profile")
        config = _make_config([agent])

        handler = _make_handler(
            find_ensemble_result=config,
            available_profiles={
                "good-profile": {"provider": "ollama", "model": "llama3"},
            },
        )

        with patch(
            "llm_orc.providers.registry.provider_registry.provider_exists",
            return_value=True,
        ):
            result = await handler.validate_ensemble({"ensemble_name": "x"})

        assert result["valid"] is True


# ---------------------------------------------------------------------------
# set_project_context
# ---------------------------------------------------------------------------


class TestSetProjectContext:
    """set_project_context replaces config_manager on the handler."""

    def test_set_project_context_updates_config_manager(self) -> None:
        handler = _make_handler()
        new_config = MagicMock()
        ctx = MagicMock()
        ctx.config_manager = new_config

        handler.set_project_context(ctx)

        assert handler._config_manager is new_config
