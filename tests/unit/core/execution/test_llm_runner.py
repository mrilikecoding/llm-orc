"""Tests for LlmAgentRunner role-loading behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest


class TestLoadRoleHandlesAbsentSystemPrompt:
    """When AgentConfig has no system_prompt (or system_prompt=None),
    the runner falls back to a default prompt rather than producing None.

    Regression: single-agent capability ensembles authored without an
    agent-level system_prompt (relying on ensemble-level default_task)
    previously errored at agent execution time with
    "unsupported operand type(s) for +: 'NoneType' and 'str'" because
    the None role prompt propagated to `models/ollama.py`'s
    `role_prompt + message` token-estimation call.
    """

    @pytest.mark.asyncio
    async def test_load_role_falls_back_when_system_prompt_omitted(self) -> None:
        """LlmAgentConfig without explicit system_prompt yields fallback prompt."""
        from llm_orc.core.execution.runners.llm_runner import LlmAgentRunner
        from llm_orc.core.execution.usage_collector import UsageCollector
        from llm_orc.schemas.agent_config import LlmAgentConfig

        config_manager = Mock()
        config_manager.get_model_profiles.return_value = {
            "test-profile": {"model": "gpt-4", "provider": "openai"},
        }
        runner = LlmAgentRunner(
            model_factory=Mock(),
            config_manager=config_manager,
            usage_collector=UsageCollector(),
            emit_event=lambda _e, _d: None,
            classify_failure=lambda _e: "unknown",
        )

        agent = LlmAgentConfig(
            name="extractor",
            model_profile="test-profile",
        )

        role = await runner._load_role_from_config(agent)

        assert role.name == "extractor"
        assert role.prompt is not None
        assert "extractor" in role.prompt

    @pytest.mark.asyncio
    async def test_load_role_falls_back_when_system_prompt_explicit_none(
        self,
    ) -> None:
        """LlmAgentConfig with explicit system_prompt=None yields fallback prompt."""
        from llm_orc.core.execution.runners.llm_runner import LlmAgentRunner
        from llm_orc.core.execution.usage_collector import UsageCollector
        from llm_orc.schemas.agent_config import LlmAgentConfig

        config_manager = Mock()
        config_manager.get_model_profiles.return_value = {
            "test-profile": {"model": "gpt-4", "provider": "openai"},
        }
        runner = LlmAgentRunner(
            model_factory=Mock(),
            config_manager=config_manager,
            usage_collector=UsageCollector(),
            emit_event=lambda _e, _d: None,
            classify_failure=lambda _e: "unknown",
        )

        agent = LlmAgentConfig(
            name="mapper",
            model_profile="test-profile",
            system_prompt=None,
        )

        role = await runner._load_role_from_config(agent)

        assert role.prompt is not None
        assert "mapper" in role.prompt

    @pytest.mark.asyncio
    async def test_load_role_uses_explicit_system_prompt(self) -> None:
        """Regression: explicit system_prompt is preserved unchanged."""
        from llm_orc.core.execution.runners.llm_runner import LlmAgentRunner
        from llm_orc.core.execution.usage_collector import UsageCollector
        from llm_orc.schemas.agent_config import LlmAgentConfig

        config_manager = Mock()
        config_manager.get_model_profiles.return_value = {
            "test-profile": {"model": "gpt-4", "provider": "openai"},
        }
        runner = LlmAgentRunner(
            model_factory=Mock(),
            config_manager=config_manager,
            usage_collector=UsageCollector(),
            emit_event=lambda _e, _d: None,
            classify_failure=lambda _e: "unknown",
        )

        agent = LlmAgentConfig(
            name="coder",
            model_profile="test-profile",
            system_prompt="You are a coding assistant.",
        )

        role = await runner._load_role_from_config(agent)

        assert role.prompt == "You are a coding assistant."
