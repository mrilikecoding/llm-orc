"""Tests for Pydantic agent config models (ADR-012)."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llm_orc.schemas.agent_config import (
    LlmAgentConfig,
    ScriptAgentConfig,
    parse_agent_config,
)


class TestLlmAgentConfigParsing:
    """Scenario: LLM agent config parsed from YAML."""

    def test_model_profile_agent(self) -> None:
        data: dict[str, Any] = {"name": "analyzer", "model_profile": "gpt4"}
        config = parse_agent_config(data)
        assert isinstance(config, LlmAgentConfig)
        assert config.name == "analyzer"
        assert config.model_profile == "gpt4"

    def test_inline_model_agent(self) -> None:
        data: dict[str, Any] = {
            "name": "quick-test",
            "model": "llama3",
            "provider": "ollama",
        }
        config = parse_agent_config(data)
        assert isinstance(config, LlmAgentConfig)
        assert config.model == "llama3"
        assert config.provider == "ollama"

    def test_llm_agent_with_all_optional_fields(self) -> None:
        data: dict[str, Any] = {
            "name": "full",
            "model_profile": "gpt4",
            "system_prompt": "You are helpful.",
            "temperature": 0.5,
            "max_tokens": 1000,
            "timeout_seconds": 30,
            "output_format": "json",
            "fallback_model_profile": "local-fallback",
        }
        config = parse_agent_config(data)
        assert isinstance(config, LlmAgentConfig)
        assert config.system_prompt == "You are helpful."
        assert config.temperature == 0.5
        assert config.max_tokens == 1000


class TestScriptAgentConfigParsing:
    """Scenario: Script agent config parsed from YAML."""

    def test_script_agent(self) -> None:
        data: dict[str, Any] = {"name": "scanner", "script": "scripts/scan.py"}
        config = parse_agent_config(data)
        assert isinstance(config, ScriptAgentConfig)
        assert config.name == "scanner"
        assert config.script == "scripts/scan.py"

    def test_script_agent_with_parameters(self) -> None:
        data: dict[str, Any] = {
            "name": "processor",
            "script": "scripts/process.py",
            "parameters": {"input_dir": "/tmp"},
        }
        config = parse_agent_config(data)
        assert isinstance(config, ScriptAgentConfig)
        assert config.parameters == {"input_dir": "/tmp"}


class TestInlineModelRequiresProvider:
    """Scenario: Inline model requires provider."""

    def test_model_without_provider_rejected(self) -> None:
        data: dict[str, Any] = {"name": "test", "model": "llama3"}
        with pytest.raises(ValidationError, match="provider"):
            parse_agent_config(data)


class TestProfileXorInlineModel:
    """Scenario: Profile XOR inline model enforced."""

    def test_both_profile_and_inline_rejected(self) -> None:
        data: dict[str, Any] = {
            "name": "test",
            "model_profile": "gpt4",
            "model": "llama3",
            "provider": "ollama",
        }
        with pytest.raises(ValidationError):
            parse_agent_config(data)

    def test_neither_profile_nor_model_rejected(self) -> None:
        """An LLM agent with no model_profile, model, or script is invalid."""
        data: dict[str, Any] = {"name": "test", "system_prompt": "hello"}
        with pytest.raises(ValidationError):
            parse_agent_config(data)


class TestUnknownFieldsRejected:
    """Scenario: Unknown fields rejected."""

    def test_extra_field_on_llm_agent(self) -> None:
        data: dict[str, Any] = {
            "name": "test",
            "model_profile": "gpt4",
            "typo_field": "oops",
        }
        with pytest.raises(ValidationError, match="typo_field"):
            parse_agent_config(data)

    def test_extra_field_on_script_agent(self) -> None:
        data: dict[str, Any] = {
            "name": "test",
            "script": "scan.py",
            "bogus": True,
        }
        with pytest.raises(ValidationError, match="bogus"):
            parse_agent_config(data)


class TestBaseAgentConfigFields:
    """Test shared BaseAgentConfig fields."""

    def test_depends_on_defaults_to_empty(self) -> None:
        data: dict[str, Any] = {"name": "test", "model_profile": "gpt4"}
        config = parse_agent_config(data)
        assert config.depends_on == []

    def test_depends_on_list(self) -> None:
        data: dict[str, Any] = {
            "name": "test",
            "model_profile": "gpt4",
            "depends_on": ["agent-a", "agent-b"],
        }
        config = parse_agent_config(data)
        assert config.depends_on == ["agent-a", "agent-b"]

    def test_fan_out_defaults_to_false(self) -> None:
        data: dict[str, Any] = {"name": "test", "model_profile": "gpt4"}
        config = parse_agent_config(data)
        assert config.fan_out is False

    def test_fan_out_true(self) -> None:
        data: dict[str, Any] = {
            "name": "test",
            "model_profile": "gpt4",
            "fan_out": True,
            "depends_on": ["upstream"],
        }
        config = parse_agent_config(data)
        assert config.fan_out is True


class TestEnsembleLoaderProducesPydanticConfigs:
    """Scenario 7: EnsembleLoader produces list[AgentConfig]."""

    def test_loader_produces_typed_agent_configs(self) -> None:
        """EnsembleLoader.load_from_file() returns AgentConfig instances."""
        from llm_orc.core.config.ensemble_config import EnsembleLoader

        ensemble_yaml = {
            "name": "typed-ensemble",
            "description": "Ensemble with typed configs",
            "agents": [
                {"name": "llm-1", "model_profile": "fast-model"},
                {"name": "llm-2", "model_profile": "quality-model"},
                {
                    "name": "scanner",
                    "script": "scripts/scan.py",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            config = loader.load_from_file(yaml_path)

            assert len(config.agents) == 3
            assert isinstance(config.agents[0], LlmAgentConfig)
            assert isinstance(config.agents[1], LlmAgentConfig)
            assert isinstance(config.agents[2], ScriptAgentConfig)
            assert config.agents[0].name == "llm-1"
            assert config.agents[2].script == "scripts/scan.py"
        finally:
            Path(yaml_path).unlink()


class TestAgentDispatcherUsesIsinstance:
    """Scenario 6: Agent type determined by isinstance."""

    def test_isinstance_dispatch_for_llm_agent(self) -> None:
        """AgentDispatcher uses isinstance for LlmAgentConfig."""
        from llm_orc.core.execution.agent_dispatcher import (
            AgentDispatcher,
        )

        config = LlmAgentConfig(name="test", model_profile="gpt4")
        # _determine_agent_type should work with AgentConfig
        result = AgentDispatcher._determine_agent_type(None, config)  # type: ignore[arg-type]
        assert result == "llm"

    def test_isinstance_dispatch_for_script_agent(self) -> None:
        """AgentDispatcher uses isinstance for ScriptAgentConfig."""
        from llm_orc.core.execution.agent_dispatcher import (
            AgentDispatcher,
        )

        config = ScriptAgentConfig(name="test", script="scan.py")
        result = AgentDispatcher._determine_agent_type(None, config)  # type: ignore[arg-type]
        assert result == "script"


class TestAgentLevelOverridesWinOverProfile:
    """Scenario 8: Agent-level overrides win over profile defaults."""

    @pytest.mark.asyncio
    async def test_agent_temperature_overrides_profile(self) -> None:
        """Agent config temperature=0.2 overrides profile temperature=0.7."""
        from unittest.mock import Mock, patch

        from llm_orc.core.execution.llm_agent_runner import (
            LlmAgentRunner,
        )
        from llm_orc.core.execution.usage_collector import (
            UsageCollector,
        )

        config_manager = Mock()
        runner = LlmAgentRunner(
            model_factory=Mock(),
            config_manager=config_manager,
            usage_collector=UsageCollector(),
            emit_event=lambda _e, _d: None,
            classify_failure=lambda _e: "unknown",
        )

        config_manager.get_model_profiles.return_value = {
            "test-profile": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": 0.7,
            },
        }
        agent = LlmAgentConfig(
            name="test",
            model_profile="test-profile",
            temperature=0.2,
        )
        resolved = (
            await runner._resolve_model_profile_to_config(agent)
        )

        assert resolved["temperature"] == 0.2


class TestIntegrationExecutorWithPydanticConfigs:
    """Scenario 9: EnsembleExecutor runs with Pydantic agent configs."""

    @pytest.mark.asyncio
    async def test_yaml_to_executor_round_trip(self) -> None:
        """Load YAML via EnsembleLoader, execute via EnsembleExecutor."""
        from unittest.mock import AsyncMock, Mock, patch

        from llm_orc.core.config.ensemble_config import EnsembleLoader
        from llm_orc.core.execution.ensemble_execution import (
            EnsembleExecutor,
        )

        ensemble_yaml = {
            "name": "round-trip-test",
            "description": "YAML-to-executor integration",
            "agents": [
                {
                    "name": "scanner",
                    "script": "echo '{\"success\": true, \"data\": \"ok\"}'",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(ensemble_yaml, f)
            yaml_path = f.name

        try:
            loader = EnsembleLoader()
            config = loader.load_from_file(yaml_path)

            assert isinstance(config.agents[0], ScriptAgentConfig)

            executor = EnsembleExecutor()
            mock_artifact = Mock()
            mock_artifact.save_execution_results = Mock()

            with patch.object(
                executor, "_artifact_manager", mock_artifact
            ):
                result = await executor.execute(config, "test input")

            assert result["status"] in [
                "completed",
                "completed_with_errors",
            ]
            assert "scanner" in result["results"]
        finally:
            Path(yaml_path).unlink()
