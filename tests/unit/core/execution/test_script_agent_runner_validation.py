"""Tests for ScriptAgentRunner opt-in output validation."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm_orc.core.execution.script_agent_runner import ScriptAgentRunner
from llm_orc.schemas.agent_config import ScriptAgentConfig


def _make_runner() -> ScriptAgentRunner:
    """Create a ScriptAgentRunner with mocked dependencies."""
    cache = Mock()
    cache.get.return_value = None
    return ScriptAgentRunner(
        script_cache=cache,
        usage_collector=Mock(),
        progress_controller=Mock(),
        emit_event=Mock(),
        project_dir=None,
    )


class TestOutputValidation:
    """Test opt-in output validation for known primitives."""

    @pytest.mark.asyncio
    async def test_valid_primitive_output_passes(self) -> None:
        """Valid output from a known primitive passes validation silently."""
        runner = _make_runner()
        valid_output = json.dumps(
            {
                "success": True,
                "input": "hello",
                "multiline": False,
                "length": 5,
            }
        )

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
            parameters={"prompt": "Enter:"},
        )

        # Mock the input handling to return our controlled output
        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=valid_output,
        ):
            response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        assert json.loads(response)["success"] is True

    @pytest.mark.asyncio
    async def test_invalid_primitive_output_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid output from a known primitive logs a warning."""
        runner = _make_runner()
        # Missing required 'success' field for GetUserInputOutput
        invalid_output = json.dumps({"unexpected": "data"})

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
            parameters={"prompt": "Enter:"},
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=invalid_output,
        ):
            with caplog.at_level(logging.WARNING):
                response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        # Output still passes through despite validation failure
        assert json.loads(response) == {"unexpected": "data"}
        assert any("validation failed" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_primitive_skips_validation(self) -> None:
        """Non-primitive scripts bypass validation entirely."""
        runner = _make_runner()
        output = json.dumps({"custom": "output"})

        agent_config = ScriptAgentConfig(
            name="custom_agent",
            script="scripts/custom/my_script.py",
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=output,
        ):
            response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        assert json.loads(response) == {"custom": "output"}

    @pytest.mark.asyncio
    async def test_dict_response_gets_serialized_then_validated(
        self,
    ) -> None:
        """Dict responses get serialized before validation."""
        runner = _make_runner()

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "input": "hi",
                "multiline": False,
                "length": 2,
            },
        ):
            response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        # Dict gets serialized to JSON string
        assert json.loads(response)["success"] is True


class TestCacheHitValidation:
    """Cache hits run through validation just like fresh executions."""

    @pytest.mark.asyncio
    async def test_cache_hit_calls_validate_primitive_output(self) -> None:
        """Cached results are validated before being returned."""
        cache = Mock()
        # Seed cache with a pre-existing result
        cache.get.return_value = {
            "output": '{"cached": true}',
            "success": True,
        }
        runner = ScriptAgentRunner(
            script_cache=cache,
            usage_collector=Mock(),
            progress_controller=Mock(),
            emit_event=Mock(),
            project_dir=None,
        )

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
        )

        with patch.object(runner, "_validate_primitive_output") as mock_validate:
            await runner.execute(agent_config, "{}")

        mock_validate.assert_called_once_with(
            "primitives/user-interaction/get_user_input.py",
            '{"cached": true}',
        )


def _make_strict_runner() -> ScriptAgentRunner:
    """Create a ScriptAgentRunner with strict_schema enabled."""
    cache = Mock()
    cache.get.return_value = None
    return ScriptAgentRunner(
        script_cache=cache,
        usage_collector=Mock(),
        progress_controller=Mock(),
        emit_event=Mock(),
        project_dir=None,
        strict_schema=True,
    )


class TestStrictSchemaValidation:
    """strict_schema=True raises ValueError on schema violations."""

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_invalid_output(self) -> None:
        """Invalid output raises ValueError when strict_schema is True."""
        runner = _make_strict_runner()
        invalid_output = json.dumps({"unexpected": "data"})  # missing required fields

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
            parameters={"prompt": "Enter:"},
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=invalid_output,
        ):
            with pytest.raises(ValueError, match="schema validation failed"):
                await runner._execute_without_cache(agent_config, "{}")

    @pytest.mark.asyncio
    async def test_non_strict_mode_warns_on_invalid_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid output logs a warning (not raises) when strict_schema is False."""
        runner = _make_runner()  # strict_schema=False by default
        invalid_output = json.dumps({"unexpected": "data"})

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
            parameters={"prompt": "Enter:"},
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=invalid_output,
        ):
            with caplog.at_level(logging.WARNING):
                response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        assert json.loads(response) == {"unexpected": "data"}
        assert any("validation failed" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_strict_mode_passes_on_valid_output(self) -> None:
        """Valid output does not raise even in strict mode."""
        runner = _make_strict_runner()
        valid_output = json.dumps(
            {"success": True, "input": "hi", "multiline": False, "length": 2}
        )

        agent_config = ScriptAgentConfig(
            name="test_agent",
            script="primitives/user-interaction/get_user_input.py",
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=valid_output,
        ):
            response, _, _ = await runner._execute_without_cache(agent_config, "{}")

        assert json.loads(response)["success"] is True


class TestModelSubstitutedFlag:
    """Script agent always returns model_substituted=False."""

    @pytest.mark.asyncio
    async def test_script_agent_never_substitutes_model(self) -> None:
        """ScriptAgentRunner.execute always returns model_substituted=False."""
        runner = _make_runner()
        output = json.dumps({"custom": "output"})

        agent_config = ScriptAgentConfig(
            name="custom_agent",
            script="scripts/custom/my_script.py",
        )

        with patch.object(
            runner,
            "_execute_with_input_handling",
            new_callable=AsyncMock,
            return_value=output,
        ):
            _, _, model_substituted = await runner._execute_without_cache(
                agent_config, "{}"
            )

        assert model_substituted is False, "Script agents never substitute models"

    @pytest.mark.asyncio
    async def test_script_execute_returns_false_on_cache_hit(self) -> None:
        """execute() returns model_substituted=False even on cache hit."""
        cache = Mock()
        cache.get.return_value = {"output": '{"cached": true}', "success": True}
        runner = ScriptAgentRunner(
            script_cache=cache,
            usage_collector=Mock(),
            progress_controller=Mock(),
            emit_event=Mock(),
            project_dir=None,
        )

        agent_config = ScriptAgentConfig(name="cached_agent", script="scripts/any.py")
        _, _, model_substituted = await runner.execute(agent_config, "{}")

        assert model_substituted is False, "Cache hits never substitute models"
