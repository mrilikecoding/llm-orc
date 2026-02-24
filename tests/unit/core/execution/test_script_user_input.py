"""Tests for script agent user input handling functionality."""

from unittest.mock import Mock

from llm_orc.core.execution.scripting.user_input_handler import ScriptUserInputHandler
from llm_orc.schemas.agent_config import LlmAgentConfig, ScriptAgentConfig


class TestScriptUserInputDetection:
    """Test detection of scripts that require user input."""

    def test_detects_get_user_input_script_reference(self) -> None:
        """Test detection when script reference points to get_user_input.py."""
        handler = ScriptUserInputHandler()

        script_ref = "primitives/user-interaction/get_user_input.py"

        result = handler.requires_user_input(script_ref)

        assert result is True

    def test_detects_script_content_with_input_function(self) -> None:
        """Test detection when script content contains input() function."""
        handler = ScriptUserInputHandler()

        script_content = """
        import json
        import sys

        user_input = input("Enter your name: ")
        print(json.dumps({"name": user_input}))
        """

        result = handler.requires_user_input(script_content)

        assert result is True

    def test_ignores_regular_scripts(self) -> None:
        """Test that regular scripts without input are not detected."""
        handler = ScriptUserInputHandler()

        script_content = """
        import json
        import sys

        print(json.dumps({"result": "success"}))
        """

        result = handler.requires_user_input(script_content)

        assert result is False

    def test_ensemble_requires_user_input_detects_interactive_agents(self) -> None:
        """Test ensemble detection of agents with interactive scripts."""
        handler = ScriptUserInputHandler()

        # Mock ensemble config with interactive agents
        mock_ensemble = Mock()
        mock_ensemble.agents = [
            ScriptAgentConfig(
                name="user_input_agent",
                script="primitives/user-interaction/get_user_input.py",
            ),
            ScriptAgentConfig(
                name="regular_agent",
                script="utils/process_data.py",
            ),
        ]

        result = handler.ensemble_requires_user_input(mock_ensemble)

        assert result is True

    def test_ensemble_requires_user_input_detects_no_interactive_agents(self) -> None:
        """Test ensemble detection when no interactive agents present."""
        handler = ScriptUserInputHandler()

        # Mock ensemble config with no interactive agents
        mock_ensemble = Mock()
        mock_ensemble.agents = [
            ScriptAgentConfig(
                name="regular_agent",
                script="utils/process_data.py",
            ),
            LlmAgentConfig(
                name="llm_agent",
                model_profile="claude-3-sonnet",
            ),
        ]

        result = handler.ensemble_requires_user_input(mock_ensemble)

        assert result is False

    def test_ensemble_requires_user_input_handles_empty_ensemble(self) -> None:
        """Test ensemble detection with empty or invalid config."""
        handler = ScriptUserInputHandler()

        # Test empty agents
        mock_ensemble = Mock()
        mock_ensemble.agents = []
        result = handler.ensemble_requires_user_input(mock_ensemble)
        assert result is False

        # Test no agents attribute
        mock_ensemble_no_attr = Mock()
        del mock_ensemble_no_attr.agents
        result = handler.ensemble_requires_user_input(mock_ensemble_no_attr)
        assert result is False
