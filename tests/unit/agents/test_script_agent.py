"""Tests for script-based agent execution."""

import json
from unittest.mock import patch

import pytest

from llm_orc.agents.script_agent import ScriptAgent


class TestScriptAgent:
    """Test script-based agent functionality."""

    def test_script_agent_creation_requires_script_or_command(self) -> None:
        """Test that script agent requires either script or command."""
        # This should fail - no script or command provided
        with pytest.raises(ValueError, match="must have either 'script' or 'command'"):
            ScriptAgent("test_agent", {})

    def test_script_agent_creation_with_script(self) -> None:
        """Test script agent creation with script content."""
        config = {"script": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.name == "test_agent"
        assert agent.script == "echo 'Hello World'"
        assert agent.command == ""
        assert agent.timeout == 60  # default timeout

    def test_script_agent_creation_with_command(self) -> None:
        """Test script agent creation with command."""
        config = {"command": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.name == "test_agent"
        assert agent.command == "echo 'Hello World'"
        assert agent.script == ""

    def test_script_agent_creation_with_custom_timeout(self) -> None:
        """Test script agent creation with custom timeout."""
        config = {"script": "echo 'test'", "timeout_seconds": 30}
        agent = ScriptAgent("test_agent", config)

        assert agent.timeout == 30

    def test_script_agent_get_agent_type(self) -> None:
        """Test script agent type identification."""
        config = {"script": "echo 'test'"}
        agent = ScriptAgent("test_agent", config)

        assert agent.get_agent_type() == "script"

    @pytest.mark.asyncio
    async def test_script_agent_execute_simple_script(self) -> None:
        """Test script agent execution with simple script returns JSON."""
        config = {"script": "echo 'Hello World'"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"] == "Hello World"

    @pytest.mark.asyncio
    async def test_script_agent_execute_simple_command(self) -> None:
        """Test script agent execution with simple command returns JSON."""
        config = {"command": "echo 'Hello Command'"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"] == "Hello Command"

    @pytest.mark.asyncio
    async def test_script_agent_receives_input_data(self) -> None:
        """Test that script agent receives input data via environment."""
        config = {"script": 'echo "Input: $INPUT_DATA"'}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test message")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"] == "Input: test message"

    @pytest.mark.asyncio
    async def test_script_agent_with_custom_environment(self) -> None:
        """Test script agent with custom environment variables."""
        config = {
            "script": 'echo "Custom: $CUSTOM_VAR"',
            "environment": {"CUSTOM_VAR": "custom_value"},
        }
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"] == "Custom: custom_value"

    @pytest.mark.asyncio
    async def test_script_agent_dangerous_command_returns_json_error(self) -> None:
        """Test that dangerous commands return JSON error."""
        config = {"command": "rm -rf /"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "Blocked dangerous command" in parsed["error"]

    @pytest.mark.asyncio
    async def test_script_agent_timeout_returns_json_error(self) -> None:
        """Test script agent timeout returns JSON error."""
        config = {"script": "echo test"}
        agent = ScriptAgent("test_agent", config)

        with patch(
            "llm_orc.agents.script_agent.subprocess.run",
            side_effect=__import__("subprocess").TimeoutExpired(cmd="", timeout=1),
        ):
            result = await agent.execute("test input")
            parsed = json.loads(result)
            assert parsed["success"] is False
            assert "timed out" in parsed["error"].lower()

    @pytest.mark.asyncio
    async def test_script_agent_called_process_error_returns_json_error(self) -> None:
        """Test script agent handling of command failures returns JSON."""
        config = {"script": "exit 1"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_script_agent_general_exception_returns_json_error(self) -> None:
        """Test script agent general exception returns JSON error."""
        config = {"script": "echo 'test'"}
        agent = ScriptAgent("test_agent", config)

        with patch(
            "llm_orc.agents.script_agent.subprocess.run",
            side_effect=OSError("Permission denied"),
        ):
            result = await agent.execute("test input")
            parsed = json.loads(result)
            assert parsed["success"] is False
            assert "error" in parsed
