"""Tests for enhanced script agent with JSON I/O support."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_orc.agents.enhanced_script_agent import EnhancedScriptAgent


class TestEnhancedScriptAgent:
    """Test enhanced script agent functionality."""

    def test_script_agent_passes_json_parameters_via_stdin(self) -> None:
        """Test that script agent passes JSON parameters via stdin."""
        # Create test script that echoes stdin
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json
data = json.loads(sys.stdin.read())
print(json.dumps({"received": data}))
"""
            )
            script_path = script_file.name

        try:
            config = {
                "script": script_path,
                "parameters": {"key1": "value1", "key2": 123},
            }
            agent = EnhancedScriptAgent("test_agent", config)

            # Mock subprocess.run to capture the stdin that would be passed
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = '{"success": true}'
                mock_run.return_value.returncode = 0

                asyncio.run(agent.execute("test input"))

                # Verify JSON was passed via stdin
                call_args = mock_run.call_args
                stdin_data = call_args.kwargs.get("input")
                assert stdin_data is not None

                parsed = json.loads(stdin_data)
                assert parsed["parameters"] == {"key1": "value1", "key2": 123}
                assert parsed["input"] == "test input"
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_parses_json_output(self) -> None:
        """Test that script agent parses JSON output from scripts."""
        # Create test script that outputs JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import json
result = {"status": "success", "value": 42, "items": ["a", "b", "c"]}
print(json.dumps(result))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            # Result should be JSON string that we parse
            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert parsed_result["status"] == "success"
            assert parsed_result["value"] == 42
            assert parsed_result["items"] == ["a", "b", "c"]
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_handles_non_json_output(self) -> None:
        """Test that script agent handles non-JSON output gracefully."""
        # Create test script that outputs plain text
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
print("This is plain text output")
print("Not JSON at all")
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            # Should return JSON string containing the raw output
            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert "output" in parsed_result
            assert "This is plain text output" in parsed_result["output"]
            assert "Not JSON at all" in parsed_result["output"]
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_passes_context_as_json(self) -> None:
        """Test that script agent passes context as structured JSON."""
        # Create test script that uses context
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json
data = json.loads(sys.stdin.read())
context = data.get('context', {})
result = {
    "user": context.get("user", "unknown"),
    "role": context.get("role", "none"),
    "session_id": context.get("session_id", "")
}
print(json.dumps(result))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            context = {"user": "alice", "role": "admin", "session_id": "abc123"}
            result = await agent.execute("test input", context)

            # Parse JSON string result
            parsed_result = json.loads(result)
            assert parsed_result["user"] == "alice"
            assert parsed_result["role"] == "admin"
            assert parsed_result["session_id"] == "abc123"
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_handles_script_errors_in_json(self) -> None:
        """Test that script agent returns errors as structured JSON."""
        # Create test script that fails
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
sys.exit(1)
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            # Should return error in structured format as JSON string
            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert parsed_result.get("success") is False
            assert "error" in parsed_result
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_resolves_script_paths(self) -> None:
        """Test that enhanced script agent uses ScriptResolver for path resolution."""
        config = {"script": "scripts/test.py"}
        agent = EnhancedScriptAgent("test_agent", config)

        # Mock the script resolver
        with patch.object(
            agent._script_resolver, "resolve_script_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/absolute/path/to/script.py"

            # Mock subprocess to avoid actual execution
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = '{"success": true}'
                mock_run.return_value.returncode = 0

                await agent.execute("test input")

                # Verify resolver was called
                mock_resolve.assert_called_once_with("scripts/test.py")

    @pytest.mark.asyncio
    async def test_script_agent_supports_different_languages(self) -> None:
        """Test that enhanced script agent supports different script languages."""
        # Test with shell script
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as script_file:
            script_file.write(
                """#!/bin/bash
echo '{"shell": "bash", "success": true}'
"""
            )
            script_path = script_file.name
            os.chmod(script_path, 0o755)

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            parsed_result = json.loads(result)
            assert parsed_result["shell"] == "bash"
            assert parsed_result["success"] is True
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_timeout_returns_json_error(self) -> None:
        """Test that timeout errors are returned as structured JSON."""
        # Create script that times out
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import time
time.sleep(10)
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path, "timeout_seconds": 0.1}
            agent = EnhancedScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            # Should return timeout error in structured format as JSON string
            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert parsed_result.get("success") is False
            assert "error" in parsed_result
            assert "timed out" in parsed_result["error"].lower()
        finally:
            Path(script_path).unlink(missing_ok=True)

    def test_enhanced_script_agent_inherits_from_base(self) -> None:
        """Test that EnhancedScriptAgent inherits from ScriptAgent."""
        from llm_orc.agents.script_agent import ScriptAgent

        config = {"script": "echo test"}
        agent = EnhancedScriptAgent("test_agent", config)

        # Should be instance of both EnhancedScriptAgent and ScriptAgent
        assert isinstance(agent, EnhancedScriptAgent)
        assert isinstance(agent, ScriptAgent)

    @pytest.mark.asyncio
    async def test_script_agent_json_merge_with_parameters(self) -> None:
        """Test that parameters are properly merged with input and context."""
        # Create test script that verifies all data is merged
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json
data = json.loads(sys.stdin.read())
result = {
    "has_input": "input" in data,
    "has_parameters": "parameters" in data,
    "has_context": "context" in data,
    "param_value": data.get("parameters", {}).get("test_param"),
    "input_value": data.get("input"),
    "context_user": data.get("context", {}).get("user")
}
print(json.dumps(result))
"""
            )
            script_path = script_file.name

        try:
            config = {
                "script": script_path,
                "parameters": {"test_param": "param_value"},
            }
            agent = EnhancedScriptAgent("test_agent", config)

            context = {"user": "test_user", "session": "123"}
            result = await agent.execute("input_data", context)

            parsed_result = json.loads(result)
            assert parsed_result["has_input"] is True
            assert parsed_result["has_parameters"] is True
            assert parsed_result["has_context"] is True
            assert parsed_result["param_value"] == "param_value"
            assert parsed_result["input_value"] == "input_data"
            assert parsed_result["context_user"] == "test_user"
        finally:
            Path(script_path).unlink(missing_ok=True)


class TestEnhancedScriptAgentUserInput:
    """Integration tests for enhanced script agent with user input support."""

    @pytest.mark.asyncio
    async def test_enhanced_script_agent_handles_user_input_during_execution(
        self,
    ) -> None:
        """Test that enhanced script agent can handle user input during execution."""
        # Create script that requires user input
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json

# Read JSON data from stdin (first line)
first_line = sys.stdin.readline()
data = json.loads(first_line)

# Simulate requesting user input
if data.get("input") == "start_interactive":
    # Output request for user input
    print(json.dumps({"type": "user_input_request", "prompt": "Enter your name:"}))
    sys.stdout.flush()

    # Wait for and read user input from stdin (next line)
    user_input = sys.stdin.readline().strip()
    print(json.dumps({"greeting": f"Hello, {user_input}!"}))
else:
    print(json.dumps({"output": "No interaction needed"}))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            # Mock the user input handler
            user_responses = ["Alice"]

            def mock_input_handler(prompt: str) -> str:
                return user_responses.pop(0)

            result = await agent.execute_with_user_input(
                "start_interactive", user_input_handler=mock_input_handler
            )

            # Should handle user input and return final result
            assert "Alice" in result
            assert "Hello" in result
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_enhanced_script_agent_fallback_for_non_interactive_scripts(
        self,
    ) -> None:
        """Test enhanced script agent fallback to normal execution."""
        # Create normal script that doesn't require user input
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json

data = json.loads(sys.stdin.read())
result = {"processed": data.get("input", ""), "type": "normal"}
print(json.dumps(result))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = EnhancedScriptAgent("test_agent", config)

            # Should work without user input handler
            result = await agent.execute_with_user_input("test_data")

            parsed_result = json.loads(result)
            assert parsed_result["processed"] == "test_data"
            assert parsed_result["type"] == "normal"
        finally:
            Path(script_path).unlink(missing_ok=True)
