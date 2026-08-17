"""Tests for script-based agent execution."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llm_orc.agents.script_agent import (
    SCRIPT_POOL_SIZE,
    SCRIPT_POOL_THREAD_PREFIX,
    ScriptAgent,
)


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
    async def test_script_agent_dangerous_command_returns_json_error(
        self,
    ) -> None:
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
    async def test_script_agent_called_process_error_returns_json_error(
        self,
    ) -> None:
        """Test script agent handling of command failures returns JSON."""
        config = {"script": "exit 1"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute("test input")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_script_agent_general_exception_returns_json_error(
        self,
    ) -> None:
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


class TestScriptAgentJsonIO:
    """Test script agent JSON I/O functionality."""

    def test_script_agent_passes_json_parameters_via_stdin(self) -> None:
        """Test that script agent passes JSON parameters via stdin."""
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
            agent = ScriptAgent("test_agent", config)

            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = '{"success": true}'
                mock_run.return_value.returncode = 0

                asyncio.run(agent.execute("test input"))

                call_args = mock_run.call_args
                stdin_data = call_args.kwargs.get("input")
                assert stdin_data is not None

                parsed = json.loads(stdin_data)
                assert parsed["parameters"] == {
                    "key1": "value1",
                    "key2": 123,
                }
                assert parsed["input"] == "test input"
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_parses_json_output(self) -> None:
        """Test that script agent parses JSON output from scripts."""
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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute("test input")

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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute("test input")

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
            agent = ScriptAgent("test_agent", config)

            context = {
                "user": "alice",
                "role": "admin",
                "session_id": "abc123",
            }
            result = await agent.execute("test input", context)

            parsed_result = json.loads(result)
            assert parsed_result["user"] == "alice"
            assert parsed_result["role"] == "admin"
            assert parsed_result["session_id"] == "abc123"
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_handles_script_errors_in_json(self) -> None:
        """Test that script agent returns errors as structured JSON."""
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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert parsed_result.get("success") is False
            assert "error" in parsed_result
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_resolves_script_paths(self) -> None:
        """Test that script agent uses ScriptResolver for paths."""
        config = {"script": "scripts/test.py"}
        agent = ScriptAgent("test_agent", config)

        with patch.object(
            agent._script_resolver, "resolve_script_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/absolute/path/to/script.py"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value.stdout = '{"success": true}'
                mock_run.return_value.returncode = 0

                await agent.execute("test input")

                mock_resolve.assert_called_once_with("scripts/test.py")

    @pytest.mark.asyncio
    async def test_script_agent_supports_different_languages(self) -> None:
        """Test that script agent supports different script languages."""
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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            parsed_result = json.loads(result)
            assert parsed_result["shell"] == "bash"
            assert parsed_result["success"] is True
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_timeout_returns_json_error(self) -> None:
        """Test that timeout errors are returned as structured JSON."""
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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute("test input")

            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert parsed_result.get("success") is False
            assert "error" in parsed_result
            assert "timed out" in parsed_result["error"].lower()
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_json_merge_with_parameters(self) -> None:
        """Test parameters are properly merged with input and context."""
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
            agent = ScriptAgent("test_agent", config)

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


class TestScriptAgentUserInput:
    """Tests for script agent with user input support."""

    @pytest.mark.asyncio
    async def test_script_agent_handles_user_input_during_execution(
        self,
    ) -> None:
        """Test that script agent can handle user input."""
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
            agent = ScriptAgent("test_agent", config)

            user_responses = ["Alice"]

            def mock_input_handler(prompt: str) -> str:
                return user_responses.pop(0)

            result = await agent.execute_with_user_input(
                "start_interactive",
                user_input_handler=mock_input_handler,
            )

            assert "Alice" in result
            assert "Hello" in result
        finally:
            Path(script_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_script_agent_fallback_for_non_interactive_scripts(
        self,
    ) -> None:
        """Test script agent fallback to normal execution."""
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
            agent = ScriptAgent("test_agent", config)

            result = await agent.execute_with_user_input("test_data")

            parsed_result = json.loads(result)
            assert parsed_result["processed"] == "test_data"
            assert parsed_result["type"] == "normal"
        finally:
            Path(script_path).unlink(missing_ok=True)


class TestScriptAgentADR001:
    """Test ADR-001 Pydantic schema-based execution."""

    async def test_execute_with_schema_success(self) -> None:
        """Test execute_with_schema with valid input."""
        from llm_orc.schemas.script_agent import ScriptAgentInput

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import sys
import json

data = json.loads(sys.stdin.read())
print(json.dumps({"success": True, "data": data["input"], "error": None}))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = ScriptAgent("test_agent", config)

            input_schema = ScriptAgentInput(
                agent_name="test_agent",
                input_data="test input",
                dependencies={},
                context={},
            )

            result = await agent.execute_with_schema(input_schema)

            assert result.success is True
            assert result.data == "test input"
            assert result.error is None
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def test_execute_with_schema_non_json_wrapped(self) -> None:
        """Test execute_with_schema wraps non-JSON output."""
        from llm_orc.schemas.script_agent import ScriptAgentInput

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
print("This is not JSON")
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = ScriptAgent("test_agent", config)

            input_schema = ScriptAgentInput(
                agent_name="test_agent",
                input_data="test",
                dependencies={},
                context={},
            )

            result = await agent.execute_with_schema(input_schema)

            assert result.success is True
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def test_execute_with_schema_execution_error(self) -> None:
        """Test execute_with_schema handles execution errors."""
        from llm_orc.schemas.script_agent import ScriptAgentInput

        config = {"script": "/nonexistent/script.py"}
        agent = ScriptAgent("test_agent", config)

        input_schema = ScriptAgentInput(
            agent_name="test_agent",
            input_data="test",
            dependencies={},
            context={},
        )

        result = await agent.execute_with_schema(input_schema)

        assert result.success is False
        assert result.error is not None

    async def test_execute_with_schema_json_success(self) -> None:
        """Test execute_with_schema_json with valid input.

        INPUT_DATA contains the full ScriptAgentInput JSON.
        INPUT_TEXT contains the raw input text.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import os
import json

input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))
output = {"success": True, "data": input_data.get("input_data"), "error": None}
print(json.dumps(output))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = ScriptAgent("test_agent", config)

            input_json = json.dumps(
                {
                    "agent_name": "test_agent",
                    "input_data": "test input",
                    "dependencies": {},
                    "context": {},
                }
            )

            result = await agent.execute_with_schema_json(input_json)

            parsed_result = json.loads(result)
            assert parsed_result["success"] is True
            assert parsed_result["data"] == "test input"
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def test_execute_with_schema_json_preserves_newlines_in_input_data(
        self,
    ) -> None:
        """INPUT_TEXT env var contains raw text with actual newlines.

        Regression: scripts doing text processing (tokenization, NLP)
        saw JSON escape sequences (\\n) instead of actual newlines,
        producing garbage tokens like "nthe" from "\\nthe".
        INPUT_TEXT provides the raw input_data field with real newlines.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(
                """#!/usr/bin/env python3
import os, json
raw = os.environ.get('INPUT_TEXT', '')
# If newlines are preserved, "Line 1" and "Line 2" are on separate lines.
# If mangled, raw contains JSON escapes like \\n instead of real newlines.
ok = '\\n' in raw and '\\\\n' not in raw
out = {"success": True, "data": raw, "has_real_newlines": ok, "error": None}
print(json.dumps(out))
"""
            )
            script_path = script_file.name

        try:
            config = {"script": script_path}
            agent = ScriptAgent("test_agent", config)

            input_json = json.dumps(
                {
                    "agent_name": "test_agent",
                    "input_data": "Line 1\nLine 2\nLine 3",
                    "dependencies": {},
                    "context": {},
                }
            )

            result = await agent.execute_with_schema_json(input_json)

            parsed_result = json.loads(result)
            assert parsed_result["success"] is True
            assert parsed_result["has_real_newlines"] is True, (
                f"INPUT_TEXT has mangled newlines: {parsed_result['data']!r}"
            )
            assert "Line 1\nLine 2" in parsed_result["data"]
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def test_execute_with_schema_json_invalid_input(self) -> None:
        """Test execute_with_schema_json with invalid input JSON."""
        config = {"script": "nonexistent.py"}
        agent = ScriptAgent("test_agent", config)

        result = await agent.execute_with_schema_json("not valid json")

        parsed_result = json.loads(result)
        assert parsed_result["success"] is False
        assert "error" in parsed_result


class TestInterpreterResolution:
    """#154: a file-backed .py script agent runs under the interpreter
    running llm-orc, not whichever python3 the operator's PATH exposes.

    The failure this prevents is asymmetric and therefore nasty: serving
    scripts that import llm_orc died while the stdlib-only scripts in the
    same ensemble kept running, leaving a half-dead pipeline rather than
    a clean failure (the 2026-08-13 misfire, #152).
    """

    def test_python_scripts_run_under_the_host_interpreter(self) -> None:
        agent = ScriptAgent("t", {"script": "x.py"})
        assert agent._get_interpreter("x.py") == [sys.executable]
        assert agent._get_interpreter("/abs/path/to/thing.python") == [sys.executable]

    def test_other_languages_are_untouched(self) -> None:
        agent = ScriptAgent("t", {"script": "x.sh"})
        assert agent._get_interpreter("x.sh") == ["bash"]
        assert agent._get_interpreter("x.bash") == ["bash"]
        assert agent._get_interpreter("x.js") == ["node"]
        assert agent._get_interpreter("x.rb") == ["ruby"]
        assert agent._get_interpreter("x.unknown") == ["bash"]

    def test_falls_back_when_the_host_interpreter_is_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty sys.executable (embedded) or a frozen bundle, where
        [sys.executable, script.py] would re-invoke the CLI with a path
        as argv[1] rather than run the script."""
        agent = ScriptAgent("t", {"script": "x.py"})
        monkeypatch.setattr(sys, "executable", "")
        assert agent._get_interpreter("x.py") == ["python3"]
        monkeypatch.setattr(sys, "executable", "/opt/llm-orc/llm-orc")
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert agent._get_interpreter("x.py") == ["python3"]

    @pytest.mark.asyncio
    async def test_a_script_importing_llm_orc_runs_without_it_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The behavioral pin, against a REAL subprocess: mocking the
        interpreter is exactly what hid this class of failure. Runs with
        PATH stripped of the venv, the misfire's own condition."""
        script = tmp_path / "needs_llm_orc.py"
        script.write_text(
            "import json, sys, llm_orc\n"
            'print(json.dumps({"ok": True, "exe": sys.executable}))\n'
        )
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        agent = ScriptAgent("t", {"script": str(script), "timeout_seconds": 60})
        result = await agent.execute("{}")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        # Asserting the interpreter identity, not just that the import
        # worked: on a box where someone pip-installed llm-orc into the
        # system python, the import alone would succeed pre-fix and this
        # pin would have no red signal at all.
        assert parsed["exe"] == sys.executable


class TestTimeoutFloor:
    """#157 review: the design promised a floor for a directly-constructed
    ScriptAgent and `.get(key, 60)` did not provide one, because the key is
    usually PRESENT and None — the exact shape model_dump produces. One
    construction site is pinned today, but #158's work touches this area."""

    def test_a_present_none_timeout_falls_back_to_the_floor(self) -> None:
        agent = ScriptAgent("t", {"script": "x.py", "timeout_seconds": None})
        assert agent.timeout == 60

    def test_an_absent_timeout_falls_back_to_the_floor(self) -> None:
        assert ScriptAgent("t", {"script": "x.py"}).timeout == 60

    def test_an_explicit_zero_is_preserved(self) -> None:
        """`or 60` would swallow it and desync the inner bound from the
        outer one, which resolves 0 to 0."""
        agent = ScriptAgent("t", {"script": "x.py", "timeout_seconds": 0})
        assert agent.timeout == 0


class TestScriptAgentsOffTheEventLoop:
    """#158: five subprocess.run calls sat directly inside async def, so
    a script agent blocked the event loop for its whole subprocess. That
    serialized agents the engine explicitly gathers, and stalled any
    model-backed agent awaiting in the same loop.
    """

    def _script(self, tmp_path: Path, body: str) -> str:
        script = tmp_path / f"s{abs(hash(body)) % 10000}.py"
        script.write_text(body)
        return str(script)

    def test_concurrent_script_agents_actually_overlap(self, tmp_path: Path) -> None:
        """4 x 1s gathered took 4.18s before this change. The bound is a
        stated number with room for a 2x-loaded machine either way, not a
        tight one."""
        path = self._script(
            tmp_path, "import time, json\ntime.sleep(1)\nprint(json.dumps({'ok': 1}))\n"
        )

        async def run() -> float:
            agents = [
                ScriptAgent(f"a{i}", {"script": path, "timeout_seconds": 30})
                for i in range(4)
            ]
            start = time.time()
            await asyncio.gather(*(a.execute("{}") for a in agents))
            return time.time() - start

        assert asyncio.run(run()) < 2.5

    def test_the_loop_stays_responsive_while_a_script_runs(
        self, tmp_path: Path
    ) -> None:
        """The only DIRECT proof the loop is free: a counter ticking on
        asyncio.sleep(0) registered 0 ticks before this change. Snapshot
        around the call rather than reading at the end, or the version
        that passes today gets written."""
        path = self._script(self._tmp(tmp_path), "import time\ntime.sleep(0.6)\n")
        ticks = 0

        async def spin() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        async def run() -> tuple[int, int]:
            spinner = asyncio.create_task(spin())
            await asyncio.sleep(0)
            before = ticks
            await ScriptAgent("a", {"script": path, "timeout_seconds": 30}).execute(
                "{}"
            )
            after = ticks
            spinner.cancel()
            return before, after

        before, after = asyncio.run(run())
        assert after > before

    def _tmp(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_the_subprocess_runs_in_the_dedicated_pool(self, tmp_path: Path) -> None:
        """With no semaphore left, this is the ONLY symptom if the work
        lands in the default pool — which is exactly what asyncio.to_thread
        would have done while everything else looked correct."""
        path = self._script(tmp_path, "import json\nprint(json.dumps({'ok': 1}))\n")
        # Observed in the PARENT: the child process reports its own
        # MainThread, so only the caller's thread reveals which pool ran it.
        seen: list[str] = []
        real_run = subprocess.run

        def _spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(threading.current_thread().name)
            return real_run(*args, **kwargs)

        with patch("llm_orc.agents.script_agent.subprocess.run", _spy):
            asyncio.run(
                ScriptAgent("a", {"script": path, "timeout_seconds": 30}).execute("{}")
            )

        assert seen, "subprocess.run was never called"
        assert SCRIPT_POOL_THREAD_PREFIX in seen[0], seen[0]

    def test_a_hanging_subprocess_still_fails_at_its_inner_bound(
        self, tmp_path: Path
    ) -> None:
        """#158 retires the OUTER timeout for script agents, so the inner
        one is the whole bound. It is also the only one that ever reaped."""
        path = self._script(tmp_path, "import time\ntime.sleep(30)\n")
        result = asyncio.run(
            ScriptAgent("a", {"script": path, "timeout_seconds": 1}).execute("{}")
        )
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "timed out" in parsed["error"].lower()

    def test_saturation_is_honest_rather_than_a_false_timeout(
        self, tmp_path: Path
    ) -> None:
        """More agents than pool slots: every one completes correctly.
        Queue delay is honest waiting, not a consumed budget — which is
        precisely what the abandoned thread-slot gate existed to ensure
        and what removing the outer timer gives for free."""
        path = self._script(
            tmp_path,
            "import time, json\ntime.sleep(0.2)\nprint(json.dumps({'ok': 1}))\n",
        )
        count = SCRIPT_POOL_SIZE + 4

        async def run() -> list[str]:
            agents = [
                ScriptAgent(f"a{i}", {"script": path, "timeout_seconds": 30})
                for i in range(count)
            ]
            return list(await asyncio.gather(*(a.execute("{}") for a in agents)))

        results = asyncio.run(run())
        assert len(results) == count
        for raw in results:
            assert json.loads(raw)["ok"] == 1
