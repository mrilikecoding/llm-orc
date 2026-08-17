"""Tests for ScriptAgentRunner._execute_interactive refactor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.agents.script_agent import ScriptAgent
from llm_orc.core.execution.progress_controller import NoOpProgressController
from llm_orc.core.execution.scripting.agent_runner import ScriptAgentRunner
from llm_orc.core.execution.scripting.cache import ScriptCache, ScriptCacheConfig
from llm_orc.core.execution.usage_collector import UsageCollector
from llm_orc.schemas.agent_config import ScriptAgentConfig


def _make_runner(
    progress_controller: object | None = None,
) -> ScriptAgentRunner:
    """Create a ScriptAgentRunner with mocked dependencies."""
    return ScriptAgentRunner(
        script_cache=Mock(),
        usage_collector=Mock(),
        progress_controller=progress_controller or Mock(),
        emit_event=Mock(),
        project_dir=None,
    )


def _make_script_agent(
    name: str = "test_agent",
    prompt: str = "What is your name?",
    script: str = "primitives/user-interaction/get_user_input.py",
    timeout: int = 60,
) -> Mock:
    """Create a mock ScriptAgent."""
    agent = Mock()
    agent.name = name
    agent.script = script
    agent.parameters = {"prompt": prompt}
    agent.timeout = timeout
    agent.environment = {}

    resolver = Mock()
    resolver.resolve_script_path.return_value = "/fake/get_user_input.py"
    agent._script_resolver = resolver
    agent._get_interpreter.return_value = ["python3"]
    return agent


class TestExecuteInteractivePausesController:
    """Test that _execute_interactive pauses the progress controller."""

    @pytest.mark.asyncio
    async def test_pauses_progress_controller(self) -> None:
        """pause_for_user_input is called with agent name and prompt."""
        controller = Mock()
        runner = _make_runner(progress_controller=controller)
        agent = _make_script_agent(prompt="Your name?")

        with (
            patch("builtins.input", return_value="Alice"),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await runner._execute_interactive(agent, "test input")

        controller.pause_for_user_input.assert_called_once_with(
            "test_agent", "Your name?"
        )


class TestExecuteInteractiveCollectsInput:
    """Test that _execute_interactive collects input via builtin input()."""

    @pytest.mark.asyncio
    async def test_collects_input_via_builtin_input(self) -> None:
        """builtin input() is called with the configured prompt."""
        runner = _make_runner()
        agent = _make_script_agent(prompt="Enter value:")

        with (
            patch("builtins.input", return_value="hello") as mock_input,
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await runner._execute_interactive(agent, "test input")

        mock_input.assert_called_once_with("Enter value: ")


class TestExecuteInteractivePipesJsonToScript:
    """Test that collected input is piped as JSON to the subprocess."""

    @pytest.mark.asyncio
    async def test_pipes_json_to_script_stdin(self) -> None:
        """subprocess.run receives JSON stdin with user response."""
        runner = _make_runner()
        agent = _make_script_agent()

        with (
            patch("builtins.input", return_value="Alice"),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await runner._execute_interactive(agent, "test input")

        call_kwargs = mock_run.call_args
        stdin_payload = call_kwargs.kwargs.get("input", call_kwargs[1].get("input", ""))
        parsed = json.loads(stdin_payload)
        assert parsed["input"] == "Alice"
        assert "parameters" in parsed


class TestExecuteInteractiveResumesController:
    """Test that _execute_interactive resumes the progress controller."""

    @pytest.mark.asyncio
    async def test_resumes_progress_after_input(self) -> None:
        """resume_from_user_input is called after subprocess completes."""
        controller = Mock()
        runner = _make_runner(progress_controller=controller)
        agent = _make_script_agent()

        with (
            patch("builtins.input", return_value="Alice"),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await runner._execute_interactive(agent, "test input")

        controller.resume_from_user_input.assert_called_once_with("test_agent")


class TestExecuteInteractiveHandlesEofError:
    """Test that EOFError from input() results in empty response piped."""

    @pytest.mark.asyncio
    async def test_handles_eoferror(self) -> None:
        """EOFError yields empty string piped to subprocess."""
        runner = _make_runner()
        agent = _make_script_agent()

        with (
            patch("builtins.input", side_effect=EOFError),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await runner._execute_interactive(agent, "test input")

        stdin_payload = mock_run.call_args.kwargs.get(
            "input", mock_run.call_args[1].get("input", "")
        )
        parsed = json.loads(stdin_payload)
        assert parsed["input"] == ""


class TestExecuteInteractiveWithNoOpController:
    """Test that NoOpProgressController doesn't crash."""

    @pytest.mark.asyncio
    async def test_works_with_noop_controller(self) -> None:
        """No crash when using NoOpProgressController."""
        controller = NoOpProgressController()
        runner = _make_runner(progress_controller=controller)
        agent = _make_script_agent()

        with (
            patch("builtins.input", return_value="test"),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            result = await runner._execute_interactive(agent, "test input")

        assert result is not None


class TestExecuteInteractiveSerializesConcurrent:
    """Test that concurrent interactive agents serialize their input()."""

    @pytest.mark.asyncio
    async def test_serializes_concurrent_interactive_agents(self) -> None:
        """Two concurrent calls execute input() sequentially."""
        runner = _make_runner()
        call_order: list[str] = []

        def slow_input(prompt: str) -> str:
            call_order.append(f"input_start_{prompt.strip()}")
            # Simulates a brief blocking period
            call_order.append(f"input_end_{prompt.strip()}")
            return "response"

        agent_a = _make_script_agent(name="agent_a", prompt="A?")
        agent_b = _make_script_agent(name="agent_b", prompt="B?")

        with (
            patch("builtins.input", side_effect=slow_input),
            patch("subprocess.run") as mock_run,
            patch("os.path.exists", return_value=True),
        ):
            mock_run.return_value = Mock(returncode=0, stdout='{"success": true}')
            await asyncio.gather(
                runner._execute_interactive(agent_a, "test"),
                runner._execute_interactive(agent_b, "test"),
            )

        # Both input calls should have happened (2 starts, 2 ends)
        input_starts = [c for c in call_order if c.startswith("input_start")]
        assert len(input_starts) == 2

        # They should be serialized: first agent's end before second's start
        # (the lock ensures no interleaving)
        a_end = call_order.index("input_end_A?")
        b_start_indices = [i for i, c in enumerate(call_order) if c == "input_start_B?"]
        if b_start_indices:
            # If B ran second, its start should be after A's end
            # (If B ran first, A's start should be after B's end — either is fine)
            assert a_end < b_start_indices[0] or call_order.index(
                "input_end_B?"
            ) < call_order.index("input_start_A?")


class TestScriptAgentTimeoutWiring:
    """#157: engine-run script agents had NO bound at all.

    ``config.get("timeout_seconds", 60)`` never saw its default, because
    ``model_dump()`` always emits the key and supplies None when unset,
    so every subprocess ran with ``timeout=None``. The outer
    ``asyncio.wait_for`` could not save it either: the blocking
    subprocess sits in an ``async def`` and stalls the event loop, so
    that timer never runs (#158).

    These pins are on the WIRING rather than on the timeout mechanism,
    because the mechanism was fine — a directly-constructed ScriptAgent
    has always honoured its timeout (test_script_agent.py). What broke
    was the number never arriving.
    """

    def _runner(self, default_timeout: int) -> ScriptAgentRunner:
        return ScriptAgentRunner(
            script_cache=ScriptCache(ScriptCacheConfig(enabled=False)),
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=None,
            performance_config={"execution": {"default_timeout": default_timeout}},
        )

    @pytest.mark.parametrize(
        ("explicit", "expected"),
        [(None, 7), (3, 3)],
        ids=["inherits-the-operator-default", "explicit-still-wins"],
    )
    def test_the_constructed_agent_gets_the_resolved_timeout(
        self, tmp_path: Path, explicit: int | None, expected: int
    ) -> None:
        script = tmp_path / "quick.py"
        script.write_text('import json\nprint(json.dumps({"ok": True}))\n')
        config = ScriptAgentConfig(
            name="quick", script=str(script), timeout_seconds=explicit
        )

        runner = self._runner(default_timeout=7)
        captured: list[ScriptAgent] = []
        real_init = ScriptAgent.__init__

        def _capture(self: ScriptAgent, *args: Any, **kwargs: Any) -> None:
            real_init(self, *args, **kwargs)
            captured.append(self)

        with patch.object(ScriptAgent, "__init__", _capture):
            asyncio.run(runner.execute(config, "{}"))

        assert captured, "no ScriptAgent was constructed"
        assert captured[0].timeout == expected

    def test_a_hanging_script_is_bounded_by_the_operator_default(
        self, tmp_path: Path
    ) -> None:
        """The behavioral half, unmocked: before this fix the sleep ran to
        completion because the bound was None. No wall-clock upper bound
        is asserted — a loaded machine must not make this flaky."""
        script = tmp_path / "hangs.py"
        script.write_text("import time\ntime.sleep(30)\n")
        config = ScriptAgentConfig(name="hangs", script=str(script))

        runner = self._runner(default_timeout=1)
        response, _model, _substituted = asyncio.run(runner.execute(config, "{}"))

        assert "timed out" in response.lower(), response
