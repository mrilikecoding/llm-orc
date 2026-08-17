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


class TestScriptAgentsSkipTheOuterTimeout:
    """#158: a script agent's outer asyncio.wait_for was a DUPLICATE of
    its inner subprocess bound and strictly worse at the job — it cannot
    reap the child, and it cannot fire during the only window it uniquely
    covers, because that window is what blocks the loop.

    Retiring it (not satisfying it vacuously) is what keeps queue delay
    from being charged against an agent's budget once script agents
    genuinely run concurrently.
    """

    def _dispatcher(self, default_timeout: int = 300) -> Any:
        from llm_orc.core.execution.phases.agent_dispatcher import AgentDispatcher

        async def _resolve(config: Any) -> dict[str, Any]:
            dumped: dict[str, Any] = config.model_dump()
            return dumped

        return AgentDispatcher(
            execution_coordinator=Mock(),
            dependency_resolver=Mock(),
            progress_controller=Mock(),
            emit_event_fn=lambda name, data: None,
            resolve_profile_fn=_resolve,
            performance_config={"execution": {"default_timeout": default_timeout}},
        )

    def test_a_script_agent_gets_no_outer_timeout(self) -> None:
        dispatcher = self._dispatcher()
        config = ScriptAgentConfig(name="s", script="x.py")
        assert asyncio.run(dispatcher._get_agent_timeout(config)) is None

    def test_an_llm_agent_still_gets_one(self) -> None:
        from llm_orc.schemas.agent_config import LlmAgentConfig

        dispatcher = self._dispatcher(default_timeout=300)
        config = LlmAgentConfig(name="m", model_profile="p")
        assert asyncio.run(dispatcher._get_agent_timeout(config)) == 300

    def test_an_explicit_timeout_on_a_script_agent_is_still_not_an_outer_bound(
        self,
    ) -> None:
        """The explicit value still reaches the SUBPROCESS via the runner;
        what it no longer does is start a timer before the work begins."""
        dispatcher = self._dispatcher()
        config = ScriptAgentConfig(name="s", script="x.py", timeout_seconds=45)
        assert asyncio.run(dispatcher._get_agent_timeout(config)) is None

    def test_a_timed_out_script_agent_keeps_todays_result_shape(
        self, tmp_path: Path
    ) -> None:
        """The contract does NOT move, asserted END TO END through the real
        dispatcher and coordinator, because that is where the contract now
        lives. An earlier version of this pin built a ScriptAgentRunner
        directly and asserted on the raw response string; review round 1
        showed a one-line mutation of the coordinator (resolving the
        default instead of skipping the timer) that flipped status from
        success to failed with the ENTIRE suite still green.

        A timeout comes back as a SUCCESSFUL agent carrying a failure
        envelope. Consumers read that shape — fan_out/coordinator.py skips
        expansion when upstream status is not success — so flipping it
        would silently unexpand a fan-out.

        Do NOT tighten the timings here. This catches a reintroduced outer
        bound at EITHER site (the dispatcher returning an int, or the
        coordinator resolving a default) by relying on the outer timer
        firing before the inner one. Those bounds are equal by
        construction, and the measured head start is only 0.4-1.0ms — the
        setup path before the interpreter spawns. It is a strict
        happens-before rather than a race (27/27 including a saturated
        box), but the margin is far smaller than it looks. The dispatcher
        route also has deterministic, timing-free coverage in
        test_a_script_agent_gets_no_outer_timeout above.
        """
        from llm_orc.core.execution.phases.agent_dispatcher import AgentDispatcher
        from llm_orc.core.execution.phases.agent_execution_coordinator import (
            AgentExecutionCoordinator,
        )
        from llm_orc.core.execution.phases.dependency_resolver import (
            DependencyResolver,
        )

        script = tmp_path / "hangs.py"
        script.write_text("import time\ntime.sleep(30)\n")
        perf = {"execution": {"default_timeout": 1}}

        runner = ScriptAgentRunner(
            script_cache=ScriptCache(ScriptCacheConfig(enabled=False)),
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=None,
            performance_config=perf,
        )

        async def _executor(cfg: Any, inp: str) -> Any:
            return await runner.execute(cfg, inp)

        async def _resolve(cfg: Any) -> dict[str, Any]:
            dumped: dict[str, Any] = cfg.model_dump()
            return dumped

        dispatcher = AgentDispatcher(
            AgentExecutionCoordinator(perf, _executor),
            DependencyResolver(lambda name: ""),
            NoOpProgressController(),
            lambda name, data: None,
            _resolve,
            perf,
        )
        config = ScriptAgentConfig(name="hangs", script=str(script))

        results = asyncio.run(dispatcher.execute_agents_in_phase([config], "{}"))
        result = results["hangs"]

        assert result.status == "success", result.error
        assert result.error is None
        assert result.response is not None
        parsed = json.loads(result.response)
        assert parsed["success"] is False
        assert "timed out" in parsed["error"].lower()


class TestFailuresAreNotCached:
    """#159: every result was cached, including failure envelopes, and
    ScriptCache replays them for a 3600s TTL on the same (script, input,
    parameters) key. One rate-limited search or one timeout under
    momentary load poisoned that key for an hour — and with
    persist_to_artifacts the entry survives a restart, so the worst case
    is cross-process rather than one in-process hour.

    Asserts on get_stats() rather than wall time: exact, cannot flake,
    and distinguishes "did not cache" from "cached but the read was
    slow", which a timing assertion cannot.
    """

    def _run_twice(self, tmp_path: Path, body: str) -> dict[str, Any]:
        script = tmp_path / "probe.py"
        script.write_text(body)
        cache = ScriptCache(ScriptCacheConfig(enabled=True))  # off by default (#160)
        runner = ScriptAgentRunner(
            script_cache=cache,
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=None,
            performance_config={"execution": {"default_timeout": 30}},
        )
        config = ScriptAgentConfig(name="probe", script=str(script))

        async def _both() -> None:
            await runner.execute(config, "{}")
            await runner.execute(config, "{}")

        asyncio.run(_both())
        stats: dict[str, Any] = cache.get_stats()
        return stats

    def test_a_failed_script_is_not_cached(self, tmp_path: Path) -> None:
        """sys.exit(3) rather than a real timeout: instant, same branch."""
        stats = self._run_twice(tmp_path, "import sys\nsys.exit(3)\n")
        assert stats["hits"] == 0
        assert stats["sets"] == 0

    def test_an_error_keyed_response_is_not_cached(self, tmp_path: Path) -> None:
        """The issue's own motivating example. web_searcher reports every
        failure as {"error": ...} and exits 0, so no success key and no
        exception envelope exist — a success-only predicate would leave
        exactly this poisoned. 0 of 33 serving scripts emit a boolean
        success at all."""
        stats = self._run_twice(
            tmp_path,
            'import json\nprint(json.dumps({"error": "rate_limited"}))\n',
        )
        assert stats["hits"] == 0
        assert stats["sets"] == 0

    def test_a_successful_script_is_still_cached(self, tmp_path: Path) -> None:
        stats = self._run_twice(
            tmp_path, 'import json\nprint(json.dumps({"success": True, "n": 1}))\n'
        )
        assert stats["hits"] == 1
        assert stats["sets"] == 1

    @pytest.mark.parametrize(
        ("body", "label"),
        [
            ('import json\nprint(json.dumps({"n": 1}))\n', "no-success-key"),
            ("print('plain prose output')\n", "prose"),
            # ScriptAgentOutput.model_dump() ALWAYS emits error: null, so an
            # implementer writing `"error" in parsed` instead of a truthiness
            # check silently stops caching every schema-path success — the
            # exact mirror of the no-success-key degradation above.
            (
                'import json\nprint(json.dumps({"success": True, "error": None}))\n',
                "benign-null-error",
            ),
        ],
    )
    def test_responses_without_a_success_key_still_cache(
        self, tmp_path: Path, body: str, label: str
    ) -> None:
        """The pin that matters most, and the one the success-case pin
        CANNOT provide: an implementer writing `not parsed.get("success")`
        without the True default stops caching every response lacking the
        key — all 33 serving scripts, every prose response — while a pin
        using {"success": true} still passes. That is exactly the "never
        cache anything" degradation."""
        stats = self._run_twice(tmp_path, body)
        assert stats["hits"] == 1, label
        assert stats["sets"] == 1, label

    def test_a_non_dict_json_response_does_not_raise(self, tmp_path: Path) -> None:
        """_parse_output returns json.loads verbatim and execute returns it
        unwrapped when it is not a dict, so a script printing an array
        yields a list. Parsing that unguarded raises TypeError and would
        kill a run that works today."""
        stats = self._run_twice(tmp_path, "print('[1, 2, 3]')\n")
        assert stats["misses"] >= 1

    def test_the_entry_carries_no_lying_success_field(self, tmp_path: Path) -> None:
        """The runner wrote a hardcoded success: True that nothing reads,
        on an entry that might hold a failure."""
        script = tmp_path / "ok.py"
        script.write_text('import json\nprint(json.dumps({"success": True}))\n')
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = ScriptAgentRunner(
            script_cache=cache,
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=None,
            performance_config={"execution": {"default_timeout": 30}},
        )
        config = ScriptAgentConfig(name="ok", script=str(script))
        asyncio.run(runner.execute(config, "{}"))

        # The cache is keyed by identity now (#160), not by the raw path.
        identity = runner._cache_identity(str(script))
        entry = cache.get(identity, {"input_data": "{}", "parameters": {}})
        assert entry is not None
        assert "success" not in entry

    def test_a_bare_success_false_is_not_cached(self, tmp_path: Path) -> None:
        """No error key, so the error clause cannot see it. This is the ONLY
        pin that dies when the success clause is deleted outright: every
        other failure in the corpus — including sys.exit(3)'s envelope,
        which carries "Script failed with exit code 3" — also sets a truthy
        error, so the error clause alone would catch them."""
        stats = self._run_twice(
            tmp_path,
            'import json\nprint(json.dumps({"success": False, "reason": "no"}))\n',
        )
        assert stats["hits"] == 0
        assert stats["sets"] == 0

    @pytest.mark.parametrize(("literal", "label"), [("0", "zero"), ("null", "null")])
    def test_a_falsy_non_bool_success_is_not_cached(
        self, tmp_path: Path, literal: str, label: str
    ) -> None:
        """Truthiness rather than `is False`, which is what the predicate's
        docstring argues for and what nothing else pins: a script emitting
        {"success": 0} or {"success": null} is reporting failure, and an
        identity check would cache both."""
        stats = self._run_twice(tmp_path, f"print('{{\"success\": {literal}}}')\n")
        assert stats["hits"] == 0, label
        assert stats["sets"] == 0, label

    def test_a_schema_path_scalar_response_does_not_raise(self, tmp_path: Path) -> None:
        """execute_with_schema_json returns RAW stdout rather than routing
        through _parse_output, so on the ScriptAgentInput dispatch shape a
        script printing an array yields a str that json-parses to a list.
        Dropping isinstance(parsed, dict) turns that into AttributeError and
        kills a run that works today — and no other pin catches it, because
        the non-schema path never produces that shape."""
        script = tmp_path / "arr.py"
        script.write_text("print('[1, 2, 3]')\n")
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = ScriptAgentRunner(
            script_cache=cache,
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=None,
            performance_config={"execution": {"default_timeout": 30}},
        )
        config = ScriptAgentConfig(name="arr", script=str(script))
        schema_input = json.dumps(
            {
                "agent_name": "arr",
                "input_data": "hi",
                "dependencies": {},
                "context": {},
            }
        )
        response, _model, _sub = asyncio.run(runner.execute(config, schema_input))

        assert response == "[1, 2, 3]"
        assert cache.get_stats()["sets"] == 1


class TestCacheIdentity:
    """#160: the cache key hashed agent_config.script, a REFERENCE, which
    in every shipped ensemble is a path. So it identified the file's name
    and never its contents: edit a script, re-run, and get the pre-edit
    result for the TTL — cross-process under persist_to_artifacts.

    Strictly worse than the cached-failure replay fixed in #159, and
    invisible to that predicate, since a stale SUCCESS caches correctly
    under any failure-skipping rule.
    """

    def _runner(
        self,
        cache: ScriptCache,
        project_dir: Path | None = None,
    ) -> ScriptAgentRunner:
        return ScriptAgentRunner(
            script_cache=cache,
            usage_collector=UsageCollector(),
            progress_controller=None,
            emit_event=lambda name, data: None,
            project_dir=project_dir,
            performance_config={"execution": {"default_timeout": 30}},
        )

    def _emit(self, value: str) -> str:
        return f'import json\nprint(json.dumps({{"v": "{value}"}}))\n'

    def test_editing_a_script_invalidates_its_entry(self, tmp_path: Path) -> None:
        """The issue's reproduction."""
        script = tmp_path / "probe.py"
        script.write_text(self._emit("one"))
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache)
        config = ScriptAgentConfig(name="probe", script=str(script))

        first, _, _ = asyncio.run(runner.execute(config, "{}"))
        script.write_text(self._emit("two"))
        second, _, _ = asyncio.run(runner.execute(config, "{}"))

        assert json.loads(first)["v"] == "one"
        assert json.loads(second)["v"] == "two"

    def test_a_project_relative_reference_is_also_invalidated(
        self, tmp_path: Path
    ) -> None:
        """THE critical pin. If the key-time resolver is built without
        threading project_dir, a project-relative reference fails to
        resolve, the identity falls back to the reference string, and this
        whole change ships INERT — while every other pin here stays green,
        because they use absolute tmp_path references that resolve either
        way. Project-relative is the shape every shipped ensemble uses."""
        scripts = tmp_path / ".llm-orc" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "probe.py").write_text(self._emit("one"))
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache, project_dir=tmp_path)
        config = ScriptAgentConfig(name="probe", script="scripts/probe.py")

        first, _, _ = asyncio.run(runner.execute(config, "{}"))
        (scripts / "probe.py").write_text(self._emit("two"))
        second, _, _ = asyncio.run(runner.execute(config, "{}"))

        assert json.loads(first)["v"] == "one"
        assert json.loads(second)["v"] == "two"

    def test_the_cross_process_case_is_invalidated(self, tmp_path: Path) -> None:
        """The worst case the issue names: a FRESH ScriptCache reading the
        persisted entry after an edit.

        The second half is not decoration. Review showed that the edit
        assertions alone pass identically when persistence is broken in
        EITHER direction (never written, or never read), because a total
        persistence failure produces the same observation as correct
        invalidation. So the pin has to watch an UNCHANGED script hit
        across a fresh cache too, or it is only proving that nothing
        persists at all.
        """
        script = tmp_path / "probe.py"
        script.write_text(self._emit("one"))
        config = ScriptAgentConfig(name="probe", script=str(script))
        cfg = ScriptCacheConfig(
            enabled=True, persist_to_artifacts=True, artifact_base_dir=tmp_path
        )

        first, _, _ = asyncio.run(self._runner(ScriptCache(cfg)).execute(config, "{}"))
        script.write_text(self._emit("two"))
        second, _, _ = asyncio.run(self._runner(ScriptCache(cfg)).execute(config, "{}"))

        assert json.loads(first)["v"] == "one"
        assert json.loads(second)["v"] == "two"

        # Unchanged from here on, so a third fresh cache MUST hit the entry
        # the second run persisted.
        third_cache = ScriptCache(cfg)
        third, _, _ = asyncio.run(self._runner(third_cache).execute(config, "{}"))

        assert json.loads(third)["v"] == "two"
        assert third_cache.get_stats()["hits"] == 1, (
            "nothing crossed the process boundary, so the edit half of this "
            "pin was passing for the wrong reason"
        )

    def test_a_script_edited_mid_run_does_not_poison_the_old_bytes(
        self, tmp_path: Path
    ) -> None:
        """The race the digest's placement opens, found by review.

        The digest is taken before the subprocess opens the file, so an edit
        landing in that window makes the run produce the NEW bytes' output
        and file it under the OLD bytes' key. The design doc first called
        this self-correcting on the next run. It is not: the old-bytes key
        is poisoned for the full TTL, and two executions end up sharing an
        entry although they ran different bytes, which is a literal
        violation of this issue's invariant rather than mere staleness.
        """
        script = tmp_path / "probe.py"
        script.write_text(self._emit("one"))
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache)
        config = ScriptAgentConfig(name="probe", script=str(script))

        real = runner._execute_without_cache

        async def edit_then_run(agent_config: Any, input_data: str) -> Any:
            # Lands in the real window: after the identity is computed,
            # before the child reads the file.
            script.write_text(self._emit("two"))
            return await real(agent_config, input_data)

        with patch.object(runner, "_execute_without_cache", new=edit_then_run):
            first, _, _ = asyncio.run(runner.execute(config, "{}"))

        assert json.loads(first)["v"] == "two", "the child ran the edited bytes"

        # Put the ORIGINAL bytes back. They must not be served the other
        # version's output.
        script.write_text(self._emit("one"))
        second, _, _ = asyncio.run(runner.execute(config, "{}"))

        assert json.loads(second)["v"] == "one"
        assert cache.get_stats()["hits"] == 0

    def test_an_unchanged_script_still_hits(self, tmp_path: Path) -> None:
        script = tmp_path / "probe.py"
        script.write_text(self._emit("one"))
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache)
        config = ScriptAgentConfig(name="probe", script=str(script))

        asyncio.run(runner.execute(config, "{}"))
        asyncio.run(runner.execute(config, "{}"))

        assert cache.get_stats()["hits"] == 1
        assert cache.get_stats()["sets"] == 1

    def test_inline_content_still_caches(self) -> None:
        """The case the current code gets RIGHT, and which a naive
        'always hash the file' fix would break: an inline reference is
        genuinely its own content."""
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache)
        config = ScriptAgentConfig(name="inline", script="echo hello")

        asyncio.run(runner.execute(config, "{}"))
        asyncio.run(runner.execute(config, "{}"))

        assert cache.get_stats()["hits"] == 1

    def test_identical_bytes_at_different_paths_do_not_share_an_entry(
        self, tmp_path: Path
    ) -> None:
        """Pins the PATH half of the pair: bytes alone would collide two
        scripts that read their own __file__ or a sibling relative path."""
        body = self._emit("same")
        (tmp_path / "a.py").write_text(body)
        (tmp_path / "b.py").write_text(body)
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache)

        for name in ("a", "b"):
            asyncio.run(
                runner.execute(
                    ScriptAgentConfig(name=name, script=str(tmp_path / f"{name}.py")),
                    "{}",
                )
            )

        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["sets"] == 2

    def test_an_unresolvable_reference_does_not_raise(self, tmp_path: Path) -> None:
        """Computing a cache key must never be the thing that reports a
        missing script; execution a moment later produces the real error."""
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache, project_dir=tmp_path)
        config = ScriptAgentConfig(name="gone", script="scripts/does_not_exist.py")

        response, _, _ = asyncio.run(runner.execute(config, "{}"))

        assert json.loads(response)["success"] is False

    def test_inline_content_with_a_nul_byte_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """One of three things the os.path.isfile guard uniquely buys.

        Written after a mutation run showed that removing that guard killed
        no pin. For inline content too long for PATH_MAX, read_bytes raises
        OSError, which the except below already catches, so the guard is
        redundant there. For inline content carrying a NUL byte it raises
        ValueError, which an `except OSError` does NOT catch, so it escapes
        _cache_identity and kills the run before the script is ever
        executed.

        Review later found two more, both non-files that resolve fine: a
        FIFO, where read_bytes blocks on open until a writer appears and so
        HANGS the agent, and a character device such as /dev/zero, which
        reads unbounded. Neither is pinned here; a hang is awkward to pin
        without a watchdog, and this pin already kills the mutant.
        """
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache, project_dir=tmp_path)
        config = ScriptAgentConfig(name="nul", script="echo\x00hello")

        # The point is that this returns rather than raising ValueError;
        # whatever the execution then makes of the reference is its business.
        asyncio.run(runner.execute(config, "{}"))

    def test_an_interactive_agent_never_reads_a_non_interactive_alias_entry(
        self, tmp_path: Path
    ) -> None:
        """The interactive skip at the GET, which nothing else pins.

        Found by review: with only the set guarded, nothing is ever written,
        so test_an_interactive_agent_is_never_cached's `hits == 0` is
        trivially true and the get-side guard could be deleted with the whole
        suite green. This pin makes a NON-interactive agent write the entry,
        then asks whether the interactive one reads it.

        The alias is live in a shipped install rather than contrived. The
        resolver normalizes hyphen to underscore, so
        primitives/user-interaction/get-user-input.py resolves to the packaged
        interactive primitive and gets the same identity, while
        ScriptUserInputHandler.requires_user_input() answers False for it.
        That is the aliasing the byte-identity introduced.
        """
        d = tmp_path / ".llm-orc" / "scripts" / "primitives" / "user_interaction"
        d.mkdir(parents=True)
        (d / "get_user_input.py").write_text(
            'import json\nprint(json.dumps({"answer": "FIRST-AGENTS-ANSWER"}))\n'
        )
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache, project_dir=tmp_path)

        alias = ScriptAgentConfig(
            name="alias",
            script="scripts/primitives/user-interaction/get-user-input.py",
        )
        asyncio.run(runner.execute(alias, "{}"))
        assert cache.get_stats()["sets"] == 1, "the alias should have cached"

        interactive = ScriptAgentConfig(
            name="ask",
            script="scripts/primitives/user_interaction/get_user_input.py",
        )
        with patch("builtins.input", return_value="typed") as mock_input:
            asyncio.run(runner.execute(interactive, "{}"))

        assert cache.get_stats()["hits"] == 0, (
            "an interactive agent was served a non-interactive alias's entry"
        )
        assert mock_input.called, "the human was never prompted; cache replayed"

    def test_an_interactive_agent_is_never_cached(self, tmp_path: Path) -> None:
        """The human's ANSWER is not part of the key, so a second identical
        interactive agent replayed the first person's answer without
        prompting. Skipped at BOTH get and set: the new key aliases several
        references onto one file, while the interactive predicate still
        judges each reference separately, so a get-only skip would let an
        interactive agent's entry be hit by a non-interactive alias."""
        scripts = tmp_path / ".llm-orc" / "scripts" / "primitives" / "user_interaction"
        scripts.mkdir(parents=True)
        (scripts / "get_user_input.py").write_text(
            'import json\nprint(json.dumps({"answer": "typed"}))\n'
        )
        cache = ScriptCache(ScriptCacheConfig(enabled=True))
        runner = self._runner(cache, project_dir=tmp_path)
        config = ScriptAgentConfig(
            name="ask", script="scripts/primitives/user_interaction/get_user_input.py"
        )

        # Patch the prompt itself: pytest raises OSError (not EOFError) when
        # a test reads stdin under capture, and the point here is the cache
        # path, not the terminal.
        with patch("builtins.input", return_value="typed"):
            asyncio.run(runner.execute(config, "{}"))
            asyncio.run(runner.execute(config, "{}"))

        assert cache.get_stats()["sets"] == 0
        assert cache.get_stats()["hits"] == 0
