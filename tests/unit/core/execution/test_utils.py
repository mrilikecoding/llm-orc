"""Tests for shared execution utility functions."""

from __future__ import annotations

from typing import Any, Literal

from llm_orc.core.execution.utils import (
    dep_name,
    result_succeeded,
    terminal_agent_names,
    terminal_failure_summary,
)
from llm_orc.schemas.agent_config import AgentConfig, LlmAgentConfig, ScriptAgentConfig


def _agent(name: str, depends_on: list[str | dict[str, Any]] | None = None) -> Any:
    return LlmAgentConfig(
        name=name, model_profile="test-profile", depends_on=depends_on or []
    )


class TestTerminalAgentNames:
    """Terminal nodes: agent names no other agent in the DAG depends on.

    Shared by ``resolve_deliverable``, ``_terminal_agents_for_ensemble``,
    and the ``ensemble:`` agent's own success rule (fail-closed-
    composition B1), so all three read the same terminal set.
    """

    def test_single_agent_is_its_own_terminal(self) -> None:
        agents = [_agent("solo")]
        assert terminal_agent_names(agents) == ["solo"]

    def test_linear_chain_last_node_is_terminal(self) -> None:
        agents = [_agent("a"), _agent("b", depends_on=["a"])]
        assert terminal_agent_names(agents) == ["b"]

    def test_join_node_is_the_sole_terminal(self) -> None:
        agents = [
            _agent("coder"),
            _agent("critic"),
            _agent("synthesizer", depends_on=["coder", "critic"]),
        ]
        assert terminal_agent_names(agents) == ["synthesizer"]

    def test_multiple_terminals_in_declaration_order(self) -> None:
        agents = [
            _agent("root"),
            _agent("left", depends_on=["root"]),
            _agent("right", depends_on=["root"]),
        ]
        assert terminal_agent_names(agents) == ["left", "right"]

    def test_dict_form_dependencies_are_recognized(self) -> None:
        agents = [
            _agent("first"),
            _agent("second", depends_on=[{"agent_name": "first"}]),
        ]
        assert terminal_agent_names(agents) == ["second"]

    def test_empty_agent_list_yields_no_terminals(self) -> None:
        assert terminal_agent_names([]) == []


class TestDepName:
    """Existing coverage for the string/dict dependency-entry accessor."""

    def test_string_dependency(self) -> None:
        assert dep_name("upstream") == "upstream"

    def test_dict_dependency(self) -> None:
        assert dep_name({"agent_name": "upstream"}) == "upstream"


class TestResultSucceeded:
    """SF2: one predicate for "this result counts as a succeeded
    terminal", shared by terminal_failure_summary and GuardEvaluator's
    dependency-succeeded check."""

    def test_success_status_succeeds(self) -> None:
        assert result_succeeded({"status": "success", "response": "ok"}) is True

    def test_partial_status_succeeds(self) -> None:
        assert result_succeeded({"status": "partial", "response": ["a", None]}) is True

    def test_failed_status_does_not_succeed(self) -> None:
        assert result_succeeded({"status": "failed", "error": "boom"}) is False

    def test_skipped_status_does_not_succeed(self) -> None:
        assert result_succeeded({"status": "skipped", "response": None}) is False

    def test_handled_failure_does_not_succeed_even_with_success_status(self) -> None:
        """X1: a node that ran only via on_dependency_failure: run still
        reports status success (it ran fine), but handled_failure marks
        it as not a succeeded terminal."""
        result = {
            "status": "success",
            "response": "refusal",
            "handled_failure": True,
        }
        assert result_succeeded(result) is False

    def test_non_dict_result_does_not_succeed(self) -> None:
        assert result_succeeded(None) is False

    def test_object_form_is_tolerated(self) -> None:
        """GuardEvaluator sees AgentResult-like objects, not only dicts."""

        class _Result:
            status = "success"
            handled_failure = False

        assert result_succeeded(_Result()) is True


def _script(
    name: str,
    depends_on: list[str | dict[str, Any]] | None = None,
    on_dependency_failure: Literal["run", "skip"] = "skip",
) -> ScriptAgentConfig:
    return ScriptAgentConfig(
        name=name,
        script="echo ok",
        depends_on=depends_on or [],
        on_dependency_failure=on_dependency_failure,
    )


class TestTerminalFailureSummary:
    """B1/X1/SF2: whether a child's terminal(s) succeeded, for
    EnsembleAgentRunner/DynamicDispatchRunner/LoopAgentRunner's own
    success rule."""

    def test_none_when_terminal_succeeded(self) -> None:
        agents: list[AgentConfig] = [_script("worker")]
        results = {"worker": {"status": "success", "response": "ok"}}
        assert terminal_failure_summary(agents, results) is None

    def test_none_when_terminal_partial(self) -> None:
        """Partial fan-out counts as success (partfanparent decision)."""
        agents: list[AgentConfig] = [_script("worker")]
        results = {"worker": {"status": "partial", "response": ["a", None]}}
        assert terminal_failure_summary(agents, results) is None

    def test_summary_when_terminal_failed(self) -> None:
        agents: list[AgentConfig] = [_script("worker")]
        results = {"worker": {"status": "failed", "error": "boom"}}
        summary = terminal_failure_summary(agents, results)
        assert summary == "worker (failed): boom"

    def test_summary_names_real_failure_not_handled_failure_alone(self) -> None:
        """X1: a sole terminal that ran only via on_dependency_failure:
        run (handled_failure) does not count as succeeded — the parent
        fails, naming the real upstream failure it handled."""
        agents: list[AgentConfig] = [
            _script("a"),
            _script("t", depends_on=["a"], on_dependency_failure="run"),
        ]
        results = {
            "a": {"status": "failed", "error": "boom"},
            "t": {
                "status": "success",
                "response": "handled",
                "handled_failure": True,
                "handled_failure_reason": "no dependency succeeded: a (failed: boom)",
            },
        }
        summary = terminal_failure_summary(agents, results)
        assert summary == (
            "t (handled failure: no dependency succeeded: a (failed: boom))"
        )

    def test_none_when_only_terminal_is_a_plain_when_skip(self) -> None:
        """SF2 whenskip/whenparent decision: a terminal skipped by a
        plain when:-false guard (no reason) is neither success nor
        failure — a child whose only terminal is when-skipped does not
        fail the parent, matching how a top-level invocation of the
        same child already reads."""
        agents: list[AgentConfig] = [_script("a"), _script("t", depends_on=["a"])]
        results = {
            "a": {"status": "success", "response": '{"ok": false}'},
            "t": {"status": "skipped", "response": None},
        }
        assert terminal_failure_summary(agents, results) is None

    def test_summary_when_terminal_is_a_dependency_cascade_skip(self) -> None:
        """A cascade skip (rule 1, carries a reason) is a real failure to
        report — unlike a plain when:-false skip."""
        agents: list[AgentConfig] = [_script("a"), _script("t", depends_on=["a"])]
        results = {
            "a": {"status": "failed", "error": "boom"},
            "t": {
                "status": "skipped",
                "response": None,
                "reason": "no dependency succeeded: a (failed: boom)",
            },
        }
        summary = terminal_failure_summary(agents, results)
        assert summary is not None
        assert "t" in summary

    def test_no_terminals_falls_open(self) -> None:
        assert terminal_failure_summary([], {}) is None
