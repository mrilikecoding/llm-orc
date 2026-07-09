"""Tests for guard predicate evaluation (conditional node execution)."""

from __future__ import annotations

import json
from typing import Any

from llm_orc.core.execution.phases.guard_evaluator import GuardEvaluator
from llm_orc.schemas.agent_config import LlmAgentConfig


class TestGuardPredicate:
    """The `when:` predicate decides whether a node runs."""

    def test_skips_when_guard_predicate_is_false(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="build",
            model_profile="gpt4",
            depends_on=["gate"],
            when="${gate.ok}",
        )
        results: dict[str, Any] = {
            "gate": {"status": "success", "response": json.dumps({"ok": False})}
        }
        assert evaluator.should_run(agent, results) is False

    def test_runs_when_guard_predicate_is_true(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="build",
            model_profile="gpt4",
            depends_on=["gate"],
            when="${gate.ok}",
        )
        results: dict[str, Any] = {
            "gate": {"status": "success", "response": json.dumps({"ok": True})}
        }
        assert evaluator.should_run(agent, results) is True

    def test_runs_when_no_guard(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="plain", model_profile="gpt4")
        assert evaluator.should_run(agent, {}) is True

    def test_equality_predicate_matches(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="coder",
            model_profile="gpt4",
            depends_on=["router"],
            when='${router.choice} == "code"',
        )
        results: dict[str, Any] = {
            "router": {"status": "success", "response": json.dumps({"choice": "code"})}
        }
        assert evaluator.should_run(agent, results) is True

    def test_equality_predicate_does_not_match(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="coder",
            model_profile="gpt4",
            depends_on=["router"],
            when='${router.choice} == "code"',
        )
        results: dict[str, Any] = {
            "router": {"status": "success", "response": json.dumps({"choice": "prose"})}
        }
        assert evaluator.should_run(agent, results) is False


class TestSkipPropagation:
    """A node whose every dependency skipped is itself skipped; a join runs
    on whichever branch fired."""

    def test_skips_when_all_dependencies_skipped(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="join", model_profile="gpt4", depends_on=["a", "b"])
        results: dict[str, Any] = {
            "a": {"status": "skipped", "response": None},
            "b": {"status": "skipped", "response": None},
        }
        assert evaluator.should_run(agent, results) is False

    def test_runs_when_any_dependency_produced(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="join", model_profile="gpt4", depends_on=["a", "b"])
        results: dict[str, Any] = {
            "a": {"status": "skipped", "response": None},
            "b": {"status": "success", "response": "ok"},
        }
        assert evaluator.should_run(agent, results) is True
