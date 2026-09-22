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

    def test_skips_when_sole_dependency_failed(self) -> None:
        """Fail-closed-composition rule 1: a failed dependency cascades
        the skip exactly like a skipped one — not just "all skipped"."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="downstream", model_profile="gpt4", depends_on=["upstream"]
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.should_run(agent, results) is False

    def test_skips_when_dependencies_mix_failed_and_skipped(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="join", model_profile="gpt4", depends_on=["a", "b"])
        results: dict[str, Any] = {
            "a": {"status": "failed", "error": "boom", "response": None},
            "b": {"status": "skipped", "response": None},
        }
        assert evaluator.should_run(agent, results) is False

    def test_runs_when_one_of_two_failed_dependencies_succeeded(self) -> None:
        """Partial failure: at least one dependency succeeded, so the
        agent still runs (rule 2's territory, not rule 1's)."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="join", model_profile="gpt4", depends_on=["a", "b"])
        results: dict[str, Any] = {
            "a": {"status": "failed", "error": "boom", "response": None},
            "b": {"status": "success", "response": "ok"},
        }
        assert evaluator.should_run(agent, results) is True


class TestDependencySkipReason:
    """Rule 1's skip record names each upstream agent and its status, with
    error text for failed ones — so a human (or a downstream script) can
    see why the node never ran."""

    def test_reason_none_when_agent_runs(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="downstream", model_profile="gpt4", depends_on=["upstream"]
        )
        results: dict[str, Any] = {"upstream": {"status": "success", "response": "ok"}}
        assert evaluator.dependency_skip_reason(agent, results) is None

    def test_reason_none_for_when_clause_skip(self) -> None:
        """A `when:`-false skip is not a dependency-cascade skip: rule 1
        is evaluated first and did not fire here, so there is no
        dependency-cascade reason to report."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="build",
            model_profile="gpt4",
            depends_on=["gate"],
            when="${gate.ok}",
        )
        results: dict[str, Any] = {
            "gate": {"status": "success", "response": '{"ok": false}'}
        }
        assert evaluator.should_run(agent, results) is False
        assert evaluator.dependency_skip_reason(agent, results) is None

    def test_reason_names_failed_dependency_with_error(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="downstream", model_profile="gpt4", depends_on=["upstream"]
        )
        results: dict[str, Any] = {
            "upstream": {
                "status": "failed",
                "error": "model 'test-fallback' not found",
                "response": None,
            },
        }
        reason = evaluator.dependency_skip_reason(agent, results)
        assert reason == (
            "no dependency succeeded: upstream (failed: "
            "model 'test-fallback' not found)"
        )

    def test_reason_names_each_dependency_in_a_mix(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="join", model_profile="gpt4", depends_on=["a", "b"])
        results: dict[str, Any] = {
            "a": {"status": "failed", "error": "boom", "response": None},
            "b": {"status": "skipped", "response": None},
        }
        reason = evaluator.dependency_skip_reason(agent, results)
        assert reason == "no dependency succeeded: a (failed: boom), b (skipped)"
