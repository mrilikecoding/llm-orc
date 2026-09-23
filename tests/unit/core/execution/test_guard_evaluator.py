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

    def test_run_marked_node_runs_despite_no_dependency_succeeding(self) -> None:
        """on_dependency_failure: run waives rule 1's cascade — a
        failure-handling node (a refusal composer) executes so it can
        read the failure it exists to report."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="shape",
            model_profile="gpt4",
            depends_on=["upstream"],
            on_dependency_failure="run",
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.should_run(agent, results) is True

    def test_run_marked_node_still_honors_when_false(self) -> None:
        """The flag only waives rule 1; `when:` semantics are unchanged
        and are still evaluated (after rule 1, per the field's contract)."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="shape",
            model_profile="gpt4",
            depends_on=["upstream"],
            on_dependency_failure="run",
            when="${upstream.ok}",
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.should_run(agent, results) is False

    def test_skips_when_sole_dependency_is_a_handled_failure(self) -> None:
        """X1: a dependency that ran only via on_dependency_failure: run
        — status success, handled_failure True — does not count as a
        succeeded dependency for the cascade rule: it is the refusal/
        deliverable, not proof the pipeline worked."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="downstream", model_profile="gpt4", depends_on=["shape"]
        )
        results: dict[str, Any] = {
            "shape": {
                "status": "success",
                "response": "refusal",
                "handled_failure": True,
            },
        }
        assert evaluator.should_run(agent, results) is False

    def test_runs_when_sole_dependency_is_a_partial_fan_out(self) -> None:
        """A gathered fan-out dependency whose status is "partial" (some
        instances succeeded, some failed) counts as a successful
        dependency for the cascade rule — the consumer still runs
        (fail-closed-composition, partial fan-out decision)."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="compiler", model_profile="gpt4", depends_on=["searcher"]
        )
        results: dict[str, Any] = {
            "searcher": {
                "status": "partial",
                "response": ["a", None],
                "fan_out": True,
            },
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

    def test_handled_failure_true_for_a_run_marked_node_with_no_success(self) -> None:
        """X1: GuardEvaluator.handled_failure is the flag EnsembleExecutor
        stamps on the node's own result once it runs."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="shape",
            model_profile="gpt4",
            depends_on=["upstream"],
            on_dependency_failure="run",
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.handled_failure(agent, results) is True
        assert evaluator.dependency_failure_text(agent, results) == (
            "upstream (failed: boom)"
        )

    def test_handled_failure_false_when_a_dependency_succeeded(self) -> None:
        """A run-marked node whose dependency actually succeeded is a
        normal run, not a handled failure — nothing to mark."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="shape",
            model_profile="gpt4",
            depends_on=["upstream"],
            on_dependency_failure="run",
        )
        results: dict[str, Any] = {"upstream": {"status": "success", "response": "ok"}}
        assert evaluator.handled_failure(agent, results) is False

    def test_handled_failure_false_when_sole_dependency_is_neutral(self) -> None:
        """runwhenskipped decision (addendum 2026-09-23): a run-marked
        node whose sole dependency is itself skipped_by_guard (a plain
        when:-false skip, nothing genuinely failed) is a normal run, not
        a failure handler — before this fix, "no dependency succeeded"
        alone (regardless of whether anything was blocking) was enough
        to mark handled_failure, so a total non-event (an intentional
        upstream guard skip) made a wrapping ensemble:/dispatch:/loop:
        node fail, naming "handled failure: b (skipped)" though nothing
        failed."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="t",
            model_profile="gpt4",
            depends_on=["b"],
            on_dependency_failure="run",
        )
        results: dict[str, Any] = {"b": {"status": "skipped", "response": None}}
        assert evaluator.handled_failure(agent, results) is False

    def test_handled_failure_false_when_flag_not_set(self) -> None:
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="downstream", model_profile="gpt4", depends_on=["upstream"]
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.handled_failure(agent, results) is False

    def test_reason_set_for_a_run_marked_node_skipped_by_its_own_when(self) -> None:
        """S1 fix (addendum 2026-09-23): a run-marked node is never
        skipped by rule 1's cascade GATE (should_run bypasses it), but
        if its OWN `when:` evaluates false and none of its dependencies
        succeeded, that skip is just as much a lost failure signal as a
        plain node's cascade skip — dependency_skip_reason no longer
        special-cases on_dependency_failure. Before this fix, this
        skip carried no reason and read as a neutral when-skip, so a
        wrapping ensemble:/dispatch:/loop: node reported success over a
        total upstream crash."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(
            name="shape",
            model_profile="gpt4",
            depends_on=["upstream"],
            on_dependency_failure="run",
            when="${upstream.ok}",
        )
        results: dict[str, Any] = {
            "upstream": {"status": "failed", "error": "boom", "response": None},
        }
        assert evaluator.should_run(agent, results) is False
        reason = evaluator.dependency_skip_reason(agent, results)
        assert reason == "no dependency succeeded: upstream (failed: boom)"

    def test_reason_none_when_every_dependency_is_itself_neutral(self) -> None:
        """The whenchain decision: a node whose sole dependency was
        itself skipped_by_guard (a plain `when:`-false skip, nothing
        genuinely failed) is cascade-skipped by rule 1 (no dependency is
        ok), but the skip is neutral, not a failure — no dependency is
        blocking either. Before this fix, cascade skips always carried
        a reason (looked like skipped_by_failure) whenever "no dependency
        ok" held, regardless of whether anything was actually blocking,
        so a whole chain of intentional when-skips read as a failure at
        the wrapping ensemble: node."""
        evaluator = GuardEvaluator()
        agent = LlmAgentConfig(name="t", model_profile="gpt4", depends_on=["b"])
        results: dict[str, Any] = {
            "b": {"status": "skipped", "response": None},
        }
        assert evaluator.should_run(agent, results) is False
        assert evaluator.dependency_skip_reason(agent, results) is None
