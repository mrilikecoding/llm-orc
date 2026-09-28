"""Tests for the engine-owned node outcome (fail-closed-composition
addendum 2026-09-23)."""

from __future__ import annotations

from typing import Any

from llm_orc.core.execution.outcome import (
    Outcome,
    child_has_errors,
    compute_outcome,
    is_blocking,
    is_neutral,
    is_ok,
    outcome_of,
    stamp_outcome,
)


class TestComputeOutcomeFromLegacyFields:
    """A result with no explicit ``outcome`` key is still classified
    correctly from its ``status``/``reason``/``handled_failure`` fields —
    the fallback every hand-built test fixture and pre-stamping result
    relies on."""

    def test_success_is_succeeded(self) -> None:
        assert compute_outcome({"status": "success"}) is Outcome.SUCCEEDED

    def test_partial_is_partial(self) -> None:
        assert compute_outcome({"status": "partial"}) is Outcome.PARTIAL

    def test_failed_is_failed(self) -> None:
        assert compute_outcome({"status": "failed", "error": "boom"}) is Outcome.FAILED

    def test_skipped_with_reason_is_skipped_by_failure(self) -> None:
        result = {"status": "skipped", "reason": "no dependency succeeded: a (failed)"}
        assert compute_outcome(result) is Outcome.SKIPPED_BY_FAILURE

    def test_skipped_without_reason_is_skipped_by_guard(self) -> None:
        assert compute_outcome({"status": "skipped"}) is Outcome.SKIPPED_BY_GUARD

    def test_handled_failure_overrides_success_status(self) -> None:
        result = {"status": "success", "handled_failure": True}
        assert compute_outcome(result) is Outcome.HANDLED_FAILURE

    def test_handled_failure_false_does_not_override(self) -> None:
        result = {"status": "success", "handled_failure": False}
        assert compute_outcome(result) is Outcome.SUCCEEDED

    def test_unrecognized_shape_is_none(self) -> None:
        assert compute_outcome({"foo": "bar"}) is None
        assert compute_outcome(None) is None

    def test_object_form_is_tolerated(self) -> None:
        class _Result:
            status = "success"
            handled_failure = False

        assert compute_outcome(_Result()) is Outcome.SUCCEEDED


class TestOutcomeOfPrefersStampedValue:
    def test_explicit_string_outcome_wins_over_legacy_shape(self) -> None:
        # A stamped "failed" outcome on an otherwise success-shaped
        # record: the stamp is authoritative, not re-derived.
        result = {"status": "success", "outcome": "failed"}
        assert outcome_of(result) is Outcome.FAILED

    def test_missing_outcome_falls_back_to_compute(self) -> None:
        result = {"status": "success"}
        assert outcome_of(result) is Outcome.SUCCEEDED

    def test_garbage_outcome_string_falls_back_to_compute(self) -> None:
        result = {"status": "success", "outcome": "not-a-real-outcome"}
        assert outcome_of(result) is Outcome.SUCCEEDED


class TestPredicates:
    def test_ok_covers_succeeded_and_partial(self) -> None:
        assert is_ok({"status": "success"}) is True
        assert is_ok({"status": "partial"}) is True

    def test_ok_excludes_handled_failure(self) -> None:
        assert is_ok({"status": "success", "handled_failure": True}) is False

    def test_ok_excludes_failed_and_skips(self) -> None:
        assert is_ok({"status": "failed"}) is False
        assert is_ok({"status": "skipped"}) is False
        assert is_ok({"status": "skipped", "reason": "x"}) is False

    def test_blocking_covers_failed_skipped_by_failure_and_handled_failure(
        self,
    ) -> None:
        assert is_blocking({"status": "failed"}) is True
        assert is_blocking({"status": "skipped", "reason": "x"}) is True
        assert is_blocking({"status": "success", "handled_failure": True}) is True

    def test_blocking_excludes_ok_and_guard_skip(self) -> None:
        assert is_blocking({"status": "success"}) is False
        assert is_blocking({"status": "partial"}) is False
        assert is_blocking({"status": "skipped"}) is False

    def test_neutral_is_guard_skip_only(self) -> None:
        assert is_neutral({"status": "skipped"}) is True
        assert is_neutral({"status": "skipped", "reason": "x"}) is False
        assert is_neutral({"status": "success"}) is False

    def test_none_result_is_neither_ok_nor_blocking_nor_neutral(self) -> None:
        assert is_ok(None) is False
        assert is_blocking(None) is False
        assert is_neutral(None) is False


class TestStampOutcome:
    """The one function every stamping site funnels through."""

    def test_stamps_succeeded_and_clears_has_errors(self) -> None:
        record: dict[str, Any] = {"status": "success", "response": "ok"}
        stamp_outcome(record)
        assert record["outcome"] == "succeeded"
        assert record["has_errors"] is False

    def test_stamps_failed_and_sets_has_errors(self) -> None:
        record: dict[str, Any] = {"status": "failed", "error": "boom"}
        stamp_outcome(record)
        assert record["outcome"] == "failed"
        assert record["has_errors"] is True

    def test_stamps_handled_failure(self) -> None:
        record: dict[str, Any] = {
            "status": "success",
            "response": "refusal",
            "handled_failure": True,
        }
        stamp_outcome(record)
        assert record["outcome"] == "handled_failure"
        assert record["has_errors"] is True

    def test_stamps_skipped_by_guard(self) -> None:
        record: dict[str, Any] = {"status": "skipped", "response": None}
        stamp_outcome(record)
        assert record["outcome"] == "skipped_by_guard"
        assert record["has_errors"] is False

    def test_returns_the_same_mutated_record(self) -> None:
        record: dict[str, Any] = {"status": "success"}
        assert stamp_outcome(record) is record


class TestChildHasErrors:
    """Whether a child ensemble/dispatch/loop execution's own subtree had
    a blocking outcome anywhere in it, read from the child's own result
    (JSON string or dict)."""

    def test_true_when_child_has_errors_field_true(self) -> None:
        response = (
            '{"status": "completed_with_errors", "has_errors": true, "results": {}}'
        )
        assert child_has_errors(response) is True

    def test_false_when_child_has_errors_field_false(self) -> None:
        response = '{"status": "completed", "has_errors": false, "results": {}}'
        assert child_has_errors(response) is False

    def test_falls_back_to_status_when_has_errors_field_absent(self) -> None:
        """A pre-addendum child result has no has_errors field yet."""
        response = '{"status": "completed_with_errors", "results": {}}'
        assert child_has_errors(response) is True

    def test_false_for_completed_status_with_no_has_errors_field(self) -> None:
        response = '{"status": "completed", "results": {}}'
        assert child_has_errors(response) is False

    def test_false_for_non_json_response(self) -> None:
        assert child_has_errors("not json") is False

    def test_false_for_non_dict_response(self) -> None:
        assert child_has_errors(None) is False
        assert child_has_errors(["a", "b"]) is False

    def test_accepts_dict_shape_directly(self) -> None:
        assert child_has_errors({"has_errors": True}) is True

    def test_accepts_loop_wrapper_shape(self) -> None:
        """A loop: node's own response wraps {output, iterations,
        terminated, has_errors} rather than a raw child execution record
        with a results key — has_errors still reads the same way."""
        response = {
            "output": {},
            "iterations": 2,
            "terminated": "max_iterations",
            "has_errors": True,
        }
        assert child_has_errors(response) is True
