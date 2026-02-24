"""Tests for fan-out result gathering (issue #73)."""

import pytest

from llm_orc.core.execution.fan_out.expander import FanOutExpander
from llm_orc.core.execution.fan_out.gatherer import FanOutGatherer


class TestFanOutGatherer:
    """Test FanOutGatherer functionality."""

    @pytest.fixture
    def expander(self) -> FanOutExpander:
        """Create a FanOutExpander instance."""
        return FanOutExpander()

    @pytest.fixture
    def gatherer(self, expander: FanOutExpander) -> FanOutGatherer:
        """Create a FanOutGatherer instance."""
        return FanOutGatherer(expander)

    def test_record_instance_result_success(self, gatherer: FanOutGatherer) -> None:
        """record_instance_result stores successful result."""
        gatherer.record_instance_result(
            instance_name="extractor[0]",
            result={"concepts": ["theme1"]},
            success=True,
        )

        # Should be tracked
        assert gatherer.has_instance_result("extractor[0]")

    def test_record_instance_result_failure(self, gatherer: FanOutGatherer) -> None:
        """record_instance_result stores failed result with error."""
        gatherer.record_instance_result(
            instance_name="extractor[1]",
            result=None,
            success=False,
            error="timeout after 60s",
        )

        # Should be tracked
        assert gatherer.has_instance_result("extractor[1]")

    def test_gather_results_orders_by_index(self, gatherer: FanOutGatherer) -> None:
        """gather_results returns results ordered by instance index."""
        # Record out of order
        gatherer.record_instance_result("extractor[2]", "result_2", True)
        gatherer.record_instance_result("extractor[0]", "result_0", True)
        gatherer.record_instance_result("extractor[1]", "result_1", True)

        result = gatherer.gather_results("extractor")

        assert result["response"] == ["result_0", "result_1", "result_2"]
        assert result["status"] == "success"
        assert result["fan_out"] is True

    def test_gather_results_partial_on_some_failures(
        self, gatherer: FanOutGatherer
    ) -> None:
        """gather_results returns partial status when some instances fail."""
        gatherer.record_instance_result("extractor[0]", "result_0", True)
        gatherer.record_instance_result("extractor[1]", None, False, error="timeout")
        gatherer.record_instance_result("extractor[2]", "result_2", True)

        result = gatherer.gather_results("extractor")

        assert result["status"] == "partial"
        assert result["response"] == ["result_0", None, "result_2"]
        assert len(result["instances"]) == 3
        assert result["instances"][1]["status"] == "failed"
        assert result["instances"][1]["error"] == "timeout"

    def test_gather_results_failed_on_all_failures(
        self, gatherer: FanOutGatherer
    ) -> None:
        """gather_results returns failed status when all instances fail."""
        gatherer.record_instance_result("extractor[0]", None, False, error="error_0")
        gatherer.record_instance_result("extractor[1]", None, False, error="error_1")

        result = gatherer.gather_results("extractor")

        assert result["status"] == "failed"
        assert result["response"] == [None, None]

    def test_gather_results_instance_statuses(self, gatherer: FanOutGatherer) -> None:
        """gather_results includes per-instance status information."""
        gatherer.record_instance_result("analyzer[0]", {"data": "a"}, True)
        gatherer.record_instance_result(
            "analyzer[1]", None, False, error="network error"
        )

        result = gatherer.gather_results("analyzer")

        assert len(result["instances"]) == 2
        assert result["instances"][0]["index"] == 0
        assert result["instances"][0]["status"] == "success"
        assert result["instances"][1]["index"] == 1
        assert result["instances"][1]["status"] == "failed"
        assert result["instances"][1]["error"] == "network error"

    def test_get_error_summary(self, gatherer: FanOutGatherer) -> None:
        """get_error_summary returns error details for failed instances."""
        gatherer.record_instance_result("extractor[0]", "ok", True)
        gatherer.record_instance_result("extractor[1]", None, False, error="timeout")
        gatherer.record_instance_result(
            "extractor[2]", None, False, error="rate limited"
        )

        summary = gatherer.get_error_summary("extractor")

        assert summary["total_instances"] == 3
        assert summary["failed_count"] == 2
        assert summary["success_count"] == 1
        assert len(summary["errors"]) == 2
        assert {"index": 1, "error": "timeout"} in summary["errors"]
        assert {"index": 2, "error": "rate limited"} in summary["errors"]

    def test_get_error_summary_no_errors(self, gatherer: FanOutGatherer) -> None:
        """get_error_summary returns empty errors list when all succeed."""
        gatherer.record_instance_result("extractor[0]", "ok", True)
        gatherer.record_instance_result("extractor[1]", "ok", True)

        summary = gatherer.get_error_summary("extractor")

        assert summary["failed_count"] == 0
        assert summary["success_count"] == 2
        assert summary["errors"] == []

    def test_has_pending_instances_true(self, gatherer: FanOutGatherer) -> None:
        """has_pending_instances returns True when not all recorded."""
        # Only record 2 of expected 3
        gatherer.record_instance_result("extractor[0]", "ok", True)
        gatherer.record_instance_result("extractor[1]", "ok", True)

        # Should have pending if we expected 3
        assert gatherer.has_pending_instances("extractor", expected_count=3)

    def test_has_pending_instances_false(self, gatherer: FanOutGatherer) -> None:
        """has_pending_instances returns False when all recorded."""
        gatherer.record_instance_result("extractor[0]", "ok", True)
        gatherer.record_instance_result("extractor[1]", "ok", True)

        assert not gatherer.has_pending_instances("extractor", expected_count=2)

    def test_get_recorded_count(self, gatherer: FanOutGatherer) -> None:
        """get_recorded_count returns number of recorded instances."""
        assert gatherer.get_recorded_count("extractor") == 0

        gatherer.record_instance_result("extractor[0]", "ok", True)
        assert gatherer.get_recorded_count("extractor") == 1

        gatherer.record_instance_result("extractor[1]", "ok", True)
        assert gatherer.get_recorded_count("extractor") == 2

    def test_clear_clears_all_recorded_results(self, gatherer: FanOutGatherer) -> None:
        """clear removes all recorded results for an agent."""
        gatherer.record_instance_result("extractor[0]", "ok", True)
        gatherer.record_instance_result("extractor[1]", "ok", True)

        gatherer.clear("extractor")

        assert gatherer.get_recorded_count("extractor") == 0

    def test_multiple_agents_tracked_independently(
        self, gatherer: FanOutGatherer
    ) -> None:
        """Multiple fan-out agents are tracked independently."""
        gatherer.record_instance_result("extractor[0]", "e0", True)
        gatherer.record_instance_result("analyzer[0]", "a0", True)
        gatherer.record_instance_result("extractor[1]", "e1", True)
        gatherer.record_instance_result("analyzer[1]", "a1", True)

        extractor_result = gatherer.gather_results("extractor")
        analyzer_result = gatherer.gather_results("analyzer")

        assert extractor_result["response"] == ["e0", "e1"]
        assert analyzer_result["response"] == ["a0", "a1"]
