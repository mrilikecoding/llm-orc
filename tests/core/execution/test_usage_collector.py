"""Tests for usage collector."""

from unittest.mock import Mock

from llm_orc.core.execution.usage_collector import UsageCollector


class TestUsageCollector:
    """Test usage collection and aggregation functionality."""

    def test_init_creates_empty_collector(self) -> None:
        """Test that initialization creates empty collector."""
        collector = UsageCollector()

        assert collector.get_agent_usage() == {}
        assert collector.get_agent_count() == 0
        assert collector.get_total_tokens() == 0
        assert collector.get_total_cost() == 0.0

    def test_reset_clears_usage_data(self) -> None:
        """Test that reset clears all usage data."""
        collector = UsageCollector()
        collector.add_manual_usage("agent1", {"total_tokens": 100})

        assert collector.get_agent_count() == 1

        collector.reset()

        assert collector.get_agent_usage() == {}
        assert collector.get_agent_count() == 0

    def test_collect_agent_usage_with_model_instance(self) -> None:
        """Test collecting usage from model instance."""
        collector = UsageCollector()

        # Mock model instance with usage
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {
            "total_tokens": 150,
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.05,
            "duration_ms": 1500,
        }

        collector.collect_agent_usage("agent1", mock_model)

        usage = collector.get_agent_usage()
        assert "agent1" in usage
        assert usage["agent1"]["total_tokens"] == 150
        assert usage["agent1"]["cost_usd"] == 0.05

    def test_collect_agent_usage_with_none_model(self) -> None:
        """Test collecting usage with None model instance."""
        collector = UsageCollector()

        collector.collect_agent_usage("agent1", None)

        assert collector.get_agent_usage() == {}
        assert not collector.has_usage_for_agent("agent1")

    def test_collect_agent_usage_model_without_usage_method(self) -> None:
        """Test collecting usage from model without get_last_usage method."""
        collector = UsageCollector()

        # Mock model without get_last_usage method
        mock_model = Mock(spec=[])  # Empty spec, no methods

        collector.collect_agent_usage("agent1", mock_model)

        assert collector.get_agent_usage() == {}

    def test_collect_agent_usage_model_returns_none(self) -> None:
        """Test collecting usage when model returns None."""
        collector = UsageCollector()

        mock_model = Mock()
        mock_model.get_last_usage.return_value = None

        collector.collect_agent_usage("agent1", mock_model)

        assert collector.get_agent_usage() == {}

    def test_add_manual_usage(self) -> None:
        """Test manually adding usage data."""
        collector = UsageCollector()

        usage_data = {
            "total_tokens": 200,
            "input_tokens": 120,
            "output_tokens": 80,
            "cost_usd": 0.08,
            "duration_ms": 2000,
        }

        collector.add_manual_usage("script_agent", usage_data)

        assert collector.has_usage_for_agent("script_agent")
        assert collector.get_agent_usage_data("script_agent") == usage_data

    def test_merge_usage(self) -> None:
        """Test merging usage data from another source."""
        collector = UsageCollector()
        collector.add_manual_usage("agent1", {"total_tokens": 100})

        other_usage = {
            "agent2": {"total_tokens": 150, "cost_usd": 0.05},
            "agent3": {"total_tokens": 200, "cost_usd": 0.08},
        }

        collector.merge_usage(other_usage)

        usage = collector.get_agent_usage()
        assert len(usage) == 3
        assert "agent1" in usage
        assert "agent2" in usage
        assert "agent3" in usage

    def test_calculate_usage_summary_basic(self) -> None:
        """Test calculating basic usage summary."""
        collector = UsageCollector()

        collector.add_manual_usage(
            "agent1",
            {
                "total_tokens": 100,
                "input_tokens": 60,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "duration_ms": 1000,
            },
        )

        collector.add_manual_usage(
            "agent2",
            {
                "total_tokens": 150,
                "input_tokens": 90,
                "output_tokens": 60,
                "cost_usd": 0.08,
                "duration_ms": 1500,
            },
        )

        summary = collector.calculate_usage_summary()

        # Check structure
        assert "agents" in summary
        assert "totals" in summary
        assert "synthesis" not in summary

        # Check agents data
        assert len(summary["agents"]) == 2
        assert "agent1" in summary["agents"]
        assert "agent2" in summary["agents"]

        # Check totals
        totals = summary["totals"]
        assert totals["total_tokens"] == 250
        assert totals["total_input_tokens"] == 150
        assert totals["total_output_tokens"] == 100
        assert totals["total_cost_usd"] == 0.13
        assert totals["total_duration_ms"] == 2500
        assert totals["agents_count"] == 2

    def test_calculate_usage_summary_with_synthesis(self) -> None:
        """Test calculating usage summary with synthesis."""
        collector = UsageCollector()

        collector.add_manual_usage(
            "agent1",
            {
                "total_tokens": 100,
                "input_tokens": 60,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "duration_ms": 1000,
            },
        )

        synthesis_usage = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.03,
            "duration_ms": 500,
        }

        summary = collector.calculate_usage_summary(synthesis_usage)

        # Check synthesis is included
        assert "synthesis" in summary
        assert summary["synthesis"] == synthesis_usage

        # Check totals include synthesis
        totals = summary["totals"]
        assert totals["total_tokens"] == 150  # 100 + 50
        assert totals["total_input_tokens"] == 90  # 60 + 30
        assert totals["total_output_tokens"] == 60  # 40 + 20
        assert totals["total_cost_usd"] == 0.08  # 0.05 + 0.03
        assert totals["total_duration_ms"] == 1500  # 1000 + 500

    def test_calculate_usage_summary_empty(self) -> None:
        """Test calculating usage summary with no data."""
        collector = UsageCollector()

        summary = collector.calculate_usage_summary()

        assert summary["agents"] == {}
        totals = summary["totals"]
        assert totals["total_tokens"] == 0
        assert totals["total_input_tokens"] == 0
        assert totals["total_output_tokens"] == 0
        assert totals["total_cost_usd"] == 0.0
        assert totals["total_duration_ms"] == 0
        assert totals["agents_count"] == 0

    def test_calculate_usage_summary_missing_fields(self) -> None:
        """Test calculating summary with missing fields in usage data."""
        collector = UsageCollector()

        # Add usage with only some fields
        collector.add_manual_usage("agent1", {"total_tokens": 100})
        collector.add_manual_usage("agent2", {"cost_usd": 0.05, "duration_ms": 1000})

        summary = collector.calculate_usage_summary()

        totals = summary["totals"]
        assert totals["total_tokens"] == 100  # Only agent1 has tokens
        assert totals["total_input_tokens"] == 0  # No agent has input_tokens
        assert totals["total_output_tokens"] == 0  # No agent has output_tokens
        assert totals["total_cost_usd"] == 0.05  # Only agent2 has cost
        assert totals["total_duration_ms"] == 1000  # Only agent2 has duration

    def test_get_total_tokens(self) -> None:
        """Test getting total tokens across all agents."""
        collector = UsageCollector()

        collector.add_manual_usage("agent1", {"total_tokens": 100})
        collector.add_manual_usage("agent2", {"total_tokens": 150})
        collector.add_manual_usage("agent3", {"other_field": "value"})  # No tokens

        assert collector.get_total_tokens() == 250

    def test_get_total_cost(self) -> None:
        """Test getting total cost across all agents."""
        collector = UsageCollector()

        collector.add_manual_usage("agent1", {"cost_usd": 0.05})
        collector.add_manual_usage("agent2", {"cost_usd": 0.08})
        collector.add_manual_usage("agent3", {"other_field": "value"})  # No cost

        assert collector.get_total_cost() == 0.13

    def test_has_usage_for_agent(self) -> None:
        """Test checking if usage exists for specific agent."""
        collector = UsageCollector()

        collector.add_manual_usage("agent1", {"total_tokens": 100})

        assert collector.has_usage_for_agent("agent1") is True
        assert collector.has_usage_for_agent("agent2") is False

    def test_get_agent_usage_data(self) -> None:
        """Test getting usage data for specific agent."""
        collector = UsageCollector()

        usage_data = {"total_tokens": 100, "cost_usd": 0.05}
        collector.add_manual_usage("agent1", usage_data)

        assert collector.get_agent_usage_data("agent1") == usage_data
        assert collector.get_agent_usage_data("nonexistent") is None

    def test_remove_agent_usage(self) -> None:
        """Test removing usage data for specific agent."""
        collector = UsageCollector()

        collector.add_manual_usage("agent1", {"total_tokens": 100})
        collector.add_manual_usage("agent2", {"total_tokens": 150})

        assert collector.get_agent_count() == 2

        collector.remove_agent_usage("agent1")

        assert collector.get_agent_count() == 1
        assert not collector.has_usage_for_agent("agent1")
        assert collector.has_usage_for_agent("agent2")

        # Removing non-existent agent should not error
        collector.remove_agent_usage("nonexistent")

    def test_get_usage_breakdown_by_metric(self) -> None:
        """Test getting usage breakdown organized by metric type."""
        collector = UsageCollector()

        collector.add_manual_usage(
            "agent1",
            {
                "total_tokens": 100,
                "input_tokens": 60,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "duration_ms": 1000,
            },
        )

        collector.add_manual_usage(
            "agent2",
            {
                "total_tokens": 150,
                "input_tokens": 90,
                "output_tokens": 60,
                "cost_usd": 0.08,
                "duration_ms": 1500,
            },
        )

        breakdown = collector.get_usage_breakdown_by_metric()

        # Check structure
        assert "tokens" in breakdown
        assert "costs" in breakdown
        assert "durations" in breakdown

        # Check tokens breakdown
        tokens = breakdown["tokens"]
        assert tokens["agent1"]["total_tokens"] == 100
        assert tokens["agent1"]["input_tokens"] == 60
        assert tokens["agent1"]["output_tokens"] == 40
        assert tokens["agent2"]["total_tokens"] == 150

        # Check costs breakdown
        costs = breakdown["costs"]
        assert costs["agent1"]["cost_usd"] == 0.05
        assert costs["agent2"]["cost_usd"] == 0.08

        # Check durations breakdown
        durations = breakdown["durations"]
        assert durations["agent1"]["duration_ms"] == 1000
        assert durations["agent2"]["duration_ms"] == 1500

    def test_get_usage_breakdown_missing_fields(self) -> None:
        """Test usage breakdown with missing fields."""
        collector = UsageCollector()

        # Add usage with only some fields
        collector.add_manual_usage("agent1", {"total_tokens": 100})

        breakdown = collector.get_usage_breakdown_by_metric()

        tokens = breakdown["tokens"]["agent1"]
        assert tokens["total_tokens"] == 100
        assert tokens["input_tokens"] == 0  # Missing field defaults to 0
        assert tokens["output_tokens"] == 0  # Missing field defaults to 0

        costs = breakdown["costs"]["agent1"]
        assert costs["cost_usd"] == 0.0  # Missing field defaults to 0.0

        durations = breakdown["durations"]["agent1"]
        assert durations["duration_ms"] == 0  # Missing field defaults to 0

    def test_get_agent_usage_returns_copy(self) -> None:
        """Test that get_agent_usage returns a copy, not reference."""
        collector = UsageCollector()

        collector.add_manual_usage("agent1", {"total_tokens": 100})

        usage1 = collector.get_agent_usage()
        usage2 = collector.get_agent_usage()

        # Modify one copy
        usage1["agent1"]["total_tokens"] = 999

        # Original should be unchanged
        assert usage2["agent1"]["total_tokens"] == 100
        agent1_data = collector.get_agent_usage_data("agent1")
        assert agent1_data is not None
        assert agent1_data["total_tokens"] == 100

    def test_complex_workflow(self) -> None:
        """Test complex workflow with multiple operations."""
        collector = UsageCollector()

        # Collect from model instances
        mock_model1 = Mock()
        mock_model1.get_last_usage.return_value = {
            "total_tokens": 100,
            "cost_usd": 0.05,
        }

        mock_model2 = Mock()
        mock_model2.get_last_usage.return_value = {
            "total_tokens": 150,
            "cost_usd": 0.08,
        }

        collector.collect_agent_usage("llm_agent1", mock_model1)
        collector.collect_agent_usage("llm_agent2", mock_model2)

        # Add manual usage for script agent
        collector.add_manual_usage(
            "script_agent",
            {
                "total_tokens": 0,  # Script agents don't use tokens
                "duration_ms": 500,
            },
        )

        # Merge additional usage
        collector.merge_usage(
            {"external_agent": {"total_tokens": 75, "cost_usd": 0.03}}
        )

        # Calculate summary
        summary = collector.calculate_usage_summary()

        assert summary["totals"]["agents_count"] == 4
        assert summary["totals"]["total_tokens"] == 325  # 100 + 150 + 0 + 75
        assert summary["totals"]["total_cost_usd"] == 0.16  # 0.05 + 0.08 + 0.03

        # Remove one agent
        collector.remove_agent_usage("script_agent")

        final_summary = collector.calculate_usage_summary()
        assert final_summary["totals"]["agents_count"] == 3
