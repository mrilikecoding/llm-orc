"""Tests for usage collector."""

from unittest.mock import Mock, patch

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

    def test_start_agent_resource_monitoring_exception(self) -> None:
        """Test start_agent_resource_monitoring when psutil raises exception."""
        collector = UsageCollector()
        
        # Mock psutil to raise exception
        with patch('llm_orc.core.execution.usage_collector.psutil') as mock_psutil:
            mock_psutil.cpu_percent.side_effect = Exception("psutil error")
            
            # When
            result = collector.start_agent_resource_monitoring("agent1")
            
            # Then - should return empty dict on exception
            assert result == {}

    def test_sample_agent_resources_no_monitoring(self) -> None:
        """Test sample_agent_resources when no monitoring was started."""
        collector = UsageCollector()
        
        # When/Then - should not raise exception
        collector.sample_agent_resources("agent1")

    def test_sample_agent_resources_with_exception(self) -> None:
        """Test sample_agent_resources when psutil raises exception."""
        collector = UsageCollector()
        
        # Start monitoring first
        collector.start_agent_resource_monitoring("agent1")
        
        # Mock psutil to raise exception during sampling
        with patch('llm_orc.core.execution.usage_collector.psutil') as mock_psutil:
            mock_psutil.cpu_percent.side_effect = Exception("psutil error")
            
            # When/Then - should not raise exception (silent handling)
            collector.sample_agent_resources("agent1")

    def test_finalize_agent_resource_monitoring_not_started(self) -> None:
        """Test finalize_agent_resource_monitoring when monitoring wasn't started."""
        collector = UsageCollector()
        
        # When
        result = collector.finalize_agent_resource_monitoring("agent1")
        
        # Then - should return empty dict
        assert result == {}

    def test_collect_agent_usage_with_none_model(self) -> None:
        """Test collect_agent_usage with None model."""
        collector = UsageCollector()
        
        # When/Then - should not raise exception
        collector.collect_agent_usage("agent1", None)

    def test_collect_agent_usage_model_without_method(self) -> None:
        """Test collect_agent_usage with model that doesn't have get_last_usage."""
        collector = UsageCollector()
        
        # Mock model without get_last_usage method
        mock_model = Mock(spec=[])  # No methods
        
        # When/Then - should not raise exception
        collector.collect_agent_usage("agent1", mock_model)

    def test_collect_agent_usage_model_returns_none(self) -> None:
        """Test collect_agent_usage when model returns None usage."""
        collector = UsageCollector()
        
        # Mock model that returns None
        mock_model = Mock()
        mock_model.get_last_usage.return_value = None
        
        # When/Then - should not raise exception
        collector.collect_agent_usage("agent1", mock_model)

    def test_get_usage_breakdown_by_metric(self) -> None:
        """Test get_usage_breakdown_by_metric method."""
        collector = UsageCollector()
        
        # Add some usage data
        collector.add_manual_usage("agent1", {
            "total_tokens": 100,
            "total_cost_usd": 0.05
        })
        collector.add_manual_usage("agent2", {
            "total_tokens": 200,
            "total_cost_usd": 0.10
        })
        
        # When
        breakdown = collector.get_usage_breakdown_by_metric()
        
        # Then
        assert "tokens" in breakdown
        assert "costs" in breakdown
        assert "durations" in breakdown
        assert breakdown["tokens"]["agent1"]["total_tokens"] == 100
        assert breakdown["tokens"]["agent2"]["total_tokens"] == 200

    def test_remove_agent_usage(self) -> None:
        """Test remove_agent_usage method."""
        collector = UsageCollector()
        
        # Add usage data
        collector.add_manual_usage("agent1", {"total_tokens": 100})
        assert collector.has_usage_for_agent("agent1")
        
        # When
        collector.remove_agent_usage("agent1")
        
        # Then
        assert not collector.has_usage_for_agent("agent1")

    def test_get_agent_usage_data(self) -> None:
        """Test get_agent_usage_data method."""
        collector = UsageCollector()
        
        # Add usage data
        usage_data = {"total_tokens": 100, "total_cost_usd": 0.05}
        collector.add_manual_usage("agent1", usage_data)
        
        # When
        retrieved = collector.get_agent_usage_data("agent1")
        
        # Then
        assert retrieved is not None
        assert retrieved["total_tokens"] == 100
        
        # Test non-existent agent
        assert collector.get_agent_usage_data("non_existent") is None

    def test_calculate_usage_summary_without_synthesis(self) -> None:
        """Test calculate_usage_summary without synthesis usage."""
        collector = UsageCollector()
        
        # Add usage data
        collector.add_manual_usage("agent1", {
            "total_tokens": 100,
            "input_tokens": 60,
            "output_tokens": 40,
            "cost_usd": 0.05,
            "duration_ms": 1000
        })
        
        # When
        summary = collector.calculate_usage_summary()
        
        # Then
        assert "agents" in summary
        assert "totals" in summary
        assert summary["totals"]["total_tokens"] == 100
        assert summary["totals"]["total_input_tokens"] == 60
        assert summary["totals"]["total_output_tokens"] == 40
        assert summary["totals"]["total_cost_usd"] == 0.05
        assert summary["totals"]["agents_count"] == 1

    def test_calculate_usage_summary_with_synthesis(self) -> None:
        """Test calculate_usage_summary with synthesis usage."""
        collector = UsageCollector()
        
        # Add agent usage
        collector.add_manual_usage("agent1", {
            "total_tokens": 100,
            "input_tokens": 60,
            "output_tokens": 40,
            "cost_usd": 0.05,
            "duration_ms": 1000
        })
        
        # Synthesis usage data
        synthesis_usage = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.02,
            "duration_ms": 500
        }
        
        # When
        summary = collector.calculate_usage_summary(synthesis_usage)
        
        # Then
        assert "synthesis" in summary
        assert summary["synthesis"] == synthesis_usage
        assert summary["totals"]["total_tokens"] == 150  # 100 + 50
        assert summary["totals"]["total_input_tokens"] == 90  # 60 + 30
        assert summary["totals"]["total_output_tokens"] == 60  # 40 + 20
        assert summary["totals"]["total_cost_usd"] == 0.07  # 0.05 + 0.02

    def test_merge_usage(self) -> None:
        """Test merge_usage method."""
        collector = UsageCollector()
        
        # Add initial usage
        collector.add_manual_usage("agent1", {"total_tokens": 100})
        
        # Prepare usage to merge
        other_usage = {
            "agent2": {"total_tokens": 200, "cost_usd": 0.10},
            "agent3": {"total_tokens": 50}
        }
        
        # When
        collector.merge_usage(other_usage)
        
        # Then
        usage = collector.get_agent_usage()
        assert "agent1" in usage
        assert "agent2" in usage
        assert "agent3" in usage
        assert usage["agent2"]["total_tokens"] == 200

    def test_sample_agent_resources_successful_sampling(self) -> None:
        """Test sample_agent_resources with successful resource sampling."""
        collector = UsageCollector()
        
        # Start monitoring first
        collector.start_agent_resource_monitoring("agent1")
        
        # Mock psutil to return specific values
        with patch('llm_orc.core.execution.usage_collector.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 75.5
            mock_psutil.virtual_memory.return_value.percent = 85.2
            
            # When - take samples
            collector.sample_agent_resources("agent1")
            collector.sample_agent_resources("agent1")  # Take multiple samples
            
            # Then - verify psutil was called
            assert mock_psutil.cpu_percent.called
            assert mock_psutil.virtual_memory.called

    def test_finalize_agent_resource_monitoring_successful(self) -> None:
        """Test finalize_agent_resource_monitoring with successful monitoring."""
        collector = UsageCollector()
        
        # Start monitoring first
        collector.start_agent_resource_monitoring("agent1")
        
        # Take some samples with mock data
        with patch('llm_orc.core.execution.usage_collector.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 70.0
            mock_psutil.virtual_memory.return_value.percent = 80.0
            collector.sample_agent_resources("agent1")
            
            # When
            result = collector.finalize_agent_resource_monitoring("agent1")
            
            # Then
            assert "peak_cpu" in result
            assert "avg_cpu" in result 
            assert "peak_memory" in result
            assert "avg_memory" in result
            assert "resource_duration_seconds" in result
            assert "resource_sample_count" in result
            assert result["peak_cpu"] >= 0
            assert result["avg_cpu"] >= 0

    def test_finalize_agent_resource_monitoring_exception(self) -> None:
        """Test finalize_agent_resource_monitoring when exception occurs."""
        collector = UsageCollector()
        
        # Start monitoring
        collector.start_agent_resource_monitoring("agent1")
        
        # Force an exception by corrupting internal state
        collector._agent_resource_metrics["agent1"] = "invalid_data"  # Not a dict
        
        # When
        result = collector.finalize_agent_resource_monitoring("agent1")
        
        # Then - should return empty dict on exception
        assert result == {}

    def test_add_manual_usage_with_total_cost_usd(self) -> None:
        """Test add_manual_usage stores data exactly as provided."""
        collector = UsageCollector()
        
        # Test with total_cost_usd field (stored as-is)
        collector.add_manual_usage("agent1", {
            "total_tokens": 100,
            "total_cost_usd": 0.05
        })
        
        # When
        usage = collector.get_agent_usage()
        
        # Then
        assert usage["agent1"]["total_cost_usd"] == 0.05
        assert usage["agent1"]["total_tokens"] == 100

    def test_get_total_cost_with_invalid_cost_types(self) -> None:
        """Test get_total_cost handles invalid cost types correctly."""
        collector = UsageCollector()
        
        # Add usage with valid and invalid cost types
        collector.add_manual_usage("agent1", {
            "total_tokens": 100,
            "cost_usd": 0.05  # Valid float
        })
        collector.add_manual_usage("agent2", {
            "total_tokens": 200,
            "cost_usd": 10  # Valid int
        })
        
        # Manually add invalid cost type to test the isinstance check
        collector._agent_usage["agent3"] = {
            "total_tokens": 50,
            "cost_usd": "invalid"  # Invalid string
        }
        
        # When
        total_cost = collector.get_total_cost()
        
        # Then - should only sum valid costs (0.05 + 10.0 = 10.05)
        assert total_cost == 10.05

    def test_collect_agent_usage_with_model_profile(self) -> None:
        """Test collect_agent_usage when model has model_profile."""
        collector = UsageCollector()
        
        # Mock model with usage and model_profile
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {
            "total_tokens": 100,
            "cost_usd": 0.05
        }
        mock_model.get_model_profile.return_value = "gpt-4"
        
        # When
        collector.collect_agent_usage("agent1", mock_model)
        
        # Then
        usage = collector.get_agent_usage()
        assert usage["agent1"]["model_profile"] == "gpt-4"

    def test_collect_agent_usage_with_resource_metrics(self) -> None:
        """Test collect_agent_usage merges resource metrics when available."""
        collector = UsageCollector()
        
        # Start resource monitoring to create metrics
        collector.start_agent_resource_monitoring("agent1")
        
        # Mock model with usage
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {
            "total_tokens": 100,
            "cost_usd": 0.05
        }
        mock_model.get_model_profile.return_value = None
        
        # When
        collector.collect_agent_usage("agent1", mock_model)
        
        # Then
        usage = collector.get_agent_usage()
        assert usage["agent1"]["total_tokens"] == 100
        # Should also have resource monitoring fields merged in
        assert "start_time" in usage["agent1"]

