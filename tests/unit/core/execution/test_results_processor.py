"""Tests for results processor."""

import time
from typing import Any
from unittest.mock import Mock

from llm_orc.core.execution.results_processor import ResultsProcessor


class TestResultsProcessor:
    """Test results processing and formatting functionality."""

    def test_create_initial_result(self) -> None:
        """Test creating initial result structure."""
        processor = ResultsProcessor()

        start_time = time.time()
        result = processor.create_initial_result("test_ensemble", "test input", 3)

        assert result["ensemble"] == "test_ensemble"
        assert result["status"] == "running"
        assert result["input"]["data"] == "test input"
        assert result["results"] == {}
        assert result["synthesis"] is None
        assert result["metadata"]["agents_used"] == 3
        assert result["metadata"]["started_at"] >= start_time

    def test_finalize_result_success(self) -> None:
        """Test finalizing result without errors."""
        processor = ResultsProcessor()

        start_time = time.time() - 1.5  # Simulate 1.5 second execution
        result = {
            "ensemble": "test",
            "status": "running",
            "metadata": {"agents_used": 2},
        }
        agent_usage = {
            "agent1": {"total_tokens": 100, "cost_usd": 0.05},
            "agent2": {"total_tokens": 150, "cost_usd": 0.08},
        }

        finalized = processor.finalize_result(result, agent_usage, False, start_time)

        assert finalized["status"] == "completed"
        assert "duration" in finalized["metadata"]
        assert "completed_at" in finalized["metadata"]
        assert "usage" in finalized["metadata"]

        # Check duration format
        duration = finalized["metadata"]["duration"]
        assert duration.endswith("s")
        assert float(duration[:-1]) > 1.0  # Should be > 1 second

    def test_finalize_result_with_errors(self) -> None:
        """Test finalizing result with errors."""
        processor = ResultsProcessor()

        start_time = time.time()
        result = {"ensemble": "test", "status": "running", "metadata": {}}

        finalized = processor.finalize_result(result, {}, True, start_time)

        assert finalized["status"] == "completed_with_errors"

    def test_calculate_usage_summary_basic(self) -> None:
        """Test calculating basic usage summary."""
        processor = ResultsProcessor()

        agent_usage = {
            "agent1": {
                "total_tokens": 100,
                "input_tokens": 60,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "duration_ms": 1000,
            },
            "agent2": {
                "total_tokens": 150,
                "input_tokens": 90,
                "output_tokens": 60,
                "cost_usd": 0.08,
                "duration_ms": 1500,
            },
        }

        summary = processor.calculate_usage_summary(agent_usage, None)

        # Check structure
        assert "agents" in summary
        assert "totals" in summary
        assert "synthesis" not in summary

        # Check agents data
        assert summary["agents"] == agent_usage

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
        processor = ResultsProcessor()

        agent_usage = {
            "agent1": {
                "total_tokens": 100,
                "input_tokens": 60,
                "output_tokens": 40,
                "cost_usd": 0.05,
                "duration_ms": 1000,
            }
        }
        synthesis_usage = {
            "total_tokens": 50,
            "input_tokens": 30,
            "output_tokens": 20,
            "cost_usd": 0.03,
            "duration_ms": 500,
        }

        summary = processor.calculate_usage_summary(agent_usage, synthesis_usage)

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
        processor = ResultsProcessor()

        summary = processor.calculate_usage_summary({}, None)

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
        processor = ResultsProcessor()

        agent_usage = {
            "agent1": {"total_tokens": 100},  # Missing other fields
            "agent2": {"cost_usd": 0.05, "duration_ms": 1000},  # Missing tokens
        }

        summary = processor.calculate_usage_summary(agent_usage, None)

        totals = summary["totals"]
        assert totals["total_tokens"] == 100  # Only agent1 has tokens
        assert totals["total_input_tokens"] == 0  # No agent has input_tokens
        assert totals["total_output_tokens"] == 0  # No agent has output_tokens
        assert totals["total_cost_usd"] == 0.05  # Only agent2 has cost
        assert totals["total_duration_ms"] == 1000  # Only agent2 has duration

    def test_process_agent_results_success(self) -> None:
        """Test processing successful agent results."""
        processor = ResultsProcessor()

        # Mock model with usage
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {
            "total_tokens": 100,
            "cost_usd": 0.05,
        }

        agent_results = [
            ("agent1", ("Response 1", mock_model)),
            ("agent2", ("Response 2", None)),  # No model (script agent)
        ]

        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        processor.process_agent_results(agent_results, results_dict, agent_usage)

        # Check results
        assert "agent1" in results_dict
        assert results_dict["agent1"]["response"] == "Response 1"
        assert results_dict["agent1"]["status"] == "success"

        assert "agent2" in results_dict
        assert results_dict["agent2"]["response"] == "Response 2"
        assert results_dict["agent2"]["status"] == "success"

        # Check usage (only agent1 has model usage)
        assert "agent1" in agent_usage
        assert agent_usage["agent1"]["total_tokens"] == 100
        assert "agent2" not in agent_usage

    def test_process_agent_results_with_errors(self) -> None:
        """Test processing agent results with errors."""
        processor = ResultsProcessor()

        agent_results = [
            ("agent1", ("Response 1", None)),
            ("agent2", None),  # Error case
            Exception("Something went wrong"),  # Exception case
        ]

        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        processor.process_agent_results(agent_results, results_dict, agent_usage)

        # Only agent1 should be processed
        assert len(results_dict) == 1
        assert "agent1" in results_dict
        assert results_dict["agent1"]["status"] == "success"

    def test_process_agent_results_model_without_usage(self) -> None:
        """Test processing results with model that has no usage."""
        processor = ResultsProcessor()

        mock_model = Mock()
        mock_model.get_last_usage.return_value = None  # No usage

        agent_results = [("agent1", ("Response", mock_model))]
        results_dict: dict[str, Any] = {}
        agent_usage: dict[str, Any] = {}

        processor.process_agent_results(agent_results, results_dict, agent_usage)

        assert "agent1" in results_dict
        assert "agent1" not in agent_usage  # No usage recorded

    def test_format_execution_summary(self) -> None:
        """Test formatting execution summary."""
        processor = ResultsProcessor()

        result = {
            "ensemble": "test_ensemble",
            "status": "completed",
            "metadata": {
                "duration": "2.50s",
                "usage": {
                    "totals": {
                        "agents_count": 3,
                        "total_tokens": 500,
                        "total_cost_usd": 0.25,
                    }
                },
            },
        }

        summary = processor.format_execution_summary(result)

        assert summary["ensemble_name"] == "test_ensemble"
        assert summary["status"] == "completed"
        assert summary["agents_count"] == 3
        assert summary["duration"] == "2.50s"
        assert summary["total_tokens"] == 500
        assert summary["total_cost_usd"] == 0.25
        assert summary["has_errors"] is False

    def test_format_execution_summary_with_errors(self) -> None:
        """Test formatting execution summary with errors."""
        processor = ResultsProcessor()

        result = {
            "ensemble": "test_ensemble",
            "status": "completed_with_errors",
            "metadata": {},
        }

        summary = processor.format_execution_summary(result)

        assert summary["has_errors"] is True

    def test_format_execution_summary_missing_data(self) -> None:
        """Test formatting summary with missing data."""
        processor = ResultsProcessor()

        result: dict[str, Any] = {}  # Empty result

        summary = processor.format_execution_summary(result)

        assert summary["ensemble_name"] == "unknown"
        assert summary["status"] == "unknown"
        assert summary["agents_count"] == 0
        assert summary["duration"] == "0.00s"
        assert summary["total_tokens"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["has_errors"] is False

    def test_get_agent_statuses(self) -> None:
        """Test extracting agent statuses."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success", "response": "OK"},
            "agent2": {"status": "failed", "error": "Error"},
            "agent3": {"response": "No status"},  # Missing status
            "agent4": "not a dict",  # Invalid format
        }

        statuses = processor.get_agent_statuses(results)

        assert statuses["agent1"] == "success"
        assert statuses["agent2"] == "failed"
        assert statuses["agent3"] == "unknown"
        assert statuses["agent4"] == "unknown"

    def test_count_successful_agents(self) -> None:
        """Test counting successful agents."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success"},
            "agent2": {"status": "failed"},
            "agent3": {"status": "success"},
            "agent4": {"no_status": True},
            "agent5": "not a dict",
        }

        count = processor.count_successful_agents(results)
        assert count == 2

    def test_count_failed_agents(self) -> None:
        """Test counting failed agents."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success"},
            "agent2": {"status": "failed"},
            "agent3": {"status": "failed"},
            "agent4": {"no_status": True},
        }

        count = processor.count_failed_agents(results)
        assert count == 2

    def test_extract_agent_responses(self) -> None:
        """Test extracting agent responses."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success", "response": "Response 1"},
            "agent2": {"status": "failed", "error": "Error"},
            "agent3": {"status": "success", "response": "Response 3"},
            "agent4": "not a dict",
        }

        responses = processor.extract_agent_responses(results)

        assert len(responses) == 2
        assert responses["agent1"] == "Response 1"
        assert responses["agent3"] == "Response 3"
        assert "agent2" not in responses  # No response field
        assert "agent4" not in responses  # Not a dict

    def test_extract_agent_errors(self) -> None:
        """Test extracting agent errors."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success", "response": "OK"},
            "agent2": {"status": "failed", "error": "Network error"},
            "agent3": {"status": "failed", "error": "Timeout"},
            "agent4": {"status": "failed"},  # No error field
            "agent5": "not a dict",
        }

        errors = processor.extract_agent_errors(results)

        assert len(errors) == 2
        assert errors["agent2"] == "Network error"
        assert errors["agent3"] == "Timeout"
        assert "agent1" not in errors  # Success
        assert "agent4" not in errors  # No error field
        assert "agent5" not in errors  # Not a dict

    def test_complex_workflow(self) -> None:
        """Test complex workflow with multiple operations."""
        processor = ResultsProcessor()

        # Create initial result
        result = processor.create_initial_result("complex_ensemble", "analyze data", 3)
        assert result["status"] == "running"

        # Process agent results
        mock_model = Mock()
        mock_model.get_last_usage.return_value = {"total_tokens": 100, "cost_usd": 0.05}

        agent_results = [
            ("analyzer", ("Analysis complete", mock_model)),
            ("validator", ("Validation passed", None)),
            ("reporter", ("Report generated", None)),
        ]

        results_dict = result["results"]
        agent_usage: dict[str, Any] = {}

        processor.process_agent_results(agent_results, results_dict, agent_usage)

        # Finalize result
        start_time = time.time() - 2.0  # 2 second execution
        finalized = processor.finalize_result(result, agent_usage, False, start_time)

        # Check final state
        assert finalized["status"] == "completed"
        assert len(finalized["results"]) == 3

        # Format summary
        summary = processor.format_execution_summary(finalized)
        assert summary["agents_count"] == 1  # Only analyzer has usage data
        assert summary["has_errors"] is False

        # Extract responses
        responses = processor.extract_agent_responses(finalized["results"])
        assert len(responses) == 3
        assert "analyzer" in responses
        assert "validator" in responses
        assert "reporter" in responses


class TestFanOutResultsProcessing:
    """Test fan-out result processing (issue #73)."""

    def test_add_fan_out_metadata(self) -> None:
        """Test adding fan-out execution statistics to result."""
        processor = ResultsProcessor()

        result: dict[str, Any] = {
            "ensemble": "test",
            "metadata": {"agents_used": 3},
        }

        fan_out_stats = {
            "extractor": {
                "total_instances": 5,
                "successful_instances": 4,
                "failed_instances": 1,
            },
        }

        processor.add_fan_out_metadata(result, fan_out_stats)

        assert "fan_out" in result["metadata"]
        assert result["metadata"]["fan_out"] == fan_out_stats

    def test_add_fan_out_metadata_empty_stats(self) -> None:
        """Test adding empty fan-out stats does not add key."""
        processor = ResultsProcessor()

        result: dict[str, Any] = {
            "ensemble": "test",
            "metadata": {"agents_used": 2},
        }

        processor.add_fan_out_metadata(result, {})

        assert "fan_out" not in result["metadata"]

    def test_count_fan_out_instances(self) -> None:
        """Test counting fan-out instances in results."""
        processor = ResultsProcessor()

        results = {
            "chunker": {"status": "success", "response": "..."},
            "extractor[0]": {"status": "success", "response": "r0"},
            "extractor[1]": {"status": "failed", "error": "timeout"},
            "extractor[2]": {"status": "success", "response": "r2"},
            "synthesizer": {"status": "success", "response": "..."},
        }

        stats = processor.count_fan_out_instances(results)

        assert stats["extractor"]["total_instances"] == 3
        assert stats["extractor"]["successful_instances"] == 2
        assert stats["extractor"]["failed_instances"] == 1

    def test_count_fan_out_instances_no_fan_out(self) -> None:
        """Test counting when no fan-out instances present."""
        processor = ResultsProcessor()

        results = {
            "agent1": {"status": "success", "response": "r1"},
            "agent2": {"status": "success", "response": "r2"},
        }

        stats = processor.count_fan_out_instances(results)

        assert stats == {}
