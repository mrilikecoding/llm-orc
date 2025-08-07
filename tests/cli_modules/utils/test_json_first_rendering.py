"""Tests for JSON-first rendering architecture."""

from llm_orc.cli_modules.utils.json_renderer import (
    render_json_as_markdown,
    render_json_as_text,
    transform_to_execution_json,
)


class TestExecutionResultsTransformation:
    """Test transformation of raw execution data to structured JSON."""

    def test_transform_complete_execution_data(self) -> None:
        """Test transformation of complete execution result with all data types."""
        raw_results = {
            "agent1": {"content": "Result 1", "status": "success"},
            "agent2": {"content": "Result 2", "status": "success"},
        }

        raw_usage = {
            "agents": {
                "agent1": {"total_tokens": 150, "total_cost_usd": 0.001},
                "agent2": {"total_tokens": 200, "total_cost_usd": 0.002},
            },
            "totals": {"total_tokens": 350, "total_cost_usd": 0.003, "agents_count": 2},
        }

        raw_metadata = {
            "adaptive_resource_management": {
                "management_type": "user_configured",
                "concurrency_decisions": [{"configured_limit": 5, "agent_count": 2}],
                "execution_metrics": {
                    "peak_cpu": 45.2,
                    "avg_cpu": 32.1,
                    "peak_memory": 78.5,
                    "avg_memory": 65.3,
                    "sample_count": 12,
                },
                "phase_metrics": [
                    {
                        "phase_index": 0,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                        "peak_cpu": 40.0,
                        "avg_cpu": 30.0,
                        "peak_memory": 75.0,
                        "avg_memory": 65.0,
                    },
                    {
                        "phase_index": 1,
                        "agent_names": ["agent2"],
                        "duration_seconds": 0.8,
                        "peak_cpu": 35.0,
                        "avg_cpu": 28.0,
                        "peak_memory": 70.0,
                        "avg_memory": 60.0,
                        "final_cpu_percent": 25.0,
                    },
                ],
            }
        }

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)

        # Test top-level structure
        assert "execution_summary" in json_result
        assert "resource_management" in json_result
        assert "agent_results" in json_result
        assert "usage_summary" in json_result

        # Test execution summary
        summary = json_result["execution_summary"]
        assert summary["total_agents"] == 2
        assert summary["successful_agents"] == 2
        assert summary["failed_agents"] == 0

        # Test resource management with 1-based phase numbering
        rm = json_result["resource_management"]
        assert rm["type"] == "user_configured"
        assert rm["concurrency_limit"] == 5

        # Execution metrics should now be computed from phase data
        # Expected: peak_cpu = max(40.0, 35.0) = 40.0
        # Expected: peak_memory = max(75.0, 70.0) = 75.0
        # Expected: avg_cpu = (30.0*1.5 + 28.0*0.8) / (1.5+0.8) = (45+22.4)/2.3 = 29.3
        # Expected: avg_memory = (65.0*1.5 + 60.0*0.8) / (1.5+0.8) = 63.3
        exec_metrics = rm["execution_metrics"]
        assert exec_metrics["peak_cpu"] == 40.0
        assert exec_metrics["peak_memory"] == 75.0
        assert abs(exec_metrics["avg_cpu"] - 29.3) < 0.1  # Allow float differences
        assert abs(exec_metrics["avg_memory"] - 63.3) < 0.1
        assert exec_metrics["has_phase_data"]
        assert len(rm["phases"]) == 2
        assert rm["phases"][0]["phase_number"] == 1  # 1-based for users
        assert rm["phases"][1]["phase_number"] == 2

        # Test agent results
        agents = json_result["agent_results"]
        assert len(agents) == 2
        assert agents[0]["name"] == "agent1"
        assert agents[0]["status"] == "success"
        assert agents[0]["content"] == "Result 1"

        # Test usage summary
        usage = json_result["usage_summary"]
        assert usage["total_tokens"] == 350
        assert usage["total_cost_usd"] == 0.003
        assert len(usage["per_agent"]) == 2

    def test_transform_handles_missing_data(self) -> None:
        """Test transformation handles missing or incomplete data gracefully."""
        # Minimal data
        raw_results = {"agent1": {"content": "Result", "status": "success"}}
        raw_usage = {"totals": {"total_tokens": 100}}
        raw_metadata: dict[
            str, dict[str, str | list[dict[str, str | int | float]]]
        ] = {}

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)

        # Should still have required structure with defaults
        assert json_result["execution_summary"]["total_agents"] == 1
        assert json_result["resource_management"]["type"] == "unknown"
        assert json_result["resource_management"]["concurrency_limit"] == 1
        assert json_result["usage_summary"]["total_tokens"] == 100


class TestJSONToPresentation:
    """Test rendering of structured JSON to text and markdown formats."""

    def test_render_json_as_text_consistent_formatting(self) -> None:
        """Test that JSON-to-text rendering produces consistent format."""
        structured_json = {
            "execution_summary": {"total_agents": 2, "successful_agents": 2},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                    }
                ],
            },
            "usage_summary": {"total_tokens": 350, "total_cost_usd": 0.003},
        }

        text_output = render_json_as_text(structured_json)

        # Test key formatting requirements
        assert "Phase 1:" in text_output  # 1-based numbering
        assert "Max concurrency limit used: 5" in text_output
        assert "Total tokens: 350" in text_output

    def test_render_json_as_markdown_consistent_formatting(self) -> None:
        """Test that JSON-to-markdown rendering produces consistent format."""
        structured_json = {
            "execution_summary": {"total_agents": 2, "successful_agents": 2},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.5,
                    }
                ],
            },
            "usage_summary": {"total_tokens": 350, "total_cost_usd": 0.003},
        }

        markdown_output = render_json_as_markdown(structured_json)

        # Test key formatting requirements
        assert "Phase 1:" in markdown_output  # 1-based numbering
        # Account for markdown bold
        assert "concurrency limit used:** 5" in markdown_output
        assert "tokens:** 350" in markdown_output

    def test_text_and_markdown_consistency(self) -> None:
        """Test text and markdown renderers produce consistent data representation."""
        structured_json = {
            "execution_summary": {"total_agents": 1},
            "resource_management": {"concurrency_limit": 3, "phases": []},
            "usage_summary": {"total_tokens": 100},
        }

        text_output = render_json_as_text(structured_json)
        markdown_output = render_json_as_markdown(structured_json)

        # Both should show same concurrency limit value (accounting for formatting)
        assert "concurrency limit used: 3" in text_output.lower()
        assert "concurrency limit used:** 3" in markdown_output.lower()

        # Both should show same token count
        assert "100" in text_output
        assert "100" in markdown_output


class TestRichMarkdownRendering:
    """Test comprehensive rich markdown rendering with full feature parity."""

    def test_render_complete_performance_summary(self) -> None:
        """Test that markdown renderer includes comprehensive performance data."""
        execution_json = {
            "execution_summary": {
                "total_agents": 3,
                "successful_agents": 3,
                "failed_agents": 0,
            },
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 5,
                "execution_metrics": {
                    "peak_cpu": 45.2,
                    "avg_cpu": 32.1,
                    "peak_memory": 78.5,
                    "avg_memory": 65.3,
                    "sample_count": 12,
                },
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1", "agent2"],
                        "duration_seconds": 1.5,
                        "peak_cpu": 40.0,
                        "avg_cpu": 30.0,
                        "peak_memory": 70.0,
                        "avg_memory": 60.0,
                        "sample_count": 5,
                    },
                    {
                        "phase_number": 2,
                        "agent_names": ["agent3"],
                        "duration_seconds": 0.8,
                        "final_cpu_percent": 25.0,
                        "final_memory_percent": 45.0,
                    },
                ],
            },
            "usage_summary": {
                "total_tokens": 450,
                "total_cost_usd": 0.005,
                "per_agent": [
                    {"name": "agent1", "tokens": 150, "cost_usd": 0.001},
                    {"name": "agent2", "tokens": 200, "cost_usd": 0.002},
                    {"name": "agent3", "tokens": 100, "cost_usd": 0.002},
                ],
            },
        }

        from llm_orc.cli_modules.utils.json_renderer import (
            render_comprehensive_markdown,
        )

        markdown_output = render_comprehensive_markdown(execution_json)

        # Test resource management section
        assert "### Resource Management" in markdown_output
        # Account for markdown bold formatting
        assert "concurrency limit used:** 5" in markdown_output
        assert "Peak usage:** CPU 45.2%" in markdown_output
        assert "Memory 78.5%" in markdown_output

        # Test per-phase metrics section
        assert "#### Per-Phase Performance" in markdown_output
        assert "**Phase 1**" in markdown_output
        assert "**Phase 2**" in markdown_output
        assert "agent1, agent2" in markdown_output  # Phase 1 agents
        assert "agent3" in markdown_output  # Phase 2 agents
        assert "1.5 seconds" in markdown_output  # Phase 1 duration
        assert "0.8 seconds" in markdown_output  # Phase 2 duration

        # Test per-agent usage section
        assert "### Per-Agent Usage" in markdown_output
        assert "**agent1**" in markdown_output
        assert "150 tokens" in markdown_output
        assert "$0.001" in markdown_output

    def test_render_handles_missing_phase_data_gracefully(self) -> None:
        """Test rendering handles missing or incomplete phase data."""
        execution_json = {
            "execution_summary": {"total_agents": 1},
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 3,
                "phases": [],  # No phase data
            },
            "usage_summary": {"total_tokens": 100, "per_agent": []},
        }

        from llm_orc.cli_modules.utils.json_renderer import (
            render_comprehensive_markdown,
        )

        markdown_output = render_comprehensive_markdown(execution_json)

        # Should still have main sections but gracefully handle missing data
        assert "### Resource Management" in markdown_output
        # Account for markdown bold formatting
        assert "concurrency limit used:** 3" in markdown_output
        # Should not crash or show empty phase sections
        assert "Phase 1" not in markdown_output


class TestModelDisplayLogic:
    """Test model display logic in agent results transformation."""
    
    def test_model_display_with_both_profile_and_model(self) -> None:
        """Test model display when both model_profile and model are present."""
        raw_results = {
            "agent1": {"content": "Result 1", "status": "success"}
        }
        raw_usage = {
            "agents": {
                "agent1": {
                    "model": "claude-3-sonnet-20240229", 
                    "model_profile": "efficient",
                    "total_tokens": 150
                }
            },
            "totals": {"total_tokens": 150, "agents_count": 1}
        }
        raw_metadata = {}

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)
        
        agent = json_result["agent_results"][0]
        assert agent["model_display"] == " (efficient → claude-3-sonnet-20240229)"
    
    def test_model_display_with_model_only(self) -> None:
        """Test model display when only model is present (no profile)."""
        raw_results = {
            "agent1": {"content": "Result 1", "status": "success"}
        }
        raw_usage = {
            "agents": {
                "agent1": {
                    "model": "claude-3-sonnet-20240229", 
                    "total_tokens": 150
                }
            },
            "totals": {"total_tokens": 150, "agents_count": 1}
        }
        raw_metadata = {}

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)
        
        agent = json_result["agent_results"][0]
        assert agent["model_display"] == " (claude-3-sonnet-20240229)"
    
    def test_model_display_with_neither(self) -> None:
        """Test model display when neither model nor profile are present."""
        raw_results = {
            "agent1": {"content": "Result 1", "status": "success"}
        }
        raw_usage = {
            "agents": {
                "agent1": {"total_tokens": 150}
            },
            "totals": {"total_tokens": 150, "agents_count": 1}
        }
        raw_metadata = {}

        json_result = transform_to_execution_json(raw_results, raw_usage, raw_metadata)
        
        agent = json_result["agent_results"][0]
        assert agent["model_display"] == ""


class TestTextRendering:
    """Test text rendering functions for uncovered lines."""
    
    def test_render_agent_results_success_with_content(self) -> None:
        """Test text rendering of successful agent with content."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_agent_results
        
        agent_results = [
            {
                "name": "agent1",
                "status": "success", 
                "content": "This is some result content",
                "model_display": " (claude-3)",
                "error": ""
            }
        ]
        
        lines = _render_text_agent_results(agent_results)
        
        assert "agent1 (claude-3):" in lines
        assert "This is some result content" in lines
    
    def test_render_agent_results_success_no_content(self) -> None:
        """Test text rendering of successful agent with no content.""" 
        from llm_orc.cli_modules.utils.json_renderer import _render_text_agent_results
        
        agent_results = [
            {
                "name": "agent1",
                "status": "success",
                "content": "",
                "model_display": "",
                "error": ""
            }
        ]
        
        lines = _render_text_agent_results(agent_results)
        
        assert "agent1:" in lines
        assert "*No content provided*" in lines
    
    def test_render_agent_results_failed_with_error(self) -> None:
        """Test text rendering of failed agent with error message."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_agent_results
        
        agent_results = [
            {
                "name": "agent1",
                "status": "failed",
                "content": "",
                "model_display": " (gpt-4)",
                "error": "Connection timeout"
            }
        ]
        
        lines = _render_text_agent_results(agent_results)
        
        assert "❌ agent1 (gpt-4):" in lines
        assert "Error: Connection timeout" in lines
    
    def test_render_agent_results_failed_no_error(self) -> None:
        """Test text rendering of failed agent without error message."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_agent_results
        
        agent_results = [
            {
                "name": "agent1", 
                "status": "failed",
                "content": "",
                "model_display": "",
                "error": ""
            }
        ]
        
        lines = _render_text_agent_results(agent_results)
        
        assert "❌ agent1:" in lines
        assert "Error: Unknown error" in lines


class TestResourceManagementRendering:
    """Test resource management text rendering functions."""
    
    def test_render_resource_management_empty(self) -> None:
        """Test resource management rendering with empty data."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_resource_management
        
        lines = _render_text_resource_management({})
        assert lines == []
    
    def test_render_resource_management_basic(self) -> None:
        """Test resource management rendering with basic data."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_resource_management
        
        rm_data = {
            "type": "user_configured",
            "concurrency_limit": 5
        }
        
        lines = _render_text_resource_management(rm_data)
        
        assert "Resource Management" in lines
        assert "Type: user_configured (fixed concurrency limits)" in lines
        assert "Max concurrency limit used: 5" in lines
    
    def test_render_resource_management_with_metrics_no_phase_data(self) -> None:
        """Test resource management rendering with execution metrics (no phase data)."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_resource_management
        
        rm_data = {
            "type": "adaptive", 
            "concurrency_limit": 3,
            "execution_metrics": {
                "peak_cpu": 45.2,
                "avg_cpu": 32.1,
                "peak_memory": 78.5,
                "avg_memory": 65.3,
                "sample_count": 12,
                "has_phase_data": False
            }
        }
        
        lines = _render_text_resource_management(rm_data)
        
        assert "Peak usage: CPU 45.2%, Memory 78.5%" in lines
        assert "Average usage: CPU 32.1%, Memory 65.3%" in lines
        assert "Monitoring: 12 samples collected" in lines
    
    def test_render_resource_management_with_phase_data(self) -> None:
        """Test resource management rendering with phase-computed metrics."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_resource_management
        
        rm_data = {
            "type": "adaptive",
            "concurrency_limit": 3,
            "execution_metrics": {
                "peak_cpu": 45.2,
                "avg_cpu": 32.1, 
                "peak_memory": 78.5,
                "avg_memory": 65.3,
                "has_phase_data": True
            }
        }
        
        lines = _render_text_resource_management(rm_data)
        
        assert "Peak usage: CPU 45.2%, Memory 78.5%" in lines
        assert "Average usage: CPU 32.1%, Memory 65.3%" in lines
        assert "Computed from per-phase performance data" in lines
        # Should not show sample count for phase data
        assert "samples collected" not in " ".join(lines)


class TestPhaseRendering:
    """Test phase rendering functions."""
    
    def test_render_phases_empty(self) -> None:
        """Test phase rendering with empty list."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_phases
        
        lines = _render_text_phases([])
        assert lines == []
    
    def test_render_phases_with_peak_avg_metrics(self) -> None:
        """Test phase rendering with peak/avg metrics and sample count."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_phases
        
        phases = [
            {
                "phase_number": 1,
                "agent_names": ["agent1", "agent2"],
                "duration_seconds": 1.5,
                "peak_cpu": 45.0,
                "avg_cpu": 32.0,
                "peak_memory": 75.0,
                "avg_memory": 65.0,
                "sample_count": 8
            }
        ]
        
        lines = _render_text_phases(phases)
        
        lines_text = " ".join(lines)
        assert "Per-Phase Performance" in lines_text
        assert "Phase 1 (2 agents)" in lines_text
        assert "Agents: agent1, agent2" in lines_text
        assert "Duration: 1.5 seconds" in lines_text
        assert "Resource usage: CPU 32.0% (peak 45.0%), Memory 65.0% (peak 75.0%)" in lines_text
        assert "Monitoring: 8 samples collected" in lines_text
    
    def test_render_phases_with_final_metrics(self) -> None:
        """Test phase rendering with final metrics (no peak/avg)."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_phases
        
        phases = [
            {
                "phase_number": 2,
                "agent_names": ["agent3"],
                "duration_seconds": 0.8,
                "final_cpu_percent": 25.0,
                "final_memory_percent": 45.0
            }
        ]
        
        lines = _render_text_phases(phases)
        
        lines_text = " ".join(lines)
        assert "Phase 2 (1 agents)" in lines_text
        assert "Agents: agent3" in lines_text
        assert "Duration: 0.8 seconds" in lines_text
        assert "Resource usage: CPU 25.0%, Memory 45.0%" in lines_text
        # Should not show monitoring info for final metrics
        assert "samples collected" not in lines_text
    
    def test_render_phases_no_agent_names(self) -> None:
        """Test phase rendering without agent names."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_phases
        
        phases = [
            {
                "phase_number": 1,
                "agent_names": [],
                "duration_seconds": 1.0
            }
        ]
        
        lines = _render_text_phases(phases)
        
        lines_text = " ".join(lines)
        assert "Phase 1 (0 agents)" in lines_text
        assert "Duration: 1.0 seconds" in lines_text
        # Should not have Agents line when no agent names
        assert "Agents:" not in lines_text


class TestUsageRendering:
    """Test usage summary rendering functions."""
    
    def test_render_usage_summary_empty(self) -> None:
        """Test usage summary rendering with no per-agent data."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_usage_summary
        
        usage = {"per_agent": []}
        lines = _render_text_usage_summary(usage)
        assert lines == []
    
    def test_render_usage_summary_basic(self) -> None:
        """Test usage summary rendering with basic agent data."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_usage_summary
        
        usage = {
            "per_agent": [
                {
                    "name": "agent1",
                    "tokens": 150,
                    "cost_usd": 0.001,
                    "duration_ms": 1500,
                    "model_display": "claude-3-sonnet"
                }
            ]
        }
        
        lines = _render_text_usage_summary(usage)
        lines_text = " ".join(lines)
        
        assert "Per-Agent Usage" in lines_text
        assert "agent1 (claude-3-sonnet): 150 tokens, $0.0010, 1500ms" in lines_text
    
    def test_render_usage_summary_with_resource_metrics(self) -> None:
        """Test usage summary rendering with resource metrics."""
        from llm_orc.cli_modules.utils.json_renderer import _render_text_usage_summary
        
        usage = {
            "per_agent": [
                {
                    "name": "agent1",
                    "tokens": 200,
                    "cost_usd": 0.002,
                    "duration_ms": 2000,
                    "model_display": "gpt-4",
                    "peak_cpu": 45.5,
                    "avg_cpu": 32.1,
                    "peak_memory": 78.2,
                    "avg_memory": 65.7
                }
            ]
        }
        
        lines = _render_text_usage_summary(usage)
        lines_text = " ".join(lines)
        
        expected = "agent1 (gpt-4): 200 tokens, $0.0020, 2000ms, CPU 32.1% (peak 45.5%), Memory 65.7% (peak 78.2%)"
        assert expected in lines_text


class TestComprehensiveRendering:
    """Test comprehensive text rendering functions."""
    
    def test_render_comprehensive_text_integration(self) -> None:
        """Test comprehensive text rendering integrates all components."""
        from llm_orc.cli_modules.utils.json_renderer import render_comprehensive_text
        
        structured_json = {
            "execution_summary": {
                "total_agents": 2,
                "successful_agents": 2,
                "failed_agents": 0
            },
            "resource_management": {
                "type": "user_configured",
                "concurrency_limit": 3,
                "execution_metrics": {
                    "peak_cpu": 40.0,
                    "avg_cpu": 30.0,
                    "peak_memory": 70.0,
                    "avg_memory": 60.0,
                    "has_phase_data": False,
                    "sample_count": 5
                },
                "phases": [
                    {
                        "phase_number": 1,
                        "agent_names": ["agent1"],
                        "duration_seconds": 1.2
                    }
                ]
            },
            "usage_summary": {
                "total_tokens": 300,
                "total_cost_usd": 0.003,
                "per_agent": [
                    {
                        "name": "agent1",
                        "tokens": 150,
                        "cost_usd": 0.0015,
                        "duration_ms": 1200,
                        "model_display": "claude-3"
                    }
                ]
            },
            "agent_results": [
                {
                    "name": "agent1",
                    "status": "success",
                    "content": "Test result",
                    "model_display": " (claude-3)"
                }
            ]
        }
        
        result = render_comprehensive_text(structured_json)
        
        # Should include all major sections
        assert "Resource Management" in result
        assert "Per-Phase Performance" in result  
        assert "Per-Agent Usage" in result
        assert "agent1 (claude-3): 150 tokens" in result
