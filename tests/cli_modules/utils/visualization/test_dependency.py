"""Comprehensive tests for dependency visualization module."""

import pytest
from rich.tree import Tree

from llm_orc.cli_modules.utils.visualization.dependency import (
    create_dependency_graph,
    create_dependency_tree,
    create_dependency_graph_with_status,
    find_final_agent,
    _group_agents_by_dependency_level,
    _calculate_agent_level,
    _create_plain_text_dependency_graph,
    _create_structured_dependency_info,
    _create_agent_statuses,
    _build_dependency_levels,
)


class TestCreateDependencyGraph:
    """Test dependency graph creation functions."""

    def test_create_dependency_graph_simple(self):
        """Test creating dependency graph with simple agents."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        result = create_dependency_graph(agents)
        
        assert result == "agent_a → agent_b"

    def test_create_dependency_graph_complex(self):
        """Test creating dependency graph with complex dependencies."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]
        
        result = create_dependency_graph(agents)
        
        assert "agent_a" in result
        assert "agent_b" in result  
        assert "agent_c" in result
        assert "→" in result

    def test_create_dependency_graph_empty(self):
        """Test creating dependency graph with empty agents list."""
        result = create_dependency_graph([])
        
        assert result == "No agents to display"

    def test_create_dependency_graph_with_status_simple(self):
        """Test creating dependency graph with status indicators."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        statuses = {"agent_a": "completed", "agent_b": "running"}
        
        result = create_dependency_graph_with_status(agents, statuses)
        
        assert "agent_a" in result
        assert "agent_b" in result
        assert "→" in result

    def test_create_dependency_graph_with_status_failed(self):
        """Test dependency graph with failed status."""
        agents = [
            {"name": "agent_a", "depends_on": []},
        ]
        statuses = {"agent_a": "failed"}
        
        result = create_dependency_graph_with_status(agents, statuses)
        
        assert "agent_a" in result

    def test_create_dependency_graph_with_status_empty(self):
        """Test dependency graph with empty agents."""
        result = create_dependency_graph_with_status([], {})
        
        assert result == "No agents to display"

    def test_create_dependency_graph_no_levels(self):
        """Test dependency graph when no levels found."""
        # This is an edge case where the grouping function returns empty
        agents = []
        
        result = create_dependency_graph_with_status(agents, {})
        
        assert result == "No agents to display"


class TestCreateDependencyTree:
    """Test dependency tree creation functions."""

    def test_create_dependency_tree_simple(self):
        """Test creating dependency tree with simple agents."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        result = create_dependency_tree(agents)
        
        assert isinstance(result, Tree)
        # Check that the tree has the expected structure
        assert result.label is not None
        assert len(result.children) > 0  # Should have level nodes

    def test_create_dependency_tree_with_status(self):
        """Test creating dependency tree with agent status."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        statuses = {"agent_a": "completed", "agent_b": "running"}
        
        result = create_dependency_tree(agents, statuses)
        
        assert isinstance(result, Tree)
        # Check that the tree has the expected structure
        assert result.label is not None
        assert len(result.children) > 0  # Should have level nodes

    def test_create_dependency_tree_empty_agents(self):
        """Test creating dependency tree with empty agents list."""
        result = create_dependency_tree([])
        
        assert isinstance(result, Tree)

    def test_create_dependency_tree_multiple_levels(self):
        """Test dependency tree with multiple dependency levels."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
            {"name": "agent_d", "depends_on": ["agent_c"]},
        ]
        
        result = create_dependency_tree(agents)
        
        assert isinstance(result, Tree)
        # Check that the tree has multiple levels (should have multiple child nodes)
        assert result.label is not None
        assert len(result.children) >= 3  # Should have at least 3 levels


class TestFindFinalAgent:
    """Test finding the final agent function."""

    def test_find_final_agent_coordinator(self):
        """Test finding final agent when coordinator exists."""
        results = {
            "agent_a": {"status": "success"},
            "coordinator": {"status": "success"},
            "agent_b": {"status": "success"},
        }
        
        result = find_final_agent(results)
        
        assert result == "coordinator"

    def test_find_final_agent_synthesizer(self):
        """Test finding final agent when synthesizer exists (no coordinator)."""
        results = {
            "agent_a": {"status": "success"},
            "synthesizer": {"status": "success"},
            "agent_b": {"status": "success"},
        }
        
        result = find_final_agent(results)
        
        assert result == "synthesizer"

    def test_find_final_agent_last_successful(self):
        """Test finding final agent as last successful when no special names."""
        results = {
            "agent_a": {"status": "success"},
            "agent_b": {"status": "failed"},
            "agent_c": {"status": "success"},
        }
        
        result = find_final_agent(results)
        
        assert result == "agent_c"

    def test_find_final_agent_no_successful(self):
        """Test finding final agent when no successful agents."""
        results = {
            "agent_a": {"status": "failed"},
            "agent_b": {"status": "failed"},
        }
        
        result = find_final_agent(results)
        
        assert result is None

    def test_find_final_agent_empty_results(self):
        """Test finding final agent with empty results."""
        result = find_final_agent({})
        
        assert result is None


class TestGroupAgentsByDependencyLevel:
    """Test grouping agents by dependency level."""

    def test_group_agents_simple(self):
        """Test grouping agents with simple dependencies."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        result = _group_agents_by_dependency_level(agents)
        
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0]["name"] == "agent_a"
        assert result[1][0]["name"] == "agent_b"

    def test_group_agents_complex(self):
        """Test grouping agents with complex dependencies."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
            {"name": "agent_d", "depends_on": ["agent_c"]},
        ]
        
        result = _group_agents_by_dependency_level(agents)
        
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result[0]) == 2  # agent_a and agent_b
        assert len(result[1]) == 1  # agent_c
        assert len(result[2]) == 1  # agent_d

    def test_group_agents_empty(self):
        """Test grouping with empty agents list."""
        result = _group_agents_by_dependency_level([])
        
        assert result == {}


class TestCalculateAgentLevel:
    """Test calculating agent dependency level."""

    def test_calculate_agent_level_no_dependencies(self):
        """Test calculating level for agent with no dependencies."""
        agent = {"name": "agent_a", "depends_on": []}
        all_agents = [agent]
        
        result = _calculate_agent_level(agent, all_agents)
        
        assert result == 0

    def test_calculate_agent_level_single_dependency(self):
        """Test calculating level for agent with single dependency."""
        agent_a = {"name": "agent_a", "depends_on": []}
        agent_b = {"name": "agent_b", "depends_on": ["agent_a"]}
        all_agents = [agent_a, agent_b]
        
        result = _calculate_agent_level(agent_b, all_agents)
        
        assert result == 1

    def test_calculate_agent_level_nested_dependencies(self):
        """Test calculating level for agent with nested dependencies."""
        agent_a = {"name": "agent_a", "depends_on": []}
        agent_b = {"name": "agent_b", "depends_on": ["agent_a"]}
        agent_c = {"name": "agent_c", "depends_on": ["agent_b"]}
        all_agents = [agent_a, agent_b, agent_c]
        
        result = _calculate_agent_level(agent_c, all_agents)
        
        assert result == 2

    def test_calculate_agent_level_missing_dependency(self):
        """Test calculating level when dependency doesn't exist."""
        agent = {"name": "agent_b", "depends_on": ["nonexistent"]}
        all_agents = [agent]
        
        result = _calculate_agent_level(agent, all_agents)
        
        assert result == 1  # Should still increment from 0


class TestCreatePlainTextDependencyGraph:
    """Test creating plain text dependency graphs."""

    def test_create_plain_text_dependency_graph_simple(self):
        """Test creating plain text graph with simple agents."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        result = _create_plain_text_dependency_graph(agents)
        
        assert len(result) >= 2
        assert "agent_a" in str(result)
        assert "agent_b" in str(result)

    def test_create_plain_text_dependency_graph_complex(self):
        """Test creating plain text graph with complex dependencies."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
            {"name": "agent_c", "depends_on": ["agent_a", "agent_b"]},
        ]
        
        result = _create_plain_text_dependency_graph(agents)
        
        assert len(result) >= 2
        graph_str = " ".join(result)
        assert "agent_a" in graph_str
        assert "agent_b" in graph_str
        assert "agent_c" in graph_str

    def test_create_plain_text_dependency_graph_empty(self):
        """Test creating plain text graph with empty agents."""
        result = _create_plain_text_dependency_graph([])
        
        assert result == ["No agents found"]


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_structured_dependency_info(self):
        """Test creating structured dependency information."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        agents_by_level, agent_statuses = _create_structured_dependency_info(agents)
        
        assert isinstance(agents_by_level, dict)
        assert isinstance(agent_statuses, dict)
        assert 0 in agents_by_level
        assert 1 in agents_by_level
        assert "agent_a" in agent_statuses
        assert "agent_b" in agent_statuses

    def test_create_agent_statuses(self):
        """Test creating initial agent status mapping."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": []},
        ]
        
        result = _create_agent_statuses(agents)
        
        assert result == {"agent_a": "pending", "agent_b": "pending"}

    def test_build_dependency_levels(self):
        """Test building dependency levels."""
        agents = [
            {"name": "agent_a", "depends_on": []},
            {"name": "agent_b", "depends_on": ["agent_a"]},
        ]
        
        result = _build_dependency_levels(agents)
        
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 1
        assert len(result[1]) == 1