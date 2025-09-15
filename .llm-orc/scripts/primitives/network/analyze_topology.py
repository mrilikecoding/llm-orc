#!/usr/bin/env python3
"""Network topology analysis script for TDD testing."""

import json
import sys


def analyze_network_topology(topology_data: dict, analysis_type: str = "centrality") -> dict:
    """Analyze network topology data and return centrality scores."""
    nodes = topology_data.get("nodes", [])
    edges = topology_data.get("edges", [])

    # Simple centrality calculation for TDD testing
    # Count connections per node
    node_connections = {node: 0 for node in nodes}

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in node_connections:
            node_connections[source] += edge.get("weight", 1.0)
        if target in node_connections:
            node_connections[target] += edge.get("weight", 1.0)

    # Convert to centrality scores (normalized)
    max_connections = max(node_connections.values()) if node_connections else 1
    centrality_scores = {
        node: connections / max_connections
        for node, connections in node_connections.items()
    }

    # Sort nodes by centrality
    node_rankings = sorted(
        nodes,
        key=lambda n: centrality_scores.get(n, 0),
        reverse=True
    )

    return {
        "centrality_scores": centrality_scores,
        "node_rankings": node_rankings,
        "analysis_metadata": {
            "algorithm": "weighted_degree_centrality",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "analysis_type": analysis_type,
        },
    }


def main() -> None:
    """Main entry point for script execution."""
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract topology data from EnhancedScriptAgent input format
        # EnhancedScriptAgent wraps data in {"input": data, "parameters": {}, "context": {}}
        actual_input = input_data.get("input", input_data)  # Fall back for direct testing
        topology_data = actual_input.get("topology_data", {})
        analysis_type = actual_input.get("analysis_type", "centrality")

        # Perform analysis
        analysis_results = analyze_network_topology(topology_data, analysis_type)

        # Return structured output
        output = {
            "success": True,
            "data": {
                "analysis_results": analysis_results
            },
            "error": None,
        }

        print(json.dumps(output))

    except Exception as e:
        # Return error output
        error_output = {
            "success": False,
            "data": None,
            "error": f"Network analysis failed: {str(e)}",
        }
        print(json.dumps(error_output))
        sys.exit(1)


if __name__ == "__main__":
    main()