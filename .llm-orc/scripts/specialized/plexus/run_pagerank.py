#!/usr/bin/env python3
"""
run_pagerank.py — PageRank on a Plexus graph export

Input (stdin): graph-export JSON (docs/schemas/graph-export.schema.json)
Output (stdout): analysis-result JSON (docs/schemas/analysis-result.schema.json)

Parameters (via ensemble config):
  damping: float (default 0.85)
  max_iterations: int (default 100)
  tolerance: float (default 1e-6)

Usage in ensemble:
  script: scripts/specialized/plexus/run_pagerank.py
  parameters:
    damping: 0.85
"""
import json
import os
import sys

from llm_orc.script_utils import unwrap_input


def pagerank(nodes, edges, damping=0.85, max_iterations=100, tolerance=1e-6):
    """Compute PageRank scores for graph nodes.

    Simple iterative implementation. For production use with large graphs,
    replace with NetworkX: ``nx.pagerank(G, alpha=damping)``.
    """
    node_ids = [n["id"] for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}

    # Build adjacency: target -> list of sources (for incoming edges)
    incoming = {nid: [] for nid in node_ids}
    outgoing_count = {nid: 0 for nid in node_ids}

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in incoming and tgt in incoming:
            incoming[tgt].append(src)
            outgoing_count[src] += 1

    # Initialize scores uniformly
    scores = {nid: 1.0 / n for nid in node_ids}

    for _ in range(max_iterations):
        new_scores = {}
        for nid in node_ids:
            rank_sum = sum(
                scores[src] / outgoing_count[src]
                for src in incoming[nid]
                if outgoing_count[src] > 0
            )
            new_scores[nid] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_scores[nid] - scores[nid]) for nid in node_ids)
        scores = new_scores
        if diff < tolerance:
            break

    return scores


def main():
    raw = sys.stdin.read()
    data, params = unwrap_input(raw, debug=os.environ.get("LLM_ORC_DEBUG"))

    # Parameters: prefer ensemble config (params), fall back to input data
    damping = params.get("damping", data.get("damping", 0.85))
    max_iterations = params.get("max_iterations", data.get("max_iterations", 100))
    tolerance = params.get("tolerance", data.get("tolerance", 1e-6))

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    scores = pagerank(nodes, edges, damping, max_iterations, tolerance)

    updates = [
        {"node_id": nid, "properties": {"pagerank_score": round(score, 6)}}
        for nid, score in scores.items()
    ]

    print(json.dumps({"updates": updates}, indent=2))


if __name__ == "__main__":
    main()
