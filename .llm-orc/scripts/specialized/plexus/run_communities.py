#!/usr/bin/env python3
"""
run_communities.py — Community detection on a Plexus graph export

Input (stdin): graph-export JSON (docs/schemas/graph-export.schema.json)
Output (stdout): analysis-result JSON (docs/schemas/analysis-result.schema.json)

Parameters (via ensemble config):
  resolution: float (default 1.0) — higher values find smaller communities

Uses a simple label propagation algorithm. For production use with large graphs,
replace with NetworkX Louvain: ``nx.community.louvain_communities(G)``.

Usage in ensemble:
  script: scripts/specialized/plexus/run_communities.py
  parameters:
    resolution: 1.0
"""
import json
import random
import sys


def label_propagation(nodes, edges, max_iterations=50, seed=42):
    """Detect communities via label propagation.

    Each node starts with its own label. In each iteration, nodes adopt the
    most frequent label among their neighbors. Converges when no labels change.
    """
    random.seed(seed)
    node_ids = [n["id"] for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}

    # Build undirected adjacency
    neighbors = {nid: [] for nid in node_ids}
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in neighbors and tgt in neighbors:
            neighbors[src].append(tgt)
            neighbors[tgt].append(src)

    # Initialize: each node is its own community
    labels = {nid: i for i, nid in enumerate(node_ids)}

    for _ in range(max_iterations):
        changed = False
        order = list(node_ids)
        random.shuffle(order)

        for nid in order:
            nbrs = neighbors[nid]
            if not nbrs:
                continue

            # Count neighbor labels
            label_counts = {}
            for nbr in nbrs:
                lbl = labels[nbr]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            # Pick most frequent (break ties randomly)
            max_count = max(label_counts.values())
            candidates = [l for l, c in label_counts.items() if c == max_count]
            new_label = random.choice(candidates)

            if labels[nid] != new_label:
                labels[nid] = new_label
                changed = True

        if not changed:
            break

    # Renumber communities consecutively from 0
    unique_labels = sorted(set(labels.values()))
    remap = {old: new for new, old in enumerate(unique_labels)}
    return {nid: remap[lbl] for nid, lbl in labels.items()}


def unwrap_input(raw_json):
    """Unwrap llm-orc envelope to get the actual graph data and parameters.

    Handles three input formats:
    1. ScriptAgentInput: {"agent_name": "...", "input_data": "<json>", ...}
    2. Legacy wrapper:   {"input": "<json or dict>", "parameters": {...}, ...}
    3. Direct:           {"nodes": [...], "edges": [...], ...}

    Returns (data_dict, parameters_dict).
    """
    envelope = json.loads(raw_json) if raw_json.strip() else {}

    # Format 1: ScriptAgentInput envelope
    input_data = envelope.get("input_data", "")
    if isinstance(input_data, str) and input_data.strip():
        try:
            return json.loads(input_data), envelope.get("parameters", {}) or {}
        except json.JSONDecodeError:
            pass

    # Format 2: Legacy wrapper {"input": ..., "parameters": ...}
    if "input" in envelope and "parameters" in envelope:
        inner = envelope["input"]
        params = envelope.get("parameters", {}) or {}
        if isinstance(inner, str) and inner.strip():
            try:
                return json.loads(inner), params
            except json.JSONDecodeError:
                return envelope, params
        if isinstance(inner, dict):
            return inner, params

    # Format 3: Direct invocation — envelope IS the data
    return envelope, {}


def main():
    raw = sys.stdin.read()
    data, _params = unwrap_input(raw)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    communities = label_propagation(nodes, edges)

    updates = [
        {"node_id": nid, "properties": {"community": community_id}}
        for nid, community_id in communities.items()
    ]

    print(json.dumps({"updates": updates}, indent=2))


if __name__ == "__main__":
    main()
