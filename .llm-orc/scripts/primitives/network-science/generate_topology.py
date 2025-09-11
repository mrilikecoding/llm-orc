#!/usr/bin/env python3
"""
generate_topology.py - Generate various network topologies for research

Usage: 
  As script agent in ensemble:
    script: primitives/network-science/generate_topology.py
    parameters:
      topology_type: "small_world"
      nodes: 100
      k: 6
      p: 0.1
      
  From command line:
    echo '{"topology_type": "random", "nodes": 50}' | python generate_topology.py
"""
import json
import random
import sys


def generate_small_world(nodes, k, p, seed=None):
    """Generate Watts-Strogatz small-world network."""
    if seed is not None:
        random.seed(seed)
    
    # Start with regular ring lattice
    edges = []
    for i in range(nodes):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % nodes
            edges.append([i, neighbor])
    
    # Rewire edges with probability p
    rewired_edges = []
    for edge in edges:
        if random.random() < p:
            # Rewire to random node
            new_target = random.randint(0, nodes - 1)
            while new_target == edge[0] or [edge[0], new_target] in rewired_edges:
                new_target = random.randint(0, nodes - 1)
            rewired_edges.append([edge[0], new_target])
        else:
            rewired_edges.append(edge)
    
    return rewired_edges


def generate_random(nodes, edge_prob, seed=None):
    """Generate Erdős–Rényi random network."""
    if seed is not None:
        random.seed(seed)
    
    edges = []
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_prob:
                edges.append([i, j])
    
    return edges


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters with defaults
    topology_type = config.get('topology_type', 'small_world')
    nodes = config.get('nodes', 50)
    seed = config.get('seed', None)
    
    try:
        if topology_type == "small_world":
            k = config.get('k', 6)
            p = config.get('p', 0.1)
            edges = generate_small_world(nodes, k, p, seed)
            topology_params = {"k": k, "p": p}
            
        elif topology_type == "random":
            edge_prob = config.get('edge_prob', 0.1)
            edges = generate_random(nodes, edge_prob, seed)
            topology_params = {"edge_prob": edge_prob}
            
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        result = {
            "success": True,
            "topology_type": topology_type,
            "nodes": nodes,
            "edges": edges,
            "edge_count": len(edges),
            "parameters": topology_params,
            "seed": seed
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "topology_type": topology_type,
            "nodes": nodes,
            "edges": []
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()