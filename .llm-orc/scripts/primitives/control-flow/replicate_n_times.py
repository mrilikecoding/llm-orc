#!/usr/bin/env python3
"""
replicate_n_times.py - Generate replication parameters for statistical studies

Usage: 
  As script agent in ensemble:
    script: primitives/control-flow/replicate_n_times.py
    parameters:
      replications: 50
      seed: 12345
      
  From command line:
    echo '{"replications": 10}' | python replicate_n_times.py
"""
import json
import random
import sys


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters with defaults
    replications = config.get('replications', 1)
    seed = config.get('seed', None)
    
    try:
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
        
        # Generate replication parameters
        replication_configs = []
        for i in range(replications):
            replication_config = {
                "replication_id": i + 1,
                "total_replications": replications,
                "random_seed": random.randint(1, 1000000) if seed is None else seed + i
            }
            replication_configs.append(replication_config)
        
        result = {
            "success": True,
            "replications": replication_configs,
            "total_replications": replications,
            "base_seed": seed
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "replications": []
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()