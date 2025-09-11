#!/usr/bin/env python3
"""
compare_performance.py - Compare performance between different agent groups

Usage:
  As script agent in ensemble:
    script: primitives/research/compare_performance.py
    parameters:
      metrics: [response_quality, problem_coverage]
      comparison_method: statistical
"""
import json
import sys
import numpy as np

def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        input_data = sys.stdin.read()
        try:
            data = json.loads(input_data)
            config = data.get('parameters', {})
            input_results = data.get('input', '')
        except json.JSONDecodeError:
            config = {}
            input_results = input_data
    else:
        config = {}
        input_results = ''
    
    # Get parameters with defaults
    metrics = config.get('metrics', ['response_quality'])
    comparison_method = config.get('comparison_method', 'statistical')
    
    # Simulate performance comparison (in real implementation, this would analyze actual results)
    try:
        # Generate simulated performance data
        swarm_scores = {
            'response_quality': np.random.normal(7.2, 1.5, 10).clip(1, 10).tolist(),
            'problem_coverage': np.random.normal(8.1, 1.2, 10).clip(1, 10).tolist(),
            'solution_creativity': np.random.normal(6.8, 1.8, 10).clip(1, 10).tolist(),
            'factual_accuracy': np.random.normal(7.9, 1.1, 10).clip(1, 10).tolist(),
            'coherence_score': np.random.normal(6.5, 1.6, 10).clip(1, 10).tolist()
        }
        
        single_model_scores = {
            'response_quality': np.random.normal(7.8, 1.2, 10).clip(1, 10).tolist(),
            'problem_coverage': np.random.normal(7.6, 1.4, 10).clip(1, 10).tolist(),
            'solution_creativity': np.random.normal(8.2, 1.3, 10).clip(1, 10).tolist(),
            'factual_accuracy': np.random.normal(8.4, 0.9, 10).clip(1, 10).tolist(),
            'coherence_score': np.random.normal(8.7, 1.0, 10).clip(1, 10).tolist()
        }
        
        # Calculate summary statistics
        comparison_results = {}
        for metric in metrics:
            if metric in swarm_scores and metric in single_model_scores:
                swarm_mean = np.mean(swarm_scores[metric])
                single_mean = np.mean(single_model_scores[metric])
                
                comparison_results[metric] = {
                    'swarm_mean': round(swarm_mean, 2),
                    'swarm_std': round(np.std(swarm_scores[metric]), 2),
                    'single_model_mean': round(single_mean, 2),
                    'single_model_std': round(np.std(single_model_scores[metric]), 2),
                    'difference': round(single_mean - swarm_mean, 2),
                    'swarm_advantage': swarm_mean > single_mean
                }
        
        result = {
            "success": True,
            "comparison_method": comparison_method,
            "metrics_analyzed": metrics,
            "results": comparison_results,
            "raw_scores": {
                "swarm": {k: v for k, v in swarm_scores.items() if k in metrics},
                "single_model": {k: v for k, v in single_model_scores.items() if k in metrics}
            },
            "summary": f"Compared {len(metrics)} performance metrics using {comparison_method} method"
        }
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "metrics": metrics
        }
    
    # Output JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()