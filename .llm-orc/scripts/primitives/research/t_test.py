#!/usr/bin/env python3
"""
t_test.py - Perform statistical t-test analysis

Usage: 
  As script agent in ensemble:
    script: primitives/research/t_test.py
    parameters:
      group1: [1.2, 1.5, 1.8, 2.1]
      group2: [2.3, 2.6, 2.9, 3.2]
      alpha: 0.05
      
  From command line:
    echo '{"group1": [1, 2, 3], "group2": [4, 5, 6]}' | python t_test.py
"""
import json
import math
import sys


def calculate_mean(values):
    """Calculate mean of values."""
    return sum(values) / len(values)


def calculate_variance(values, mean):
    """Calculate sample variance."""
    if len(values) <= 1:
        return 0
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def welch_t_test(group1, group2):
    """Perform Welch's t-test for unequal variances."""
    n1, n2 = len(group1), len(group2)
    
    if n1 == 0 or n2 == 0:
        raise ValueError("Groups cannot be empty")
    
    mean1, mean2 = calculate_mean(group1), calculate_mean(group2)
    var1, var2 = calculate_variance(group1, mean1), calculate_variance(group2, mean2)
    
    # Welch's t-statistic
    pooled_se = math.sqrt(var1/n1 + var2/n2)
    if pooled_se == 0:
        raise ValueError("Cannot compute t-test: pooled standard error is zero")
    
    t_stat = (mean1 - mean2) / pooled_se
    
    # Welch's degrees of freedom approximation
    if var1/n1 + var2/n2 == 0:
        df = n1 + n2 - 2
    else:
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    return t_stat, df, mean1, mean2, var1, var2


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters
    group1 = config.get('group1', [])
    group2 = config.get('group2', [])
    alpha = config.get('alpha', 0.05)
    
    try:
        if not group1 or not group2:
            raise ValueError("Both group1 and group2 must contain data")
        
        t_stat, df, mean1, mean2, var1, var2 = welch_t_test(group1, group2)
        
        # Simple p-value approximation (for demonstration)
        # In practice, you'd use a proper t-distribution CDF
        abs_t = abs(t_stat)
        if abs_t > 2.58:
            p_value_approx = "< 0.01"
            significant = alpha >= 0.01
        elif abs_t > 1.96:
            p_value_approx = "< 0.05"
            significant = alpha >= 0.05
        else:
            p_value_approx = "> 0.05"
            significant = False
        
        effect_size = (mean1 - mean2) / math.sqrt((var1 + var2) / 2)
        
        result = {
            "success": True,
            "t_statistic": t_stat,
            "degrees_of_freedom": df,
            "p_value_approx": p_value_approx,
            "significant": significant,
            "alpha": alpha,
            "group1_stats": {
                "mean": mean1,
                "variance": var1,
                "n": len(group1)
            },
            "group2_stats": {
                "mean": mean2,
                "variance": var2,
                "n": len(group2)
            },
            "effect_size_cohens_d": effect_size
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "t_statistic": None,
            "significant": False
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()