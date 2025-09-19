#!/usr/bin/env python3
"""JSON merge primitive for llm-orc (ADR-001).

Primitive: JSON merge utility
Input: JSON data to merge
Output: Merged JSON result
Depends: json, os
"""

import json
import os
from typing import Any, Dict


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def main() -> None:
    """Merge JSON data from dependencies and input."""
    try:
        # Get input data from environment
        input_data_str = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_data_str)

        # Parse source data
        source_data_str = input_data.get("input_data", "{}")
        if source_data_str.startswith("{") or source_data_str.startswith("["):
            source_data = json.loads(source_data_str)
        else:
            source_data = {"input": source_data_str}

        # Merge with dependency data
        dependencies = input_data.get("dependencies", {})
        merged_data = source_data.copy() if isinstance(source_data, dict) else {"source": source_data}

        # Merge each dependency into the result
        for key, value in dependencies.items():
            if isinstance(value, dict):
                merged_data = deep_merge(merged_data, value)
            else:
                merged_data[key] = value

        # Create successful output
        result = {
            "success": True,
            "data": {
                "merged": merged_data,
                "dependency_count": len(dependencies),
                "merge_type": "deep"
            },
            "error": None,
            "agent_requests": []
        }

    except json.JSONDecodeError as e:
        result = {
            "success": False,
            "data": None,
            "error": f"JSON parsing failed: {e}",
            "agent_requests": []
        }
    except Exception as e:
        result = {
            "success": False,
            "data": None,
            "error": f"Merge failed: {e}",
            "agent_requests": []
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()