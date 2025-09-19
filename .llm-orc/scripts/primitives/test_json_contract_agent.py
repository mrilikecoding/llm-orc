#!/usr/bin/env python3
"""Test script for JSON contract validation between ScriptResolver and EnsembleExecutor."""

import json
import os
import sys


def main():
    """Main script logic for JSON contract testing."""
    # Get input data from environment (passed by EnsembleExecutor)
    input_data_str = os.environ.get("INPUT_DATA", "{}")

    try:
        # Parse input data as JSON
        if input_data_str.startswith("{") or input_data_str.startswith("["):
            input_data = json.loads(input_data_str)
        else:
            # Treat as plain string if not JSON
            input_data = {"input_data": input_data_str}
    except json.JSONDecodeError:
        # Fallback for non-JSON input
        input_data = {"input_data": input_data_str}

    # Extract agent name and other fields from parsed input
    agent_name = input_data.get("agent_name", "test_agent")
    actual_input = input_data.get("input_data", "")
    context = input_data.get("context", {})
    dependencies = input_data.get("dependencies", {})

    # Create output that conforms to ScriptAgentOutput schema
    output = {
        "success": True,
        "data": {
            "processed_input": actual_input,
            "agent_name": agent_name,
            "context_keys": list(context.keys()),
            "dependency_count": len(dependencies),
            "message": "JSON contract validation test completed"
        },
        "error": None,
        "agent_requests": []
    }

    # Output as JSON
    print(json.dumps(output))


if __name__ == "__main__":
    main()