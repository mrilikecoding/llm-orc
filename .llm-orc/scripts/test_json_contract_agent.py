#!/usr/bin/env python3
"""Test script for JSON contract validation integration."""

import json
import os
import sys


def main():
    """Test script that processes ScriptAgentInput and produces ScriptAgentOutput."""
    try:
        # Get input data from environment (set by EnsembleExecutor)
        input_data = os.environ.get("INPUT_DATA", "{}")
        agent_params = os.environ.get("AGENT_PARAMETERS", "{}")

        # Parse input according to ScriptAgentInput schema
        input_json = json.loads(input_data)
        params_json = json.loads(agent_params)

        # Validate required ScriptAgentInput fields exist
        if not isinstance(input_json, dict):
            raise ValueError("Input must be a dictionary")

        agent_name = input_json.get("agent_name", "test_agent")
        input_text = input_json.get("input_data", "")
        context = input_json.get("context", {})
        dependencies = input_json.get("dependencies", {})

        # Process the input
        processed_text = f"Processed: {input_text}"

        # Create ScriptAgentOutput format
        output = {
            "success": True,
            "data": {
                "processed_input": processed_text,
                "agent_name": agent_name,
                "context_keys": list(context.keys()),
                "dependency_count": len(dependencies)
            },
            "error": None,
            "agent_requests": []
        }

        # Output as JSON
        print(json.dumps(output))

    except Exception as e:
        # Error output in ScriptAgentOutput format
        error_output = {
            "success": False,
            "data": None,
            "error": str(e),
            "agent_requests": []
        }
        print(json.dumps(error_output))
        sys.exit(1)


if __name__ == "__main__":
    main()