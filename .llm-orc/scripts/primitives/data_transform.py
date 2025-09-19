#!/usr/bin/env python3
"""
# Primitive: Transform data according to specified type
# Input: TestDataTransformInputSchema
# Output: TestDataTransformOutputSchema
# Depends:

data_transform.py - Data transformation primitive for ADR-002 composable system

This primitive implements the universal Primitive interface for data transformation.
It demonstrates dependency resolution and type-safe composition patterns.
"""
import json
import os
import sys


def main():
    """Main entry point for the primitive."""
    try:
        # Read input data from environment (set by PrimitiveComposer)
        input_json = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_json)

        # Extract parameters from ScriptAgentInput structure
        agent_name = input_data.get("agent_name", "data_transform")
        context = input_data.get("context", {})
        dependencies = input_data.get("dependencies", {})

        # Get source data from dependencies (resolved by PrimitiveComposer)
        source_data = dependencies.get("source_data", context.get("source_data"))
        transform_type = context.get("transform_type", "json")

        # Perform transformation based on type
        if transform_type == "json":
            transformed_data = {
                "original": source_data,
                "transformed": True,
                "transform_type": transform_type,
                "agent": agent_name
            }
        elif transform_type == "uppercase":
            transformed_data = str(source_data).upper() if source_data else ""
        elif transform_type == "length":
            transformed_data = len(str(source_data)) if source_data else 0
        else:
            transformed_data = {"wrapped": source_data, "type": transform_type}

        # Return output in ScriptAgentOutput format
        result = {
            "success": True,
            "data": {
                "transformed_data": transformed_data,
                "source_data": source_data,
                "transform_type": transform_type,
            },
            "error": None,
            "agent_requests": []
        }

    except Exception as e:
        result = {
            "success": False,
            "data": None,
            "error": str(e),
            "agent_requests": []
        }

    # Output JSON for primitive composition system
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()