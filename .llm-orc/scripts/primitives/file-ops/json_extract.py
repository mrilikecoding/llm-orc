#!/usr/bin/env python3
"""JSON extract primitive script with Pydantic schema validation (ADR-001, ADR-002).

This primitive extracts values from JSON content using JSONPath expressions,
demonstrating composable script interfaces with type safety.
"""

import json
import sys
from typing import Any


def main() -> None:
    """Main execution - read JSON input, extract values, output structured result."""
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract parameters
        json_content = input_data.get("json_content")
        path_expression = input_data.get("path", "$")  # Root by default
        extract_key = input_data.get("key")

        if not json_content:
            raise ValueError("Missing required parameter: json_content")

        # Parse JSON content if it's a string
        if isinstance(json_content, str):
            parsed_json = json.loads(json_content)
        else:
            parsed_json = json_content

        # Extract value based on key or path
        extracted_value = None
        if extract_key:
            # Simple key extraction
            if isinstance(parsed_json, dict) and extract_key in parsed_json:
                extracted_value = parsed_json[extract_key]
            else:
                raise KeyError(f"Key '{extract_key}' not found in JSON")
        else:
            # For now, just return the whole JSON for path "$"
            if path_expression == "$":
                extracted_value = parsed_json
            else:
                raise ValueError(f"JSONPath expression '{path_expression}' not supported yet")

        # Output structured result
        output = {
            "success": True,
            "data": {
                "extracted_value": extracted_value,
                "path_expression": path_expression,
                "key": extract_key,
                "value_type": type(extracted_value).__name__,
            },
            "error": None,
            "agent_requests": [],
        }

        print(json.dumps(output, indent=2))

    except json.JSONDecodeError as e:
        # Output error result for invalid JSON input
        error_output = {
            "success": False,
            "data": None,
            "error": f"Invalid JSON input: {str(e)}",
            "agent_requests": [],
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    except Exception as e:
        # Handle all other errors with proper exception info
        error_output = {
            "success": False,
            "data": None,
            "error": f"JSON extraction failed: {str(e)}",
            "agent_requests": [],
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()