#!/usr/bin/env python3
"""JSON extraction primitive for llm-orc (ADR-001).

Primitive: JSON extraction utility
Input: JSON data with extraction path
Output: Extracted JSON value
Depends: json, os
"""

import json
import os


def main() -> None:
    """Extract a value from JSON data based on provided path."""
    try:
        # Get input data from environment
        input_data_str = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_data_str)

        # Extract parameters
        source_data_str = input_data.get("input_data", "{}")
        extraction_path = input_data.get("context", {}).get("path", "")

        # Parse source data
        if source_data_str.startswith("{") or source_data_str.startswith("["):
            source_data = json.loads(source_data_str)
        else:
            source_data = {"value": source_data_str}

        # Extract value using path (support simple dot notation)
        extracted_value = source_data
        if extraction_path:
            for key in extraction_path.split("."):
                if isinstance(extracted_value, dict) and key in extracted_value:
                    extracted_value = extracted_value[key]
                else:
                    extracted_value = None
                    break

        # Create successful output
        result = {
            "success": True,
            "data": {
                "extracted": extracted_value,
                "path": extraction_path,
                "source_type": type(source_data).__name__
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
            "error": f"Extraction failed: {e}",
            "agent_requests": []
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()