#!/usr/bin/env python3
"""Test script for script agent functionality."""

import json
import sys

def main():
    # Read JSON input from stdin
    if not sys.stdin.isatty():
        try:
            input_data = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            input_data = {}
    else:
        input_data = {}
    
    # Process the input
    result = {
        "success": True,
        "message": "Hello from test script!",
        "input_received": input_data,
        "parameters": input_data.get("parameters", {}),
        "context": input_data.get("context", {})
    }
    
    # Output JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()