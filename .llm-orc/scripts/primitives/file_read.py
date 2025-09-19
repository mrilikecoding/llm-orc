#!/usr/bin/env python3
"""File read primitive for llm-orc (ADR-001).

Primitive: File reading utility
Input: File path to read
Output: File contents and metadata
Depends: json, os, pathlib
"""

import json
import os
from pathlib import Path


def main() -> None:
    """Read file contents based on provided path."""
    try:
        # Get input data from environment
        input_data_str = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_data_str)

        # Extract file path from context or dependencies
        file_path = (
            input_data.get("context", {}).get("file_path") or
            input_data.get("dependencies", {}).get("file_path") or
            input_data.get("input_data", "")
        )

        if not file_path:
            raise ValueError("No file path provided")

        # Read the file
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file contents
        with open(path_obj, 'r', encoding='utf-8') as f:
            contents = f.read()

        # Get file metadata
        stat_info = path_obj.stat()

        # Create successful output
        result = {
            "success": True,
            "data": {
                "contents": contents,
                "file_path": str(path_obj.absolute()),
                "size_bytes": stat_info.st_size,
                "lines": len(contents.splitlines()),
                "encoding": "utf-8"
            },
            "error": None,
            "agent_requests": []
        }

    except FileNotFoundError as e:
        result = {
            "success": False,
            "data": None,
            "error": f"File not found: {e}",
            "agent_requests": []
        }
    except PermissionError as e:
        result = {
            "success": False,
            "data": None,
            "error": f"Permission denied: {e}",
            "agent_requests": []
        }
    except Exception as e:
        result = {
            "success": False,
            "data": None,
            "error": f"File read failed: {e}",
            "agent_requests": []
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()