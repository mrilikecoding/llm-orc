#!/usr/bin/env python3
"""File write primitive for llm-orc (ADR-001).

Primitive: File writing utility
Input: File path and content to write
Output: Write operation result and metadata
Depends: json, os, pathlib
"""

import json
import os
from pathlib import Path


def main() -> None:
    """Write content to file based on provided path and data."""
    try:
        # Get input data from environment
        input_data_str = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_data_str)

        # Extract file path and content
        file_path = (
            input_data.get("context", {}).get("file_path") or
            input_data.get("dependencies", {}).get("file_path") or
            input_data.get("context", {}).get("path")
        )

        content = (
            input_data.get("dependencies", {}).get("content") or
            input_data.get("input_data", "")
        )

        if not file_path:
            raise ValueError("No file path provided")

        # Ensure parent directory exists
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        with open(path_obj, 'w', encoding='utf-8') as f:
            if isinstance(content, dict) or isinstance(content, list):
                # Write JSON content
                json.dump(content, f, indent=2)
                content_type = "json"
            else:
                # Write text content
                f.write(str(content))
                content_type = "text"

        # Get file metadata after write
        stat_info = path_obj.stat()

        # Create successful output
        result = {
            "success": True,
            "data": {
                "file_path": str(path_obj.absolute()),
                "bytes_written": stat_info.st_size,
                "content_type": content_type,
                "encoding": "utf-8",
                "operation": "write"
            },
            "error": None,
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
            "error": f"File write failed: {e}",
            "agent_requests": []
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()