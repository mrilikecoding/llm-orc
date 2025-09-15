#!/usr/bin/env python3
"""Script that attempts to read protected files for error handling testing."""

import json
import sys
import os


def read_protected_file(target_file: str) -> dict:
    """Attempt to read a protected file to trigger permission errors."""
    try:
        # Try to read the protected file
        with open(target_file, 'r') as f:
            content = f.read()

        return {
            "success": True,
            "data": {
                "file_content": content,
                "file_size": len(content)
            },
            "error": None
        }

    except PermissionError as e:
        # This is the expected error - chain it properly
        raise RuntimeError(f"Script failed to access protected file '{target_file}': {str(e)}") from e

    except FileNotFoundError as e:
        # File doesn't exist
        raise RuntimeError(f"Script target file not found: '{target_file}'") from e

    except Exception as e:
        # Any other error
        raise RuntimeError(f"Script encountered unexpected error reading '{target_file}': {str(e)}") from e


def main() -> None:
    """Main entry point for script execution."""
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract file path from EnhancedScriptAgent input format
        actual_input = input_data.get("input", input_data)

        # Get target file from parameters or default
        target_file = actual_input.get("target_file", "/root/protected_file.txt")

        # Attempt to read the protected file
        result = read_protected_file(target_file)

        print(json.dumps(result))

    except Exception as e:
        # Return error output with proper chaining preserved
        error_output = {
            "success": False,
            "data": None,
            "error": f"Protected file read failed: {str(e)}",
            "error_type": type(e).__name__,
            "original_error": str(e.__cause__) if e.__cause__ else None
        }
        print(json.dumps(error_output))
        sys.exit(1)


if __name__ == "__main__":
    main()