#!/usr/bin/env python3
"""
read_file.py - Read file content and output as JSON

Usage: 
  As script agent in ensemble:
    script: primitives/file-ops/read_file.py
    parameters:
      path: "data/input.txt"
      encoding: "utf-8"
      
  From command line:
    echo '{"path": "file.txt"}' | python read_file.py
"""
import json
import sys
from pathlib import Path


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters with defaults
    file_path = config.get('path', 'input.txt')
    encoding = config.get('encoding', 'utf-8')
    
    # Execute primitive operation
    try:
        content = Path(file_path).read_text(encoding=encoding)
        result = {
            "success": True,
            "content": content,
            "path": str(file_path),
            "size": len(content)
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "path": str(file_path)
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()