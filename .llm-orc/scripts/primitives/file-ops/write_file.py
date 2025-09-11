#!/usr/bin/env python3
"""
write_file.py - Write content to file and output status as JSON

Usage: 
  As script agent in ensemble:
    script: primitives/file-ops/write_file.py
    parameters:
      path: "output/result.txt"
      content: "Hello world"
      encoding: "utf-8"
      
  From command line:
    echo '{"path": "output.txt", "content": "Hello"}' | python write_file.py
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
    file_path = config.get('path', 'output.txt')
    content = config.get('content', '')
    encoding = config.get('encoding', 'utf-8')
    
    # Execute primitive operation
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_text(content, encoding=encoding)
        
        result = {
            "success": True,
            "path": str(file_path),
            "size": len(content),
            "bytes_written": len(content.encode(encoding))
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