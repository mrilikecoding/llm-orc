#!/usr/bin/env python3
"""
json_extract.py - Extract specific fields from JSON data

Usage: 
  As script agent in ensemble:
    script: primitives/data-transform/json_extract.py
    parameters:
      json_data: '{"name": "Alice", "age": 30}'
      fields: ["name", "age"]
      
  From command line:
    echo '{"json_data": "{\"x\": 1}", "fields": ["x"]}' | python json_extract.py
"""
import json
import sys


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters
    json_data = config.get('json_data', '{}')
    fields = config.get('fields', [])
    
    try:
        # Parse the input JSON
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Extract specified fields
        extracted = {}
        missing_fields = []
        
        for field in fields:
            if field in data:
                extracted[field] = data[field]
            else:
                missing_fields.append(field)
        
        result = {
            "success": True,
            "extracted": extracted,
            "missing_fields": missing_fields,
            "total_fields": len(fields),
            "extracted_count": len(extracted)
        }
    except json.JSONDecodeError as e:
        result = {
            "success": False,
            "error": f"Invalid JSON: {str(e)}",
            "extracted": {}
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "extracted": {}
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()