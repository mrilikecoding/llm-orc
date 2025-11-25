#!/usr/bin/env python3
import os
import json

# Test if we can get user input
try:
    name = input("Enter your name: ")
    result = {
        "success": True,
        "input": name,
        "message": f"Hello {name}!"
    }
except Exception as e:
    result = {
        "success": False,
        "error": str(e),
        "input": ""
    }

print(json.dumps(result, indent=2))