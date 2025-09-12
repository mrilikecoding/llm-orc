#!/usr/bin/env python3
"""
get_user_input.py - Get user input and output as JSON

Usage: 
  As script agent in ensemble:
    script: primitives/user-interaction/get_user_input.py
    parameters:
      prompt: "Enter your feedback:"
      multiline: false
      
  From command line:
    echo '{"prompt": "Your name?"}' | python get_user_input.py
"""
import json
import os
import sys


def main():
    # Read configuration from environment variables if available, otherwise from stdin
    if "AGENT_PARAMETERS" in os.environ:
        # Running in interactive mode - get config from environment
        config = json.loads(os.environ.get("AGENT_PARAMETERS", "{}"))
    elif not sys.stdin.isatty():
        # Running in normal mode - get config from stdin
        config = json.loads(sys.stdin.read())
    else:
        config = {}
    
    # Get parameters with defaults
    prompt = config.get('prompt', 'Enter input:')
    multiline = config.get('multiline', False)
    
    try:
        if multiline:
            print(f"{prompt} (Enter blank line to finish)")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            user_input = "\n".join(lines)
        else:
            user_input = input(f"{prompt} ")
        
        result = {
            "success": True,
            "input": user_input,
            "multiline": multiline,
            "length": len(user_input)
        }
    except (EOFError, KeyboardInterrupt):
        result = {
            "success": False,
            "error": "User cancelled input",
            "input": ""
        }
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "input": ""
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()