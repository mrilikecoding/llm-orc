#!/usr/bin/env python3
"""
confirm_action.py - Get yes/no confirmation from user

Usage: 
  As script agent in ensemble:
    script: primitives/user-interaction/confirm_action.py
    parameters:
      prompt: "Proceed with analysis?"
      default: "n"
      
  From command line:
    echo '{"prompt": "Continue?"}' | python confirm_action.py
"""
import json
import sys


def main():
    # Read configuration from stdin
    if not sys.stdin.isatty():
        config = json.loads(sys.stdin.read())
    else:
        config = {}

    # Extract parameters from EnhancedScriptAgent format
    # Format: {"input": "...", "parameters": {...}, "context": {...}}
    parameters = config.get('parameters', config)

    # Get parameters with defaults
    prompt = parameters.get('prompt', 'Continue?')
    default = parameters.get('default', 'n').lower()
    
    try:
        # Format prompt with default indication
        if default == 'y':
            full_prompt = f"{prompt} [Y/n]: "
        else:
            full_prompt = f"{prompt} [y/N]: "
        
        user_input = input(full_prompt).strip().lower()
        
        # Handle default case
        if user_input == "":
            user_input = default
        
        # Determine confirmation
        confirmed = user_input in ['y', 'yes', 'true', '1']
        
        result = {
            "success": True,
            "confirmed": confirmed,
            "input": user_input,
            "prompt": prompt
        }
    except (EOFError, KeyboardInterrupt):
        result = {
            "success": False,
            "confirmed": False,
            "error": "User cancelled confirmation",
            "input": ""
        }
    except Exception as e:
        result = {
            "success": False,
            "confirmed": False,
            "error": str(e),
            "input": ""
        }
    
    # Output JSON for downstream agents
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()