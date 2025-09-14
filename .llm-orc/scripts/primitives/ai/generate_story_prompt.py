#!/usr/bin/env python3
"""Story prompt generator with dynamic parameter generation (ADR-001).

This primitive script generates contextual story prompts and creates
AgentRequest objects to gather user input, demonstrating dynamic
parameter generation capabilities.
"""

import json
import sys
from typing import Any


def generate_cyberpunk_prompt(character_type: str) -> str:
    """Generate a cyberpunk-themed prompt based on character type."""
    prompts = {
        "protagonist": (
            "You are a cyber-enhanced detective in Neo-Tokyo 2185. "
            "Neon lights reflect off rain-slicked streets as you track "
            "a rogue AI through the city's data networks. "
            "What is your character's name and their unique cybernetic enhancement?"
        ),
        "antagonist": (
            "You control a powerful megacorporation in the sprawling "
            "metroplex. Your neural implants grant you access to vast "
            "data streams and corporate warfare tools. "
            "What is your corporation's name and its dark secret?"
        ),
        "support": (
            "You run a black market cybernetics shop in the underground. "
            "Modified humans and AIs seek your expertise for upgrades "
            "beyond legal limits. "
            "What specialized modification do you secretly offer?"
        ),
    }

    return prompts.get(character_type, prompts["protagonist"])


def main() -> None:
    """Main execution - read input, generate prompt, output AgentRequest."""
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())

        # Extract parameters
        character_type = input_data.get("character_type", "protagonist")
        theme = input_data.get("theme", "cyberpunk")

        # Generate dynamic prompt based on theme and character
        if theme == "cyberpunk":
            prompt = generate_cyberpunk_prompt(character_type)
        else:
            prompt = f"Create a {character_type} character for a {theme} story."

        # Create AgentRequest for user_input agent
        agent_request = {
            "target_agent_type": "user_input",
            "parameters": {
                "prompt": prompt,
                "multiline": True,
                "context": {
                    "theme": theme,
                    "character_type": character_type,
                    "generator": "generate_story_prompt.py"
                }
            },
            "priority": 1
        }

        # Output successful result with AgentRequest
        output = {
            "success": True,
            "data": {
                "generated_prompt": prompt,
                "theme": theme,
                "character_type": character_type
            },
            "error": None,
            "agent_requests": [agent_request]
        }

        print(json.dumps(output, indent=2))

    except json.JSONDecodeError as e:
        # Output error result
        error_output = {
            "success": False,
            "data": None,
            "error": f"Invalid JSON input: {str(e)}",
            "agent_requests": []
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    except Exception as e:
        # Handle unexpected errors
        error_output = {
            "success": False,
            "data": None,
            "error": f"Script execution failed: {str(e)}",
            "agent_requests": []
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()