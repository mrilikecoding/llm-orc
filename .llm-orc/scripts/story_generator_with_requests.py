#!/usr/bin/env python3
"""Story generator that outputs AgentRequest objects for user input."""

import json
import os


def main():
    """Generate story and output AgentRequest for user input."""
    try:
        # Get input data from environment (set by EnsembleExecutor)
        input_data = os.environ.get("INPUT_DATA", "{}")

        # Parse input according to ScriptAgentInput schema
        try:
            input_json = json.loads(input_data)
        except json.JSONDecodeError:
            input_json = {"input_data": input_data}

        story_theme = input_json.get("context", {}).get("theme", "cyberpunk")

        # Generate story fragment
        story_fragment = f"In the neon-lit streets of Neo-Tokyo 2087, a {story_theme} tale begins..."

        # Create AgentRequest for user input
        agent_request = {
            "target_agent_type": "user_input",
            "parameters": {
                "prompt": f"What is the protagonist's name in this {story_theme} story?",
                "validation_pattern": r"^[A-Za-z\s]{2,30}$",
                "retry_message": "Please enter a valid name (2-30 characters, letters only)"
            },
            "priority": 1
        }

        # Create ScriptAgentOutput with AgentRequest
        output = {
            "success": True,
            "data": {
                "story_fragment": story_fragment,
                "theme": story_theme,
                "character_prompt_generated": True
            },
            "error": None,
            "agent_requests": [agent_request]
        }

        # Output as JSON
        print(json.dumps(output))

    except Exception as e:
        # Error output in ScriptAgentOutput format
        error_output = {
            "success": False,
            "data": None,
            "error": str(e),
            "agent_requests": []
        }
        print(json.dumps(error_output))


if __name__ == "__main__":
    main()