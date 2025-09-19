#!/usr/bin/env python3
"""
# Primitive: Collect input from user with prompt
# Input: TestUserInputSchema
# Output: TestUserOutputSchema
# Depends:

get_user_input.py - User interaction primitive for ADR-002 composable system

This primitive implements the universal Primitive interface for collecting user input.
It demonstrates the type-safe composition patterns defined in ADR-002.
"""
import json
import os
import sys


def main():
    """Main entry point for the primitive."""
    try:
        # Read input data from environment (set by PrimitiveComposer)
        input_json = os.environ.get("INPUT_DATA", "{}")
        input_data = json.loads(input_json)

        # Extract parameters from ScriptAgentInput structure
        agent_name = input_data.get("agent_name", "get_user_input")
        context = input_data.get("context", {})
        prompt = context.get("prompt", "Enter input:")

        # Simulate user input for testing (in real scenarios this would be interactive)
        if "test" in agent_name.lower() or os.environ.get("TESTING_MODE"):
            user_response = f"Response to: {prompt}"
        else:
            user_response = input(f"{prompt} ")

        # Return output in ScriptAgentOutput format
        result = {
            "success": True,
            "data": {
                "user_response": user_response,
                "prompt_used": prompt,
            },
            "error": None,
            "agent_requests": []
        }

    except Exception as e:
        result = {
            "success": False,
            "data": None,
            "error": str(e),
            "agent_requests": []
        }

    # Output JSON for primitive composition system
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()