"""Minimal test primitives that don't require library submodule.

This module provides test implementations of primitive scripts that mirror
the functionality of library primitives without requiring the llm-orchestra-library
submodule to be initialized. These are used to ensure tests pass in isolation.

Per ADR-006: Library-Based Primitives Architecture, tests should be independent
of external library dependencies while still validating integration behavior.
"""

import json
from pathlib import Path
from typing import Any


class TestPrimitiveFactory:
    """Creates test primitive scripts in isolated temp directories.

    This factory generates minimal Python scripts that implement the same
    JSON I/O interfaces as real library primitives, but with mocked behavior
    suitable for testing.
    """

    @staticmethod
    def create_user_input_script(tmp_path: Path, language: str = "python") -> Path:
        """Create a test user_input primitive.

        Args:
            tmp_path: Temporary directory for script creation
            language: Language for the script (currently only Python supported)

        Returns:
            Path to the created script file
        """
        script = tmp_path / "user_input.py"
        script.write_text("""#!/usr/bin/env python3
\"\"\"Test user input primitive that mocks user interaction.

Follows ADR-006 bridge primitive pattern with JSON I/O.
In tests, returns mock data instead of actual user input.
\"\"\"
import json
import os
import sys

def main():
    try:
        # Get structured input from environment
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

        # Extract parameters
        prompt = input_data.get('prompt', 'Enter value: ')
        validation_pattern = input_data.get('validation_pattern')
        retry_message = input_data.get('retry_message', 'Please enter a valid value')

        # In tests, use mock input instead of actual input()
        # This allows tests to run without user interaction
        mock_input = input_data.get('mock_user_input', 'test_user_input')

        # Simulate validation if pattern provided
        validation_passed = True
        if validation_pattern:
            import re
            validation_passed = bool(re.match(validation_pattern, mock_input))

        # Return structured output matching real primitive interface
        result = {
            "success": True,
            "data": mock_input,
            "user_input": mock_input,
            "input_length": len(mock_input),
            "validation_passed": validation_passed,
            "received_dynamic_parameters": input_data,
            "metadata": {
                "prompt_used": prompt,
                "is_test_mode": True
            }
        }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "data": None
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        script.chmod(0o755)
        return script

    @staticmethod
    def create_subprocess_executor(tmp_path: Path) -> Path:
        """Create a test subprocess executor bridge primitive.

        Args:
            tmp_path: Temporary directory for script creation

        Returns:
            Path to the created script file
        """
        script = tmp_path / "subprocess_executor.py"
        script.write_text("""#!/usr/bin/env python3
\"\"\"Test subprocess executor that mocks external command execution.

Implements ADR-006 bridge primitive pattern for testing without
actually executing external commands.
\"\"\"
import json
import os
import sys

def main():
    try:
        # Get structured input from environment
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

        # Extract parameters
        command = input_data.get('command')
        working_dir = input_data.get('working_dir', '.')
        env_vars = input_data.get('env_vars', {})
        timeout = input_data.get('timeout', 30)

        if not command:
            result = {
                "success": False,
                "error": "command parameter required",
                "stdout": "",
                "stderr": "Missing required parameter: command",
                "return_code": 1
            }
        else:
            # Mock successful execution
            result = {
                "success": True,
                "stdout": f"Mock execution of: {command}",
                "stderr": "",
                "return_code": 0,
                "metadata": {
                    "command_executed": command,
                    "working_dir": working_dir,
                    "timeout_used": timeout,
                    "is_test_mode": True
                }
            }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "return_code": 1
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        script.chmod(0o755)
        return script

    @staticmethod
    def create_node_executor(tmp_path: Path) -> Path:
        """Create a test Node.js executor bridge primitive.

        Args:
            tmp_path: Temporary directory for script creation

        Returns:
            Path to the created script file
        """
        script = tmp_path / "node_executor.py"
        script.write_text("""#!/usr/bin/env python3
\"\"\"Test Node.js executor that mocks JavaScript execution.

Implements ADR-006 bridge primitive pattern for testing without
requiring Node.js to be installed.
\"\"\"
import json
import os
import sys

def main():
    try:
        # Get structured input from environment
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

        # Extract parameters
        script_content = input_data.get('script')
        script_path = input_data.get('script_path')
        data = input_data.get('data', {})
        timeout = input_data.get('timeout', 30)

        if not script_content and not script_path:
            result = {
                "success": False,
                "error": "Either script or script_path parameter required"
            }
        else:
            # Mock JavaScript execution with simple transformations
            mock_js_result = {
                "input_received": data,
                "script_type": "inline" if script_content else "file",
                "mock_processing": "JavaScript execution simulated"
            }

            result = {
                "success": True,
                "data": mock_js_result,
                "metadata": {
                    "script_content_length": (
                        len(script_content) if script_content else 0
                    ),
                    "script_path": script_path,
                    "timeout_used": timeout,
                    "is_test_mode": True
                }
            }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        script.chmod(0o755)
        return script

    @staticmethod
    def create_file_read_script(tmp_path: Path) -> Path:
        """Create a test file read primitive.

        Args:
            tmp_path: Temporary directory for script creation

        Returns:
            Path to the created script file
        """
        script = tmp_path / "file_read.py"
        script.write_text("""#!/usr/bin/env python3
\"\"\"Test file read primitive that mocks file operations.
\"\"\"
import json
import os
import sys

def main():
    try:
        input_data = json.loads(os.environ.get('INPUT_DATA', '{}'))

        file_path = input_data.get('file_path')
        encoding = input_data.get('encoding', 'utf-8')

        if not file_path:
            result = {
                "success": False,
                "error": "file_path parameter required"
            }
        else:
            # Mock file content
            mock_content = f"Mock content from {file_path}"
            result = {
                "success": True,
                "data": mock_content,
                "file_content": mock_content,
                "file_size": len(mock_content),
                "encoding_used": encoding,
                "metadata": {
                    "file_path": file_path,
                    "is_test_mode": True
                }
            }

        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        script.chmod(0o755)
        return script

    @classmethod
    def setup_test_primitives_dir(cls, tmp_path: Path) -> Path:
        """Setup complete test primitives directory with all common primitives.

        Args:
            tmp_path: Temporary directory for primitives creation

        Returns:
            Path to the primitives directory containing all test scripts
        """
        primitives_dir = tmp_path / "primitives"
        primitives_dir.mkdir(parents=True, exist_ok=True)

        # Create all standard test primitives
        cls.create_user_input_script(primitives_dir)
        cls.create_subprocess_executor(primitives_dir)
        cls.create_node_executor(primitives_dir)
        cls.create_file_read_script(primitives_dir)

        return primitives_dir

    @classmethod
    def create_custom_primitive(
        cls, tmp_path: Path, script_name: str, script_content: str
    ) -> Path:
        """Create a custom primitive script with specified content.

        Args:
            tmp_path: Temporary directory for script creation
            script_name: Name of the script file (should end with .py)
            script_content: Full Python script content

        Returns:
            Path to the created script file
        """
        script = tmp_path / script_name
        script.write_text(script_content)
        script.chmod(0o755)
        return script


def create_test_primitive_with_json_contract(
    tmp_path: Path,
    agent_name: str,
    mock_agent_requests: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a test primitive that outputs AgentRequest objects.

    This function creates a test script that demonstrates the JSON contract
    pattern from ADR-001, where scripts can output structured AgentRequest
    objects for inter-agent communication.

    Args:
        tmp_path: Temporary directory for script creation
        agent_name: Name for the test agent
        mock_agent_requests: Optional list of AgentRequest objects to output

    Returns:
        Path to the created script file
    """
    if mock_agent_requests is None:
        mock_agent_requests = [
            {
                "target_agent_type": "user_input",
                "parameters": {
                    "prompt": "What is the character's name?",
                    "validation_pattern": "^[A-Za-z\\s]{2,30}$",
                    "retry_message": "Please enter a valid name (2-30 characters)",
                },
                "priority": 1,
            }
        ]

    script = tmp_path / f"{agent_name}.py"
    script_content = f'''#!/usr/bin/env python3
"""Test script agent that outputs AgentRequest objects for coordination.

Demonstrates ADR-001 JSON contract pattern for inter-agent communication.
"""
import json
import os

def main():
    try:
        # Get input data
        input_data = json.loads(os.environ.get('INPUT_DATA', '{{}}'))

        # Mock processing based on input
        theme = input_data.get('theme', 'generic')
        character_type = input_data.get('character_type', 'protagonist')

        # Generate mock story fragment
        story_fragment = f"A {{theme}} {{character_type}} story begins..."

        # Output with AgentRequest objects
        output = {{
            "success": True,
            "data": {{
                "story_fragment": story_fragment,
                "prompt_generated": True,
                "theme": theme,
                "character_type": character_type
            }},
            "error": None,
            "agent_requests": {json.dumps(mock_agent_requests)},
            "metadata": {{
                "agent_name": "{agent_name}",
                "is_test_mode": True
            }}
        }}

        print(json.dumps(output))

    except Exception as e:
        error_output = {{
            "success": False,
            "data": None,
            "error": str(e),
            "agent_requests": []
        }}
        print(json.dumps(error_output))

if __name__ == "__main__":
    main()
'''

    script.write_text(script_content)
    script.chmod(0o755)
    return script
