"""Enhanced script agent with JSON I/O support."""

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_orc.agents.script_agent import ScriptAgent
from llm_orc.core.execution.script_resolver import ScriptResolver


class EnhancedScriptAgent(ScriptAgent):
    """Enhanced script agent that supports JSON I/O and script resolution."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize enhanced script agent with configuration.

        Args:
            name: Agent name
            config: Agent configuration including script and parameters
        """
        super().__init__(name, config)
        self._script_resolver = ScriptResolver()
        self.parameters = config.get("parameters", {})

    async def execute(
        self, input_data: str, context: dict[str, Any] | None = None
    ) -> str:
        """Execute the script with JSON I/O support.

        Args:
            input_data: Input data for the script
            context: Optional context variables

        Returns:
            JSON string output from script or error as JSON string
        """
        if context is None:
            context = {}

        try:
            # Resolve script path using ScriptResolver
            if self.script:
                resolved_script = self._script_resolver.resolve_script_path(self.script)
            else:
                resolved_script = None

            # Prepare JSON input for the script
            json_input = {
                "input": input_data,
                "parameters": self.parameters,
                "context": context,
            }
            json_input_str = json.dumps(json_input)

            # Execute the script with JSON input
            if resolved_script:
                # Check if resolved script is a file path or inline content
                if os.path.exists(resolved_script):
                    result = await self._execute_script_file(
                        resolved_script, json_input_str
                    )
                else:
                    # Inline script content
                    result = await self._execute_inline_script(
                        resolved_script, json_input_str
                    )
            else:
                # Execute command directly
                result = await self._execute_command_with_json(
                    self.command, json_input_str
                )

            # Try to parse output as JSON, but always return string
            parsed_result = self._parse_output(result)
            if isinstance(parsed_result, dict):
                return json.dumps(parsed_result)
            return parsed_result

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Script timed out after {self.timeout} seconds",
                }
            )
        except subprocess.CalledProcessError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Script failed with exit code {e.returncode}",
                    "stderr": e.stderr if e.stderr else "",
                }
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def execute_with_user_input(
        self,
        input_data: str,
        context: dict[str, Any] | None = None,
        user_input_handler: Callable[[str], str] | None = None,
    ) -> str:
        """Execute the script with support for user input during execution.

        Args:
            input_data: Input data for the script
            context: Optional context variables
            user_input_handler: Optional handler for user input requests

        Returns:
            JSON string output from script or error as JSON string
        """
        if context is None:
            context = {}

        try:
            # Simple approach - try interactive execution if handler provided
            if user_input_handler:
                return await self._execute_interactive(
                    input_data, context, user_input_handler
                )
            else:
                # Fall back to normal execution
                return await self.execute(input_data, context)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _execute_interactive(
        self,
        input_data: str,
        context: dict[str, Any],
        user_input_handler: Callable[[str], str],
    ) -> str:
        """Execute script with interactive user input support.

        Args:
            input_data: Input data for the script
            context: Context variables
            user_input_handler: Handler for user input requests

        Returns:
            Final script output as JSON string
        """
        # Resolve script path
        if self.script:
            resolved_script = self._script_resolver.resolve_script_path(self.script)
        else:
            resolved_script = None

        # Prepare JSON input
        json_input = {
            "input": input_data,
            "parameters": self.parameters,
            "context": context,
        }
        json_input_str = json.dumps(json_input)

        # For the minimal implementation, we'll use subprocess with interactive I/O
        if resolved_script and os.path.exists(resolved_script):
            return await self._execute_script_interactive(
                resolved_script, json_input_str, user_input_handler
            )
        else:
            # Fall back to normal execution for non-file scripts
            return await self.execute(input_data, context)

    async def _execute_script_interactive(
        self,
        script_path: str,
        json_input: str,
        user_input_handler: Callable[[str], str],
    ) -> str:
        """Execute script file with interactive user input support.

        Args:
            script_path: Path to script file
            json_input: JSON input to pass initially
            user_input_handler: Handler for user input requests

        Returns:
            Final script output
        """
        # Determine interpreter and prepare environment
        interpreter = self._get_interpreter(script_path)
        env = os.environ.copy()
        env.update(self.environment)

        # Start the process with stdin/stdout pipes
        process = subprocess.Popen(
            interpreter + [script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            return await self._handle_interactive_process(
                process, json_input, user_input_handler
            )
        except subprocess.TimeoutExpired:
            process.kill()
            return json.dumps(
                {
                    "success": False,
                    "error": f"Script timed out after {self.timeout} seconds",
                }
            )
        except Exception as e:
            process.kill()
            return json.dumps({"success": False, "error": str(e)})
        finally:
            self._cleanup_process(process)

    async def _handle_interactive_process(
        self,
        process: subprocess.Popen[str],
        json_input: str,
        user_input_handler: Callable[[str], str],
    ) -> str:
        """Handle the interactive process communication.

        Args:
            process: The subprocess
            json_input: Initial JSON input
            user_input_handler: Handler for user input requests

        Returns:
            Final script output as JSON string
        """
        # Send initial JSON input
        if process.stdin:
            process.stdin.write(json_input + "\n")
            process.stdin.flush()

        # Read and process output lines
        output_lines = self._read_and_process_output(process, user_input_handler)

        # Wait for process to complete
        process.wait(timeout=self.timeout)

        # Return final result
        return self._format_final_result(output_lines)

    def _read_and_process_output(
        self,
        process: subprocess.Popen[str],
        user_input_handler: Callable[[str], str],
    ) -> list[str]:
        """Read output lines and handle user input requests.

        Args:
            process: The subprocess
            user_input_handler: Handler for user input requests

        Returns:
            List of output lines
        """
        output_lines: list[str] = []
        if not process.stdout:
            return output_lines

        while True:
            line = process.stdout.readline()
            if not line:
                break

            output_lines.append(line.strip())

            # Check if this line is a user input request
            if self._handle_user_input_request(line, process, user_input_handler):
                continue

        return output_lines

    def _handle_user_input_request(
        self,
        line: str,
        process: subprocess.Popen[str],
        user_input_handler: Callable[[str], str],
    ) -> bool:
        """Handle a user input request if the line contains one.

        Args:
            line: Output line to check
            process: The subprocess
            user_input_handler: Handler for user input requests

        Returns:
            True if a user input request was handled, False otherwise
        """
        try:
            parsed_line = json.loads(line.strip())
            if (
                isinstance(parsed_line, dict)
                and parsed_line.get("type") == "user_input_request"
            ):
                prompt = parsed_line.get("prompt", "")
                user_response = user_input_handler(prompt)
                if process.stdin:
                    process.stdin.write(user_response + "\n")
                    process.stdin.flush()
                return True
        except json.JSONDecodeError:
            pass
        return False

    def _format_final_result(self, output_lines: list[str]) -> str:
        """Format the final result from output lines.

        Args:
            output_lines: List of output lines

        Returns:
            Final result as JSON string
        """
        if not output_lines:
            return json.dumps({"success": True, "output": ""})

        # Try to return the last JSON line that's not a user input request
        for line in reversed(output_lines):
            try:
                parsed = json.loads(line)
                if (
                    isinstance(parsed, dict)
                    and parsed.get("type") != "user_input_request"
                ):
                    return json.dumps(parsed)
            except json.JSONDecodeError:
                continue

        # If no JSON found, return the full output
        return json.dumps({"output": "\n".join(output_lines)})

    def _cleanup_process(self, process: subprocess.Popen[str]) -> None:
        """Clean up process resources.

        Args:
            process: The subprocess to clean up
        """
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    async def _execute_script_file(self, script_path: str, json_input: str) -> str:
        """Execute a script file with JSON input via stdin.

        Args:
            script_path: Path to the script file
            json_input: JSON input to pass via stdin

        Returns:
            Script output (stdout)
        """
        # Determine interpreter based on file extension
        interpreter = self._get_interpreter(script_path)

        # Prepare environment
        env = os.environ.copy()
        env.update(self.environment)

        # Execute script with JSON input via stdin
        result = subprocess.run(
            interpreter + [script_path],
            input=json_input,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            check=True,
        )

        return result.stdout

    async def _execute_inline_script(self, script_content: str, json_input: str) -> str:
        """Execute inline script content with JSON input.

        Args:
            script_content: Script content to execute
            json_input: JSON input to pass via stdin

        Returns:
            Script output (stdout)
        """
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("set -euo pipefail\n")
            f.write(script_content)
            script_path = f.name

        try:
            os.chmod(script_path, 0o755)

            # Execute with JSON input via stdin
            env = os.environ.copy()
            env.update(self.environment)

            result = subprocess.run(
                ["/bin/bash", script_path],
                input=json_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                check=True,
            )

            return result.stdout

        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _execute_command_with_json(self, command: str, json_input: str) -> str:
        """Execute a command with JSON input via stdin.

        Args:
            command: Command to execute
            json_input: JSON input to pass via stdin

        Returns:
            Command output (stdout)
        """
        import shlex

        # Parse command safely
        args = shlex.split(command)
        if not args:
            raise RuntimeError("Empty command provided")

        # Validate executable safety
        self._validate_executable_safety(args[0])

        # Execute with JSON input via stdin
        env = os.environ.copy()
        env.update(self.environment)

        result = subprocess.run(
            args,
            input=json_input,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            check=True,
        )

        return result.stdout

    def _get_interpreter(self, script_path: str) -> list[str]:
        """Get the appropriate interpreter for a script file.

        Args:
            script_path: Path to the script file

        Returns:
            List of interpreter command parts
        """
        ext = Path(script_path).suffix.lower()

        interpreters = {
            ".py": ["python3"],
            ".python": ["python3"],
            ".sh": ["bash"],
            ".bash": ["bash"],
            ".js": ["node"],
            ".javascript": ["node"],
            ".rb": ["ruby"],
            ".ruby": ["ruby"],
        }

        return interpreters.get(ext, ["bash"])

    def _parse_output(self, output: str) -> dict[str, Any] | str:
        """Parse script output as JSON if possible.

        Args:
            output: Raw script output

        Returns:
            Parsed JSON dict or dict with raw output
        """
        output = output.strip()

        if not output:
            return {"success": True, "output": ""}

        try:
            # Try to parse as JSON
            parsed: dict[str, Any] = json.loads(output)
            return parsed
        except json.JSONDecodeError:
            # Return as structured output with raw text
            return {"success": True, "output": output}
