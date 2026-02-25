"""Script-based agent execution for hybrid LLM/script workflows."""

import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_orc.core.execution.scripting.resolver import ScriptResolver
from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput


class ScriptEnvironmentManager:
    """Manages environment variable preparation for script execution."""

    def __init__(
        self, base_environment: dict[str, str], agent_parameters: dict[str, Any]
    ):
        """Initialize with base environment and agent parameters.

        Args:
            base_environment: Base environment variables
            agent_parameters: Agent-specific parameters
        """
        self.base_environment = base_environment
        self.agent_parameters = agent_parameters

    def prepare_environment(
        self, input_data: str, direct_json: bool = False
    ) -> dict[str, str]:
        """Prepare environment with proper JSON handling.

        Args:
            input_data: Input data (raw or JSON)
            direct_json: If True, pass input_data directly as ScriptAgentInput JSON

        Returns:
            Environment dictionary with INPUT_DATA and AGENT_PARAMETERS
        """
        env = os.environ.copy()
        env.update(self.base_environment)

        if direct_json:
            # Full ScriptAgentInput JSON for schema-aware scripts
            env["INPUT_DATA"] = input_data
            # Raw input text with real newlines for text-processing
            # scripts (avoids JSON escape sequences like \n appearing
            # as literal characters).
            try:
                parsed = json.loads(input_data)
                env["INPUT_TEXT"] = parsed.get("input_data", "")
            except (json.JSONDecodeError, TypeError):
                env["INPUT_TEXT"] = input_data
            env["AGENT_PARAMETERS"] = json.dumps(self.agent_parameters)
        else:
            # Parse json_input to extract ScriptAgentInput for environment variables
            try:
                parsed_input = json.loads(input_data)
                # Extract the actual input data which should be ScriptAgentInput JSON
                actual_input = parsed_input.get("input", "")
                # Set INPUT_DATA environment variable for script compatibility
                if isinstance(actual_input, str):
                    env["INPUT_DATA"] = actual_input
                    env["INPUT_TEXT"] = actual_input
                else:
                    env["INPUT_DATA"] = json.dumps(actual_input)
                    env["INPUT_TEXT"] = json.dumps(actual_input)
                # Set AGENT_PARAMETERS for additional parameters
                env["AGENT_PARAMETERS"] = json.dumps(parsed_input.get("parameters", {}))
            except (json.JSONDecodeError, KeyError):
                # Fallback to passing raw input_data as INPUT_DATA
                env["INPUT_DATA"] = input_data
                env["INPUT_TEXT"] = input_data
                env["AGENT_PARAMETERS"] = "{}"

        return env


class ScriptAgent:
    """Agent that executes scripts/commands as part of an ensemble.

    Supports JSON I/O, script resolution, and Pydantic schema validation.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        project_dir: Path | None = None,
    ):
        """Initialize script agent with configuration.

        Args:
            name: Agent name
            config: Agent configuration including script and parameters
            project_dir: Optional project directory for script resolution
        """
        self.name = name
        self.config = config
        self.script = config.get("script", "")
        self.command = config.get("command", "")
        self.timeout = config.get("timeout_seconds", 60)
        self.environment = config.get("environment", {})

        if not self.script and not self.command:
            raise ValueError(
                f"Script agent {name} must have either 'script' or 'command'"
            )

        self._script_resolver = ScriptResolver(project_dir=project_dir)
        self.parameters = config.get("parameters", {})
        self._env_manager = ScriptEnvironmentManager(self.environment, self.parameters)

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
            resolved_script = (
                self._script_resolver.resolve_script_path(self.script)
                if self.script
                else None
            )

            json_input_str = json.dumps(
                {
                    "input": input_data,
                    "parameters": self.parameters,
                    "context": context,
                }
            )

            result = await self._run_script(resolved_script, json_input_str)

            parsed_result = self._parse_output(result)
            if isinstance(parsed_result, dict):
                return json.dumps(parsed_result)
            return parsed_result

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "success": False,
                    "error": (f"Script timed out after {self.timeout} seconds"),
                }
            )
        except subprocess.CalledProcessError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": (f"Script failed with exit code {e.returncode}"),
                    "stderr": e.stderr if e.stderr else "",
                }
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _run_script(
        self, resolved_script: str | None, json_input_str: str
    ) -> str:
        """Run the resolved script and return its output."""
        if resolved_script and os.path.exists(resolved_script):
            return await self._execute_script_file(resolved_script, json_input_str)
        if resolved_script:
            return await self._execute_inline_script(resolved_script, json_input_str)
        return await self._execute_command_with_json(self.command, json_input_str)

    async def execute_with_schema(
        self, input_schema: ScriptAgentInput
    ) -> ScriptAgentOutput:
        """Execute script with Pydantic schema validation (ADR-001).

        Args:
            input_schema: Validated input schema

        Returns:
            Validated output schema

        Raises:
            ValueError: If schema validation fails
        """
        try:
            # Execute script with schema data
            raw_result = await self.execute(
                input_schema.input_data,
                {
                    "dependencies": input_schema.dependencies,
                    **input_schema.context,
                },
            )

            # Parse the raw JSON result
            result_dict = json.loads(raw_result)

            # Validate and return as ScriptAgentOutput schema
            return ScriptAgentOutput(**result_dict)

        except json.JSONDecodeError as e:
            # Handle non-JSON output
            return ScriptAgentOutput(
                success=False,
                error=f"Script output is not valid JSON: {str(e)}",
                data=raw_result if "raw_result" in locals() else None,
            )
        except Exception as e:
            # Handle execution errors with proper exception chaining (ADR-003)
            error_msg = f"Schema-based execution failed for {input_schema.agent_name}"
            return ScriptAgentOutput(success=False, error=f"{error_msg}: {str(e)}")

    async def execute_with_schema_json(self, input_json: str) -> str:
        """Execute script with ScriptAgentInput JSON directly (ADR-001).

        This method bypasses the double-wrapping issue by passing
        ScriptAgentInput JSON directly to the script via environment
        variables, maintaining JSON contract validation between
        ScriptResolver and EnsembleExecutor.

        Args:
            input_json: ScriptAgentInput JSON string

        Returns:
            JSON string output from script
        """
        try:
            # Validate the input is proper ScriptAgentInput
            input_schema = ScriptAgentInput.model_validate_json(input_json)

            # Resolve script path using ScriptResolver
            if self.script:
                resolved_script = self._script_resolver.resolve_script_path(self.script)
            else:
                resolved_script = None

            # Execute the script with ScriptAgentInput JSON directly
            if resolved_script and os.path.exists(resolved_script):
                result = await self._execute_script_file_with_schema_json(
                    resolved_script, input_json
                )
            elif resolved_script:
                # Inline script - execute with ScriptAgentInput JSON via stdin
                result = await self._execute_inline_script_with_schema_json(
                    resolved_script, input_json
                )
            else:
                # No script defined - fallback to regular execution
                schema_result = await self.execute_with_schema(input_schema)
                return json.dumps(schema_result.model_dump())

            return result

        except Exception as e:
            # Return error in ScriptAgentOutput format
            error_output = ScriptAgentOutput(
                success=False,
                error=f"Schema JSON execution failed: {str(e)}",
                data=None,
            )
            return json.dumps(error_output.model_dump())

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

        # For the minimal implementation, use subprocess with interactive I/O
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
        process = subprocess.Popen(  # nosec B603
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
                    "error": (f"Script timed out after {self.timeout} seconds"),
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

    def _prepare_script_environment(self, json_input: str) -> dict[str, str]:
        """Prepare environment variables for script execution.

        Args:
            json_input: JSON input to parse for environment variables

        Returns:
            Environment dictionary with INPUT_DATA and AGENT_PARAMETERS
        """
        return self._env_manager.prepare_environment(json_input, direct_json=False)

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
        env = self._prepare_script_environment(json_input)

        # Execute script with JSON input via stdin and environment variables
        result = subprocess.run(  # nosec B603
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
            os.chmod(script_path, 0o700)  # Only owner can read/write/execute

            # Execute with JSON input via stdin
            env = os.environ.copy()
            env.update(self.environment)

            # Parse json_input to extract ScriptAgentInput for env variables
            try:
                input_data = json.loads(json_input)
                # Extract the actual input data
                actual_input = input_data.get("input", "")
                # Set INPUT_DATA environment variable for script compat
                if isinstance(actual_input, str):
                    env["INPUT_DATA"] = actual_input
                else:
                    env["INPUT_DATA"] = json.dumps(actual_input)
                # Set AGENT_PARAMETERS for additional parameters
                env["AGENT_PARAMETERS"] = json.dumps(input_data.get("parameters", {}))
            except (json.JSONDecodeError, KeyError):
                # Fallback to passing raw json_input as INPUT_DATA
                env["INPUT_DATA"] = json_input
                env["AGENT_PARAMETERS"] = "{}"

            result = subprocess.run(  # nosec B603
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
        # Parse command safely
        args = shlex.split(command)
        if not args:
            raise RuntimeError("Empty command provided")

        # Validate executable safety
        self._validate_executable_safety(args[0])

        # Execute with JSON input via stdin
        env = os.environ.copy()
        env.update(self.environment)

        result = subprocess.run(  # nosec B603
            args,
            input=json_input,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            check=True,
        )

        return result.stdout

    async def _execute_script_file_with_schema_json(
        self, script_path: str, input_json: str
    ) -> str:
        """Execute script file with ScriptAgentInput JSON directly.

        Args:
            script_path: Path to the script file
            input_json: ScriptAgentInput JSON string

        Returns:
            Script output (stdout)
        """
        # Determine interpreter based on file extension
        interpreter = self._get_interpreter(script_path)

        # Prepare environment with ScriptAgentInput JSON directly
        env = self._env_manager.prepare_environment(input_json, direct_json=True)

        # Execute script with ScriptAgentInput JSON in env AND stdin
        result = subprocess.run(  # nosec B603
            interpreter + [script_path],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            input=input_json,
            check=True,
        )

        return result.stdout.strip()

    async def _execute_inline_script_with_schema_json(
        self, script_content: str, input_json: str
    ) -> str:
        """Execute inline script with ScriptAgentInput JSON via stdin.

        Args:
            script_content: Inline script command to execute
            input_json: ScriptAgentInput JSON string

        Returns:
            Script output (stdout)
        """
        # Execute inline script with JSON input via stdin
        result = subprocess.run(  # nosec B602
            script_content,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            input=input_json,
            check=True,
        )

        return result.stdout.strip()

    def _validate_executable_safety(self, executable: str) -> None:
        """Validate that the executable is safe to run."""
        # Block obvious dangerous commands
        dangerous_commands = {
            "rm",
            "rmdir",
            "del",
            "format",
            "fdisk",
            "mkfs",
            "dd",
            "sudo",
            "su",
            "chmod",
            "chown",
            "passwd",
            "useradd",
            "userdel",
        }

        # Extract just the command name (handle paths)
        cmd_name = Path(executable).name

        if cmd_name in dangerous_commands:
            raise RuntimeError(
                f"Blocked dangerous command: {cmd_name}. "
                f"Script agents cannot execute system modification "
                f"commands."
            )

    def get_agent_type(self) -> str:
        """Return the agent type."""
        return "script"

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
