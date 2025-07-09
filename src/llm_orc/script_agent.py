"""Script-based agent execution for hybrid LLM/script workflows."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class ScriptAgent:
    """Agent that executes scripts/commands as part of an ensemble."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize script agent with configuration."""
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

    async def execute(
        self, input_data: str, context: dict[str, Any] | None = None
    ) -> str:
        """Execute the script/command with input data."""
        if context is None:
            context = {}

        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.environment)

        # Add input data to environment
        env["INPUT_DATA"] = input_data

        # Add context variables to environment
        for key, value in context.items():
            env[f"CONTEXT_{key.upper()}"] = str(value)

        try:
            if self.script:
                # Execute script content
                result = await self._execute_script(self.script, env)
            else:
                # Execute command directly
                result = await self._execute_command(self.command, env)

            return result

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Script agent {self.name} timed out after {self.timeout}s"
            ) from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Script agent {self.name} failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Script agent {self.name} error: {str(e)}") from e

    async def _execute_script(self, script_content: str, env: dict[str, str]) -> str:
        """Execute script content in a temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write("#!/bin/bash\n")
            f.write("set -euo pipefail\n")  # Exit on error, undefined vars
            f.write(script_content)
            script_path = f.name

        try:
            # Make script executable
            os.chmod(script_path, 0o755)

            # Execute script
            result = subprocess.run(
                ["/bin/bash", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                check=True,
            )

            return result.stdout

        finally:
            # Clean up temporary file
            Path(script_path).unlink(missing_ok=True)

    async def _execute_command(self, command: str, env: dict[str, str]) -> str:
        """Execute command directly."""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
            check=True,
        )

        return result.stdout

    def get_agent_type(self) -> str:
        """Return the agent type."""
        return "script"
