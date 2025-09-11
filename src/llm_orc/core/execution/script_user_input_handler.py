"""Handler for script agent user input requirements."""

from typing import Any

from llm_orc.core.communication.protocol import MessageProtocol


class ScriptUserInputHandler:
    """Handles detection and management of script user input requirements."""

    def requires_user_input(self, script_ref_or_content: str) -> bool:
        """Check if a script reference or content requires user input.

        Args:
            script_ref_or_content: Either a script reference path or script content

        Returns:
            True if the script requires user input, False otherwise
        """
        # Check if it's a reference to get_user_input.py
        if "get_user_input.py" in script_ref_or_content:
            return True

        # Check if script content contains input() function calls
        if "input(" in script_ref_or_content:
            return True

        return False

    def ensemble_requires_user_input(self, ensemble_config: Any) -> bool:
        """Check if an ensemble configuration contains agents that require user input.

        Args:
            ensemble_config: Ensemble configuration object with agents list

        Returns:
            True if any agent in the ensemble requires user input, False otherwise
        """
        if not hasattr(ensemble_config, "agents") or not ensemble_config.agents:
            return False

        for agent_config in ensemble_config.agents:
            if not isinstance(agent_config, dict):
                continue

            # Check if this is a script agent
            if agent_config.get("type") != "script":
                continue

            # Check the script reference or content
            script_ref = agent_config.get("script", "")
            if self.requires_user_input(script_ref):
                return True

        return False

    async def handle_input_request(
        self,
        input_request: dict[str, Any],
        protocol: MessageProtocol,
        conversation_id: str,
        cli_input_collector: Any,
    ) -> str:
        """Handle user input request from script agent.

        Args:
            input_request: Dictionary containing input request details
            protocol: Communication protocol for message passing
            conversation_id: ID of the conversation
            cli_input_collector: CLI component that collects user input

        Returns:
            User input as string
        """
        prompt = input_request.get("prompt", "Enter input: ")
        result = await cli_input_collector.collect_input(prompt)
        return str(result)
