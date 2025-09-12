"""Handler for script agent user input requirements with event emission."""

from collections.abc import Callable, Coroutine
from typing import Any

from llm_orc.core.communication.protocol import MessageProtocol
from llm_orc.visualization.events import EventFactory


class ScriptUserInputHandler:
    """Handles detection and management of script user input requirements."""

    def __init__(
        self, event_emitter: Callable[[Any], Coroutine[Any, Any, None]] | None = None
    ) -> None:
        """Initialize the handler with optional event emitter.

        Args:
            event_emitter: Optional async function to emit events
        """
        self.event_emitter = event_emitter

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
        ensemble_name: str | None = None,
        execution_id: str | None = None,
    ) -> str:
        """Handle user input request from script agent.

        Args:
            input_request: Dictionary containing input request details
            protocol: Communication protocol for message passing
            conversation_id: ID of the conversation
            cli_input_collector: CLI component that collects user input
            ensemble_name: Name of the ensemble being executed
            execution_id: ID of the current execution

        Returns:
            User input as string
        """
        prompt = input_request.get("prompt", "Enter input: ")
        agent_name = input_request.get("agent_name", "script_agent")
        script_path = input_request.get("script_path", "")

        # Emit STREAMING_PAUSED event
        if self.event_emitter:
            paused_event = EventFactory.streaming_paused(
                ensemble_name=ensemble_name or "unknown",
                execution_id=execution_id or "unknown",
                reason="waiting_for_user_input",
            )
            await self.event_emitter(paused_event)

        # Emit USER_INPUT_REQUIRED event
        if self.event_emitter:
            input_required_event = EventFactory.user_input_required(
                agent_name=agent_name,
                ensemble_name=ensemble_name or "unknown",
                execution_id=execution_id or "unknown",
                prompt=prompt,
                script_path=script_path,
            )
            await self.event_emitter(input_required_event)

        # Collect user input
        result = await cli_input_collector.collect_input(prompt)
        user_input = str(result)

        # Emit USER_INPUT_RECEIVED event
        if self.event_emitter:
            input_received_event = EventFactory.user_input_received(
                agent_name=agent_name,
                ensemble_name=ensemble_name or "unknown",
                execution_id=execution_id or "unknown",
                user_input=user_input,
                script_path=script_path,
            )
            await self.event_emitter(input_received_event)

        # Emit STREAMING_RESUMED event
        if self.event_emitter:
            resumed_event = EventFactory.streaming_resumed(
                ensemble_name=ensemble_name or "unknown",
                execution_id=execution_id or "unknown",
                reason="user_input_received",
            )
            await self.event_emitter(resumed_event)

        return user_input
