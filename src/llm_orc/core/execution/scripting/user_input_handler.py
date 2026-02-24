"""Handler for script agent user input requirements."""

from typing import Any

from llm_orc.core.validation import LLMResponseGenerator


class ScriptUserInputHandler:
    """Handles detection and management of script user input requirements.

    Supports two modes:
    - Interactive mode (test_mode=False): Uses real stdin for user input
    - Test mode (test_mode=True): Uses LLM simulation for automated testing
    """

    def __init__(
        self,
        test_mode: bool = False,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the handler with optional test mode.

        Args:
            test_mode: If True, use LLM simulation for user input
            llm_config: LLM simulation configuration per agent
        """
        self.test_mode = test_mode
        self.llm_simulators: dict[str, LLMResponseGenerator] = {}

        if test_mode and llm_config:
            self._initialize_simulators(llm_config)

    def requires_user_input(self, script_ref_or_content: str) -> bool:
        """Check if a script reference or content requires user input.

        Args:
            script_ref_or_content: Either a script reference path or script content

        Returns:
            True if the script requires user input, False otherwise
        """
        interactive_scripts = ("get_user_input.py", "confirm_action.py")
        for script_name in interactive_scripts:
            if script_name in script_ref_or_content:
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

        from llm_orc.schemas.agent_config import ScriptAgentConfig

        for agent_config in ensemble_config.agents:
            if not isinstance(agent_config, ScriptAgentConfig):
                continue

            if self.requires_user_input(agent_config.script):
                return True

        return False

    def _initialize_simulators(self, llm_config: dict[str, Any]) -> None:
        """Initialize LLM simulators from configuration.

        Args:
            llm_config: Dictionary mapping agent names to LLM configs
        """
        for agent_name, agent_llm_config in llm_config.items():
            model = agent_llm_config.get("model", "qwen3:0.6b")
            persona = agent_llm_config.get("persona", "helpful_user")
            cached_responses = agent_llm_config.get("cached_responses", {})

            self.llm_simulators[agent_name] = LLMResponseGenerator(
                model=model,
                persona=persona,
                response_cache=cached_responses,
            )

    async def get_user_input(
        self, agent_name: str, prompt: str, context: dict[str, Any]
    ) -> str:
        """Get user input - either from LLM simulation or real stdin.

        Args:
            agent_name: Name of the agent requesting input
            prompt: Prompt to display to user
            context: Execution context for LLM simulation

        Returns:
            User input as string

        Raises:
            RuntimeError: If test mode enabled but no simulator configured
            NotImplementedError: If interactive mode (not implemented in this method)
        """
        if self.test_mode:
            # Test mode - use LLM simulation
            if agent_name not in self.llm_simulators:
                raise RuntimeError(
                    f"No LLM simulator configured for agent: {agent_name}"
                )

            simulator = self.llm_simulators[agent_name]
            return await simulator.generate_response(prompt, context)

        # Interactive mode - use real stdin
        # This is not implemented here as it requires proper terminal handling
        raise NotImplementedError(
            "Interactive mode should use handle_input_request method"
        )

    async def handle_input_request(
        self,
        input_request: dict[str, Any],
        conversation_id: str,
        cli_input_collector: Any,
    ) -> str:
        """Handle user input request from script agent.

        Args:
            input_request: Dictionary containing input request details
            conversation_id: ID of the conversation
            cli_input_collector: CLI component that collects user input

        Returns:
            User input as string
        """
        _ = conversation_id  # reserved for future per-conversation state
        prompt = input_request.get("prompt", "Enter input: ")

        result = await cli_input_collector.collect_input(prompt)
        return str(result)
