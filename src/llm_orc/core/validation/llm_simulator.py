"""LLM-based user response generator for test mode simulation."""

import hashlib
import json
from typing import Any

from llm_orc.models.ollama import OllamaModel


class LLMResponseGenerator:
    """Generates contextual user responses using small LLMs."""

    def __init__(
        self,
        model: str = "qwen3:0.6b",
        persona: str = "helpful_user",
        system_prompt: str | None = None,
        response_cache: dict[str, str] | None = None,
    ) -> None:
        """Initialize the LLM response generator.

        Args:
            model: Model name to use (default: qwen3:0.6b)
            persona: Persona type for system prompt
            system_prompt: Optional custom system prompt
            response_cache: Optional cache for deterministic responses
        """
        self.model = model
        self.persona = persona
        self.system_prompt = system_prompt or self._default_persona_prompts()[persona]
        self.response_cache = response_cache or {}
        self.conversation_history: list[dict[str, str]] = []
        self.llm_client = OllamaModel(model_name=model)

    async def generate_response(self, prompt: str, context: dict[str, Any]) -> str:
        """Generate contextual response using LLM.

        Args:
            prompt: User input prompt to respond to
            context: Execution context (previous outputs, etc.)

        Returns:
            Generated response string
        """
        cache_key = self._create_cache_key(prompt, context)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        llm_prompt = self._build_llm_prompt(prompt, context)

        try:
            response = await self._call_llm(llm_prompt)
        except Exception as e:
            raise RuntimeError(f"LLM simulation failed for model {self.model}") from e

        self.response_cache[cache_key] = response

        self.conversation_history.append(
            {
                "prompt": prompt,
                "response": response,
                "context": json.dumps(context),
            }
        )

        return response

    def _default_persona_prompts(self) -> dict[str, str]:
        """Default system prompts for common personas.

        Returns:
            Dictionary mapping persona names to system prompts
        """
        return {
            "helpful_user": (
                "You are simulating a helpful user responding to prompts. "
                "Provide realistic, contextually appropriate responses. "
                "Keep responses concise (1-2 sentences) unless asked for detail."
            ),
            "critical_reviewer": (
                "You are simulating a critical code reviewer. "
                "Point out potential issues, edge cases, and improvements. "
                "Be constructive but thorough."
            ),
            "domain_expert": (
                "You are simulating a domain expert with deep knowledge. "
                "Provide technically accurate, detailed responses. "
                "Reference best practices and potential pitfalls."
            ),
        }

    def _create_cache_key(self, prompt: str, context: dict[str, Any]) -> str:
        """Create deterministic cache key from prompt and context.

        Args:
            prompt: User input prompt
            context: Execution context

        Returns:
            SHA256 hash as hex string
        """
        cache_input = {
            "prompt": prompt,
            "context": context,
            "persona": self.persona,
            "history_length": len(self.conversation_history),
        }

        return hashlib.sha256(
            json.dumps(cache_input, sort_keys=True).encode()
        ).hexdigest()

    def _build_llm_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        """Build LLM prompt with context and persona.

        Args:
            prompt: User input prompt
            context: Execution context

        Returns:
            Formatted prompt for LLM
        """
        context_str = "\n".join(
            f"{k}: {v}" for k, v in context.items() if v is not None
        )
        return f"Context:\n{context_str}\n\nPrompt: {prompt}"

    async def _call_llm(self, prompt: str) -> str:
        """Call local LLM (Ollama) for response generation.

        Args:
            prompt: Formatted prompt for LLM

        Returns:
            Generated response

        Raises:
            RuntimeError: If LLM call fails
        """
        response = await self.llm_client.generate_response(
            message=prompt, role_prompt=self.system_prompt
        )
        return response
