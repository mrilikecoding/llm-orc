"""Schemas for conversational agent system (ADR-005)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ConversationConfig(BaseModel):
    """Configuration for agent conversation behavior."""

    max_turns: int = 1
    state_key: str | None = None
    triggers_conversation: bool = False


class ConversationalDependency(BaseModel):
    """Dependency that can trigger multiple times based on conditions."""

    agent_name: str
    condition: str | None = None
    max_executions: int = 1
    requires_all: bool = True


class ConversationalAgent(BaseModel):
    """Agent with conversation support for both script and LLM agents."""

    name: str
    type: str | None = None  # Optional explicit type for backward compatibility
    script: str | None = None  # Script agents have script path
    model_profile: str | None = None  # LLM agents have model profile
    prompt: str | None = None  # LLM agents may have prompts
    config: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[ConversationalDependency] = Field(default_factory=list)
    conversation: ConversationConfig | None = None

    @field_validator("script", "model_profile")
    @classmethod
    def validate_agent_type_fields(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        """Validate that agent has either script or model_profile, not both."""
        values = info.data if hasattr(info, "data") else {}

        script = values.get("script") if info.field_name != "script" else v
        model_profile = (
            values.get("model_profile") if info.field_name != "model_profile" else v
        )

        # Only validate when we have all required fields
        if info.field_name == "model_profile":  # Validate on the last field
            if not script and not model_profile:
                raise ValueError(
                    "Agent must have either 'script' or 'model_profile' field"
                )
            if script and model_profile:
                raise ValueError(
                    "Agent cannot have both 'script' and 'model_profile' fields"
                )

        return v


class ConversationLimits(BaseModel):
    """Global limits for conversation execution."""

    max_total_turns: int = 20
    max_agent_executions: dict[str, int] = Field(default_factory=dict)
    timeout_seconds: int = 300
    turn_timeout_seconds: int = 60  # Per-turn timeout
    parallel_agent_limit: int = 5  # Max agents per turn


class ConversationTurn(BaseModel):
    """Record of a single conversation turn."""

    turn_number: int
    agent_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    execution_time: float
    timestamp: datetime


class ConversationState(BaseModel):
    """Persistent state across conversation turns."""

    turn_count: int = 0
    agent_execution_count: dict[str, int] = Field(default_factory=dict)
    accumulated_context: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

    def should_execute_agent(self, agent: ConversationalAgent) -> bool:
        """Check if agent should execute based on conversation limits."""
        current_executions = self.agent_execution_count.get(agent.name, 0)
        max_executions = agent.conversation.max_turns if agent.conversation else 1

        return current_executions < max_executions

    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate dependency condition against current state safely."""
        if not condition:
            return True

        # Safe evaluation context with restricted builtins and safe functions
        safe_context = {
            "turn_count": self.turn_count,
            "context": self.accumulated_context,
            "history": self.conversation_history,
            "len": len,  # Allow len() function
            "__builtins__": {},  # Restrict dangerous builtins
        }

        try:
            # Use eval with restricted context for safe evaluation
            # nosec B307: Eval is used with restricted context for condition evaluation
            return bool(eval(condition, safe_context))  # nosec B307
        except Exception:
            # If evaluation fails, default to False
            return False

    def record_agent_turn(
        self,
        agent: ConversationalAgent,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        execution_time: float,
    ) -> None:
        """Record a complete agent turn, updating all state atomically."""
        from datetime import datetime

        # Increment turn count
        self.turn_count += 1

        # Update agent execution count
        current_count = self.agent_execution_count.get(agent.name, 0)
        self.agent_execution_count[agent.name] = current_count + 1

        # Update accumulated context using state_key if available
        context_key = (
            agent.conversation.state_key
            if agent.conversation and agent.conversation.state_key
            else agent.name
        )
        self.accumulated_context[context_key] = output_data

        # Create and store conversation turn
        turn = ConversationTurn(
            turn_number=self.turn_count,
            agent_name=agent.name,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            timestamp=datetime.now(),
        )
        self.conversation_history.append(turn)

    def can_continue_conversation(self, limits: ConversationLimits) -> bool:
        """Check if conversation can continue based on global limits."""
        # Check total turn limit
        if self.turn_count >= limits.max_total_turns:
            return False

        # Check per-agent execution limits
        for agent_name, max_executions in limits.max_agent_executions.items():
            current_executions = self.agent_execution_count.get(agent_name, 0)
            if current_executions >= max_executions:
                return False

        return True

    def get_recent_turns(self, count: int) -> list[ConversationTurn]:
        """Get the most recent N conversation turns."""
        if count <= 0:
            return []

        return self.conversation_history[-count:]

    def get_agent_last_output(self, agent_name: str) -> dict[str, Any] | None:
        """Get the most recent output from a specific agent."""
        # Search backwards through history for most recent turn from this agent
        for turn in reversed(self.conversation_history):
            if turn.agent_name == agent_name:
                return turn.output_data

        return None

    def has_agent_executed(self, agent_name: str) -> bool:
        """Check if an agent has executed at least once."""
        return self.agent_execution_count.get(agent_name, 0) > 0

    def reset_conversation(self) -> None:
        """Reset all conversation state to initial values."""
        self.turn_count = 0
        self.agent_execution_count.clear()
        self.accumulated_context.clear()
        self.conversation_history.clear()


class ConversationalEnsemble(BaseModel):
    """Ensemble supporting multi-turn conversations."""

    name: str
    agents: list[ConversationalAgent]
    conversation_limits: ConversationLimits


class ConversationResult(BaseModel):
    """Result of a conversational ensemble execution."""

    final_state: dict[str, Any]
    conversation_history: list[ConversationTurn]
    turn_count: int
    completion_reason: str
