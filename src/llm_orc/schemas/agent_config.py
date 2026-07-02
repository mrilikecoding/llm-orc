"""Pydantic agent config models (ADR-012).

Discriminated union for agent configurations: LLM, Script, and
(future) Ensemble agent types. Replaces dict[str, Any] agent configs
with validated, typed models.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentConfig(BaseModel):
    """Shared fields for all agent types."""

    model_config = ConfigDict(extra="forbid")

    name: str
    depends_on: list[str | dict[str, Any]] = Field(default_factory=list)
    fan_out: bool = False
    input_key: str | None = None

    # Guard predicate (control-flow primitive): the node is skipped at
    # execution time when this evaluates false against upstream results.
    # e.g. "${gate.ok}" or "${gate.ok} == false". None means always run.
    when: str | None = None

    timeout_seconds: int | None = None

    # Fan-out instance metadata (runtime only, set by FanOutExpander)
    fan_out_chunk: Any | None = Field(default=None, exclude=True)
    fan_out_index: int | None = Field(default=None, exclude=True)
    fan_out_total: int | None = Field(default=None, exclude=True)
    fan_out_original: str | None = Field(default=None, exclude=True)


class LlmAgentConfig(BaseAgentConfig):
    """Config for an LLM agent — sends input to a language model."""

    model_profile: str | None = None
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    options: dict[str, Any] | None = None
    ollama_format: str | dict[str, Any] | None = None
    output_format: str | None = None
    fallback_model_profile: str | None = None

    @model_validator(mode="after")
    def validate_model_source(self) -> "LlmAgentConfig":
        """Enforce Invariants 2 and 3.

        - Invariant 2: model without provider is invalid.
        - Invariant 3: model_profile XOR (model + provider).
        """
        has_profile = self.model_profile is not None
        has_model = self.model is not None
        has_provider = self.provider is not None

        if has_profile and has_model:
            msg = (
                "model_profile and model are mutually exclusive — "
                "use one or the other, not both"
            )
            raise ValueError(msg)

        if has_model and not has_provider:
            msg = "provider is required when model is specified"
            raise ValueError(msg)

        if has_provider and not has_model:
            msg = "model is required when provider is specified"
            raise ValueError(msg)

        if not has_profile and not has_model:
            msg = "LLM agent requires either model_profile or model + provider"
            raise ValueError(msg)

        return self


class ScriptAgentConfig(BaseAgentConfig):
    """Config for a script agent — executes a script subprocess."""

    script: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class EnsembleAgentConfig(BaseAgentConfig):
    """Config for an ensemble agent — recursively executes another ensemble."""

    ensemble: str  # Static ensemble reference, resolved at load time


class LoopSpec(BaseModel):
    """The body + termination policy of a loop node (control-flow primitive)."""

    model_config = ConfigDict(extra="forbid")

    body: str  # ensemble run once per iteration
    until: str  # predicate over the body output; stop when true
    max_iterations: int = Field(gt=0)  # mandatory hard bound; no unbounded loops
    carry: str | None = None  # body-output field fed to the next iteration


class LoopAgentConfig(BaseAgentConfig):
    """Config for a loop agent — re-runs a body ensemble under a bound."""

    loop: LoopSpec


class DynamicDispatchAgentConfig(BaseAgentConfig):
    """Config for a dynamic-dispatch agent — resolves and runs an ensemble
    chosen at runtime from an upstream stage's output.

    Unlike EnsembleAgentConfig's load-time `ensemble` reference, `dispatch` is
    a `${ref}` template resolved at execution time against upstream results to
    the target ensemble's name.
    """

    dispatch: str

    # Runtime-only: the target ensemble name resolved at the phase layer from
    # `dispatch` against results_dict (set by DispatchResolver, read by the
    # runner). Excluded from serialization, like the fan-out runtime fields.
    dispatch_resolved: str | None = Field(default=None, exclude=True)


AgentConfig = (
    LlmAgentConfig
    | ScriptAgentConfig
    | EnsembleAgentConfig
    | LoopAgentConfig
    | DynamicDispatchAgentConfig
)


def parse_agent_config(data: dict[str, Any]) -> AgentConfig:
    """Parse a raw dict into the correct AgentConfig subtype.

    Discriminates by key presence:
    - 'script' -> ScriptAgentConfig
    - 'loop' -> LoopAgentConfig
    - 'dispatch' -> DynamicDispatchAgentConfig
    - 'ensemble' -> EnsembleAgentConfig
    - 'model_profile' or 'model' -> LlmAgentConfig
    """
    if "script" in data:
        return ScriptAgentConfig(**data)
    if "loop" in data:
        return LoopAgentConfig(**data)
    if "dispatch" in data:
        return DynamicDispatchAgentConfig(**data)
    if "ensemble" in data:
        return EnsembleAgentConfig(**data)
    if "model_profile" in data or "model" in data:
        return LlmAgentConfig(**data)
    # Try LlmAgentConfig to get a useful validation error
    return LlmAgentConfig(**data)
