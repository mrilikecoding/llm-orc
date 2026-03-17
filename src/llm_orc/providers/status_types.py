"""Pydantic models for provider status and ensemble runnability."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    """Status of an agent's runnability."""

    AVAILABLE = "available"
    MISSING_PROFILE = "missing_profile"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"


class OllamaProviderStatus(BaseModel):
    """Status of the Ollama provider."""

    available: bool
    models: list[str] = Field(default_factory=list)
    model_count: int = 0
    reason: str = ""


class CloudProviderStatus(BaseModel):
    """Status of a cloud provider (API key check only)."""

    available: bool
    reason: str = ""


class EndpointStatus(BaseModel):
    """Status of a single OpenAI-compatible endpoint."""

    base_url: str
    available: bool
    models: list[str] = Field(default_factory=list)
    profiles: list[str] = Field(default_factory=list)
    reason: str = ""


class OpenAICompatibleStatus(BaseModel):
    """Status of the OpenAI-compatible provider (multi-endpoint)."""

    available: bool
    endpoints: list[EndpointStatus] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    model_count: int = 0


class AgentRunnability(BaseModel):
    """Runnability status of a single agent."""

    name: str
    profile: str
    provider: str = ""
    status: AgentStatus = AgentStatus.AVAILABLE
    alternatives: list[str] = Field(default_factory=list)


class EnsembleRunnability(BaseModel):
    """Runnability status of an ensemble."""

    ensemble: str
    runnable: bool
    agents: list[AgentRunnability]
