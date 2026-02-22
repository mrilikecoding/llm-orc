"""Validation framework data models."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ValidationLayer(StrEnum):
    """Validation layer types."""

    STRUCTURAL = "structural"
    SCHEMA = "schema"
    BEHAVIORAL = "behavioral"
    QUANTITATIVE = "quantitative"
    SEMANTIC = "semantic"


class StructuralValidationConfig(BaseModel):
    """Structural validation configuration."""

    required_agents: list[str] = Field(default_factory=list)
    max_execution_time: float | None = None
    min_execution_time: float | None = None


class SchemaValidationConfig(BaseModel):
    """Schema validation for agent outputs."""

    agent: str
    output_schema: str
    required_fields: list[str] = Field(default_factory=list)


class BehavioralAssertion(BaseModel):
    """Behavioral assertion with Python expression."""

    name: str
    description: str
    assertion: str


class QuantitativeMetric(BaseModel):
    """Quantitative metric with optional threshold."""

    metric: str
    threshold: str | None = None
    description: str
    record: bool = False


class SemanticValidationConfig(BaseModel):
    """Semantic validation using LLM-as-judge."""

    enabled: bool = False
    validator_model: str | None = None
    criteria: list[str] = Field(default_factory=list)


class ValidationConfig(BaseModel):
    """Complete validation configuration for ensemble."""

    structural: StructuralValidationConfig | None = None
    schema_validations: list[SchemaValidationConfig] = Field(default_factory=list)
    behavioral: list[BehavioralAssertion] = Field(default_factory=list)
    quantitative: list[QuantitativeMetric] = Field(default_factory=list)
    semantic: SemanticValidationConfig | None = None
    strict_schema_validation: bool = False


class LLMSimulationConfig(BaseModel):
    """LLM simulation configuration for agent."""

    agent: str
    model: str = "qwen3:0.6b"
    persona: str = "helpful_user"
    cached_responses: dict[str, str] = Field(default_factory=dict)


class TestModeConfig(BaseModel):
    """Test mode configuration for ensemble."""

    enabled: bool = False
    llm_simulation: list[LLMSimulationConfig] = Field(default_factory=list)


class EnsembleExecutionResult(BaseModel):
    """Results from ensemble execution."""

    ensemble_name: str
    execution_order: list[str]
    agent_outputs: dict[str, dict[str, Any]]
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationLayerResult(BaseModel):
    """Results from a single validation layer."""

    layer: ValidationLayer
    passed: bool
    errors: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Complete validation results for ensemble."""

    ensemble_name: str
    passed: bool
    results: dict[str, ValidationLayerResult | None]
    timestamp: datetime = Field(default_factory=datetime.now)
