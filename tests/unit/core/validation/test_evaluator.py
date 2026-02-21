"""Unit tests for ValidationEvaluator."""

from typing import Any

import pytest

from llm_orc.core.validation.evaluator import ValidationEvaluator
from llm_orc.core.validation.models import (
    BehavioralAssertion,
    EnsembleExecutionResult,
    QuantitativeMetric,
    SchemaValidationConfig,
    SemanticValidationConfig,
    StructuralValidationConfig,
)


class TestValidationEvaluator:
    """Test ValidationEvaluator class."""

    @pytest.mark.asyncio
    async def test_structural_validation_none_config(self) -> None:
        """Test structural validation with None config returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )

        result = await evaluator._evaluate_structural(results, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_structural_validation_missing_required_agents(self) -> None:
        """Test structural validation detects missing required agents."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        config = StructuralValidationConfig(required_agents=["agent1", "agent2"])

        result = await evaluator._evaluate_structural(results, config)

        assert result is not None
        assert result.passed is False
        assert any("Missing required agents" in error for error in result.errors)
        assert "agent2" in result.errors[0]

    @pytest.mark.asyncio
    async def test_structural_validation_exceeds_max_execution_time(self) -> None:
        """Test structural validation detects execution time exceeding max."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=10.0,
        )
        config = StructuralValidationConfig(max_execution_time=5.0)

        result = await evaluator._evaluate_structural(results, config)

        assert result is not None
        assert result.passed is False
        assert any("exceeds max" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_structural_validation_below_min_execution_time(self) -> None:
        """Test structural validation detects execution time below min."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        config = StructuralValidationConfig(min_execution_time=5.0)

        result = await evaluator._evaluate_structural(results, config)

        assert result is not None
        assert result.passed is False
        assert any("below min" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_schema_validation_empty_configs(self) -> None:
        """Test schema validation with empty configs returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )

        result = await evaluator._evaluate_schema(results, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_schema_validation_agent_not_found(self) -> None:
        """Test schema validation handles agent not in results."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        configs = [
            SchemaValidationConfig(
                agent="agent2",
                output_schema="test_schema",
                required_fields=["field1"],
            )
        ]

        result = await evaluator._evaluate_schema(results, configs)

        assert result is not None
        assert result.passed is False
        assert any("not found" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_schema_validation_missing_required_fields(self) -> None:
        """Test schema validation detects missing required fields."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        configs = [
            SchemaValidationConfig(
                agent="agent1",
                output_schema="test_schema",
                required_fields=["field1", "field2"],
            )
        ]

        result = await evaluator._evaluate_schema(results, configs)

        assert result is not None
        assert result.passed is False
        assert any("missing required fields" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_behavioral_validation_empty_assertions(self) -> None:
        """Test behavioral validation with empty assertions returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )

        result = await evaluator._evaluate_behavioral(results, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_behavioral_validation_assertion_fails(self) -> None:
        """Test behavioral validation with failing assertion."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        assertions = [
            BehavioralAssertion(
                name="test_assertion",
                description="Test that should fail",
                assertion="len(execution_order) > 5",
            )
        ]

        result = await evaluator._evaluate_behavioral(results, assertions)

        assert result is not None
        assert result.passed is False
        assert any("failed" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_behavioral_validation_assertion_exception(self) -> None:
        """Test behavioral validation handles assertion exceptions."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        assertions = [
            BehavioralAssertion(
                name="bad_assertion",
                description="Assertion with error",
                assertion="undefined_variable > 0",
            )
        ]

        result = await evaluator._evaluate_behavioral(results, assertions)

        assert result is not None
        assert result.passed is False
        assert any("raised exception" in error for error in result.errors)

    def test_evaluate_assertion_exception(self) -> None:
        """Test assertion evaluation raises RuntimeError on exception."""
        evaluator = ValidationEvaluator()

        with pytest.raises(RuntimeError, match="Assertion evaluation failed"):
            evaluator._evaluate_assertion("1 / 0", {})

    @pytest.mark.asyncio
    async def test_quantitative_validation_empty_metrics(self) -> None:
        """Test quantitative validation with empty metrics returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )

        result = await evaluator._evaluate_quantitative(results, [])

        assert result is None

    @pytest.mark.asyncio
    async def test_quantitative_validation_metric_not_found(self) -> None:
        """Test quantitative validation handles unknown metric."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        metrics = [
            QuantitativeMetric(
                metric="unknown_metric",
                description="Unknown metric",
                threshold="> 0",
            )
        ]

        result = await evaluator._evaluate_quantitative(results, metrics)

        assert result is not None
        assert result.passed is False
        assert any("not found" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_quantitative_validation_threshold_fails(self) -> None:
        """Test quantitative validation detects threshold failure."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=10.0,
        )
        metrics = [
            QuantitativeMetric(
                metric="execution_time",
                description="Execution time",
                threshold="< 5.0",
            )
        ]

        result = await evaluator._evaluate_quantitative(results, metrics)

        assert result is not None
        assert result.passed is False
        assert any("failed threshold" in error for error in result.errors)

    def test_evaluate_threshold_invalid_value(self) -> None:
        """Test threshold evaluation with invalid threshold value."""
        evaluator = ValidationEvaluator()

        with pytest.raises(ValueError, match="Invalid threshold value"):
            evaluator._evaluate_threshold(5.0, "> not_a_number")

    def test_evaluate_threshold_invalid_format(self) -> None:
        """Test threshold evaluation with invalid threshold format."""
        evaluator = ValidationEvaluator()

        with pytest.raises(ValueError, match="Invalid threshold format"):
            evaluator._evaluate_threshold(5.0, "invalid")

    @pytest.mark.asyncio
    async def test_semantic_validation_none_config(self) -> None:
        """Test semantic validation with None config returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )

        result = await evaluator._evaluate_semantic(results, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_validation_disabled(self) -> None:
        """Test semantic validation with disabled config returns None."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        config = SemanticValidationConfig(enabled=False)

        result = await evaluator._evaluate_semantic(results, config)

        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_validation_no_validator_model(self) -> None:
        """Test semantic validation with no validator model specified."""
        evaluator = ValidationEvaluator()
        results = EnsembleExecutionResult(
            ensemble_name="test",
            execution_order=["agent1"],
            agent_outputs={"agent1": {"result": "test"}},
            execution_time=1.0,
        )
        config = SemanticValidationConfig(
            enabled=True, validator_model=None, criteria=["test"]
        )

        result = await evaluator._evaluate_semantic(results, config)

        assert result is not None
        assert result.passed is False
        assert any("no validator model" in error for error in result.errors)


class TestEvalSandboxSecurity:
    """Security tests for _evaluate_assertion sandbox.

    These tests verify that LLM output in the evaluation context cannot
    bypass the restricted-builtins sandbox.
    """

    def test_import_builtin_not_available(self) -> None:
        """__import__ is not available in restricted globals."""
        evaluator = ValidationEvaluator()
        with pytest.raises(RuntimeError, match="Assertion evaluation failed"):
            evaluator._evaluate_assertion("__import__('os')", {})

    def test_builtins_key_in_context_does_not_override_sandbox(self) -> None:
        """__builtins__ key in LLM-provided context does not restore dangerous builtins.

        Before the fix, `restricted_globals.update(context)` would merge a
        malicious `__builtins__` dict into globals, bypassing the sandbox.
        After the fix, context is passed as locals; Python ignores __builtins__
        in locals when resolving built-in names.
        """
        evaluator = ValidationEvaluator()
        malicious_context: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
        with pytest.raises(RuntimeError, match="Assertion evaluation failed"):
            evaluator._evaluate_assertion("__import__('os')", malicious_context)

    def test_context_key_named_len_shadows_builtin(self) -> None:
        """A context key named 'len' shadows the builtin in the eval's local scope.

        This is not a sandbox escape — it causes a TypeError — but it confirms
        that context keys take precedence over builtins in the locals namespace.
        """
        evaluator = ValidationEvaluator()
        context: dict[str, Any] = {"len": "not_a_function"}
        with pytest.raises(RuntimeError, match="Assertion evaluation failed"):
            evaluator._evaluate_assertion("len([1, 2, 3])", context)

    def test_context_key_named_sorted_shadows_builtin(self) -> None:
        """A context key named 'sorted' shadows the builtin in local scope."""
        evaluator = ValidationEvaluator()
        context: dict[str, Any] = {"sorted": None}
        with pytest.raises(RuntimeError, match="Assertion evaluation failed"):
            evaluator._evaluate_assertion("sorted([3, 1, 2])[0] == 1", context)

    def test_valid_assertion_with_context_still_works(self) -> None:
        """Legitimate context-based assertions work after the locals fix."""
        evaluator = ValidationEvaluator()
        context = {"execution_order": ["agent1", "agent2"], "agent_outputs": {}}
        result = evaluator._evaluate_assertion("len(execution_order) == 2", context)
        assert result is True
