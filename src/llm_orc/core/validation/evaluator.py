"""Validation evaluator for ensemble execution results."""

import operator
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_orc.core.validation.llm_simulator import LLMResponseGenerator
from llm_orc.core.validation.models import (
    BehavioralAssertion,
    EnsembleExecutionResult,
    QuantitativeMetric,
    SchemaValidationConfig,
    SemanticValidationConfig,
    StructuralValidationConfig,
    ValidationConfig,
    ValidationLayer,
    ValidationLayerResult,
    ValidationResult,
)


class ValidationEvaluator:
    """Evaluates validation criteria after ensemble execution."""

    async def evaluate(
        self,
        ensemble_name: str,
        results: EnsembleExecutionResult,
        validation_config: ValidationConfig,
    ) -> ValidationResult:
        """Evaluate all validation criteria.

        Args:
            ensemble_name: Name of the ensemble being validated
            results: Execution results to validate
            validation_config: Validation criteria configuration

        Returns:
            Complete validation result with pass/fail status
        """
        validation_results: dict[str, ValidationLayerResult | None] = {
            "structural": await self._evaluate_structural(
                results, validation_config.structural
            ),
            "schema": await self._evaluate_schema(
                results, validation_config.schema_validations
            ),
            "behavioral": await self._evaluate_behavioral(
                results, validation_config.behavioral
            ),
            "quantitative": await self._evaluate_quantitative(
                results, validation_config.quantitative
            ),
            "semantic": await self._evaluate_semantic(
                results, validation_config.semantic
            ),
        }

        return ValidationResult(
            ensemble_name=ensemble_name,
            passed=all(r.passed for r in validation_results.values() if r is not None),
            results=validation_results,
            timestamp=datetime.now(),
        )

    async def _evaluate_structural(
        self,
        results: EnsembleExecutionResult,
        config: StructuralValidationConfig | None,
    ) -> ValidationLayerResult | None:
        """Evaluate structural validation criteria.

        Args:
            results: Execution results
            config: Structural validation configuration

        Returns:
            Validation layer result or None if not configured
        """
        if config is None:
            return None

        errors: list[str] = []
        details: dict[str, Any] = {}

        if config.required_agents:
            missing_agents = set(config.required_agents) - set(results.execution_order)
            if missing_agents:
                errors.append(
                    f"Missing required agents: {', '.join(sorted(missing_agents))}"
                )
            details["required_agents"] = list(config.required_agents)
            details["executed_agents"] = results.execution_order

        if config.max_execution_time is not None:
            if results.execution_time > config.max_execution_time:
                errors.append(
                    f"Execution time {results.execution_time:.2f}s "
                    f"exceeds max {config.max_execution_time:.2f}s"
                )
            details["max_execution_time"] = config.max_execution_time

        if config.min_execution_time is not None:
            if results.execution_time < config.min_execution_time:
                errors.append(
                    f"Execution time {results.execution_time:.2f}s "
                    f"below min {config.min_execution_time:.2f}s"
                )
            details["min_execution_time"] = config.min_execution_time

        details["actual_execution_time"] = results.execution_time

        return ValidationLayerResult(
            layer=ValidationLayer.STRUCTURAL,
            passed=len(errors) == 0,
            errors=errors,
            details=details,
        )

    async def _evaluate_schema(
        self,
        results: EnsembleExecutionResult,
        configs: list[SchemaValidationConfig],
    ) -> ValidationLayerResult | None:
        """Evaluate schema validation criteria.

        Args:
            results: Execution results
            configs: Schema validation configurations

        Returns:
            Validation layer result or None if not configured
        """
        if not configs:
            return None

        errors: list[str] = []
        details: dict[str, Any] = {}

        for config in configs:
            agent_output = results.agent_outputs.get(config.agent)

            if agent_output is None:
                errors.append(f"Agent {config.agent} not found in execution results")
                continue

            missing_fields = [
                field for field in config.required_fields if field not in agent_output
            ]

            if missing_fields:
                errors.append(
                    f"Agent {config.agent} missing required fields: "
                    f"{', '.join(missing_fields)}"
                )

            details[config.agent] = {
                "output_schema": config.output_schema,
                "required_fields": config.required_fields,
                "present_fields": list(agent_output.keys()),
            }

        return ValidationLayerResult(
            layer=ValidationLayer.SCHEMA,
            passed=len(errors) == 0,
            errors=errors,
            details=details,
        )

    async def _evaluate_behavioral(
        self,
        results: EnsembleExecutionResult,
        assertions: list[BehavioralAssertion],
    ) -> ValidationLayerResult | None:
        """Evaluate behavioral validation criteria.

        Args:
            results: Execution results
            assertions: Behavioral assertions to evaluate

        Returns:
            Validation layer result or None if not configured
        """
        if not assertions:
            return None

        errors: list[str] = []
        details: dict[str, Any] = {}

        execution_context = {
            "execution_order": results.execution_order,
            "agent_outputs": results.agent_outputs,
            "results": results.agent_outputs,  # Alias for backward compatibility
            "execution_time": results.execution_time,
            "ensemble_name": results.ensemble_name,
            "Path": Path,
        }

        for assertion in assertions:
            try:
                result = self._evaluate_assertion(
                    assertion.assertion, execution_context
                )
                details[assertion.name] = {
                    "description": assertion.description,
                    "passed": result,
                    "assertion": assertion.assertion,
                }

                if not result:
                    errors.append(
                        f"Assertion '{assertion.name}' failed: {assertion.description}"
                    )
            except Exception as e:
                errors.append(f"Assertion '{assertion.name}' raised exception: {e!s}")
                details[assertion.name] = {
                    "description": assertion.description,
                    "error": str(e),
                    "assertion": assertion.assertion,
                }

        return ValidationLayerResult(
            layer=ValidationLayer.BEHAVIORAL,
            passed=len(errors) == 0,
            errors=errors,
            details=details,
        )

    def _evaluate_assertion(self, assertion: str, context: dict[str, Any]) -> bool:
        """Evaluate a Python assertion expression safely.

        Args:
            assertion: Python expression to evaluate
            context: Execution context for evaluation

        Returns:
            Boolean result of assertion

        Raises:
            Exception: If assertion evaluation fails
        """
        restricted_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "all": all,
                "any": any,
                "sum": sum,
                "min": min,
                "max": max,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "reversed": reversed,
            }
        }

        restricted_globals.update(context)

        try:
            result: bool = bool(eval(assertion, restricted_globals, {}))
            return result
        except Exception as e:
            raise RuntimeError(f"Assertion evaluation failed: {assertion}") from e

    async def _evaluate_quantitative(
        self,
        results: EnsembleExecutionResult,
        metrics: list[QuantitativeMetric],
    ) -> ValidationLayerResult | None:
        """Evaluate quantitative validation criteria.

        Args:
            results: Execution results
            metrics: Quantitative metrics to evaluate

        Returns:
            Validation layer result or None if not configured
        """
        if not metrics:
            return None

        errors: list[str] = []
        details: dict[str, Any] = {}

        metric_values = {
            "execution_time": results.execution_time,
            "agent_count": len(results.execution_order),
        }

        for metric in metrics:
            metric_value = metric_values.get(metric.metric)

            if metric_value is None:
                errors.append(f"Metric '{metric.metric}' not found in results")
                continue

            details[metric.metric] = {
                "value": metric_value,
                "threshold": metric.threshold,
                "description": metric.description,
            }

            if metric.threshold:
                passed = self._evaluate_threshold(metric_value, metric.threshold)
                details[metric.metric]["passed"] = passed

                if not passed:
                    errors.append(
                        f"Metric '{metric.metric}' failed threshold: "
                        f"{metric_value} not {metric.threshold}"
                    )

        return ValidationLayerResult(
            layer=ValidationLayer.QUANTITATIVE,
            passed=len(errors) == 0,
            errors=errors,
            details=details,
        )

    def _evaluate_threshold(self, value: float, threshold: str) -> bool:
        """Evaluate if value meets threshold criteria.

        Args:
            value: Metric value to check
            threshold: Threshold expression (e.g., "< 30", ">= 0.5")

        Returns:
            True if threshold is met, False otherwise
        """
        threshold = threshold.strip()

        ops = {
            ">=": operator.ge,
            "<=": operator.le,
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
        }

        for op_str, op_func in ops.items():
            if threshold.startswith(op_str):
                try:
                    threshold_value = float(threshold[len(op_str) :].strip())
                    result: bool = bool(op_func(value, threshold_value))
                    return result
                except ValueError as e:
                    raise ValueError(f"Invalid threshold value: {threshold}") from e

        raise ValueError(f"Invalid threshold format: {threshold}")

    async def _evaluate_semantic(
        self,
        results: EnsembleExecutionResult,
        config: SemanticValidationConfig | None,
    ) -> ValidationLayerResult | None:
        """Evaluate semantic validation criteria using LLM-as-judge.

        Args:
            results: Execution results
            config: Semantic validation configuration

        Returns:
            Validation layer result or None if not configured
        """
        if config is None or not config.enabled:
            return None

        errors: list[str] = []
        details: dict[str, Any] = {}

        if not config.validator_model:
            errors.append(
                "Semantic validation enabled but no validator model specified"
            )
            return ValidationLayerResult(
                layer=ValidationLayer.SEMANTIC,
                passed=False,
                errors=errors,
                details=details,
            )

        try:
            validator = LLMResponseGenerator(
                model=config.validator_model,
                persona="critical_reviewer",
            )

            outputs_summary = "\n".join(
                f"{agent}: {output}" for agent, output in results.agent_outputs.items()
            )

            for criterion in config.criteria:
                prompt = (
                    f"Evaluate if the following agent outputs meet this criterion: "
                    f"{criterion}\n\nOutputs:\n{outputs_summary}\n\n"
                    f"Respond with 'PASS' or 'FAIL' followed by justification."
                )

                try:
                    response = await validator.generate_response(prompt, {})
                    passed = response.strip().upper().startswith("PASS")

                    details[criterion] = {
                        "passed": passed,
                        "justification": response,
                    }

                    if not passed:
                        errors.append(f"Semantic criterion failed: {criterion}")
                except Exception as llm_error:
                    errors.append(
                        f"LLM validation failed for criterion '{criterion}': "
                        f"Model '{config.validator_model}' unavailable"
                    )
                    details[criterion] = {
                        "passed": False,
                        "error": f"LLM unavailable: {llm_error!s}",
                    }

        except Exception as e:
            errors.append(f"Semantic validation initialization failed: {e!s}")
            details["error"] = str(e)

        return ValidationLayerResult(
            layer=ValidationLayer.SEMANTIC,
            passed=len(errors) == 0,
            errors=errors,
            details=details,
        )
