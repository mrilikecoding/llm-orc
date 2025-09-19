"""Primitive composition engine for chaining script agents (ADR-001)."""

import json
import os
import subprocess
from typing import Any

from llm_orc.core.execution.primitive_registry import PrimitiveRegistry
from llm_orc.core.execution.script_resolver import ScriptResolver
from llm_orc.schemas.script_agent import ScriptAgentInput, ScriptAgentOutput


class PrimitiveComposer:
    """Engine for composing and executing chained primitive script agents."""

    def __init__(self) -> None:
        """Initialize the primitive composer."""
        self._registry = PrimitiveRegistry()
        self._resolver = ScriptResolver()
        self._execution_cache: dict[str, dict[str, Any]] = {}

    def compose_primitives(self, composition_config: dict[str, Any]) -> dict[str, Any]:
        """Compose primitives into an executable chain based on configuration.

        Args:
            composition_config: Configuration defining the primitive composition

        Returns:
            Composition metadata and execution plan
        """
        primitives = composition_config.get("primitives", [])

        # Validate all primitives exist
        for primitive in primitives:
            script_name = primitive.get("script", "")
            try:
                self._registry.get_primitive_info(script_name)
            except FileNotFoundError as e:
                raise ValueError(f"Primitive not found: {script_name}") from e

        # Resolve execution order based on dependencies
        execution_order = self._resolve_execution_order(composition_config)

        return {
            "name": composition_config.get("name", "unnamed_composition"),
            "description": composition_config.get("description", ""),
            "execution_order": execution_order,
            "primitive_count": len(primitives),
            "valid": True,
        }

    def validate_composition(
        self, composition_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that a composition configuration is type-safe and executable.

        Args:
            composition_config: Configuration to validate

        Returns:
            Validation result with any errors or warnings
        """
        validation_result: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        primitives = composition_config.get("primitives", [])

        # Check for type compatibility between chained primitives
        for primitive in primitives:
            # Check if primitive declares input/output types
            if "input_type" in primitive and "output_type" in primitive:
                # Find dependent primitives
                dependencies = primitive.get("dependencies", {})
                for dep_ref in dependencies.values():
                    if "." in dep_ref:  # References another primitive's output
                        dep_name = dep_ref.split(".")[0]
                        dep_primitive = next(
                            (p for p in primitives if p["name"] == dep_name), None
                        )
                        if dep_primitive and "output_type" in dep_primitive:
                            if primitive["input_type"] != dep_primitive["output_type"]:
                                validation_result["errors"].append(
                                    f"Type mismatch: {primitive['name']} expects "
                                    f"{primitive['input_type']} but {dep_name} "
                                    f"outputs {dep_primitive['output_type']}"
                                )
                                validation_result["valid"] = False

            # Validate primitive exists
            script_name = primitive.get("script", "")
            try:
                self._registry.get_primitive_info(script_name)
            except FileNotFoundError:
                validation_result["errors"].append(
                    f"Primitive script not found: {script_name}"
                )
                validation_result["valid"] = False

        return validation_result

    def execute_composition(
        self, composition_config: dict[str, Any], input_data: ScriptAgentInput
    ) -> ScriptAgentOutput:
        """Execute a composed primitive chain with the given input.

        Args:
            composition_config: Configuration defining the primitive composition
            input_data: Input data conforming to ScriptAgentInput schema

        Returns:
            Final output from the composition chain
        """
        primitives = composition_config.get("primitives", [])
        execution_order = self._resolve_execution_order(composition_config)

        # Track outputs from each primitive for dependency resolution
        primitive_outputs: dict[str, Any] = {}

        for primitive_name in execution_order:
            primitive = next(p for p in primitives if p["name"] == primitive_name)

            # Prepare input for this primitive
            primitive_input = self._prepare_primitive_input(
                primitive, input_data, primitive_outputs
            )

            # Execute the primitive
            try:
                result = self._execute_primitive(primitive, primitive_input)
                primitive_outputs[primitive_name] = result
            except Exception as e:
                return ScriptAgentOutput(
                    success=False,
                    error=f"Primitive '{primitive_name}' failed: {e}",
                    data=None,
                    agent_requests=[],
                )

        # Return the final output (from last primitive or combined results)
        final_primitive = execution_order[-1] if execution_order else ""
        final_output = primitive_outputs.get(final_primitive, {})

        return ScriptAgentOutput(
            success=True,
            data={
                "composition_result": final_output.get("data", {}),
                "all_outputs": primitive_outputs,
            },
            error=None,
            agent_requests=[],
        )

    def _resolve_execution_order(self, composition_config: dict[str, Any]) -> list[str]:
        """Resolve the correct execution order based on primitive dependencies.

        Args:
            composition_config: Configuration with primitives and dependencies

        Returns:
            List of primitive names in execution order
        """
        primitives = composition_config.get("primitives", [])

        # Simple topological sort for dependency resolution
        # Create dependency graph
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}

        # Initialize all primitives
        for primitive in primitives:
            name = primitive["name"]
            graph[name] = []
            in_degree[name] = 0

        # Build dependency edges
        for primitive in primitives:
            name = primitive["name"]
            dependencies = primitive.get("dependencies", {})
            for dep_ref in dependencies.values():
                if "." in dep_ref:  # References another primitive
                    dep_name = dep_ref.split(".")[0]
                    if dep_name in graph:
                        graph[dep_name].append(name)
                        in_degree[name] += 1

        # Topological sort
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(execution_order) != len(primitives):
            raise ValueError("Circular dependency detected in primitive composition")

        return execution_order

    def _prepare_primitive_input(
        self,
        primitive: dict[str, Any],
        base_input: ScriptAgentInput,
        previous_outputs: dict[str, Any],
    ) -> ScriptAgentInput:
        """Prepare input for a primitive based on dependencies and mappings.

        Args:
            primitive: Primitive configuration
            base_input: Original input to the composition
            previous_outputs: Outputs from previously executed primitives

        Returns:
            Prepared input for the primitive
        """
        # Start with base input
        context = dict(base_input.context)
        dependencies = dict(base_input.dependencies)

        # Add primitive-specific context
        if "context" in primitive:
            context.update(primitive["context"])

        # Resolve dependencies
        if "dependencies" in primitive:
            dependencies.update(
                self._resolve_primitive_dependencies(
                    primitive["dependencies"], previous_outputs
                )
            )

        return ScriptAgentInput(
            agent_name=primitive["name"],
            input_data=base_input.input_data,
            context=context,
            dependencies=dependencies,
        )

    def _execute_primitive(
        self, primitive: dict[str, Any], primitive_input: ScriptAgentInput
    ) -> dict[str, Any]:
        """Execute a single primitive with the given input.

        Args:
            primitive: Primitive configuration
            primitive_input: Input data for the primitive

        Returns:
            Primitive execution result
        """
        script_name = primitive["script"]
        # For primitives, try the primitives directory first
        primitives_path = f"primitives/{script_name}"
        try:
            script_path = self._resolver.resolve_script_path(primitives_path)
        except FileNotFoundError:
            # Fall back to the original script name
            script_path = self._resolver.resolve_script_path(script_name)

        # Prepare environment
        env = os.environ.copy()
        env["INPUT_DATA"] = primitive_input.model_dump_json()

        # Execute the primitive script
        result = subprocess.run(
            [script_path], capture_output=True, text=True, env=env, timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"Script execution failed: {result.stderr}")

        # Parse and validate output
        try:
            output_data = json.loads(result.stdout.strip())
            validated_output = ScriptAgentOutput.model_validate(output_data)
            return validated_output.model_dump()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON output: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Output validation failed: {e}") from e

    def _resolve_primitive_dependencies(
        self, deps: dict[str, Any], previous_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve primitive dependencies from previous outputs.

        Args:
            deps: Raw dependency configuration
            previous_outputs: Outputs from previously executed primitives

        Returns:
            Resolved dependencies dictionary
        """
        resolved = {}
        for key, value in deps.items():
            if isinstance(value, str) and "." in value:
                # Reference to another primitive's output
                ref_parts = value.split(".")
                if len(ref_parts) == 2:
                    ref_primitive, ref_field = ref_parts
                    if ref_primitive in previous_outputs:
                        output_data = previous_outputs[ref_primitive].get("data", {})
                        resolved[key] = output_data.get(ref_field, value)
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
