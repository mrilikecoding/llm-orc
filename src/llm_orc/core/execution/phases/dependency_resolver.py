"""Dependency resolution for agent execution chains."""

import json
from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.utils import SUCCEEDED_STATUSES, dep_name
from llm_orc.schemas.agent_config import (
    AgentConfig,
    DynamicDispatchAgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    LoopAgentConfig,
    ScriptAgentConfig,
)

# Node types whose runner hands input_data verbatim to a fresh child
# executor (ensemble_execution); they share one input contract.
ChildExecutionConfig = (
    EnsembleAgentConfig | LoopAgentConfig | DynamicDispatchAgentConfig
)


def _unsupported_consumer_type(
    agent_name: str, agent_config: AgentConfig
) -> ValueError:
    """The no-fall-through contract error (issue #202): raised for any
    consumer type outside the LLM/script/ensemble/loop/dispatch set. The
    radius is the whole run by design: an unrecognized consumer is an
    engine bug, not a per-agent runtime error, so the phase-wide
    comprehension in ensemble_execution takes every sibling down loudly
    (PR 203 round 2, finding 4)."""
    return ValueError(
        f"Unsupported consumer type for agent '{agent_name}': "
        f"{type(agent_config).__name__}. The dependency input contract "
        f"covers LLM, script, ensemble, loop, and dispatch agents."
    )


class DependencyResolver:
    """Resolves agent dependencies and enhances input with dependency results."""

    def __init__(
        self,
        role_resolver: Callable[[str], str | None],
        dependency_config_lookup: Callable[[str], AgentConfig | None],
        ensemble_terminal_agents: Callable[[str], list[str]],
    ) -> None:
        """Initialize resolver with role description function.

        ``dependency_config_lookup`` and ``ensemble_terminal_agents`` let
        the resolver tell an ``ensemble:`` dependency apart from any
        other and look up its child's terminal agent names, to render
        terminal responses instead of the raw execution record
        (fail-closed-composition D). SF6: required rather than
        optional-and-only-used-together — an ensemble_execution.py
        construction site that drops them used to degrade silently back
        to the pre-D full-JSON-record behavior; a caller with no
        meaningful lookup passes an explicit stub (``lambda name: None``
        / ``lambda ref: []``) instead, so the degradation is visible in
        the call site rather than absorbed here.
        """
        self._get_agent_role_description = role_resolver
        self._get_dependency_config = dependency_config_lookup
        self._ensemble_terminal_agents = ensemble_terminal_agents

    def enhance_input_with_dependencies(
        self,
        base_input: str,
        dependent_agents: list[AgentConfig],
        results_dict: dict[str, Any],
    ) -> dict[str, str]:
        """Enhance input with dependency results for each dependent agent.

        Returns a dictionary mapping agent names to their enhanced input.
        Each agent gets only the results from their specific dependencies.

        For script agents, returns JSON-formatted input with a
        'dependencies' dict containing upstream results.

        For LLM agents, returns text-formatted input with natural language
        context about previous agent results.
        """
        return {
            agent_config.name: self._compute_agent_input(
                agent_config, base_input, results_dict
            )
            for agent_config in dependent_agents
        }

    def _compute_agent_input(
        self,
        agent_config: AgentConfig,
        base_input: str,
        results_dict: dict[str, Any],
    ) -> str:
        """Compute the enhanced input string for a single agent.

        Args:
            agent_config: Agent to compute input for.
            base_input: Original ensemble input.
            results_dict: Accumulated results from earlier agents.

        Returns:
            Input string for the agent.
        """
        agent_name = agent_config.name
        dependencies = agent_config.depends_on
        is_script_agent = isinstance(agent_config, ScriptAgentConfig)

        if not dependencies:
            return self._no_dependency_input(agent_name, agent_config, base_input)

        # Apply input_key selection (ADR-014)
        effective_results, input_key_error = self._apply_input_key_selection(
            agent_config, results_dict
        )
        if input_key_error:
            return input_key_error

        if is_script_agent:
            dep_results_dict = self._extract_dependency_results_as_dict(
                dependencies, effective_results
            )
            return self._build_script_input(agent_name, base_input, dep_results_dict)

        if isinstance(agent_config, ChildExecutionConfig):
            selected = self._selected_child_value(agent_config, effective_results)
            if selected is not None:
                return selected

        dependency_results = self._extract_successful_dependency_results(
            dependencies, effective_results, agent_config
        )

        if isinstance(agent_config, ChildExecutionConfig):
            return self._child_contract_input(
                agent_config, base_input, dependency_results
            )

        if isinstance(agent_config, LlmAgentConfig):
            if agent_config.input_scope == "dependencies":
                return self._joined_dependency_results(dependency_results)
            if dependency_results:
                return self._build_enhanced_input_with_dependencies(
                    agent_name, base_input, dependency_results
                )
            return self._build_enhanced_input_no_dependencies(agent_name, base_input)

        raise _unsupported_consumer_type(agent_name, agent_config)

    def _no_dependency_input(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        base_input: str,
    ) -> str:
        """Input for a node with no dependencies (issue #202)."""
        if isinstance(agent_config, ScriptAgentConfig):
            return self._build_script_input(agent_name, base_input, {})
        if isinstance(agent_config, LlmAgentConfig | ChildExecutionConfig):
            return base_input
        raise _unsupported_consumer_type(agent_name, agent_config)

    def _selected_child_value(
        self,
        agent_config: ChildExecutionConfig,
        effective_results: dict[str, Any],
    ) -> str | None:
        """The input_key-selected value for a child-execution node
        (ADR-014), or None when the composition path should build the
        input (input_key unset, or the first dependency not successful)."""
        if not agent_config.input_key:
            return None
        first_dep = dep_name(agent_config.depends_on[0])
        dep_result = effective_results.get(first_dep, {})
        if dep_result.get("status") == "success":
            return str(dep_result.get("response", ""))
        return None

    def child_input_key_contract_error(
        self, agent_config: AgentConfig, results_dict: dict[str, Any]
    ) -> str | None:
        """None when a child-execution node's ``input_key`` contract is
        satisfiable; otherwise an error naming the upstream agent and
        the key (fail-closed-composition, the input_key/depends_on[0]
        decision).

        ``input_key`` on an ``ensemble:``/``loop:``/``dispatch:`` node
        selects ``depends_on[0]``'s response verbatim (ADR-014) — the
        node's ENTIRE input, not one dependency among several. If
        ``depends_on[0]`` did not succeed, there is no honest verbatim
        value to hand the child, even when some OTHER dependency did
        succeed: silently composing a different, unrequested input shape
        (base input plus the other deps' data) would let a failed step
        reach the child as a success by another route. This is checked
        and the agent failed before it ever runs, the same way a fan-out
        agent's own ``input_key`` contract failure already works. A
        fan-out original (``fan_out: true``) is excluded — FanOutCoordinator
        already applies this exact check ahead of expansion, with its
        own error text; this method only covers the non-fan-out case.
        """
        if not isinstance(agent_config, ChildExecutionConfig):
            return None
        if agent_config.fan_out or not agent_config.input_key:
            return None
        if not agent_config.depends_on:
            return None
        first_dep = dep_name(agent_config.depends_on[0])
        dep_result = results_dict.get(first_dep, {})
        status = dep_result.get("status") if isinstance(dep_result, dict) else None
        if status == "success":
            return None
        return (
            f"Agent '{agent_config.name}' cannot run: input_key "
            f"'{agent_config.input_key}' selects from upstream agent "
            f"'{first_dep}', which did not succeed (status: {status!r})"
        )

    def _child_contract_input(
        self,
        agent_config: ChildExecutionConfig,
        base_input: str,
        dependency_results: list[str],
    ) -> str:
        """One input contract for child-execution nodes (issue #202).

        The ``ensemble:``, ``loop:``, and ``dispatch:`` runners hand
        ``input_data`` verbatim to a fresh child executor, so the child is
        a new execution, not an agent in the dependency chain. The base
        input is followed by dependency data blocks (honoring
        ``input_scope``); the input_key-verbatim rule is applied in
        _compute_agent_input before composition. LLM instruction sentences
        are never part of this contract: they exist only for LLM consumers.
        """
        if agent_config.input_scope == "dependencies":
            return self._joined_dependency_results(dependency_results)
        if dependency_results:
            deps_text = self._joined_dependency_results(dependency_results)
            return f"{base_input}\n\n{deps_text}"
        return base_input

    @staticmethod
    def _joined_dependency_results(dependency_results: list[str]) -> str:
        """Dependency data blocks, joined (input_scope: dependencies)."""
        return "\n\n".join(dependency_results)

    def _apply_input_key_selection(
        self,
        agent_config: AgentConfig,
        results_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Apply input_key selection to upstream results (ADR-014).

        Returns (effective_results, error_or_none). When input_key is
        set, the first dependency's response is replaced with the
        selected key's value. On error, returns an error message.
        """
        input_key = agent_config.input_key
        if not input_key or not agent_config.depends_on:
            return results_dict, None

        first_dep = dep_name(agent_config.depends_on[0])
        dep_result = results_dict.get(first_dep, {})

        if dep_result.get("status") != "success":
            return results_dict, None

        response = dep_result.get("response", "")

        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return results_dict, (
                f"input_key error: upstream '{first_dep}' output "
                f"is not dict-shaped (not valid JSON)"
            )

        if not isinstance(parsed, dict):
            return results_dict, (
                f"input_key error: upstream '{first_dep}' output "
                f"is not dict (got {type(parsed).__name__})"
            )

        if input_key not in parsed:
            available = list(parsed.keys())
            return results_dict, (
                f"input_key error: key '{input_key}' not found "
                f"in upstream '{first_dep}' output. "
                f"Available keys: {available}"
            )

        selected = parsed[input_key]
        new_response = selected if isinstance(selected, str) else json.dumps(selected)

        modified = dict(results_dict)
        modified[first_dep] = {
            **dep_result,
            "response": new_response,
        }
        return modified, None

    def _extract_successful_dependency_results(
        self,
        dependencies: list[str | dict[str, Any]],
        results_dict: dict[str, Any],
        consumer_config: AgentConfig | None = None,
    ) -> list[str]:
        """Extract dependency results with role attribution.

        Despite the name (kept for the extracted-helper test), this
        covers every dependency PRESENT in ``results_dict``, not only
        successful ones: a failed or skipped dependency renders as a
        named ``(status): detail`` block instead of silently vanishing
        (fail-closed-composition rule 2 — a failed step never reaches a
        downstream consumer as an unqualified success). A dependency
        entirely absent from ``results_dict`` is still omitted.

        Args:
            dependencies: List of dependency names (str or dict form)
            results_dict: Dictionary of previous agent results
            consumer_config: The dependent agent's own config. When it is
                an LLM agent, a successful ``ensemble:`` dependency's
                response renders as its child's terminal agent responses
                instead of the raw execution record (fail-closed-
                composition D). The dependency selected by the consumer's
                own ``input_key`` is left verbatim — that selection
                already happened in ``_apply_input_key_selection``.

        Returns:
            List of formatted dependency result strings
        """
        input_key_dep = self._input_key_selected_dep_name(consumer_config)
        dependency_results = []
        for dep in dependencies:
            agent_dep_name = dep_name(dep)
            if agent_dep_name not in results_dict:
                continue
            result = results_dict[agent_dep_name]
            dep_role = self._get_agent_role_description(agent_dep_name)
            role_text = f" ({dep_role})" if dep_role else ""

            if result.get("status") in SUCCEEDED_STATUSES:
                response = result["response"]
                if (
                    isinstance(consumer_config, LlmAgentConfig)
                    and agent_dep_name != input_key_dep
                ):
                    response = self._render_fan_out_or_ensemble_dependency(
                        agent_dep_name, result
                    )
                dependency_results.append(
                    f"Agent {agent_dep_name}{role_text}:\n{response}"
                )
            else:
                dependency_results.append(
                    self._render_non_success_dependency_block(
                        agent_dep_name, role_text, result
                    )
                )

        return dependency_results

    @staticmethod
    def _render_non_success_dependency_block(
        agent_dep_name: str, role_text: str, result: dict[str, Any]
    ) -> str:
        """A named block for a dependency that did not succeed (failed or
        skipped) — fail-closed-composition rule 2: the consumer sees
        which upstream agent failed and why, in the same shape the
        engine already uses for a failed ensemble terminal
        (``_render_terminal_block``)."""
        status = result.get("status", "failed")
        detail = result.get("error") or result.get("reason") or result.get("response")
        detail_text = detail or "no response"
        return f"Agent {agent_dep_name}{role_text} ({status}): {detail_text}"

    @staticmethod
    def _input_key_selected_dep_name(
        consumer_config: AgentConfig | None,
    ) -> str | None:
        """The dependency name already resolved by the consumer's own
        ``input_key`` (ADR-014), or None. That dependency's response was
        already replaced with the selected value in
        ``_apply_input_key_selection`` and must not be re-rendered."""
        if consumer_config is None or not consumer_config.input_key:
            return None
        if not consumer_config.depends_on:
            return None
        return dep_name(consumer_config.depends_on[0])

    def _render_fan_out_or_ensemble_dependency(
        self, dep_agent_name: str, result: dict[str, Any]
    ) -> Any:
        """A ``success``/``partial`` dependency's response for an LLM
        consumer, ahead of the generic ``Agent X:`` wrap.

        A genuinely empty gathered fan-out (the upstream array was
        ``[]``) renders as an explicit statement rather than an empty
        block (SF4) — checked before the ensemble-specific render, since
        it applies to any fan-out dependency, not only an ``ensemble:``
        one. Otherwise falls through to
        ``_render_ensemble_dependency``.
        """
        response = result.get("response")
        if result.get("fan_out") and isinstance(response, list) and not response:
            return (
                f"Agent {dep_agent_name}: produced zero instances "
                "(upstream list was empty)"
            )
        return self._render_ensemble_dependency(dep_agent_name, result)

    def _render_ensemble_dependency(
        self, dep_agent_name: str, result: dict[str, Any]
    ) -> Any:
        """An ``ensemble:`` dependency's response for an LLM consumer
        (fail-closed-composition D). Falls through to the raw response
        unchanged when the dependency isn't an ensemble agent (SF6: the
        lookups themselves are always wired — a caller with nothing
        meaningful to look up passes an explicit stub).
        """
        response = result.get("response")
        dep_config = self._get_dependency_config(dep_agent_name)
        if not isinstance(dep_config, EnsembleAgentConfig):
            return response
        terminals = self._ensemble_terminal_agents(dep_config.ensemble)
        instance_errors = self._fan_out_instance_errors(result)
        return self._render_ensemble_response(response, terminals, instance_errors)

    @staticmethod
    def _fan_out_instance_errors(result: dict[str, Any]) -> dict[int, str]:
        """``index -> error`` for each failed instance of a gathered
        fan-out dependency (a ``partial`` status), so a failed instance's
        block can name why it has nothing to show instead of rendering
        a bare "(no result)" (fail-closed-composition, partial fan-out
        decision)."""
        instances = result.get("instances")
        if not isinstance(instances, list):
            return {}
        errors: dict[int, str] = {}
        for item in instances:
            if not isinstance(item, dict) or item.get("status") != "failed":
                continue
            index = item.get("index")
            error = item.get("error")
            if isinstance(index, int) and error:
                errors[index] = str(error)
        return errors

    def _render_ensemble_response(
        self,
        response: Any,
        terminals: list[str],
        instance_errors: dict[int, str] | None = None,
    ) -> str:
        """One labeled block per terminal agent. A plain (non-fan-out)
        ensemble dependency is a single child result; a gathered fan-out
        dependency is a list of them, one per instance."""
        if isinstance(response, list):
            errors = instance_errors or {}
            blocks = [
                self._render_child_result(
                    item, terminals, index=idx, error=errors.get(idx)
                )
                for idx, item in enumerate(response)
            ]
            return "\n\n".join(blocks)
        return self._render_child_result(response, terminals, index=None)

    def _render_child_result(
        self,
        raw: Any,
        terminals: list[str],
        index: int | None,
        error: str | None = None,
    ) -> str:
        """Terminal blocks for a single child result. A child result that
        doesn't parse into the expected shape — the child failed, or the
        ensemble reference didn't resolve so ``terminals`` is empty —
        renders what's there honestly instead of going silently empty.
        ``error`` is the failed fan-out instance's own error text
        (``None`` for a non-fan-out or successful item).
        """
        parsed = self._parse_child_result(raw)
        results = parsed.get("results") if parsed is not None else None
        if not isinstance(results, dict):
            return self._render_raw_fallback(raw, index, error)
        names = terminals or list(results.keys())
        blocks = [
            self._render_terminal_block(name, results.get(name), index)
            for name in names
        ]
        return "\n\n".join(blocks)

    @staticmethod
    def _parse_child_result(raw: Any) -> dict[str, Any] | None:
        """Parse a child result to a dict, or None when it doesn't."""
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _render_raw_fallback(
        raw: Any, index: int | None, error: str | None = None
    ) -> str:
        """Honest fallback for a child result that isn't a parseable
        child-result dict — never silently empty, and never a bare
        "None" for a failed fan-out instance that has an error to name.
        """
        label = f"[{index}]" if index is not None else "result"
        if raw is None:
            return f"{label} (failed): {error}" if error else f"{label}: (no result)"
        text = raw if isinstance(raw, str) else json.dumps(raw)
        return f"{label}:\n{text}"

    @staticmethod
    def _render_terminal_block(name: str, agent_result: Any, index: int | None) -> str:
        """One labeled block for a single terminal agent's result."""
        label = f"{name}[{index}]" if index is not None else name
        if not isinstance(agent_result, dict):
            return f"{label}: (no result)"
        if agent_result.get("status") == "success":
            return f"{label}:\n{agent_result.get('response')}"
        status = agent_result.get("status", "failed")
        detail = agent_result.get("error") or agent_result.get("response")
        return f"{label} ({status}): {detail or 'no response'}"

    def _extract_dependency_results_as_dict(
        self, dependencies: list[str | dict[str, Any]], results_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract dependency results as a dict, keyed by agent name.

        Every dependency PRESENT in ``results_dict`` is included
        regardless of status (fail-closed-composition rule 2): the
        ``ScriptAgentInput`` shape (``dependencies: dict[str, Any]``)
        already carries a raw result's ``status``/``error`` honestly, so
        a script consumer can read ``dependencies[name]["status"]``
        directly instead of a failed or skipped dependency silently
        vanishing from the dict. A dependency entirely absent from
        ``results_dict`` is still omitted.

        Args:
            dependencies: List of dependency agent names (str or dict form)
            results_dict: Dictionary of previous agent results

        Returns:
            Dictionary mapping dependency names to their results
        """
        return {
            agent_dep_name: results_dict[agent_dep_name]
            for agent_dep_name in (dep_name(dep) for dep in dependencies)
            if agent_dep_name in results_dict
        }

    def _build_script_input(
        self, agent_name: str, base_input: str, dependencies: dict[str, Any]
    ) -> str:
        """Build JSON-formatted input for script agents.

        Args:
            agent_name: Name of the script agent
            base_input: Original input text
            dependencies: Dict of dependency results

        Returns:
            JSON string conforming to ScriptAgentInput schema
        """
        script_input = {
            "agent_name": agent_name,
            "input_data": base_input,
            "context": {},
            "dependencies": dependencies,
        }
        return json.dumps(script_input)

    def _build_enhanced_input_with_dependencies(
        self, agent_name: str, base_input: str, dependency_results: list[str]
    ) -> str:
        """Build enhanced input with dependency results.

        Args:
            agent_name: Name of the target agent
            base_input: Original input text
            dependency_results: List of formatted dependency result strings

        Returns:
            Enhanced input string with dependencies
        """
        deps_text = "\n\n".join(dependency_results)
        return (
            f"Please respond to the following input, "
            f"taking into account the results from the previous agents "
            f"in the dependency chain.\n\n"
            f"Original Input:\n{base_input}\n\n"
            f"Previous Agent Results (for your reference):\n"
            f"{deps_text}\n\n"
            f"Please provide your own analysis, building upon "
            f"(but not simply repeating) the previous results."
        )

    def _build_enhanced_input_no_dependencies(
        self, agent_name: str, base_input: str
    ) -> str:
        """Build enhanced input for agent without dependencies.

        Args:
            agent_name: Name of the target agent
            base_input: Original input text

        Returns:
            Simple enhanced input string
        """
        return base_input

    def has_dependencies(self, agent_config: AgentConfig) -> bool:
        """Check if an agent has dependencies."""
        return bool(agent_config.depends_on)

    def get_dependencies(self, agent_config: AgentConfig) -> list[str | dict[str, Any]]:
        """Get list of dependencies for an agent."""
        return agent_config.depends_on

    def dependencies_satisfied(
        self, agent_config: AgentConfig, completed_agents: set[str]
    ) -> bool:
        """Check if all dependencies for an agent are satisfied."""
        dependencies = self.get_dependencies(agent_config)
        return all(dep_name(dep) in completed_agents for dep in dependencies)

    @staticmethod
    def _fan_out_child_input(chunk: Any) -> str:
        """Fan-out instance input for a child-execution node (issue #202).

        The fan-out coordinator already applied input_key selection; each
        instance is a fresh child execution, so it receives its chunk
        verbatim — no "Processing chunk N of M" wrapper, which is an LLM
        framing.
        """
        if isinstance(chunk, str):
            return chunk
        return json.dumps(chunk)

    @staticmethod
    def is_fan_out_instance_config(agent_config: AgentConfig) -> bool:
        """Check if an agent config is a fan-out instance.

        Args:
            agent_config: Agent configuration to check

        Returns:
            True if this is a fan-out instance configuration
        """
        return agent_config.fan_out_original is not None

    def prepare_fan_out_instance_input(
        self,
        instance_config: AgentConfig,
        base_input: str,
    ) -> str:
        """Prepare input for a fan-out instance.

        Args:
            instance_config: Instance configuration with fan_out_* metadata
            base_input: Original ensemble input

        Returns:
            Prepared input string (JSON for scripts, the chunk verbatim for
            child-execution nodes, text for LLMs)
        """
        chunk = instance_config.fan_out_chunk
        index = instance_config.fan_out_index or 0
        total = instance_config.fan_out_total or 1
        name = instance_config.name

        if isinstance(instance_config, ScriptAgentConfig):
            return self._build_fan_out_script_input(
                name, chunk, index, total, base_input
            )
        if isinstance(instance_config, ChildExecutionConfig):
            return self._fan_out_child_input(chunk)
        return self._build_fan_out_llm_input(chunk, index, total, base_input)

    def _build_fan_out_script_input(
        self,
        agent_name: str,
        chunk: Any,
        chunk_index: int,
        total_chunks: int,
        base_input: str,
    ) -> str:
        """Build JSON input for a fan-out script agent instance."""
        script_input = {
            "agent_name": agent_name,
            "input": chunk,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "base_input": base_input,
            "context": {},
        }
        return json.dumps(script_input)

    def _build_fan_out_llm_input(
        self,
        chunk: Any,
        chunk_index: int,
        total_chunks: int,
        base_input: str,
    ) -> str:
        """Build text input for a fan-out LLM agent instance."""
        # Convert chunk to string if needed
        if isinstance(chunk, dict):
            chunk_text = json.dumps(chunk)
        else:
            chunk_text = str(chunk)

        return (
            f"Processing chunk {chunk_index + 1} of {total_chunks}.\n\n"
            f"Original task: {base_input}\n\n"
            f"Chunk content:\n{chunk_text}"
        )

    def filter_by_dependency_status(
        self,
        agents: list[AgentConfig],
        completed_agents: set[str],
        with_dependencies: bool = True,
    ) -> list[AgentConfig]:
        """Filter agents based on dependency satisfaction status.

        Args:
            agents: List of agent configurations
            completed_agents: Set of agent names that have completed
            with_dependencies: If True, return agents WITH satisfied dependencies.
                             If False, return agents WITHOUT dependencies.
        """
        if with_dependencies:
            return [
                agent
                for agent in agents
                if self.has_dependencies(agent)
                and self.dependencies_satisfied(agent, completed_agents)
            ]
        else:
            return [agent for agent in agents if not self.has_dependencies(agent)]

    @staticmethod
    def get_agent_input(input_data: str | dict[str, str], agent_name: str) -> str:
        """Get appropriate input for an agent from uniform or per-agent input.

        Args:
            input_data: A string for uniform input, or a dict mapping
                       agent names to their specific enhanced input
            agent_name: Name of the agent to get input for

        Returns:
            Input string for the specified agent
        """
        if isinstance(input_data, dict):
            return input_data.get(agent_name, "")
        return input_data
