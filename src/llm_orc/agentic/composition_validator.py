"""Composition Validator — gatekeeper on the ``compose_ensemble`` path.

Per ``docs/agentic-serving/system-design.md`` §Composition Validator
(L1) and §Integration Contracts (Orchestrator Tool Dispatch → Composition
Validator, Composition Validator ↔ Ensemble Engine). Validates a
proposed composed ensemble against AS-2, AS-6, and the project-level
invariants it shares with the load path — all before the ensemble is
written to disk.

Six rejection branches plus one accept path:

1. ``invalid_agent_schema`` — Pydantic rejects the agent dict (e.g. an
   LLM agent without ``model_profile`` or ``model`` + ``provider``).
2. ``missing_dependency`` — an agent ``depends_on`` a sibling that is
   not in the proposed ensemble.
3. ``internal_dependency_cycle`` — agents form a cycle within the
   proposed ensemble's own dep graph.
4. ``invalid_fan_out`` — an agent has ``fan_out: true`` without
   ``depends_on``.
5. ``missing_primitive`` — an agent references a model profile,
   script, or ensemble that does not exist. AS-6 composition-time
   strictness: load-time tolerates dangling ensemble references
   silently, but composition must enforce "existing primitives only."
6. ``cross_ensemble_cycle`` — the proposed ensemble closes a cycle
   with existing ensembles. Delegates to
   :func:`validate_ensemble_reference_graph` — the same public routine
   the load path calls (FC-6).
7. ``depth_limit_exceeded`` — the proposed graph walks deeper than the
   configured depth limit (project-level Invariant 8). Moves the error
   left from :class:`EnsembleAgentRunner`'s runtime check.

The validator is a pure inspection — it never opens a write handle on
the local tier. Tool Dispatch performs the write only when validation
accepts; on rejection no partial state persists (AS-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import ValidationError

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    _check_missing_dependencies,
    _validate_fan_out_dependencies,
    assert_no_cycles,
    compute_reference_graph_depth,
    validate_ensemble_reference_graph,
)
from llm_orc.core.execution.scripting.resolver import (
    ScriptNotFoundError,
    ScriptResolver,
)
from llm_orc.schemas.agent_config import (
    AgentConfig,
    EnsembleAgentConfig,
    LlmAgentConfig,
    ScriptAgentConfig,
    parse_agent_config,
)

RejectionKind = Literal[
    "invalid_agent_schema",
    "missing_dependency",
    "internal_dependency_cycle",
    "invalid_fan_out",
    "missing_primitive",
    "cross_ensemble_cycle",
    "depth_limit_exceeded",
]


@dataclass(frozen=True)
class CompositionRequest:
    """A proposed ensemble to validate.

    ``agents`` carries raw dicts as they arrived from the orchestrator
    LLM — Pydantic parsing happens inside the validator so schema
    errors surface as a typed :class:`CompositionRejected` rather than
    bubbling out as exceptions.
    """

    name: str
    description: str
    agents: list[dict[str, Any]] = field(default_factory=list)
    raw_output: bool = False


@dataclass(frozen=True)
class CompositionAccepted:
    """Validation passed; the typed ``config`` is ready to persist."""

    config: EnsembleConfig


@dataclass(frozen=True)
class CompositionRejected:
    """Validation failed; ``reason`` names the specific invariant.

    Tool Dispatch converts this into a
    ``ToolCallError(kind="invocation_failed", reason=reason)`` — the
    ReAct loop observes the error and the orchestrator LLM may replan.
    """

    kind: RejectionKind
    reason: str


CompositionOutcome = CompositionAccepted | CompositionRejected


class PrimitiveRegistry(Protocol):
    """Narrow facade over primitive existence checks.

    Decouples the validator from concrete ``ConfigurationManager`` and
    ``ScriptResolver`` classes. Tests pass a handwritten double with
    scripted answers; production wires a ``ConfigManagerPrimitiveRegistry``
    that delegates to the real config manager and script resolver.
    """

    def model_profile_exists(self, name: str) -> bool: ...

    def script_exists(self, ref: str) -> bool: ...

    def ensemble_exists(self, name: str) -> bool: ...

    def ensemble_search_dirs(self) -> list[str]: ...


class CompositionValidator:
    """Orchestrates the composition-time validation stack."""

    def __init__(
        self,
        *,
        primitives: PrimitiveRegistry,
        depth_limit: int = 5,
    ) -> None:
        self._primitives = primitives
        self._depth_limit = depth_limit

    def validate(self, request: CompositionRequest) -> CompositionOutcome:
        """Run the six validation branches in order.

        Ordering is deliberate — earlier checks are cheaper and surface
        the most actionable error first. Schema → intra-ensemble deps
        → primitive existence → cross-ensemble graph → depth.
        """
        parsed = _parse_agents(request.agents)
        if isinstance(parsed, CompositionRejected):
            return parsed

        dep_outcome = _validate_internal_dependencies(parsed)
        if dep_outcome is not None:
            return dep_outcome

        primitive_outcome = _validate_primitive_existence(parsed, self._primitives)
        if primitive_outcome is not None:
            return primitive_outcome

        search_dirs = self._primitives.ensemble_search_dirs()
        try:
            validate_ensemble_reference_graph(request.name, parsed, search_dirs)
        except ValueError as exc:
            return CompositionRejected(
                kind="cross_ensemble_cycle",
                reason=str(exc),
            )

        depth = compute_reference_graph_depth(request.name, parsed, search_dirs)
        if depth > self._depth_limit:
            return CompositionRejected(
                kind="depth_limit_exceeded",
                reason=(
                    f"composed ensemble '{request.name}' has reference "
                    f"graph depth {depth} > limit {self._depth_limit}"
                ),
            )

        config = EnsembleConfig(
            name=request.name,
            description=request.description,
            agents=parsed,
            raw_output=request.raw_output,
        )
        return CompositionAccepted(config=config)


def _parse_agents(
    agent_dicts: list[dict[str, Any]],
) -> list[AgentConfig] | CompositionRejected:
    parsed: list[AgentConfig] = []
    for agent_dict in agent_dicts:
        try:
            parsed.append(parse_agent_config(agent_dict))
        except ValidationError as exc:
            return CompositionRejected(
                kind="invalid_agent_schema",
                reason=f"agent {agent_dict.get('name', '?')}: {exc}",
            )
        except ValueError as exc:
            return CompositionRejected(
                kind="invalid_agent_schema",
                reason=f"agent {agent_dict.get('name', '?')}: {exc}",
            )
    return parsed


def _validate_internal_dependencies(
    agents: list[AgentConfig],
) -> CompositionRejected | None:
    try:
        _check_missing_dependencies(agents)
    except ValueError as exc:
        return CompositionRejected(kind="missing_dependency", reason=str(exc))
    try:
        _validate_fan_out_dependencies(agents)
    except ValueError as exc:
        return CompositionRejected(kind="invalid_fan_out", reason=str(exc))
    try:
        assert_no_cycles(agents)
    except ValueError as exc:
        return CompositionRejected(kind="internal_dependency_cycle", reason=str(exc))
    return None


class LocalEnsembleWriter(Protocol):
    """Persists a validated :class:`EnsembleConfig` to the local tier.

    Tool Dispatch calls :meth:`write` only after
    :class:`CompositionValidator` returns :class:`CompositionAccepted`
    — AS-2's "validated before loading" contract. Implementations raise
    :class:`EnsembleWriteError` on collision or I/O failure so Tool
    Dispatch can surface a typed tool error without a partial state.
    """

    def write(self, config: EnsembleConfig) -> str:
        """Write the ensemble and return the absolute path."""
        ...


class EnsembleWriteError(ValueError):
    """Raised when a composed ensemble cannot be persisted.

    Inherits :class:`ValueError` so Tool Dispatch can narrow on a
    single exception type for the whole validation-plus-write surface.
    """


class ConfigManagerPrimitiveRegistry:
    """Production adapter: delegates to ConfigurationManager + ScriptResolver.

    The orchestrator's tool surface wires this adapter into
    :class:`CompositionValidator`. Tests use the handwritten double in
    ``tests/unit/agentic/test_composition_validator.py`` instead.
    """

    def __init__(
        self,
        config_manager: ConfigurationManager,
        script_resolver: ScriptResolver | None = None,
    ) -> None:
        self._config_manager = config_manager
        self._script_resolver = script_resolver or ScriptResolver()

    def model_profile_exists(self, name: str) -> bool:
        profiles = self._config_manager.get_model_profiles()
        return name in profiles

    def script_exists(self, ref: str) -> bool:
        try:
            self._script_resolver.resolve_script_path(ref)
        except (ScriptNotFoundError, FileNotFoundError):
            return False
        return True

    def ensemble_exists(self, name: str) -> bool:
        for directory in self.ensemble_search_dirs():
            dir_path = Path(directory)
            for ext in ("yaml", "yml"):
                if (dir_path / f"{name}.{ext}").exists():
                    return True
        return False

    def ensemble_search_dirs(self) -> list[str]:
        return [str(path) for path in self._config_manager.get_ensembles_dirs()]


class ConfigManagerEnsembleWriter:
    """Production adapter: writes to the local tier via ConfigurationManager.

    Mirrors :class:`EnsembleCrudHandler.get_local_ensembles_dir` so the
    orchestrator's composition path writes to the same directory the
    MCP ``create_ensemble`` tool writes to. Collision (same name
    already present) is a failure — composition never overwrites.
    """

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self._config_manager = config_manager

    def write(self, config: EnsembleConfig) -> str:
        local_dir = self._resolve_local_ensembles_dir()
        target = local_dir / f"{config.name}.yaml"
        if target.exists():
            raise EnsembleWriteError(
                f"ensemble '{config.name}' already exists at {target}"
            )
        local_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(_render_ensemble_yaml(config))
        return str(target)

    def _resolve_local_ensembles_dir(self) -> Path:
        ensemble_dirs = self._config_manager.get_ensembles_dirs()
        for path in ensemble_dirs:
            path_str = str(path)
            if ".llm-orc" in path_str and "library" not in path_str:
                return path
        if ensemble_dirs:
            return ensemble_dirs[0]
        raise EnsembleWriteError(
            "no local ensembles directory is configured; "
            "run `llm-orc init` or create .llm-orc/ensembles/"
        )


def _render_ensemble_yaml(config: EnsembleConfig) -> str:
    """Serialize an EnsembleConfig to the on-disk YAML shape.

    Uses :meth:`BaseModel.model_dump` with ``exclude_none=True`` so
    agent dicts omit optional fields that were never set, matching the
    terse hand-authored ensemble YAML the loader round-trips.
    """
    agents = [agent.model_dump(exclude_none=True) for agent in config.agents]
    data: dict[str, Any] = {
        "name": config.name,
        "description": config.description,
        "agents": agents,
    }
    if config.raw_output:
        data["raw_output"] = True
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _validate_primitive_existence(
    agents: list[AgentConfig],
    primitives: PrimitiveRegistry,
) -> CompositionRejected | None:
    for agent in agents:
        if isinstance(agent, LlmAgentConfig):
            if agent.model_profile is not None and not primitives.model_profile_exists(
                agent.model_profile
            ):
                return CompositionRejected(
                    kind="missing_primitive",
                    reason=(
                        f"agent '{agent.name}' references model_profile "
                        f"'{agent.model_profile}' which does not exist in the library"
                    ),
                )
            # agent.model + agent.provider paths do not go through the library
            # — they name an inline model, which satisfies AS-6 by construction
            # (the provider is one of a fixed set, the model is a provider string).
        elif isinstance(agent, ScriptAgentConfig):
            if not primitives.script_exists(agent.script):
                return CompositionRejected(
                    kind="missing_primitive",
                    reason=(
                        f"agent '{agent.name}' references script "
                        f"'{agent.script}' which does not exist in the library"
                    ),
                )
        elif isinstance(agent, EnsembleAgentConfig):
            if not primitives.ensemble_exists(agent.ensemble):
                return CompositionRejected(
                    kind="missing_primitive",
                    reason=(
                        f"agent '{agent.name}' references ensemble "
                        f"'{agent.ensemble}' which does not exist in the library"
                    ),
                )
    return None
