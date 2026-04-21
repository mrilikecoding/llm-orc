"""Orchestrator Tool Dispatch — closed five-tool surface (ADR-003).

Per ``docs/agentic-serving/system-design.md`` §Orchestrator Tool
Dispatch (L2) and §Integration Contracts (Orchestrator Runtime →
Orchestrator Tool Dispatch). The Runtime calls ``dispatch(call)`` with
an ``InternalToolCall``; this module routes by name through the closed
set or returns a typed tool error for names outside the set.

The closed-set property is structurally enforced: the five tool names
live in ``TOOL_NAMES`` and correspond to five async methods on this
class. FC-5 checks the count of public async methods whose names are
in ``TOOL_NAMES`` — exactly five.

WP-C wires ``invoke_ensemble`` and ``list_ensembles`` to the Ensemble
Engine. ``compose_ensemble`` (WP-G), ``query_knowledge`` (WP-I), and
``record_outcome`` (WP-I) return typed not-yet-wired errors so the
closed-set property holds from WP-C onward.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from llm_orc.core.config.config_manager import ConfigurationManager
from llm_orc.core.config.ensemble_config import EnsembleConfig, EnsembleLoader

TOOL_NAMES: frozenset[str] = frozenset(
    {
        "invoke_ensemble",
        "compose_ensemble",
        "list_ensembles",
        "query_knowledge",
        "record_outcome",
    }
)
"""The closed tool set committed by ADR-003."""


ToolErrorKind = Literal[
    "unknown_tool",
    "not_yet_wired",
    "invocation_failed",
    "invalid_arguments",
]


@dataclass(frozen=True)
class InternalToolCall:
    """A tool call emitted by the orchestrator LLM.

    ``arguments`` is pre-parsed by the Runtime — the orchestrator LLM
    emits JSON-string arguments per OpenAI convention; the Runtime
    parses before handing off so JSON handling is centralized.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallSuccess:
    """A successful tool result surfaced to the orchestrator's context."""

    id: str
    name: str
    content: Any


@dataclass(frozen=True)
class ToolCallError:
    """A failed tool call surfaced as an observation, not an exception.

    The ReAct loop continues with this result — the LLM sees the
    error, may adjust its plan, and emits the next tool call.
    """

    id: str
    name: str
    kind: ToolErrorKind
    reason: str


ToolCallResult = ToolCallSuccess | ToolCallError


class EnsembleRuntimeExecutor(Protocol):
    """Minimum Ensemble Engine surface the Tool Dispatch calls at runtime.

    ``EnsembleExecutor`` satisfies this structurally; tests pass a
    handwritten double for focused verification. Declared with
    ``Awaitable`` rather than ``async def`` — mypy preserves the return
    type through ``await`` this way where ``async def`` in a Protocol
    degrades the inferred type to ``Any``.
    """

    def execute(
        self, config: EnsembleConfig, input_data: str
    ) -> Awaitable[dict[str, Any]]: ...


EnsembleExecutorProvider = Callable[[], EnsembleRuntimeExecutor]
"""Factory for a fresh executor per invocation.

Invariant 10: immutable infrastructure shared but mutable state is
freshly created per invocation.
"""


class OrchestratorToolDispatch:
    """Closed five-tool dispatch surface (ADR-003, FC-5)."""

    def __init__(
        self,
        *,
        config_manager: ConfigurationManager,
        executor_provider: EnsembleExecutorProvider,
    ) -> None:
        self._config_manager = config_manager
        self._executor_provider = executor_provider

    async def dispatch(self, call: InternalToolCall) -> ToolCallResult:
        """Route a tool call by name through the closed set.

        Match-case makes the five committed tools visible at the
        dispatch site. A name outside the set falls through to the
        ``_`` arm and becomes a typed ``unknown_tool`` error — the
        ReAct loop continues with the error as an observation.
        """
        match call.name:
            case "invoke_ensemble":
                return await self.invoke_ensemble(call.id, call.arguments)
            case "compose_ensemble":
                return await self.compose_ensemble(call.id, call.arguments)
            case "list_ensembles":
                return await self.list_ensembles(call.id, call.arguments)
            case "query_knowledge":
                return await self.query_knowledge(call.id, call.arguments)
            case "record_outcome":
                return await self.record_outcome(call.id, call.arguments)
            case _:
                return ToolCallError(
                    id=call.id,
                    name=call.name,
                    kind="unknown_tool",
                    reason=(
                        f"tool '{call.name}' is not in the orchestrator's "
                        "committed tool set"
                    ),
                )

    async def invoke_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Resolve an ensemble by name and execute it via Ensemble Engine."""
        name = arguments.get("name")
        input_data = arguments.get("input", "")
        if not isinstance(name, str) or not name:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble requires 'name' (non-empty string)",
            )
        if not isinstance(input_data, str):
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invalid_arguments",
                reason="invoke_ensemble 'input' must be a string",
            )

        loader = EnsembleLoader()
        ensemble_config: EnsembleConfig | None = None
        for directory in self._config_manager.get_ensembles_dirs():
            ensemble_config = loader.find_ensemble(str(directory), name)
            if ensemble_config is not None:
                break
        if ensemble_config is None:
            return ToolCallError(
                id=id_,
                name="invoke_ensemble",
                kind="invocation_failed",
                reason=f"ensemble '{name}' not found in the library",
            )

        executor = self._executor_provider()
        result: dict[str, Any] = await executor.execute(ensemble_config, input_data)
        return ToolCallSuccess(id=id_, name="invoke_ensemble", content=result)

    async def compose_ensemble(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Compose a new ensemble (WP-G)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="compose_ensemble",
            kind="not_yet_wired",
            reason="compose_ensemble lands in WP-G (Composition Validator)",
        )

    async def list_ensembles(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Enumerate available ensembles across all configured tiers."""
        del arguments
        loader = EnsembleLoader()
        entries: list[dict[str, Any]] = []
        for directory in self._config_manager.get_ensembles_dirs():
            for ensemble in loader.list_ensembles(str(directory)):
                entries.append(
                    {
                        "name": ensemble.name,
                        "description": ensemble.description,
                        "path": ensemble.relative_path,
                    }
                )
        return ToolCallSuccess(id=id_, name="list_ensembles", content=entries)

    async def query_knowledge(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Query the knowledge graph (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="query_knowledge",
            kind="not_yet_wired",
            reason="query_knowledge lands in WP-I (Plexus Adapter)",
        )

    async def record_outcome(
        self, id_: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Record a routing decision or outcome (WP-I Plexus Adapter)."""
        del arguments
        return ToolCallError(
            id=id_,
            name="record_outcome",
            kind="not_yet_wired",
            reason="record_outcome lands in WP-I (Plexus Adapter)",
        )
