"""Plexus Adapter — mediates knowledge-graph interaction (ADR-009, WP-I).

Per ``docs/serving.md`` §Plexus Adapter (L1 Domain
Policy) and §Integration Contracts (Orchestrator Tool Dispatch → Plexus
Adapter). Per ADR-009 and AS-8 the Adapter is the single place
Plexus-aware code lives — Tool Dispatch and (WP-K) Calibration Gate see
a uniform surface regardless of whether Plexus is active. This is what
makes AS-8 (Plexus is optional) structurally enforceable.

WP-I ships the **skeleton with no-op fallbacks**. Tool Dispatch's
``query_knowledge`` and ``record_outcome`` switch from typed
``not_yet_wired`` errors to delegating through this Adapter; in
stateless deployments every call returns a well-formed empty / ack
value. The Plexus-absent branch satisfies FC-7 (every Plexus-facing
path covered in stateless mode).

WP-K replaces the no-op method bodies with real plexus MCP client
calls. The public surface, Tool Dispatch wiring, and Adapter
construction do not change — the integration is body-swap territory.

The Adapter exists as a class (not module-level functions) so WP-K can
inject the plexus MCP client through ``__init__`` without touching call
sites. WP-I's constructor takes no parameters; WP-K extends the
signature when the client surface is committed.
"""

from __future__ import annotations

from typing import Any


class PlexusAdapter:
    """Plexus-facing tool surface owned by the Adapter (ADR-009).

    The two methods correspond to Tool Dispatch's Plexus-facing tools
    (``query_knowledge``, ``record_outcome``). WP-I bodies are no-op
    fallbacks; WP-K replaces them with real plexus MCP calls behind
    the same shape.
    """

    async def query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Query the knowledge graph.

        WP-I: returns a well-formed empty result so the orchestrator's
        ReAct loop continues normally (scenarios.md §query_knowledge
        returns empty gracefully when Plexus is absent — ADR-009 clause
        on stateless degradation). The orchestrator LLM sees an empty
        ``results`` list and adapts its plan.
        """
        del arguments
        return {"results": [], "context": ""}

    async def record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Record a routing decision or outcome to the knowledge graph.

        WP-I: returns an acknowledgement immediately without writing.
        ADR-009's contract for ``record_outcome`` is "returns
        acknowledgement promptly" — the no-op fallback satisfies this
        trivially. WP-K writes asynchronously through the plexus MCP
        client and still returns immediately (eventual consistency at
        the enrichment layer).
        """
        del arguments
        return {"acknowledged": True}
