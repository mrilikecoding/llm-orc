"""Tests for the Plexus Adapter no-op fallbacks (ADR-009, WP-I).

Per ``docs/agentic-serving/system-design.md`` §Plexus Adapter (L1
Domain Policy). WP-I ships only the no-op fallbacks; the Plexus-active
branches land in WP-K. These tests cover the shipped contract — what
``query_knowledge`` and ``record_outcome`` see when Plexus is absent.

Covers scenarios (``docs/agentic-serving/scenarios.md``):

* §query_knowledge returns empty gracefully when Plexus is absent —
  here at unit scope (the Adapter's ``query`` returns a well-formed
  empty result).
* §record_outcome writes asynchronously without blocking the ReAct
  loop — here at unit scope (the Adapter's ``record`` returns an
  acknowledgement immediately).
"""

from __future__ import annotations

import pytest

from llm_orc.agentic.plexus_adapter import PlexusAdapter


class TestQueryNoOp:
    """``query`` returns a well-formed empty result when Plexus is absent."""

    @pytest.mark.asyncio
    async def test_query_returns_empty_results(self) -> None:
        adapter = PlexusAdapter()
        result = await adapter.query({"topic": "anything"})
        assert result == {"results": [], "context": ""}

    @pytest.mark.asyncio
    async def test_query_does_not_inspect_arguments(self) -> None:
        """No-op fallback ignores arguments — Plexus-absent has nothing to query.

        A future regression that started reading argument fields without
        the Plexus client wired in would surface here. The same call
        with different arguments must produce the same empty result.
        """
        adapter = PlexusAdapter()
        first = await adapter.query({"topic": "alpha", "limit": 5})
        second = await adapter.query({"different": "shape entirely"})
        assert first == second == {"results": [], "context": ""}


class TestRecordNoOp:
    """``record`` returns acknowledgement promptly when Plexus is absent.

    ADR-009: the orchestrator's ReAct loop must not block on ingestion.
    The no-op fallback satisfies this trivially — the call returns
    synchronously and acknowledges receipt.
    """

    @pytest.mark.asyncio
    async def test_record_returns_acknowledgement(self) -> None:
        adapter = PlexusAdapter()
        result = await adapter.record(
            {"ensemble_name": "composed-x", "quality_signal": "positive"}
        )
        assert result == {"acknowledged": True}

    @pytest.mark.asyncio
    async def test_record_does_not_inspect_arguments(self) -> None:
        adapter = PlexusAdapter()
        first = await adapter.record({"ensemble_name": "x"})
        second = await adapter.record({"completely": "different shape"})
        assert first == second == {"acknowledged": True}
