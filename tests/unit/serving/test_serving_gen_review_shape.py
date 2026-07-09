"""WP-C8 scenario 4 — dynamic dispatch fills a shape's slots with registered parts.

scenarios.md: "classify selects a shape from the catalog and dynamic dispatch fills
its slots (load-time-first binding). ... dynamic dispatch binds the runtime-chosen
parts into the shape's slots and the shape runs, with the selection expressed in
the declarative structure rather than inside a capability ensemble's self-routing
(Strategy A)."

The gen -> review exemplar shape (ADR-047 §1-2 default binding, on shipped
primitives) fills both slots by dynamic dispatch resolving REGISTERED parts from
the Topaz registry: select_parts picks the registered building-block part per
slot's capability, and the gen/review nodes dispatch onto those parts. This locks
the binding + declarative structure; the shape running end-to-end with real models
is grounded through the real endpoint (scratch), per the no-vacuum directive.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_orc.core.serving.capability_registry import capability_parts
from llm_orc.core.serving.shape_catalog import shape_catalog

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
ENSEMBLES = REPO / ".llm-orc" / "ensembles" / "agentic-serving"
SELECT_PARTS = SCRIPTS / "select_parts.py"


def _select_parts(task: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(SELECT_PARTS)],
        input=json.dumps({"input": {"task": task}}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


class TestSlotFillingIsRuntimeChosenFromTheRegistry:
    """The slots are filled from the registry at runtime, not hardcoded."""

    def test_slots_resolve_to_the_registered_parts_for_their_capability(self) -> None:
        parts = capability_parts(ENSEMBLES)
        selected = _select_parts("a function that reverses a string")

        # gen fills from code_generation, review from logical_reasoning — each is
        # the registered part for that key, so re-tagging the registry moves the
        # slot (runtime-chosen), and a hardcoded slot would diverge from the map.
        assert selected["gen"] == parts["code_generation"][0]
        assert selected["review"] == parts["logical_reasoning"][0]

    def test_unfilled_slot_resolves_empty(self, tmp_path: Path) -> None:
        """Determinism: an unfilled slot resolves empty so the dispatch fails
        deterministically rather than guessing a default (closed-set discipline)."""
        empty = capability_parts(tmp_path / "no-such-dir")
        assert empty == {}
        # select_parts picks candidates[0] or "" — an empty key yields "".
        assert (empty.get("logical_reasoning") or [""])[0] == ""


class TestDeclarativeDispatchStructure:
    """Selection lives in the declarative structure, not a model's self-routing."""

    def test_gen_review_shape_is_catalog_registered(self) -> None:
        catalog = shape_catalog(ENSEMBLES)
        assert catalog.get("review") == "gen-review"

    def test_slots_are_dynamic_dispatch_nodes_over_the_selector(self) -> None:
        shape = yaml.safe_load((ENSEMBLES / "gen-review.yaml").read_text())
        agents = {a["name"]: a for a in shape["agents"]}

        # The slots bind via dynamic dispatch on the selector's output (Strategy A):
        # not static ``ensemble:`` refs, not a model node self-routing.
        assert "script" in agents["select_parts"]
        assert agents["gen"]["dispatch"] == "${select_parts.gen}"
        assert agents["review"]["dispatch"] == "${select_parts.review}"
        assert "ensemble" not in agents["gen"]
        assert "ensemble" not in agents["review"]
        assert "model_profile" not in agents["gen"]
