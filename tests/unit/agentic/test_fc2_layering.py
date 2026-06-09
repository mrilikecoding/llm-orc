"""Static inspection tests for Fitness Criterion FC-2 (WP-B4).

Per ``docs/agentic-serving/system-design.agents.md`` §Fitness Criteria:

    FC-2 | Dependency edges point from higher layer to same-or-lower
    layer only | Static inspection of module imports against L0-L3
    assignment | 0 violations | Layering rule (Dependency Graph)

Layer assignment (per system-design.agents.md §Dependency Graph,
"Layering (inner → outer; post-Cycle 4)"):

* **L0** — Core: Ensemble Engine (lives at
  ``llm_orc.core.execution.ensemble_execution``).
* **L1** — Domain Policy: Composition Validator, Budget Controller,
  Autonomy Policy, Calibration Gate, Plexus Adapter, and the
  conditional Calibration Signal Channel (lands at WP-H4 per ADR-016).
* **L2** — Runtime: Result Summarizer Harness, Orchestrator Tool
  Dispatch, Orchestrator Runtime, and the new Conversation Compaction
  (WP-E4) and Tier-Escalation Router (WP-G4-1).
* **L3** — Entry: Serving Layer, Session Registry, Bootstrapping
  Pipeline, Orchestrator Configuration.

The layering rule (system-design.md §"Layering rule"): edges point
from a higher layer to a same-or-lower layer; never upward — **with
one narrow exception per ADR-016 (conditional acceptance):** a
read-only signal channel may flow from L0 (Ensemble Engine dispatch
outputs) to L1 (Calibration Signal Channel module). Pre-declared
below as an annotated allowed exception so the test is ready when
WP-H4 lands the channel.

Contract modules (``orchestrator_chunk``, ``session_start``,
``structural_errors``) carry typed value objects across layer
boundaries; they are layer-neutral and exempt from the layering
rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Module → layer assignment. Pre-declares the not-yet-landed Cycle 4
# modules so this test is ready when they ship — until then, the
# import scan simply does not find them.
_LAYER_MAP: dict[str, int] = {
    # L0 — Core
    "llm_orc.core.execution.ensemble_execution": 0,
    # L1 — Domain Policy
    "llm_orc.agentic.composition_validator": 1,
    "llm_orc.agentic.budget_controller": 1,
    "llm_orc.agentic.autonomy_policy": 1,
    "llm_orc.agentic.calibration_gate": 1,
    "llm_orc.agentic.plexus_adapter": 1,
    "llm_orc.agentic.calibration_signal_channel": 1,  # WP-H4 per ADR-016
    "llm_orc.agentic.dispatch_event_substrate": 1,  # WP-A6 per ADR-023
    "llm_orc.agentic.session_artifact_store": 1,  # WP-E6 per ADR-025
    "llm_orc.agentic.session_action_record": 1,  # WP-LB-K per ADR-037
    "llm_orc.agentic.sibling_interface_extractor": 1,  # loop-back #7 per ADR-039
    # L2 — Runtime
    "llm_orc.agentic.result_summarizer_harness": 2,
    "llm_orc.agentic.orchestrator_tool_dispatch": 2,
    "llm_orc.agentic.orchestrator_runtime": 2,
    "llm_orc.agentic.tool_call_validation_guard": 2,  # WP-C4 per ADR-017
    "llm_orc.agentic.conversation_compaction": 2,  # WP-E4 per ADR-012
    "llm_orc.agentic.tier_router": 2,  # WP-G4-1 per ADR-015
    "llm_orc.agentic.tier_router_audit": 2,  # WP-G4-2 per ADR-018
    "llm_orc.agentic.orchestrator_context_event_sink": 2,  # WP-C6 per ADR-023
    "llm_orc.agentic.dispatch_pipeline": 2,  # WP-A7 per ADR-027
    "llm_orc.agentic.ensemble_backed_roles": 2,  # WP-A7 per ADR-027/028/029
    "llm_orc.agentic.single_step_enforcer": 2,  # WP-LB-B per ADR-033
    "llm_orc.agentic.loop_driver": 2,  # WP-LB-B per ADR-033
    "llm_orc.agentic.delegation_rate_meter": 2,  # WP-LB-J per ADR-036 §Decision 3
    "llm_orc.agentic.artifact_bridge": 2,  # WP-LB-D per ADR-034
    # L3 — Entry
    "llm_orc.agentic.client_tool_action_terminal": 3,  # WP-LB-C per ADR-034
    "llm_orc.agentic.session_registry": 3,
    "llm_orc.agentic.session_artifacts": 3,  # WP-D4 per ADR-013
    "llm_orc.agentic.orchestrator_config": 3,
    "llm_orc.agentic.operator_terminal_event_sink": 3,  # WP-B6 per ADR-023
    "llm_orc.agentic.inference_wait_heartbeat": 3,  # WP-B6 piece 5 per ADR-023
    # Serving Layer lives outside src/llm_orc/agentic/ (in web/api/)
    # and Bootstrapping Pipeline is deferred (WP-J); both omitted
    # from the scoped layer map.
}
"""Module → L0–L3 assignment derived from system-design.agents.md."""

_CONTRACT_MODULES: frozenset[str] = frozenset(
    {
        "llm_orc.agentic.orchestrator_chunk",
        "llm_orc.agentic.session_start",
        "llm_orc.agentic.dispatch_envelope",  # WP-D6 per ADR-024
    }
)
"""Layer-neutral value-type / contract modules.

These carry typed shapes across layer boundaries (``SessionContext``,
``VisibilityEvent``, etc.) and are not behaviorally coupled to any
single layer. They are exempt from the layering rule.
"""

_ALLOWED_UPWARD_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        # ADR-016 (conditional acceptance — first-deployment evidence
        # pending): the Ensemble Engine (L0) emits read-only,
        # calibration-data-only signals to the Calibration Signal
        # Channel (L1). This is the single narrow upward exception
        # ADR-016 amends ADR-002 to permit. Enforced at runtime by
        # the channel's mechanism (e) schema validation.
        (
            "llm_orc.core.execution.ensemble_execution",
            "llm_orc.agentic.calibration_signal_channel",
        ),
    }
)
"""Edges that may legitimately point from a lower layer to a higher one."""


_AGENTIC_DIR = Path(__file__).resolve().parents[3] / "src" / "llm_orc" / "agentic"
_ENSEMBLE_ENGINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "llm_orc"
    / "core"
    / "execution"
    / "ensemble_execution.py"
)


def _imports_in(module_path: Path) -> set[str]:
    """Return every runtime ``llm_orc.agentic.*`` or
    ``llm_orc.core.execution.ensemble_execution`` import in this file.

    Imports inside ``if TYPE_CHECKING:`` blocks are excluded — they
    are a Python idiom for breaking circular type-annotation imports
    and do not execute at runtime, so they cannot violate the
    layering rule.
    """
    tree = ast.parse(module_path.read_text())
    skip_ids = _type_checking_node_ids(tree)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skip_ids:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_in_scope(alias.name):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_in_scope(node.module):
                imports.add(node.module)
    return imports


def _type_checking_node_ids(tree: ast.Module) -> set[int]:
    """Collect ``id()`` of every AST node nested under an ``if TYPE_CHECKING:``."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for sub in ast.walk(node):
                if sub is not node:
                    skip.add(id(sub))
    return skip


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _is_in_scope(module_name: str) -> bool:
    """Whether this module participates in the FC-2 layering scope."""
    return module_name.startswith("llm_orc.agentic") or module_name == (
        "llm_orc.core.execution.ensemble_execution"
    )


def _path_to_module(module_path: Path) -> str:
    """Convert a Path into its dotted module name within the project."""
    if module_path == _ENSEMBLE_ENGINE_PATH:
        return "llm_orc.core.execution.ensemble_execution"
    return f"llm_orc.agentic.{module_path.stem}"


def _agentic_module_paths() -> list[Path]:
    """All ``src/llm_orc/agentic/*.py`` files excluding ``__init__.py``."""
    return [p for p in sorted(_AGENTIC_DIR.glob("*.py")) if p.name != "__init__.py"]


class TestLayeringRule:
    """FC-2 — every import edge respects the layering rule (or is annotated)."""

    def test_every_agentic_import_edge_respects_the_layering_rule(self) -> None:
        violations: list[str] = []

        scan_paths = [*_agentic_module_paths(), _ENSEMBLE_ENGINE_PATH]
        for source_path in scan_paths:
            source_module = _path_to_module(source_path)
            if source_module not in _LAYER_MAP:
                continue
            source_layer = _LAYER_MAP[source_module]

            for target in _imports_in(source_path):
                if target in _CONTRACT_MODULES:
                    continue
                if target not in _LAYER_MAP:
                    continue  # caught by the coverage test below
                if (source_module, target) in _ALLOWED_UPWARD_EDGES:
                    continue

                target_layer = _LAYER_MAP[target]
                if target_layer > source_layer:
                    violations.append(
                        f"  {source_module} (L{source_layer}) → "
                        f"{target} (L{target_layer})"
                    )

        assert violations == [], (
            "FC-2 layering rule violated — the following import edges "
            "point from a lower layer to a higher layer without an "
            "annotated exception:\n" + "\n".join(violations) + "\n\n"
            "Per system-design.agents.md §Dependency Graph, edges must "
            "point from a higher layer to a same-or-lower layer "
            "(L3 → L2 → L1 → L0). The only allowed upward exception is "
            "the ADR-016 read-only signal channel "
            "(Ensemble Engine → Calibration Signal Channel)."
        )

    def test_layer_map_covers_every_behavioral_agentic_module(self) -> None:
        """Fail closed when a new agentic module lands without a layer decision.

        Contract modules (orchestrator_chunk, session_start) are
        layer-neutral and exempt; the structural-error base class
        lives outside the agentic package. Every other ``*.py`` under
        ``src/llm_orc/agentic/`` must have an explicit layer entry.
        """
        unclassified: list[str] = []
        for path in _agentic_module_paths():
            module = _path_to_module(path)
            if module in _CONTRACT_MODULES:
                continue
            if module not in _LAYER_MAP:
                unclassified.append(module)

        assert unclassified == [], (
            "Unclassified agentic modules — add a layer assignment to "
            "_LAYER_MAP (or to _CONTRACT_MODULES if the module is a "
            "layer-neutral value-type carrier):\n"
            + "\n".join(f"  {m}" for m in unclassified)
        )
