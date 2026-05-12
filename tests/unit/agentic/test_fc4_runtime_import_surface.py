"""Static inspection tests for Fitness Criterion FC-4.

Per ``docs/agentic-serving/system-design.md`` §Fitness Criteria, FC-4
(as amended by Design Amendment #3):

    Orchestrator Runtime imports only Budget Controller and
    Orchestrator Tool Dispatch — no Result Summarizer Harness, no
    Plexus, no config, no Autonomy, no Calibration. Result Summarizer
    Harness is interposed on the ``invoke_ensemble`` return path by
    Orchestrator Tool Dispatch (Amendment #3) | Static import check |
    Exact match.

The Runtime's import surface is the structural enforcement of
ADR-003's closed-set property at the reasoning layer: if the Runtime
can only reach Budget and Tool Dispatch, no summarizer, Plexus,
Autonomy, or Calibration code can leak into the orchestrator LLM's
mental model. Summarization is a Tool-Dispatch-side concern — the
Runtime remains unaware that results are summarized at all.

Accepted (read: contract types only, no behavioral dependencies):

- Budget Controller (L1 policy — the primary FC-4 allowance).
- Orchestrator Tool Dispatch (L2 — primary FC-4 allowance).
- session_start — contract type ``SessionContext`` crosses the
  Serving Layer → Runtime boundary. Neutral.
- orchestrator_chunk — contract types on the same edge. Neutral.
- models.base — ``ToolCallingResponse`` is the unified tool-calling
  shape that satisfies the ``OrchestratorLLM`` Protocol. Neutral.

Forbidden:

- result_summarizer_harness — per Amendment #3 the Harness is
  interposed by Tool Dispatch on ``invoke_ensemble``'s return path.
  The Runtime is unaware of summarization; importing the Harness from
  Runtime would re-couple the two and defeat AS-7's structural
  enforcement.
- orchestrator_config — L3 config; Session state resolution belongs
  in Serving Layer, not Runtime.
- session_registry — L3 registry; Runtime gets state through
  ``SessionContext``, not by reaching back into the registry.
- Future Plexus Adapter, Autonomy Policy, Calibration Gate modules
  once they exist (placeholders checked by name so this test fails
  closed when those modules land).
"""

from __future__ import annotations

import ast
from pathlib import Path

from llm_orc.agentic import orchestrator_runtime

_RUNTIME_MODULE_PATH = Path(orchestrator_runtime.__file__)

_ALLOWED_AGENTIC_IMPORTS = frozenset(
    {
        "llm_orc.agentic.budget_controller",
        "llm_orc.agentic.conversation_compaction",  # WP-E4 per ADR-012
        "llm_orc.agentic.orchestrator_chunk",
        "llm_orc.agentic.orchestrator_tool_dispatch",
        "llm_orc.agentic.session_start",
    }
)
"""Modules the Runtime is allowed to import from the scoped agentic layer.

The Conversation Compaction entry was added by WP-E4 per ADR-012 and
system-design.agents.md §"Orchestrator Runtime → Conversation
Compaction" — the Runtime invokes ``compact()`` at every turn
boundary; the resulting compacted messages array is what flows into
the next ``generate_with_tools`` call."""

_FORBIDDEN_AGENTIC_IMPORTS = frozenset(
    {
        # Summarization is a Tool-Dispatch-side concern per Amendment #3.
        "llm_orc.agentic.result_summarizer_harness",
        "llm_orc.agentic.orchestrator_config",
        "llm_orc.agentic.session_registry",
        # The three modules below do not exist yet; naming them here
        # makes the test fail closed when they land.
        "llm_orc.agentic.plexus_adapter",
        "llm_orc.agentic.autonomy_policy",
        "llm_orc.agentic.calibration_gate",
    }
)
"""Modules FC-4 forbids the Runtime from importing."""


def _agentic_imports_in(module_path: Path) -> set[str]:
    """Return every ``llm_orc.agentic.*`` module the file imports.

    Walks ``ast.Import`` and ``ast.ImportFrom`` nodes. Catches both
    direct ``import x`` and ``from x import Y`` forms at any
    position in the file. Names from
    ``from llm_orc.agentic.X import Y, Z`` collapse to the parent
    module path ``llm_orc.agentic.X``.
    """
    tree = ast.parse(module_path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("llm_orc.agentic"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("llm_orc.agentic"):
                imports.add(node.module)
    return imports


class TestRuntimeImportSurface:
    """FC-4 is a negative invariant: certain modules must not appear."""

    def test_runtime_does_not_import_any_forbidden_agentic_module(self) -> None:
        imports = _agentic_imports_in(_RUNTIME_MODULE_PATH)
        leaked = imports & _FORBIDDEN_AGENTIC_IMPORTS
        assert leaked == set(), (
            f"Orchestrator Runtime imports forbidden modules: {sorted(leaked)}. "
            "FC-4 requires the Runtime to stay ignorant of Plexus, Autonomy, "
            "Calibration, Session Registry, and Orchestrator Configuration — "
            "Session state reaches Runtime through SessionContext; Plexus / "
            "Autonomy / Calibration are Tool-Dispatch-side concerns."
        )

    def test_runtime_only_imports_allow_listed_agentic_modules(self) -> None:
        """Positive check: every agentic import is on the allow list.

        Catches imports the forbidden list does not enumerate (e.g.,
        a future module landing that no one has decided the Runtime's
        relationship to). Fails closed so every new agentic module
        pulls this test and forces an explicit allow/deny decision.
        """
        imports = _agentic_imports_in(_RUNTIME_MODULE_PATH)
        unexpected = imports - _ALLOWED_AGENTIC_IMPORTS
        assert unexpected == set(), (
            f"Orchestrator Runtime imports unexpected agentic modules: "
            f"{sorted(unexpected)}. FC-4 keeps Runtime's dependency set "
            "narrow; add to the allow list only after confirming the new "
            "module belongs in Runtime's reasoning surface, or route the "
            "dependency through Tool Dispatch instead."
        )
