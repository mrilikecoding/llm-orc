"""Dispatch-discoverability test for the serving ``explainer`` seat (WP-A8).

Scenario 2 (scenarios.md "An explain turn routes through the same skeleton and
returns prose, not a file"): ``classify`` routes an explain turn to the target
``explainer`` (locked in ``test_serving_classify.py``), so the serving
ensemble's ``dispatch: "${classify.target}"`` seat resolves the name
``explainer`` against ``.llm-orc/ensembles``. Dispatch discovery is
non-recursive — ``_find_ensemble_in_dirs`` matches ``{dir}/{name}.yaml`` at the
top level only — so a nested ``agentic-serving/explainer.yaml`` needs a
top-level entry to be dispatchable (WP-A8 discovery note b: the same top-level
symlink ``code-seat`` needs).
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.config.ensemble_config import _find_ensemble_in_dirs

REPO = Path(__file__).resolve().parents[3]
ENSEMBLES = REPO / ".llm-orc" / "ensembles"


def test_explainer_seat_is_dispatch_discoverable_top_level() -> None:
    """The ``explainer`` target resolves via the exact mechanism the seat's
    dynamic dispatch uses, so a real explain turn can fill the seat."""
    config = _find_ensemble_in_dirs("explainer", [str(ENSEMBLES)])
    assert config is not None
    assert config.name == "explainer"
