"""Unit tests for the build-round router nodes (issue #100).

The TDD retry loop's body is a deterministic router: ``route`` picks the
round shape from the carried input (fresh TDD round, or the held round when
the carry bears the HELD TESTS sentinel), ``dispatch`` runs it, and
``dispatch_unwrap`` peels the dispatch wrapper back to the bare ADR-024
round envelope so the loop's ``until``/``carry`` predicates keep reading
``${diagnostics.*}`` unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
ROUTE = SCRIPTS / "route_round.py"
UNWRAP = SCRIPTS / "dispatch_unwrap.py"

_HELD_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"


def _run(script: Path, payload: dict[str, Any]) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_route_sends_a_fresh_turn_to_the_full_tdd_round() -> None:
    out = _run(ROUTE, {"input": "Write is_even(n) in even.py"})
    assert out["target"] == "build-gated-round"
    assert out["round_input"] == "Write is_even(n) in even.py"


def test_route_sends_a_held_carry_to_the_code_only_round() -> None:
    carry = (
        "Write is_even(n) in even.py\n\n"
        "[Previous round rejected: tests did not pass.]\n\n"
        f"{_HELD_MARKER}\n```python\ndef test_even():\n    assert is_even(4)\n```"
    )
    out = _run(ROUTE, {"input": carry})
    assert out["target"] == "build-code-round"
    assert out["round_input"] == carry


def test_dispatch_unwrap_restores_the_bare_round_envelope() -> None:
    envelope = {
        "status": "success",
        "primary": "x = 1",
        "diagnostics": {"accept": True},
    }
    child_result = {
        "ensemble": "build-gated-round",
        "deliverable": json.dumps(envelope),
        "results": {},
    }
    payload = {"dependencies": {"dispatch": {"response": json.dumps(child_result)}}}
    assert _run(UNWRAP, payload) == envelope


def test_dispatch_unwrap_emits_empty_object_when_deliverable_is_missing() -> None:
    payload = {"dependencies": {"dispatch": {"response": json.dumps({"x": 1})}}}
    assert _run(UNWRAP, payload) == {}


# --- wiring: the loop body routes, and both round shapes are resolvable ---


def _load(name: str) -> Any:
    from llm_orc.core.config.ensemble_config import _find_ensemble_in_dirs

    ensembles = REPO / ".llm-orc" / "ensembles"
    return _find_ensemble_in_dirs(name, [str(ensembles)])


def test_both_round_shapes_are_dispatch_discoverable_top_level() -> None:
    """route's dispatch resolves names non-recursively at the ensembles top
    level, so both targets need top-level entries (the symlink convention)."""
    for name in ("build-round", "build-code-round"):
        config = _load(name)
        assert config is not None, name
        assert config.name == name


def test_build_gated_loop_body_is_the_router() -> None:
    config = _load("build-gated")
    assert config is not None
    round_agent = next(a for a in config.agents if a.name == "round")
    assert round_agent.loop.body == "build-round"


def test_build_code_round_has_no_test_writer_and_gate_reads_gather() -> None:
    """The held round: code-only against the carried spec; the gate carries
    round 1's adequacy from gather's held flag (no judge seat)."""
    config = _load("build-code-round")
    assert config is not None
    names = {a.name for a in config.agents}
    assert "test_writer" not in names
    assert "judge" not in names
    gate = next(a for a in config.agents if a.name == "accept_gate")
    assert set(gate.depends_on) == {"executor", "gather"}
