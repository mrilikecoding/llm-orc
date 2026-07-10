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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
ENSEMBLES = REPO / ".llm-orc" / "ensembles" / "agentic-serving"
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


def test_build_code_round_code_writer_has_timeout_headroom() -> None:
    """The held-round input is the system's longest prompt (turn + reject
    report + held tests); the 300s default timed out the 3-call
    code-generator, shipped empty code, and killed the one convertible
    retry (spike 2026-07-10). The node needs explicit headroom, bounded
    by the dispatch/loop budget above it."""
    config = _load("build-code-round")
    assert config is not None
    writer = next(a for a in config.agents if a.name == "code_writer")
    assert writer.timeout_seconds == 600


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


# --- integration: the real held-round graph through the real engine ---
# build-code-round has no model nodes, so with an echo code-generator the
# whole held path (route -> dispatch -> gather -> executor -> gate ->
# envelope -> unwrap) runs hermetically.

_HELD_CARRY = (
    "Write is_even(n) in even.py\n\n"
    "[Previous round rejected: tests did not pass."
    " Executor report: test_even: AssertionError().]\n\n"
    f"{_HELD_MARKER}\n```python\n"
    "def test_even():\n    assert is_even(4) is True\n"
    "def test_odd():\n    assert is_even(3) is False\n```"
)

# single line, no backslashes: the script resolver treats a value containing
# '\' as a file path (same constraint as the endpoint fixture's echo seat)
_ECHO_CODE_GENERATOR = (
    "name: code-generator\n"
    "description: deterministic echo seat for the held-round integration test\n"
    "agents:\n"
    "  - name: out\n"
    "    script: \"echo 'def is_even(n): return n % 2 == 0'\"\n"
)


@pytest.mark.asyncio
async def test_held_carry_runs_the_code_only_round_end_to_end() -> None:
    """A held carry drives the REAL build-round graph: route picks
    build-code-round, dispatch executes it, gather holds the carried tests,
    the executor passes the echoed code against them, the gate carries round
    1's adequacy, and unwrap restores the bare accepted envelope the loop's
    ``until`` predicate reads."""
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        ens = proj / "ensembles"
        ens.mkdir()
        scripts = proj / "scripts" / "agentic_serving"
        scripts.parent.mkdir()
        shutil.copytree(SCRIPTS, scripts)
        shutil.copy(ENSEMBLES / "build-round.yaml", ens / "build-round.yaml")
        shutil.copy(ENSEMBLES / "build-code-round.yaml", ens / "build-code-round.yaml")
        (ens / "code-generator.yaml").write_text(_ECHO_CODE_GENERATOR)

        config = EnsembleLoader().load_from_file(str(ens / "build-round.yaml"))
        executor = ExecutorFactory.create_root_executor(project_dir=proj)
        mock_artifact = Mock()
        mock_artifact.save_execution_results = Mock()
        with patch.object(executor, "_artifact_manager", mock_artifact):
            result = await executor.execute(config, _HELD_CARRY)

    unwrap = result["results"]["unwrap"]
    assert unwrap["status"] == "success", unwrap
    envelope = json.loads(unwrap["response"])
    diag = envelope["diagnostics"]
    assert diag["tests_pass"] is True, diag
    assert diag["tests_adequate"] is True
    assert "carried" in diag["accept_reason"]
    assert diag["accept"] is True
    assert diag["held_round"] is True
    assert "def is_even" in envelope["primary"]
