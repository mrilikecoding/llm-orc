"""#181: the artifact the gate accepted is the artifact that ships.

``accept_gather`` repairs the candidate (injected workspace imports, trimmed
trailing prose) BEFORE the executor runs the suite. The envelope used to
re-derive the code from the raw ``code_writer`` text, so the bytes the gate
verified were not the bytes the client wrote. These pins drive the real chain
of scripts (gather -> executor -> judge -> accept_gate -> envelope, and the
re-fix and write-tests tails) and assert on the bytes in the emitted envelope.

Every workspace here is a ``tmp_path``; nothing reads the current directory's
``.llm-orc`` state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"

_PRICING = "def discount(price, percent):\n    return price - price * percent // 100\n"
_TOTAL_TESTS = (
    "```python\n"
    "from solution import total\n"
    "def test_t():\n"
    "    assert total(200) == 180\n"
    "```\n"
)
_REQUIREMENT = "write total in solution.py that applies a 10 percent discount"


def _node(script: str, deps: dict[str, Any], input_data: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"dependencies": deps}
    if input_data:
        payload["input_data"] = input_data
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _dep(value: Any) -> dict[str, str]:
    return {"response": value if isinstance(value, str) else json.dumps(value)}


def _sub_ensemble(terminal_text: str) -> str:
    return json.dumps(
        {
            "ensemble": "x",
            "status": "completed",
            "results": {"out": {"response": terminal_text, "status": "success"}},
        }
    )


def _context(requirement: str, workspace: dict[str, str]) -> str:
    blocks = ""
    for path, body in workspace.items():
        indented = "\n".join(f"  {line}" if line else "" for line in body.splitlines())
        blocks += f"assistant: [read {path}]\n{indented}\n\n"
    return f"{blocks}\n\nCurrent request: {requirement}"


def _build_chain(
    code_writer: str,
    tests_writer: str,
    workspace: dict[str, str],
    requirement: str = _REQUIREMENT,
) -> dict[str, Any]:
    """The build-gated-round tail with every node real."""
    gather = _node(
        "accept_gather.py",
        {
            "code_writer": _dep(_sub_ensemble(code_writer)),
            "test_writer": _dep(_sub_ensemble(tests_writer)),
        },
        input_data=_context(requirement, workspace),
    )
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    judge = _node("adequacy_check.py", {"executor": _dep(executor)})
    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "judge": _dep(judge)}
    )
    envelope = _node(
        "build_gated_envelope.py",
        {
            "code_writer": _dep(_sub_ensemble(code_writer)),
            "executor": _dep(executor),
            "accept_gate": _dep(accept_gate),
        },
    )
    return {
        "gather": gather,
        "executor": executor,
        "accept_gate": accept_gate,
        "envelope": envelope,
    }


def _run_pytest(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
        cwd=directory,
        capture_output=True,
        text=True,
    )


def test_a_shipped_code_carries_the_import_the_gate_verified_with(
    tmp_path: Path,
) -> None:
    """The issue's measured example: the only defect is a missing workspace
    import. Verified bytes == shipped bytes, and the shipped file runs."""
    raw = "def total(price):\n    return discount(price, 10)\n"
    result = _build_chain(
        f"```python\n{raw}```\n", _TOTAL_TESTS, {"pricing.py": _PRICING}
    )

    assert result["accept_gate"]["accept"] is True, result["accept_gate"]["reason"]
    shipped = result["envelope"]["artifacts"][0]["content"]
    assert shipped == result["executor"]["code"]
    assert "from pricing import discount" in shipped
    assert result["envelope"]["primary"] == shipped

    (tmp_path / "pricing.py").write_text(_PRICING)
    (tmp_path / "solution.py").write_text(shipped + "\n")
    (tmp_path / "test_solution.py").write_text(result["executor"]["tests"])
    run = _run_pytest(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr


def test_b_a_deliverable_needing_no_repair_ships_byte_identical() -> None:
    raw = "def total(price):\n    return price - price // 10"
    result = _build_chain(f"```python\n{raw}\n```\n", _TOTAL_TESTS, {})

    assert result["accept_gate"]["accept"] is True
    assert result["envelope"]["primary"] == raw
    assert result["envelope"]["structured"]["content"] == raw
    assert result["envelope"]["artifacts"][0]["content"] == raw


def test_c_trimmed_trailing_prose_is_not_shipped() -> None:
    """The other gather repair: a prose line left inside the fence is trimmed
    before the run. It must not reach the client either."""
    body = "def total(price):\n    return price - price // 10"
    fenced = f"```python\n{body}\nUsage: call total(200) to get 180.\n```\n"
    result = _build_chain(fenced, _TOTAL_TESTS, {})

    assert result["accept_gate"]["accept"] is True, result["accept_gate"]["reason"]
    assert result["envelope"]["primary"] == body
    assert result["envelope"]["primary"] == result["executor"]["code"]


def test_d_a_held_round_ships_the_verified_bytes() -> None:
    """build-code-round: no test_writer seat, the carry's held block is the
    spec. Same gather repair, same envelope."""
    held = (
        f"{_REQUIREMENT}\n\n[HELD TESTS: round 1 spec; regenerate ONLY the code]\n"
        f"{_TOTAL_TESTS}"
    )
    raw = "def total(price):\n    return discount(price, 10)\n"
    gather = _node(
        "accept_gather.py",
        {"code_writer": _dep(_sub_ensemble(f"```python\n{raw}```\n"))},
        input_data=_context(held, {"pricing.py": _PRICING}),
    )
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    envelope = _node(
        "build_gated_envelope.py",
        {
            "code_writer": _dep(_sub_ensemble(f"```python\n{raw}```\n")),
            "executor": _dep(executor),
            "accept_gate": _dep({"accept": True, "reason": "ok"}),
            "gather": _dep(gather),
        },
    )

    assert gather["held"] is True
    assert executor["tests_pass"] is True, executor["report"]
    assert envelope["primary"] == executor["code"]
    assert "from pricing import discount" in envelope["primary"]


def test_e_refix_ships_the_bytes_the_executor_ran() -> None:
    """The re-fix tail: select derives the candidate once, the executor runs
    it, refix_envelope ships it. A model edit that echoes a test block next
    to the fix must ship without the test block, and as run."""
    fix = "def restock(n):\n    return n + 1\n"
    model_edit = f"```python\n{fix}```\n```python\ndef test_x():\n    pass\n```\n"
    gathered = {
        "deterministic_code": "",
        "visible_test": (
            "from solution import restock\n"
            "def test_restock():\n    assert restock(1) == 2\n"
        ),
        "prior_code": "def restock(n):\n    return n\n",
        "task": "fix restock",
        "target_file": "solution.py",
        "target_path": "solution.py",
        "workspace": {},
    }
    selected = _node(
        "refix_select.py",
        {
            "gather": _dep(gathered),
            "model_edit": _dep(_sub_ensemble(model_edit)),
        },
    )
    executor = _node("accept_executor.py", {"select": _dep(selected)})
    envelope = _node(
        "refix_envelope.py", {"select": _dep(selected), "executor": _dep(executor)}
    )

    assert envelope["diagnostics"]["accept"] is True
    assert envelope["primary"] == executor["code"]
    assert envelope["primary"] == fix.strip()


def test_f_write_tests_ships_the_tests_the_executor_ran() -> None:
    """The tests deliverable: tests_envelope ships the executor's echo of the
    tests (after the executor's own repairs), never a re-derivation."""
    gather = _node(
        "tests_gather.py",
        {"test_writer": _dep(_sub_ensemble(_TOTAL_TESTS))},
        input_data=_context("write tests for total in solution.py", {}),
    )
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "judge": _dep({})}
    )
    envelope = _node(
        "tests_envelope.py",
        {"executor": _dep(executor), "accept_gate": _dep(accept_gate)},
    )

    assert envelope["primary"] == executor["tests"]
    assert envelope["artifacts"][0]["content"] == executor["tests"]


# --- the envelope fails closed when it lacks the verified code ------------

_UNAVAILABLE = "verified code unavailable"
_RAW_CODE = "def total(price):\n    return price - price // 10\n"


def _executor_with(code: Any) -> str:
    return json.dumps({"tests_pass": True, "n_tests": 1, "code": code})


# Each shape still reads as a pass to accept_gate's lenient extraction.
_UNVERIFIED_EXECUTOR_RESPONSES = {
    "junk-prefixed-json": "note\n" + _executor_with("x = 1"),
    "json-list": json.dumps([{"tests_pass": True, "code": "x = 1"}]),
    "no-code-key": json.dumps({"tests_pass": True, "n_tests": 1}),
    "code-null": _executor_with(None),
    "code-int": _executor_with(5),
    "code-empty": _executor_with(""),
}


def _gate_and_envelope(executor_response: str) -> dict[str, Any]:
    """A hand-built executor dependency run through the real accept_gate and
    build_gated_envelope."""
    accept_gate = _node(
        "accept_gate.py",
        {
            "executor": {"response": executor_response},
            "judge": _dep({"tests_adequate": True}),
        },
    )
    envelope = _node(
        "build_gated_envelope.py",
        {
            "code_writer": _dep(_sub_ensemble(f"```python\n{_RAW_CODE}```\n")),
            "executor": {"response": executor_response},
            "accept_gate": _dep(accept_gate),
        },
    )
    return {"accept_gate": accept_gate, "envelope": envelope}


def _client_outcome(envelope: dict[str, Any]) -> dict[str, Any]:
    seat = json.dumps({"results": {"gather": _dep("..."), "envelope": _dep(envelope)}})
    resolve = _dep(
        {
            "target": "build-gated",
            "kind": "python_module",
            "file": "solution.py",
            "build": True,
            "dispatch_input": "write it",
            "is_build_ask": True,
        }
    )
    deps: dict[str, Any] = {"resolve": resolve, "seat": {"response": seat}}
    seat_contract = _node("seat_contract.py", deps)
    shape = _node("shape.py", {**deps, "seat_contract": _dep(seat_contract)})
    form_gate = _node("form_gate.py", {"shape": _dep(shape)})
    return _node("emit.py", {"form_gate": _dep(form_gate)})


def test_h_a_rejected_round_carries_the_code_the_failure_report_describes() -> None:
    """The reject envelope feeds the next round; its code is the executed
    candidate (repairs included), not the raw extraction."""
    wrong_tests = _TOTAL_TESTS.replace("== 180", "== 999")
    raw = "def total(price):\n    return discount(price, 10)\n"
    result = _build_chain(
        f"```python\n{raw}```\n", wrong_tests, {"pricing.py": _PRICING}
    )

    assert result["accept_gate"]["accept"] is False
    envelope = result["envelope"]
    assert envelope["diagnostics"]["accept"] is False
    assert envelope["primary"] == result["executor"]["code"]
    assert "from pricing import discount" in envelope["primary"]


@pytest.mark.parametrize("label", sorted(_UNVERIFIED_EXECUTOR_RESPONSES))
def test_g_an_envelope_without_the_verified_code_never_accepts(label: str) -> None:
    result = _gate_and_envelope(_UNVERIFIED_EXECUTOR_RESPONSES[label])

    assert result["accept_gate"]["accept"] is True
    diag = result["envelope"]["diagnostics"]
    assert diag["accept"] is False
    assert _UNAVAILABLE in diag["accept_reason"]
    assert result["envelope"]["primary"] not in ("None", "5")

    outcome = _client_outcome(result["envelope"])
    assert outcome["finish"] is True
    assert _UNAVAILABLE in outcome["content"]
    assert "tool_calls" not in outcome
