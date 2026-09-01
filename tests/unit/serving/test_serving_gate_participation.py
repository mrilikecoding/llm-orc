"""#171: the deliverable must participate in its own acceptance.

``accept = tests_pass AND tests_adequate`` (accept_gate.py). Neither input
can observe that the deliverable is wrong: the executor's ``_materialize``
shadows the target file only, so tests that exercise another workspace
module pass against any deliverable at all, and ``adequacy_check.py`` is a
static analysis of the tests alone and never sees the code. Measured through
the real chain, a suite satisfied entirely by the workspace accepted junk
content on two of three target-file shapes, and every import dialect
(``import X`` + attribute, aliased import, star import, a facade re-export,
and the injector's own diversion) defeated a static "tests reference the
candidate" rule.

The fix (design: docs/plans/2026-08-31-171-gate-participation-design.md) is
a RUNTIME ABLATION CONTROL in accept_executor.py: when the suite passes and
the deliverable is non-empty, run one more child with the SAME workspace and
repaired tests but the deliverable's bytes absent. If that control also
passes, the tests never exercised the deliverable, and accept must come out
False with a path-free, honest reason — flowing through accept_gate (build-
gated) and refix_envelope (re-fix) as accept_reason.

Driven via subprocess exactly as the L0 engine runs a script node, matching
every sibling harness in this directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"

_HELD_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"

_NONPARTICIPATION_REASON = "the tests never exercise the deliverable"

_RESTOCK_INVENTORY = "def restock(item, n):\n    return n + 1\n"
_RESTOCK_TESTS_BODY = (
    "from inventory import restock\n"
    "def test_restock():\n"
    "    assert restock('x', 2) == 3\n"
)


def _node(script: str, deps: dict[str, Any], input_data: str = "") -> dict[str, Any]:
    """One script node, as the engine runs it."""
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


def _sub_ensemble_response(terminal_text: str) -> str:
    return json.dumps(
        {
            "ensemble": "x",
            "status": "completed",
            "results": {"out": {"response": terminal_text, "status": "success"}},
        }
    )


def _build_gated(
    code_writer: str,
    tests_writer: str,
    requirement: str,
    *,
    workspace_file: str = "inventory.py",
    workspace_body: str = _RESTOCK_INVENTORY,
) -> dict[str, Any]:
    """gather -> executor -> judge -> accept_gate -> envelope, all real."""
    context = (
        f"assistant: [read {workspace_file}]\n"
        + "".join(f"  {line}\n" for line in workspace_body.splitlines())
        + f"\n\nCurrent request: {requirement}"
    )
    gather = _node(
        "accept_gather.py",
        {
            "code_writer": _dep(_sub_ensemble_response(code_writer)),
            "test_writer": _dep(_sub_ensemble_response(tests_writer)),
        },
        input_data=context,
    )
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    judge = _node("adequacy_check.py", {"executor": _dep(executor)})
    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "judge": _dep(judge)}
    )
    envelope = _node(
        "build_gated_envelope.py",
        {
            "code_writer": _dep(_sub_ensemble_response(code_writer)),
            "executor": _dep(executor),
            "accept_gate": _dep(accept_gate),
        },
    )
    return {
        "gather": gather,
        "executor": executor,
        "judge": judge,
        "accept_gate": accept_gate,
        "envelope": envelope,
    }


# --- the issue's own reproduction table (target_file shapes) --------------


@pytest.mark.parametrize(
    ("label", "requirement"),
    [
        ("no target file named", "fix restock so it adds two"),
        ("wrong-named target file", "fix restock so it adds two, in helpers.py"),
    ],
)
def test_workspace_satisfied_tests_no_longer_accept_wrong_code(
    label: str, requirement: str
) -> None:
    """The issue's table: with inventory.py in the workspace and the
    requirement naming no file, or a DIFFERENT file, the tests are satisfied
    by the workspace regardless of the deliverable — a wrong function used
    to ship on both shapes."""
    result = _build_gated(
        code_writer="```python\ndef restock(item, n):\n    return n\n```\n",  # wrong
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement=requirement,
    )

    assert result["executor"]["tests_pass"] is True, label
    assert result["accept_gate"]["accept"] is False, label
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON, label
    assert result["envelope"]["diagnostics"]["accept"] is False, label


@pytest.mark.parametrize(
    ("label", "junk_code"),
    [
        ("assignment-only", "x = 1"),
        ("wrong-function", "def restock(item, n):\n    return n"),
        ("comment-only", "# TODO: later"),
    ],
)
def test_workspace_satisfied_tests_no_longer_accept_junk_content(
    label: str, junk_code: str
) -> None:
    """The issue's junk-content sweep, target_file unnamed."""
    result = _build_gated(
        code_writer=f"```python\n{junk_code}\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="fix restock so it adds two",
    )

    assert result["executor"]["tests_pass"] is True, label
    assert result["accept_gate"]["accept"] is False, label
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON, label


def test_workspace_satisfied_junk_with_a_named_target_file_also_refuses() -> None:
    """The +1 integration test: both dimensions together (junk content AND
    a named-but-wrong target file), proving the two compose."""
    result = _build_gated(
        code_writer="```python\n# TODO: later\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="fix restock so it adds two, in helpers.py",
    )

    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


def test_the_genuinely_wrong_target_file_shape_is_unaffected() -> None:
    """The issue's third row: naming the REAL target file shadows it with
    the wrong code, so the workspace's own copy is gone and the tests fail
    honestly — this shape was never broken and must stay a plain test
    failure, not a participation refusal."""
    result = _build_gated(
        code_writer="```python\ndef restock(item, n):\n    return n\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="fix restock so it adds two, in inventory.py",
    )

    assert result["executor"]["tests_pass"] is False
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] != _NONPARTICIPATION_REASON


# --- the import-dialect wrong-accept set (killed the static rule) ---------


def test_import_module_then_attribute_call_refuses_junk() -> None:
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\nimport inventory\n"
            "def test_restock():\n    assert inventory.restock('x', 2) == 3\n```\n"
        ),
        requirement="fix restock",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


def test_aliased_import_refuses_junk() -> None:
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\nimport inventory as inv\n"
            "def test_restock():\n    assert inv.restock('x', 2) == 3\n```\n"
        ),
        requirement="fix restock",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


def test_star_import_refuses_junk() -> None:
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\nfrom inventory import *\n"
            "def test_restock():\n    assert restock('x', 2) == 3\n```\n"
        ),
        requirement="fix restock",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


def test_facade_reexport_refuses_junk() -> None:
    """Two workspace files: inventory.py (the real implementation) and
    facade.py (re-exporting it) — tests import from the facade, never the
    inventory module by name."""
    context = (
        "assistant: [read inventory.py]\n"
        + "".join(f"  {line}\n" for line in _RESTOCK_INVENTORY.splitlines())
        + "assistant: [read facade.py]\n"
        "  from inventory import restock\n"
        "\n\nCurrent request: fix restock"
    )
    gather = _node(
        "accept_gather.py",
        {
            "code_writer": _dep(_sub_ensemble_response("```python\nx = 1\n```\n")),
            "test_writer": _dep(
                _sub_ensemble_response(
                    "```python\nfrom facade import restock\n"
                    "def test_restock():\n    assert restock('x', 2) == 3\n```\n"
                )
            ),
        },
        input_data=context,
    )
    assert len(gather["workspace"]) == 2
    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    judge = _node("adequacy_check.py", {"executor": _dep(executor)})
    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "judge": _dep(judge)}
    )

    assert executor["tests_pass"] is True
    assert accept_gate["accept"] is False
    assert accept_gate["reason"] == _NONPARTICIPATION_REASON


def test_injected_import_diversion_refuses_junk() -> None:
    """The self-inflicted shape (WA-2b): tests reference the workspace's
    name bare, with no import at all — accept_gather's OWN injector adds
    one. Distinct from the accept_gather-level pin (which proves the
    injector no longer diverts a name the CANDIDATE defines): here the
    candidate does NOT define the name, so injection still (correctly)
    happens, and the ablation is what has to catch the resulting
    non-participation."""
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\ndef test_restock():\n    assert restock('x', 2) == 3\n```\n"
        ),
        requirement="fix restock",
    )
    assert "from inventory import restock" in result["gather"]["tests"]
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


def _executor_with_env(
    gathered: dict[str, Any], env_overrides: dict[str, str]
) -> dict[str, Any]:
    """The executor driven directly, with environment overrides — for
    exercising the budget knobs (LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT /
    _BUDGET), which ``_node`` has no way to thread through."""
    payload = json.dumps({"dependencies": {"gather": _dep(gathered)}})
    env = {**os.environ, **env_overrides}
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / "accept_executor.py")],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_control_runs_for_unittest_testcase_suites() -> None:
    """F1 (review, BLOCKER): ``_enumerate_tests`` returns ``None`` for ANY
    nested ``test_*`` def, including every TestCase method — so this shape
    takes ``_run_sandboxed``'s legacy single-run branch, which used to
    hardcode ``participates=True`` unconditionally. The whole junk sweep
    above ships verbatim in the TestCase dialect, and it is a common one
    (9/55 recorded live suites, per review)."""
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\nimport unittest\nimport inventory\n"
            "class TestRestock(unittest.TestCase):\n"
            "    def test_restock(self):\n"
            "        self.assertEqual(inventory.restock('x', 2), 3)\n```\n"
        ),
        requirement="fix restock",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == _NONPARTICIPATION_REASON


# --- held-round and re-fix: the routes the adequacy seam cannot see -------


def test_held_round_ablation_still_refuses_a_workspace_satisfied_junk_code() -> None:
    """Held-round participation pin: accept_gate._resolve_adequacy carries
    round 1's verdict unconditionally on a held round (no judge seat on
    this route), so the executor's ablation is the ONLY seam that can catch
    a non-participating candidate here."""
    context = "assistant: [read inventory.py]\n" + "".join(
        f"  {line}\n" for line in _RESTOCK_INVENTORY.splitlines()
    )
    held_input = (
        context
        + "\n\nCurrent request: fix restock\n\n"
        + _HELD_MARKER
        + "\n```python\n"
        + _RESTOCK_TESTS_BODY
        + "```"
    )
    gather = _node(
        "accept_gather.py",
        {"code_writer": _dep(_sub_ensemble_response("```python\nx = 1\n```\n"))},
        input_data=held_input,
    )
    assert gather["held"] is True

    executor = _node("accept_executor.py", {"gather": _dep(gather)})
    assert executor["tests_pass"] is True

    accept_gate = _node(
        "accept_gate.py", {"executor": _dep(executor), "gather": _dep(gather)}
    )
    assert accept_gate["tests_adequate"] is True  # carried, per #100
    assert accept_gate["accept"] is False
    assert accept_gate["reason"] == _NONPARTICIPATION_REASON


def test_refix_ablation_refuses_junk_against_an_unrelated_visible_test() -> None:
    """Re-fix participation pin: refix_envelope's own accept formula has no
    adequacy seat at all (``tests_pass and candidate_present and not
    inert``) — this is the other route accept_gate cannot see."""
    selected = {
        "requirement": "fix calc.py",
        "code": "x = 1\n",
        "tests": "def test_unrelated():\n    assert 1 + 1 == 2\n",
        "target_file": "calc.py",
        "edit_kind": "model",
        "smoke_only": False,
    }
    executor = _node("accept_executor.py", {"select": _dep(selected)})
    assert executor["tests_pass"] is True

    envelope = _node(
        "refix_envelope.py", {"select": _dep(selected), "executor": _dep(executor)}
    )
    assert envelope["diagnostics"]["accept"] is False
    assert envelope["diagnostics"]["accept_reason"] == _NONPARTICIPATION_REASON


# --- healthy shapes still accept (the over-refusal direction) -------------


def test_a_healthy_build_that_genuinely_exercises_the_deliverable_still_accepts() -> (
    None
):
    result = _build_gated(
        code_writer="```python\ndef restock(item, n):\n    return n + 1\n```\n",
        tests_writer=(
            "```python\nfrom solution import restock\n"
            "def test_restock():\n    assert restock('x', 2) == 3\n```\n"
        ),
        requirement="write restock in solution.py",
        workspace_file="",
        workspace_body="",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True
    assert result["accept_gate"]["reason"] != _NONPARTICIPATION_REASON


def test_a_mutation_pattern_test_where_the_deliverable_is_the_mutator_accepts() -> None:
    """adequacy_check's own documented mutation-pattern shape: the compared
    name is call-free but was passed to a real call earlier in the same
    test body — the deliverable IS what mutates it, so an empty-code
    control cannot reproduce the pass."""
    result = _build_gated(
        code_writer=(
            "```python\ndef add_todo(todos, item):\n    todos.append(item)\n```\n"
        ),
        tests_writer=(
            "```python\nfrom solution import add_todo\n"
            "def test_add_todo():\n"
            "    todos = []\n"
            "    add_todo(todos, 'x')\n"
            "    assert todos == ['x']\n```\n"
        ),
        requirement="write add_todo in solution.py",
        workspace_file="",
        workspace_body="",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True


def test_write_tests_shape_skips_the_ablation_on_empty_code() -> None:
    """#98: write-tests turns carry code: "" by design — an ablation there
    is vacuous (there is nothing to prove necessary) and would refuse every
    write-tests turn."""
    gathered: dict[str, Any] = {
        "requirement": "write tests for restock",
        "code": "",
        "tests": _RESTOCK_TESTS_BODY,
        "workspace": {"inventory.py": _RESTOCK_INVENTORY},
        "target_file": "",
    }
    executor = _node(
        "accept_executor.py",
        {"gather": _dep(gathered)},
        input_data=str(gathered["requirement"]),
    )
    assert executor["tests_pass"] is True
    assert executor["participates"] is True
    assert executor.get("participation_reason", "") == ""


# --- budget accounting -----------------------------------------------------

_ABLATION_BUDGET_REASON = (
    "the runtime ablation control could not run within the aggregate budget"
)


def test_the_control_child_is_counted_inside_the_aggregate_budget() -> None:
    """The control is one extra child inside _run_children's aggregate wall
    budget, not an unbounded add-on — a passing suite plus the control must
    stay well inside the budget on an ordinary turn."""
    start = time.monotonic()
    result = _node(
        "accept_executor.py",
        {
            "gather": _dep(
                {
                    "requirement": "adds",
                    "code": "def add(a, b):\n    return a + b\n",
                    "tests": "def test_add():\n    assert add(1, 2) == 3\n",
                    "workspace": {},
                    "target_file": "",
                }
            )
        },
    )
    elapsed = time.monotonic() - start
    assert result["tests_pass"] is True
    assert result["participates"] is True
    assert elapsed < 10, "the control must not meaningfully slow an ordinary turn"


def test_a_budget_overrun_on_the_last_child_fails_closed_not_silently_accepts() -> None:
    """F2 (review, BLOCKER — replaces the prior inert budget pin M2
    flagged, which passed whether or not the control ever ran). The
    aggregate-budget check happens BEFORE spawning each child, so a child
    that overruns AFTER the last pre-spawn check leaves ``failures`` empty:
    the loop reports a clean pass, and the control's own budget guard then
    skips it too. Silently defaulting ``participates=True`` there is a
    wrong-accept with no evidence behind it — the fix is to fail CLOSED,
    naming the budget honestly (never the generic participation reason,
    which would claim a verdict never reached)."""
    tests = "import time\ndef test_a():\n    time.sleep(0.5)\n    assert True\n"
    gathered = {
        "requirement": "junk",
        "code": "x = 1",
        "tests": tests,
        "workspace": {},
        "target_file": "",
    }
    result = _executor_with_env(
        gathered,
        {
            "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "5",
            "LLM_ORC_ACCEPT_EXECUTOR_BUDGET": "0.2",
        },
    )
    assert result["tests_pass"] is True, "the real (single) child genuinely passed"
    assert result["participates"] is False, "starved must never default to a pass"
    assert result["participation_reason"] == _ABLATION_BUDGET_REASON


# --- F3: the control must empty the destination, not keep the stale copy --
#
# Keeping the workspace's stale copy of the target file at the destination
# (skipping the shadow write rather than shadowing with empty bytes) answers
# "was this necessary GIVEN WHAT THE CLIENT ALREADY HAS" — the brief's own
# wording is "the deliverable's bytes ABSENT FROM EVERY DESTINATION they were
# written to", which includes the target-file shadow. Two real recorded live
# turns (arm0-run2/turn-07 todo.py, 138-arm0-calibration/turn-L2 ledger.py)
# are additive edits: the covered function is unchanged, the deliverable adds
# a genuinely new one the suite does not cover. Keeping the stale copy at the
# destination wrongly refused both.


def test_additive_edit_where_the_covered_function_is_unchanged_still_accepts() -> None:
    """F3 (review, wrong-reject, live-reachable): the deliverable adds a NEW
    function alongside the unchanged, already-covered one. The suite only
    ever exercises the old function, so a control that reverts the
    destination to the stale copy proves nothing (the stale copy alone
    already satisfies the suite) — the control must empty the destination
    entirely, and only THEN does a failing control prove the write really
    was necessary."""
    stale = "def restock(item, n):\n    return n + 1\n"
    additive = (
        "def restock(item, n):\n    return n + 1\ndef audit(item):\n    return item\n"
    )
    result = _build_gated(
        code_writer=f"```python\n{additive}\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="add audit() to inventory.py",
        workspace_file="inventory.py",
        workspace_body=stale,
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True
    assert result["accept_gate"]["reason"] != _NONPARTICIPATION_REASON


def test_byte_identical_resubmission_still_accepts() -> None:
    """F3 sibling (#166's own healthy-build pin restored below to this
    exact shape): the deliverable is byte-identical to the stale workspace
    copy. Under the briefed control (destination EMPTIED, not reverted) an
    identical resubmission still proves necessity — any real content
    differs from an empty destination, whether or not it happens to match
    what was already there."""
    same = "def restock(item, n):\n    return n + 1\n"
    result = _build_gated(
        code_writer=f"```python\n{same}\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="fix restock so it adds two, in inventory.py",
        workspace_file="inventory.py",
        workspace_body=same,
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True


# --- named bounds (documented, not fixed) ----------------------------------


def test_named_bound_control_granularity_can_diverge_from_per_test_isolation() -> None:
    """NAMED BOUND (#171 review F4, MAJOR — documented, not fixed). The
    ablation control always runs as ONE combined process; the real suite
    runs per-test isolated (a fresh subprocess, fresh workspace copy, per
    test). A workspace module with cross-test state can pass every test
    when isolated (the real run's granularity) yet fail when the SAME tests
    run together in one process (the control's granularity) for a reason
    that has nothing to do with the deliverable — the control then reads
    "necessary" and non-participating junk ships. Measured live rate: 0/46
    recorded turns hit this shape; matching granularity would cost one
    extra per-test-isolated subprocess SET on every ablation run for a
    divergence not yet observed live, so the bound is recorded here rather
    than closed (design doc: "Named bound" section)."""
    context = (
        "assistant: [read counter.py]\n"
        "  _state = {'n': 0}\n"
        "  def next_id():\n"
        "      _state['n'] += 1\n"
        "      return _state['n']\n"
        "\n\nCurrent request: fix next_id"
    )
    gather = _node(
        "accept_gather.py",
        {
            "code_writer": _dep(_sub_ensemble_response("```python\nx = 1\n```\n")),
            "test_writer": _dep(
                _sub_ensemble_response(
                    "```python\nimport counter\n"
                    "def test_first_id():\n    assert counter.next_id() == 1\n"
                    "def test_second_id():\n    assert counter.next_id() == 1\n```\n"
                )
            ),
        },
        input_data=context,
    )
    executor = _node("accept_executor.py", {"gather": _dep(gather)})

    # real run: per-test isolated, each test gets a FRESH counter.py — both pass
    assert executor["tests_pass"] is True
    # the combined control's own cross-test state leak makes it fail for a
    # reason unrelated to the (junk) deliverable — documented, not closed
    assert executor["participates"] is True


def test_named_bound_a_length_only_assert_cannot_distinguish_content() -> None:
    """NAMED BOUND (design brief "Conditions and bounds": the ablation
    proves the deliverable's BYTES were necessary, not that any particular
    behavior was observed). A suite asserting only
    ``len(open('solution.py').read()) > 0`` passes the control with junk
    content too — any non-empty content differs from an empty control, so
    necessity is "proven" without the test ever checking what the content
    IS. Review M1: measured to flip junk from refused to accepted."""
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer=(
            "```python\n"
            "def test_solution_is_non_empty():\n"
            "    assert len(open('solution.py').read()) > 0\n"
            "```\n"
        ),
        requirement="write something",
        workspace_file="",
        workspace_body="",
    )
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True


# --- end to end through the real serving chain (#155's lesson) ------------


def _serving_tail(target: str, seat_response: str, file: str) -> dict[str, Any]:
    """seat_contract -> shape -> form_gate -> emit, all real."""
    resolve = _dep(
        {
            "target": target,
            "kind": "python_module",
            "file": file,
            "build": True,
            "dispatch_input": "fix it",
            "is_build_ask": True,
        }
    )
    deps: dict[str, Any] = {"resolve": resolve, "seat": {"response": seat_response}}
    seat_contract = _node("seat_contract.py", deps)
    shape = _node("shape.py", {**deps, "seat_contract": _dep(seat_contract)})
    form_gate = _node("form_gate.py", {"shape": _dep(shape)})
    return _node("emit.py", {"form_gate": _dep(form_gate)})


def _seat_wire(envelope: dict[str, Any]) -> str:
    return json.dumps({"results": {"gather": _dep("..."), "envelope": _dep(envelope)}})


def test_end_to_end_the_ablation_refusal_reaches_the_client_path_free() -> None:
    result = _build_gated(
        code_writer="```python\nx = 1\n```\n",
        tests_writer="```python\n" + _RESTOCK_TESTS_BODY + "```\n",
        requirement="fix restock so it adds two",
    )
    assert result["envelope"]["diagnostics"]["accept"] is False

    outcome = _serving_tail(
        "build-gated", _seat_wire(result["envelope"]), "solution.py"
    )
    text = json.dumps(outcome)

    assert str(Path.home()) not in text
    assert _NONPARTICIPATION_REASON in text
