"""Unit tests for the serving accept gate (WP-D8): sandboxed executor + AND gate.

The accept gate is the verification-ladder rung above form-gate: does the produced
code RUN and pass its tests (deterministic executor, sandboxed in a subprocess per
ODP-1 so a runaway test cannot hang the serve), and does an isolated judge find the
tests adequate. The gate ANDs the two (ADR-048 §1; scenarios.md "accept = tests_pass
AND tests_adequate"). Executor and gate are deterministic and hermetic here; the
judge is the ensemble's only model call and is fed as a fixed verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"
GATE = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_gate.py"

CORRECT = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\n"
WRONG = "def celsius_to_fahrenheit(c):\n    return c * 9 / 5\n"  # missing + 32
REAL_TESTS = (
    "def test_freezing():\n    assert celsius_to_fahrenheit(0) == 32\n"
    "def test_boiling():\n    assert celsius_to_fahrenheit(100) == 212\n"
)
TRIVIAL_TESTS = "def test_callable():\n    assert callable(celsius_to_fahrenheit)\n"


def _executor(
    requirement: str,
    code: str,
    tests: str,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.dumps({"requirement": requirement, "code": code, "tests": tests})
    env = {
        **os.environ,
        "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "3",
        **(env_overrides or {}),
    }
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _gate(executor_resp: dict[str, Any], judge_resp: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "dependencies": {
                "executor": {"response": json.dumps(executor_resp)},
                "judge": {"response": json.dumps(judge_resp)},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


# --- executor: deterministic ground truth, sandboxed in a subprocess ---


def test_executor_passes_correct_code_against_real_tests() -> None:
    r = _executor("c * 9/5 + 32", CORRECT, REAL_TESTS)
    assert r["tests_pass"] is True
    assert r["n_tests"] == 2


def test_executor_fails_wrong_code() -> None:
    r = _executor("c * 9/5 + 32", WRONG, REAL_TESTS)
    assert r["tests_pass"] is False


def test_executor_is_fooled_by_trivial_tests() -> None:
    # The executor passes trivial tests — that is the point; the judge catches it.
    r = _executor("c * 9/5 + 32", CORRECT, TRIVIAL_TESTS)
    assert r["tests_pass"] is True


def test_executor_sandbox_times_out_on_a_runaway_test() -> None:
    # ODP-1: the executor runs the produced tests in a subprocess with a timeout,
    # so a runaway test is reported as a failure, not a frozen serve.
    runaway = "def test_hang():\n    while True:\n        pass\n"
    r = _executor("loops forever", CORRECT, runaway)
    assert r["tests_pass"] is False
    assert "timeout" in r["report"].lower()


def test_aggregate_budget_bounds_a_many_hang_suite() -> None:
    # Per-test isolation can spawn up to 21 children, each hanging for the
    # full per-child timeout — an interactive serve cannot absorb that. The
    # aggregate budget caps total wall time across the suite and reports it.
    hang_five = "".join(
        f"def test_hang_{i}():\n    while True:\n        pass\n" for i in range(5)
    )
    start = time.monotonic()
    r = _executor(
        "loops forever x5",
        CORRECT,
        hang_five,
        env_overrides={
            "LLM_ORC_ACCEPT_EXECUTOR_TIMEOUT": "1",
            "LLM_ORC_ACCEPT_EXECUTOR_BUDGET": "2",
        },
    )
    elapsed = time.monotonic() - start
    assert elapsed < 10
    assert r["tests_pass"] is False
    assert "aggregate budget exhausted after 2s" in r["report"]
    assert "of 5 tests not run" in r["report"]


def test_aggregate_budget_does_not_fire_on_a_normal_suite() -> None:
    r = _executor("c * 9/5 + 32", CORRECT, REAL_TESTS)
    assert r["tests_pass"] is True
    assert "aggregate budget" not in r["report"]


def test_executor_passes_the_contract_through_for_the_judge() -> None:
    # The judge seat receives {requirement, code, tests} + execution result via
    # the executor's passthrough (isolation: contract + artifact + result only).
    r = _executor("the requirement", CORRECT, REAL_TESTS)
    assert r["requirement"] == "the requirement"
    assert r["code"] == CORRECT
    assert r["tests"] == REAL_TESTS


# --- gate: accept = tests_pass AND tests_adequate (orthogonal catches) ---


def test_gate_accepts_when_both_signals_green() -> None:
    g = _gate({"tests_pass": True}, {"tests_adequate": True})
    assert g["accept"] is True


def test_gate_rejects_when_judge_inadequate() -> None:
    g = _gate({"tests_pass": True}, {"tests_adequate": False})
    assert g["accept"] is False
    assert g["tests_pass"] is True
    assert g["tests_adequate"] is False


def test_gate_rejects_when_executor_fails() -> None:
    g = _gate({"tests_pass": False}, {"tests_adequate": True})
    assert g["accept"] is False


def _gate_no_judge(
    executor_resp: dict[str, Any], gather_resp: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Drive the gate as the held round wires it: executor + gather, no judge."""
    deps: dict[str, Any] = {"executor": {"response": json.dumps(executor_resp)}}
    if gather_resp is not None:
        deps["gather"] = {"response": json.dumps(gather_resp)}
    out = subprocess.run(
        [sys.executable, str(GATE)],
        input=json.dumps({"dependencies": deps}),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def test_gate_carries_round1_adequacy_on_a_held_round() -> None:
    """The TDD retry round (issue #100): the held path only fires when round
    1's judge passed the tests, so with no judge seat the gate carries that
    verdict deterministically — the executor stays the live ground truth."""
    g = _gate_no_judge({"tests_pass": True}, {"held": True})
    assert g["tests_adequate"] is True
    assert g["accept"] is True
    assert "carried" in g["reason"]


def test_gate_held_round_still_rejects_when_tests_fail() -> None:
    g = _gate_no_judge({"tests_pass": False}, {"held": True})
    assert g["accept"] is False
    assert g["tests_adequate"] is True
    # the carried verdict must not mask the live failure cause
    assert "tests did not pass" in g["reason"]


def test_gate_without_judge_and_not_held_rejects() -> None:
    """No judge and no held flag is a miswired shape, not a free pass."""
    g = _gate_no_judge({"tests_pass": True}, {"held": False})
    assert g["accept"] is False
    g2 = _gate_no_judge({"tests_pass": True})
    assert g2["accept"] is False


def test_gate_reads_the_verdict_through_a_sub_ensemble_judge_envelope() -> None:
    """With the judge extracted to its own ensemble (#84), the judge dep's
    response is the nested sub-ensemble result envelope, not bare model
    JSON — the gate must peel it (the escaped nesting also defeats the
    regex fallback, which silently rejected every round)."""
    judge_child = json.dumps(
        {
            "ensemble": "adequacy-judge",
            "status": "completed",
            "results": {
                "judge": {
                    "response": json.dumps(
                        {"tests_adequate": True, "reason": "real behavior checks"}
                    ),
                    "status": "success",
                }
            },
        }
    )
    payload = json.dumps(
        {
            "dependencies": {
                "executor": {"response": json.dumps({"tests_pass": True})},
                "judge": {"response": judge_child},
            }
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATE)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    verdict = json.loads(out)
    assert verdict["tests_adequate"] is True
    assert verdict["accept"] is True


def test_quoted_string_false_from_the_judge_does_not_pass_the_gate() -> None:
    """Small models sometimes quote booleans; bool("false") is True in
    Python, which would wave inadequate tests through the gate."""
    verdict = _gate(
        {"tests_pass": True, "report": "all passed"},
        {"tests_adequate": "false", "reason": "trivial asserts"},
    )
    assert verdict["tests_adequate"] is False
    assert verdict["accept"] is False


LEAKY_STATE_CODE = "todos = []\ndef add(item):\n    todos.append(item)\n"
LEAKY_STATE_TESTS = (
    "def test_one():\n    add('a')\n    assert len(todos) == 1\n"
    "def test_two():\n    add('a')\n    add('b')\n    assert len(todos) == 2\n"
)
LEAKY_FILE_TESTS = (
    "import os, json\n"
    "def save(t):\n    json.dump(t, open('t.json', 'w'))\n"
    "def test_writes():\n    save([1])\n    assert os.path.exists('t.json')\n"
    "def test_fresh():\n    assert not os.path.exists('t.json')\n"
)


def test_executor_isolates_module_state_across_tests() -> None:
    result = _executor("add todos", LEAKY_STATE_CODE, LEAKY_STATE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_isolates_filesystem_across_tests() -> None:
    result = _executor("save todos", "", LEAKY_FILE_TESTS)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_executor_still_fails_genuinely_wrong_code() -> None:
    result = _executor(
        "adds",
        "def add(a, b):\n    return a - b\n",
        "def test_add():\n    assert add(1, 2) == 3\n",
    )
    assert result["tests_pass"] is False
    assert "test_add" in result["report"]


STRAY_ASSERT_TESTS = (
    "def test_overwrite():\n"
    "    save_todos(['a'])\n"
    "    assert load\n"
    "    assert load_todos() == ['a']\n"
)
STRAY_CODE = (
    "_d = {}\n"
    "def save_todos(t):\n    _d['t'] = t\n"
    "def load_todos():\n    return _d.get('t', [])\n"
)


def test_bare_name_assert_is_sanitized_before_execution() -> None:
    result = _executor("storage", STRAY_CODE, STRAY_ASSERT_TESTS)
    assert result["tests_pass"] is True
    assert result["tests_sanitized"] == 1
    assert "assert load\n" not in result["tests"]
    assert "assert load_todos() == ['a']" in result["tests"]


def test_value_bearing_asserts_are_never_sanitized() -> None:
    tests = "def test_add():\n    assert add(1, 2) == 3\n"
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_pass"] is True
    assert result["tests_sanitized"] == 0
    assert "assert add(1, 2) == 3" in result["tests"]


def test_bare_assert_on_an_assigned_local_is_kept() -> None:
    # 'assert result' on a test-local CAN be a real truthiness check —
    # only names never assigned in the tests source are value-free.
    tests = "def test_t():\n    result = add(1, 2)\n    assert result\n"
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_sanitized"] == 0
    assert "assert result" in result["tests"]


def test_missing_stdlib_and_pytest_imports_are_injected() -> None:
    tests = (
        "def test_raises():\n"
        "    with pytest.raises(ValueError):\n"
        "        boom()\n"
        "def test_file():\n"
        "    open('x.txt', 'w').write('1')\n"
        "    assert os.path.exists('x.txt')\n"
    )
    code = "def boom():\n    raise ValueError('no')\n"
    result = _executor("boom raises", code, tests)
    assert result["tests_pass"] is True
    assert result["tests_imports_injected"] == 2
    assert result["tests"].startswith("import os\nimport pytest\n")


def test_non_whitelisted_unbound_names_are_not_injected() -> None:
    tests = "def test_x():\n    assert requests.get is not None\n"
    result = _executor("http", "", tests)
    assert result["tests_imports_injected"] == 0
    assert result["tests_pass"] is False


def test_already_imported_modules_are_not_reinjected() -> None:
    tests = "import os\ndef test_x():\n    assert os.sep\n"
    result = _executor("sep", "", tests)
    assert result["tests_imports_injected"] == 0


# --- final-review findings: C1, I1, I2 ---


def test_bare_assert_on_a_loop_bound_name_is_kept() -> None:
    # C1: _ASSIGNED_NAME_RE only matched plain 'name = ...', so a bare
    # assert on a for-loop-bound name was (wrongly) treated as value-free
    # and stripped — silently turning a real falsy-value check into a
    # no-op and letting wrong code through.
    code = "def all_flags():\n    return [1, 0, 1]\n"
    tests = (
        "def test_all_true():\n"
        "    for flag in all_flags():\n"
        "        _ = flag\n"
        "        assert flag\n"
    )
    result = _executor("flags", code, tests)
    assert result["tests_sanitized"] == 0
    assert result["tests_pass"] is False  # 0 is falsy — the suite must reject


def test_nested_test_defs_fall_back_to_legacy_whole_run() -> None:
    # I1: a test_* def nested under a module-level if/try/for must not be
    # silently dropped by the per-test isolation path — fall back to one
    # legacy whole-suite run so completeness matches the pre-isolation
    # executor.
    tests = "if True:\n    def test_hidden():\n        assert add(1, 1) == 3\n"
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_pass"] is False
    assert "test_hidden" in result["report"]


def test_injection_never_shadows_a_code_bound_name() -> None:
    # I2: an injected 'import os' must not rebind a code-defined 'os' in
    # the shared runner namespace — that would flip a genuine code bug
    # (os = None) into a false pass.
    code = "os = None\n"
    tests = "def test_os():\n    assert os is not None\n"
    result = _executor("os check", code, tests)
    assert result["tests_imports_injected"] == 0
    assert result["tests_pass"] is False


def test_unparseable_tests_skip_injection_and_sanitization() -> None:
    result = _executor("broken", "x = 1\n", "def test_x(:\n")
    assert result["tests_sanitized"] == 0
    assert result["tests_imports_injected"] == 0
    assert result["tests_pass"] is False


# --- unbound-callable excision (spike 2026-07-10: `file_exists` tic) ---

_SAVE_LOAD_CODE = (
    "import json\n"
    "import os\n"
    "def save_todos(todos):\n"
    "    with open('todos.json', 'w') as f:\n"
    "        json.dump(todos, f)\n"
    "def load_todos():\n"
    "    if not os.path.exists('todos.json'):\n"
    "        return []\n"
    "    with open('todos.json') as f:\n"
    "        return json.load(f)\n"
)

# verbatim spike exemplar (turn6_s2 r1 / turn6_s5 r1+r2): file_exists is
# bound nowhere — not a module, so import injection cannot reach it
_UNBOUND_CALLABLE_TEST = (
    "def test_save_todos_saves_todos_to_file():\n"
    '    todos = ["Buy groceries", "Walk the dog"]\n'
    "    save_todos(todos)\n"
    '    assert file_exists("todos.json")\n'
)

_GOOD_ROUNDTRIP_TEST = (
    "def test_load_todos_returns_saved_todos():\n"
    '    todos = ["Buy groceries", "Walk the dog"]\n'
    "    save_todos(todos)\n"
    "    loaded = load_todos()\n"
    "    assert loaded == todos\n"
)


def test_unbound_callable_test_is_excised_and_the_rest_run() -> None:
    """A test calling a name bound nowhere (not tests, not code, not a
    builtin, not injectable) can never run — no code regeneration defines
    it. Excise that one test, run the remainder, echo the excised suite."""
    result = _executor(
        "save/load todos",
        _SAVE_LOAD_CODE,
        _UNBOUND_CALLABLE_TEST + _GOOD_ROUNDTRIP_TEST,
    )
    assert result["tests_excised"] == 1
    assert "file_exists" not in result["tests"]
    assert "test_load_todos_returns_saved_todos" in result["tests"]
    assert result["tests_pass"] is True


def test_excision_that_would_drop_all_tests_leaves_the_suite_unchanged() -> None:
    """Bound: excising everything would judge an empty suite — leave the
    suite alone so the round rejects honestly on the real NameError."""
    result = _executor("save/load todos", _SAVE_LOAD_CODE, _UNBOUND_CALLABLE_TEST)
    assert result["tests_excised"] == 0
    assert "file_exists" in result["tests"]
    assert result["tests_pass"] is False


def test_excised_to_empty_suite_still_rejects_at_the_gate() -> None:
    """The gate never gets weaker than 'at least one adequate test ran':
    when every test calls an unbound name, excision declines, the raw
    suite NameErrors in the executor, and the round rejects regardless of
    the judge's adequacy verdict on the echoed tests."""
    executor_resp = _executor(
        "save/load todos", _SAVE_LOAD_CODE, _UNBOUND_CALLABLE_TEST
    )
    check = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "adequacy_check.py"
    payload = json.dumps(
        {
            "input_data": "save/load todos",
            "dependencies": {"executor": {"response": json.dumps(executor_resp)}},
        }
    )
    judge_out = subprocess.run(
        [sys.executable, str(check)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    verdict = _gate(executor_resp, json.loads(judge_out))
    assert verdict["accept"] is False
    assert verdict["tests_pass"] is False


def test_call_to_a_code_bound_name_is_never_excised() -> None:
    result = _executor("save/load todos", _SAVE_LOAD_CODE, _GOOD_ROUNDTRIP_TEST)
    assert result["tests_excised"] == 0
    assert result["tests_pass"] is True


def test_unguarded_os_remove_is_wrapped_in_suppress() -> None:
    """Verbatim spike exemplar (turn6_s4 r1+r2): setup deletes a file that
    never exists in the fresh per-test sandbox; FileNotFoundError fires
    before the assert. The wrap makes setup idempotent without touching
    what the test asserts about the code."""
    tests = (
        "def test_load_todos_returns_empty_list_on_missing_file():\n"
        "    import os\n"
        '    os.remove("todos.json")\n'
        "    assert load_todos() == []\n"
    )
    result = _executor("save/load todos", _SAVE_LOAD_CODE, tests)
    assert result["tests_removals_guarded"] == 1
    assert "with contextlib.suppress(FileNotFoundError):" in result["tests"]
    assert "import contextlib" in result["tests"]
    assert result["tests_pass"] is True


def test_os_unlink_is_guarded_like_os_remove() -> None:
    tests = (
        "import os\n"
        "def test_missing_file():\n"
        '    os.unlink("todos.json")\n'
        "    assert load_todos() == []\n"
    )
    result = _executor("save/load todos", _SAVE_LOAD_CODE, tests)
    assert result["tests_removals_guarded"] == 1
    assert result["tests_pass"] is True


def test_removal_already_inside_try_or_with_is_left_alone() -> None:
    tests = (
        "import os\n"
        "import contextlib\n"
        "def test_missing_file():\n"
        "    with contextlib.suppress(FileNotFoundError):\n"
        '        os.remove("todos.json")\n'
        "    assert load_todos() == []\n"
        "def test_missing_file_try():\n"
        "    try:\n"
        '        os.remove("todos.json")\n'
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    assert load_todos() == []\n"
    )
    result = _executor("save/load todos", _SAVE_LOAD_CODE, tests)
    assert result["tests_removals_guarded"] == 0
    assert result["tests_pass"] is True


# --- inverted exception-expectation rewrite (spike 2026-07-10) ---

_RAISING_SAVE = (
    "def save_todos(todos):\n"
    "    if todos is None:\n"
    "        raise TypeError('todos must be a list')\n"
    "    return list(todos)\n"
)


def test_inverted_expectation_with_both_branches_false_is_rewritten() -> None:
    """Verbatim spike exemplar (turn6_s2 r1+r2): both branches assert
    False — no implementation can pass, and the code that correctly
    raised TypeError was punished for it. The messages name the intent
    (raise expected); rewrite to the canonical pytest.raises form."""
    tests = (
        "def test_save_todos_handles_exceptions():\n"
        "    try:\n"
        "        save_todos(None)\n"
        "    except TypeError:\n"
        '        assert False, "Expected TypeError"\n'
        "    else:\n"
        '        assert False, "Did not raise TypeError"\n'
    )
    result = _executor("save todos", _RAISING_SAVE, tests)
    assert result["tests_raises_rewritten"] == 1
    assert "with pytest.raises(TypeError):" in result["tests"]
    assert "import pytest" in result["tests"]
    assert result["tests_pass"] is True


def test_inverted_expectation_with_bare_handler_and_true_else_rewrites() -> None:
    """Verbatim spike exemplar (turn1_s5 r2): the specific handler asserts
    False with 'expected ValueError', a bare handler asserts False, and
    the else is a vacuous assert True — the assert-False belongs after
    the call / in else, exactly the positional garble the prompt rule
    tried to teach."""
    code = (
        "def add_todo(item):\n"
        "    if item is None:\n"
        "        raise ValueError('item required')\n"
    )
    tests = (
        "def test_add_todo_raises_value_error_for_none():\n"
        "    try:\n"
        "        add_todo(None)\n"
        "    except ValueError as e:\n"
        '        assert False, "expected ValueError"\n'
        "    except:\n"
        '        assert False, "unexpected exception"\n'
        "    else:\n"
        '        assert True, "no exception raised"\n'
    )
    result = _executor("add todo", code, tests)
    assert result["tests_raises_rewritten"] == 1
    assert "with pytest.raises(ValueError):" in result["tests"]
    assert result["tests_pass"] is True


def test_expectation_with_a_real_else_assert_is_left_alone() -> None:
    """turn6_s5's lookalike: the else carries a real value assert, so the
    test is satisfiable by correct non-raising code — rewriting to
    pytest.raises would BREAK a correct implementation. Ambiguous stays
    untouched."""
    tests = (
        "def test_load_todos_returns_empty_list_if_file_not_found():\n"
        "    try:\n"
        "        load_todos()\n"
        "    except FileNotFoundError:\n"
        '        assert False, "Expected FileNotFoundError"\n'
        "    else:\n"
        "        assert load_todos() == []\n"
    )
    result = _executor("save/load todos", _SAVE_LOAD_CODE, tests)
    assert result["tests_raises_rewritten"] == 0
    assert result["tests_pass"] is True


def test_canonical_expect_raise_try_form_is_left_alone() -> None:
    """The correct idiom (assert False AFTER the call, handler passes) has
    an assert in the try body — outside the exact signature."""
    tests = (
        "def test_rejects_none():\n"
        "    try:\n"
        "        save_todos(None)\n"
        '        assert False, "expected TypeError"\n'
        "    except TypeError:\n"
        "        pass\n"
    )
    result = _executor("save todos", _RAISING_SAVE, tests)
    assert result["tests_raises_rewritten"] == 0
    assert result["tests_pass"] is True


def test_handler_message_that_does_not_declare_expectation_is_left_alone() -> None:
    """'except E: assert False' whose message does NOT say the raise was
    expected could be a legitimate must-not-raise guard — ambiguous, so
    untouched (reject stays honest)."""
    tests = (
        "def test_never_raises_for_lists():\n"
        "    try:\n"
        "        save_todos([1])\n"
        "    except TypeError:\n"
        '        assert False, "should not raise for a list"\n'
        "    else:\n"
        "        pass\n"
    )
    result = _executor("save todos", _RAISING_SAVE, tests)
    assert result["tests_raises_rewritten"] == 0
    assert result["tests_pass"] is True


def test_call_to_a_fixture_parameter_is_never_excised() -> None:
    """A called name bound as a test parameter (pytest fixture) is bound."""
    tests = (
        "def test_uses_factory(make_thing):\n"
        "    thing = make_thing()\n"
        "    assert thing != 0\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
    )
    result = _executor("adds", "def add(a, b):\n    return a + b\n", tests)
    assert result["tests_excised"] == 0
