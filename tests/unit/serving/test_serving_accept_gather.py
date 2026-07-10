"""Unit tests for the build-gated ``gather`` node + executor dependency-mode (WP-D8).

``gather`` assembles the accept-gate's ``{requirement, code, tests}`` from the two
sub-ensemble seats (test_writer, code_writer), peeling the nested ensemble
envelopes and stripping code fences. The executor then reads that assembled
contract from its ``gather`` dependency (its flat direct-payload mode, exercised
in test_serving_accept_gate.py, is preserved).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

# Add agentic_serving to path for direct imports of _workspace
sys.path.insert(0, str(REPO / ".llm-orc" / "scripts" / "agentic_serving"))
from accept_gather import _workspace  # type: ignore[import-not-found]  # noqa: E402

GATHER = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_gather.py"
EXECUTOR = REPO / ".llm-orc" / "scripts" / "agentic_serving" / "accept_executor.py"

TESTS = "def test_even():\n    assert is_even(4) is True"
CODE = "def is_even(n):\n    return n % 2 == 0"


def _sub_ensemble_response(terminal_text: str) -> str:
    """A sub-ensemble node's response: the nested ``{ensemble, results}`` envelope
    the engine returns for an ``ensemble:`` node (peeled by _terminal)."""
    return json.dumps(
        {
            "ensemble": "x",
            "status": "completed",
            "results": {"out": {"response": terminal_text, "status": "success"}},
        }
    )


def _gather(criteria: str, tests_terminal: str, code_terminal: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_data": criteria,
            "dependencies": {
                "test_writer": {"response": _sub_ensemble_response(tests_terminal)},
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
            },
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATHER)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _executor_from_gather(gather_out: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_data": gather_out.get("requirement", ""),
            "dependencies": {"gather": {"response": json.dumps(gather_out)}},
        }
    )
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


def _gather_code_only(criteria: str, code_terminal: str) -> dict[str, Any]:
    """Drive gather as the held round wires it: no test_writer dependency."""
    payload = json.dumps(
        {
            "input_data": criteria,
            "dependencies": {
                "code_writer": {"response": _sub_ensemble_response(code_terminal)},
            },
        }
    )
    out = subprocess.run(
        [sys.executable, str(GATHER)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


_HELD_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"


def test_gather_reads_held_tests_when_test_writer_is_absent() -> None:
    """The TDD retry round (issue #100): build-code-round has no test_writer;
    gather reads the held tests from the carry's sentinel block and marks the
    contract held so the gate carries round 1's adequacy verdict."""
    criteria = (
        "Write is_even(n) in even.py\n\n"
        "[Previous round rejected: tests did not pass."
        " Executor report: test_even: AssertionError().]\n\n"
        f"{_HELD_MARKER}\n```python\n{TESTS}\n```"
    )
    out = _gather_code_only(criteria, CODE)
    assert out["tests"] == TESTS
    assert out["code"] == CODE
    assert out["held"] is True
    assert "[HELD TESTS" not in out["requirement"]
    assert "Write is_even(n) in even.py" in out["requirement"]


def test_gather_with_test_writer_ignores_a_marker_in_the_turn_text() -> None:
    """A fresh round whose turn text happens to contain the sentinel string
    still takes its tests from test_writer (worst case is a rejected round,
    never a wrong accept)."""
    criteria = f"Write is_even(n). By the way: {_HELD_MARKER}"
    out = _gather(criteria, TESTS, CODE)
    assert out["tests"] == TESTS
    assert out.get("held", False) is False


def test_gather_assembles_requirement_code_tests_from_seats() -> None:
    out = _gather("Write is_even(n).", TESTS, CODE)
    assert out["requirement"] == "Write is_even(n)."
    assert out["code"] == CODE
    assert out["tests"] == TESTS


def test_gather_code_drops_a_seat_emitted_test_fence_from_the_deliverable() -> None:
    """Seat models sometimes emit the code and a copy of the tests as two
    fences; joining them pollutes the shipped file with embedded tests (live
    finding 2026-07-09: models.py shipped with the test suite appended). The
    CODE extraction drops pure-test blocks when a non-test block exists."""
    chatty_code = (
        "Here is the code:\n```python\n" + CODE + "\n```\n"
        "And the tests:\n```python\n" + TESTS + "\n```\n"
    )
    out = _gather("Write is_even(n).", TESTS, chatty_code)
    assert out["code"] == CODE
    assert "def test_" not in out["code"]


def test_gather_tests_keep_test_fences() -> None:
    """The TESTS extraction must not drop test blocks (they are the point)."""
    chatty_tests = "```python\n" + TESTS + "\n```"
    out = _gather("Write is_even(n).", chatty_tests, CODE)
    assert out["tests"] == TESTS


def test_gather_strips_markdown_fences_from_seat_output() -> None:
    fenced_code = "Here you go:\n```python\n" + CODE + "\n```\n"
    fenced_tests = "```python\n" + TESTS + "\n```"
    out = _gather("Write is_even(n).", fenced_tests, fenced_code)
    assert out["code"] == CODE
    assert out["tests"] == TESTS


def test_executor_reads_contract_from_gather_dependency() -> None:
    gathered = _gather("Write is_even(n).", TESTS, CODE)
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 1
    # passthrough carries the contract + artifact to the judge
    assert result["code"] == CODE
    assert result["tests"] == TESTS


def test_gather_strips_conversation_context_from_the_requirement() -> None:
    """With rung-1 context threading, the shape's base input carries the
    conversation ahead of the 'Current request:' marker. The requirement the
    verifier chain echoes (and the judge reads) is the clean turn only —
    conversation context must not reach verifier seats (ADR-048 isolation).
    """
    criteria = (
        "Conversation so far:\n"
        "user: write is_even in even.py\nassistant: [wrote even.py]"
        "\n\nCurrent request: Write is_even(n)."
    )
    out = _gather(criteria, TESTS, CODE)
    assert out["requirement"] == "Write is_even(n)."


def test_gather_skips_shell_fences_when_extracting_code() -> None:
    """Seat models often append a '```bash pytest ...```' usage block; joining
    it into the Python deliverable is a SyntaxError at the executor (battery
    finding 2026-07-08). Only python/untagged fences are code."""
    chatty = (
        "Here is the code:\n```python\n" + CODE + "\n```\n"
        "Run it with:\n```bash\npytest test_even.py\n```\n"
    )
    out = _gather("Write is_even(n).", TESTS, chatty)
    assert out["code"] == CODE


def test_gather_trims_trailing_chatter_until_the_code_parses() -> None:
    """Seat models sometimes leave a prose example line inside the fence
    (ladder finding 2026-07-08: a Unicode-arrow usage line → SyntaxError at
    the executor). Trailing non-parsing lines are dropped, bounded, until the
    deliverable parses; valid code stays byte-identical."""
    chatty = CODE + "\nstack.push(3); stack.push(1) → assert stack.min() == 1\n"
    out = _gather("Write is_even(n).", TESTS, chatty)
    assert out["code"] == CODE


def test_gather_gives_up_cleanly_when_nothing_parses() -> None:
    out = _gather("Write is_even(n).", TESTS, "→ not code at all →\n→ still not →")
    assert out["code"] == "→ not code at all →\n→ still not →"


def test_gather_extracts_conversation_written_files_as_workspace() -> None:
    """Files written earlier in the conversation render as '[wrote <path>]'
    blocks in the context; gather extracts them so the executor can
    materialize them in the sandbox ("add tests for it" imports the module
    the conversation built)."""
    criteria = (
        "Conversation so far:\n"
        "user: write is_even in even.py\n"
        "assistant: [wrote even.py]\n"
        "def is_even(n):\n"
        "    return n % 2 == 0\n"
        "user: thanks\n"
        "\n\nCurrent request: add tests for it in test_even.py"
    )
    out = _gather(criteria, TESTS, CODE)
    assert out["workspace"] == {"even.py": "def is_even(n):\n    return n % 2 == 0"}


def test_gather_skips_truncated_write_blocks() -> None:
    criteria = (
        "Conversation so far:\n"
        "assistant: [wrote big.py (truncated)]\n"
        "xxxx\n"
        "\n\nCurrent request: add tests"
    )
    out = _gather(criteria, TESTS, CODE)
    assert out["workspace"] == {}


def test_edit_turn_deliverable_shadows_the_stale_workspace_copy() -> None:
    """An edit turn ('add min() to the Stack class in stack.py') produces code
    destined for stack.py, but the sandbox also materializes the OLD stack.py
    from the conversation — and tests importing 'from stack import ...' hit
    the stale copy (arm-A rerun finding 2026-07-08: AttributeError('min')).
    gather names the target file; the executor writes the code there,
    shadowing the stale copy, mirroring what the client's write will do.
    """
    criteria = (
        "Conversation so far:\n"
        "assistant: [wrote stack.py]\n"
        "class Stack:\n"
        "    def push(self, item):\n"
        "        pass\n"
        "\n\nCurrent request: Add a min() method to the Stack class in stack.py"
    )
    new_code = (
        "class Stack:\n"
        "    def __init__(self):\n"
        "        self.items = []\n"
        "    def push(self, item):\n"
        "        self.items.append(item)\n"
        "    def min(self):\n"
        "        return min(self.items) if self.items else None\n"
    )
    tests = (
        "from stack import Stack\n\n"
        "def test_min():\n"
        "    s = Stack()\n"
        "    s.push(3)\n"
        "    s.push(1)\n"
        "    assert s.min() == 1\n"
    )
    gathered = _gather(criteria, tests, new_code)
    assert gathered["target_file"] == "stack.py"
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True


def test_gather_injects_missing_workspace_imports() -> None:
    """Generated tests sometimes use a workspace module's names without
    importing them (ladder finding 2026-07-08: NameError('Stack') on every
    test). When a deliverable references top-level names defined by a
    workspace module and has no import for it, gather prepends one."""
    criteria = (
        "Conversation so far:\n"
        "assistant: [wrote stack.py]\n"
        "class Stack:\n"
        "    def push(self, item):\n"
        "        pass\n"
        "\n\nCurrent request: add tests for the Stack class"
    )
    tests_without_import = "def test_push():\n    stack = Stack()\n    stack.push(1)\n"
    out = _gather(criteria, tests_without_import, CODE)
    assert out["tests"].startswith("from stack import Stack\n")


def test_gather_leaves_deliverables_with_imports_untouched() -> None:
    criteria = (
        "Conversation so far:\n"
        "assistant: [wrote stack.py]\n"
        "class Stack:\n"
        "    pass\n"
        "\n\nCurrent request: add tests"
    )
    tests_with_import = "from stack import Stack\n\ndef test_push():\n    s = Stack()\n"
    out = _gather(criteria, tests_with_import, CODE)
    assert out["tests"] == tests_with_import.strip()


def test_executor_materializes_workspace_files_in_the_sandbox() -> None:
    """Tests that import a conversation-built module pass when the workspace
    carries it — the sandbox is no longer blind to conversation-known files."""
    gathered = {
        "requirement": "add tests for is_even",
        "code": "from even import is_even\n\ndef helper():\n    return is_even(2)",
        "tests": (
            "from even import is_even\n\n"
            "def test_even():\n    assert is_even(4) is True\n"
        ),
        "workspace": {"even.py": "def is_even(n):\n    return n % 2 == 0"},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True


def test_executor_runs_unittest_class_style_tests() -> None:
    """Seat models often emit unittest.TestCase classes; the runner must
    collect and run those, not just top-level test_* functions (long-horizon
    drive finding 2026-07-09: 'no test_* functions found')."""
    gathered = {
        "requirement": "is_even",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "tests": (
            "import unittest\n\n"
            "class TestIsEven(unittest.TestCase):\n"
            "    def test_even(self):\n"
            "        self.assertTrue(is_even(4))\n"
            "    def test_odd(self):\n"
            "        self.assertFalse(is_even(3))\n"
        ),
        "workspace": {},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True
    assert result["n_tests"] == 2


def test_async_test_functions_are_executed_not_silently_passed() -> None:
    """An async def test_* returns a coroutine when called; uncollected, it
    raised nothing and counted as a PASS — a wrong-accept channel (found
    2026-07-09 during the cross-task adequacy scan). The runner must await
    it and report its real verdict."""
    gathered = {
        "requirement": "is_even",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "tests": ("async def test_wrong_expectation():\n    assert is_even(3) is True"),
        "workspace": {},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is False
    assert result["n_tests"] == 1


def test_passing_async_tests_pass() -> None:
    gathered = {
        "requirement": "is_even",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "tests": "async def test_even():\n    assert is_even(4) is True",
        "workspace": {},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True


def test_failure_report_names_the_failing_source_line() -> None:
    """A bare 'AssertionError()' gives the retry round nothing to move on
    (live finding 2026-07-09: the same wrong expectation regenerated across
    rounds). The report carries the failing line so the next round sees
    WHICH expectation disagreed with the workspace."""
    gathered = {
        "requirement": "is_even",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "tests": ("def test_wrong_expectation():\n    assert is_even(3) is True\n"),
        "workspace": {},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is False
    assert "assert is_even(3) is True" in result["report"]


def test_executor_reports_unittest_class_failures() -> None:
    gathered = {
        "requirement": "is_even",
        "code": "def is_even(n):\n    return n % 2 == 1",  # wrong
        "tests": (
            "import unittest\n\n"
            "class TestIsEven(unittest.TestCase):\n"
            "    def test_even(self):\n"
            "        self.assertTrue(is_even(4))\n"
        ),
        "workspace": {},
    }
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is False


def test_read_block_materializes_into_the_workspace() -> None:
    context = (
        "user: write tests for existing storage.py\n"
        "assistant: [read storage.py]\n"
        "def put(k, v):\n"
        "    return (k, v)"
    )
    workspace = _workspace(context)
    assert workspace["storage.py"] == "def put(k, v):\n    return (k, v)"


def test_failed_and_oversize_read_lines_never_materialize() -> None:
    context = (
        "assistant: [read gone.py (failed)] Error: ENOENT\n"
        "assistant: [read big.py (oversize)]"
    )
    workspace = _workspace(context)
    assert workspace == {}


def test_truncated_wrote_block_still_never_materializes() -> None:
    context = "assistant: [wrote storage.py (truncated)]\ndef put(k"
    assert _workspace(context) == {}
