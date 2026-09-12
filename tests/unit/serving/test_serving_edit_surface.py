"""#182 slice B-1: an edit never drops the prior module's surface.

Evidence: docs/plans/2026-09-11-daily-driver-probe/README.md, turn 2 — "add
remove() to todo/storage.py" (a pure edit of a 40-line existing module).
code-generator's prompt says "show the code change directly"; the coder
returned the one method alone, at module top level. The build-gated round
ran that fragment as the WHOLE module and refused with "tests did not
pass" (true, but not why) after two rounds; nothing shipped.

The fix (design: docs/plans/2026-09-11-182-existing-repo-shape-design.md,
slice B-1): accept_gather computes the target file's PRIOR public surface
(#171's public_top_level_names, reused) whenever its prior body is visible
in the rendered context — a client "[read <path>]" block or a
conversation-written "[wrote <path>]" block. accept_executor checks the
deliverable's own top-level names against that surface BEFORE any
sandboxed execution (a pure AST comparison, no subprocess — the cheapest
correct seam, run ahead of both the real test run and the #171 ablation
control): a deliverable missing any prior public name refuses honestly,
naming what it dropped, instead of whatever incidental test failure the
missing surface happens to cause.

Driven via subprocess exactly as the L0 engine runs a script node, matching
every sibling harness in this directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"
AGENTIC = REPO / ".llm-orc" / "ensembles" / "agentic-serving"
CODE_GENERATOR_YAML = AGENTIC / "code-generator.yaml"

# The seed repo's real todo/storage.py (docs/plans/2026-09-11-daily-driver-
# probe/seed-repo.tgz) — its only top-level PUBLIC name is the class itself
# (add/list/complete/_load/_save are methods, not top-level bindings).
_STORAGE_PY_PRIOR = '''"""JSON-backed todo storage."""

from __future__ import annotations

import json
from pathlib import Path


class TodoStore:
    """Persist todos as a list of dicts in a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, todos: list[dict]) -> None:
        self.path.write_text(json.dumps(todos, indent=2), encoding="utf-8")

    def add(self, text: str) -> dict:
        todos = self._load()
        todo = {"id": len(todos) + 1, "text": text, "done": False}
        todos.append(todo)
        self._save(todos)
        return todo

    def list(self) -> list[dict]:
        return self._load()

    def complete(self, todo_id: int) -> dict:
        todos = self._load()
        for todo in todos:
            if todo["id"] == todo_id:
                todo["done"] = True
                self._save(todos)
                return todo
        raise KeyError(todo_id)
'''

# The 13-turn ladder's turn-2 shape ("add a complete_todo function to
# todo.py"): a short conversation-written todo.py with two top-level
# functions — this is the prior that passes ONLY because the coder happens
# to rewrite it whole (brief B-1: "the ladder's turn 2 ... passes only
# because the prior module is five lines").
_TODO_PY_PRIOR = (
    "def add_todo(todos, item):\n"
    "    todos.append({'text': item, 'done': False})\n"
    "\n"
    "\n"
    "def list_todos(todos):\n"
    "    return todos\n"
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


def _context(read_path: str, prior_body: str, requirement: str) -> str:
    """A client '[read <path>]' block rendering ``prior_body``, followed by
    the turn's request — the fenced-block grammar accept_gather._workspace
    parses (a two-space indent per body line, blank lines kept blank)."""
    indented = "\n".join(
        f"  {line}" if line else "" for line in prior_body.splitlines()
    )
    return (
        f"assistant: [read {read_path}]\n{indented}\n\nCurrent request: {requirement}"
    )


def _build_gated_edit(
    context: str, code_writer: str, tests_writer: str
) -> dict[str, Any]:
    """gather -> executor -> judge -> accept_gate -> envelope, all real —
    the build-gated-round shape, with a prior file visible in context."""
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


_REMOVE_TESTS = (
    "```python\n"
    "import storage\n\n"
    "def test_remove_deletes_a_todo():\n"
    "    store = storage.TodoStore('rm-t.json')\n"
    "    store.add('a')\n"
    "    store.remove(1)\n"
    "    assert store.list() == []\n"
    "```\n"
)

_REMOVE_REQUIREMENT = "add a remove method to TodoStore in storage.py"


# --- (a) the probe's turn-2 shape: fragment refuses, naming the class -----


def test_a_bare_method_fragment_refuses_naming_the_dropped_class() -> None:
    """Red before this slice: the fragment fails at the executor with a
    genuine NameError/AttributeError (TodoStore undefined), and the gate
    reports 'tests did not pass' — accurate, but useless for the next
    round. Now it refuses naming exactly what the fragment dropped."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = (
        "```python\n"
        "def remove(self, todo_id):\n"
        "    todos = self._load()\n"
        "    todos = [t for t in todos if t['id'] != todo_id]\n"
        "    self._save(todos)\n"
        "```\n"
    )
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    assert result["gather"]["prior_surface"] == ["TodoStore"]
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == (
        "the deliverable for storage.py no longer defines TodoStore; "
        "an edit must ship the whole updated file"
    )
    assert result["envelope"]["diagnostics"]["accept"] is False


# --- (b) whole-file edit keeping the class and adding remove -> accepts ---


def test_b_whole_file_edit_keeping_the_class_and_adding_remove_accepts() -> None:
    """The #171 control still runs and passes: nothing about the surface
    check should block a real, complete edit."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    new_storage = _STORAGE_PY_PRIOR.rstrip("\n") + (
        "\n\n    def remove(self, todo_id):\n"
        "        todos = [t for t in self._load() if t['id'] != todo_id]\n"
        "        self._save(todos)\n"
    )
    code_writer = f"```python\n{new_storage}\n```\n"
    result = _build_gated_edit(context, code_writer, _REMOVE_TESTS)

    assert result["gather"]["prior_surface"] == ["TodoStore"]
    assert result["executor"]["tests_pass"] is True, result["executor"]["report"]
    assert result["accept_gate"]["accept"] is True, result["accept_gate"]["reason"]


# --- (c) the bound: a whole-file edit that drops a public name refuses ----


def test_c_whole_file_edit_dropping_a_public_function_refuses() -> None:
    """A syntactically complete, otherwise-passing whole-file edit still
    refuses when it silently drops a name the prior module publicly
    defined — the named bound (brief B-1 step 5)."""
    context = _context(
        "todo.py", _TODO_PY_PRIOR, "add a complete_todo function to todo.py"
    )
    tests_writer = (
        "```python\n"
        "from todo import add_todo, complete_todo\n\n"
        "def test_add_and_complete():\n"
        "    todos = []\n"
        "    add_todo(todos, 'x')\n"
        "    complete_todo(todos, 0)\n"
        "    assert todos[0]['done'] is True\n"
        "```\n"
    )
    # keeps add_todo, adds complete_todo, DROPS list_todos entirely — the
    # turn's own tests never reference list_todos, so they would pass.
    code_writer = (
        "```python\n"
        "def add_todo(todos, item):\n"
        "    todos.append({'text': item, 'done': False})\n\n"
        "def complete_todo(todos, index):\n"
        "    todos[index]['done'] = True\n"
        "```\n"
    )
    result = _build_gated_edit(context, code_writer, tests_writer)

    assert result["gather"]["prior_surface"] == ["add_todo", "list_todos"]
    assert result["accept_gate"]["accept"] is False
    assert result["accept_gate"]["reason"] == (
        "the deliverable for todo.py no longer defines list_todos; "
        "an edit must ship the whole updated file"
    )


# --- (d) byte-identical resubmission still accepts (#171's OK-9 shape) ----


def test_d_byte_identical_resubmission_still_accepts() -> None:
    """A whole-file resubmission identical to the prior keeps its surface
    trivially (nothing changed) and must not be refused for that."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    code_writer = f"```python\n{_STORAGE_PY_PRIOR}\n```\n"
    tests_writer = (
        "```python\n"
        "from storage import TodoStore\n\n"
        "def test_add_assigns_id():\n"
        "    store = TodoStore('add-t.json')\n"
        "    assert store.add('a')['id'] == 1\n"
        "```\n"
    )
    result = _build_gated_edit(context, code_writer, tests_writer)

    assert result["gather"]["prior_surface"] == ["TodoStore"]
    assert result["executor"]["tests_pass"] is True, result["executor"]["report"]
    assert result["accept_gate"]["accept"] is True, result["accept_gate"]["reason"]


# --- (e) no prior visible -> surface empty, main's behavior unaffected ----


def test_e_greenfield_no_prior_visible_leaves_prior_surface_empty() -> None:
    """No file has been read or written this conversation (the ladder's
    turn-1 / control shape) — nothing to compare the deliverable against,
    so the check must never fire."""
    tests_writer = (
        "```python\nfrom add import add\n"
        "def test_add():\n    assert add(2, 3) == 5\n```\n"
    )
    result = _build_gated_edit(
        "Write a function that adds two numbers in add.py",
        "```python\ndef add(a, b):\n    return a + b\n```\n",
        tests_writer,
    )

    assert result["gather"]["prior_surface"] == []
    assert result["executor"]["tests_pass"] is True
    assert result["accept_gate"]["accept"] is True


def test_e_ladder_turn_2_shape_fragment_refuses_whole_file_accepts() -> None:
    """The 13-turn ladder's turn 2 ('add a complete_todo function to
    todo.py'): today it passes only because the prior is five lines and the
    coder happens to rewrite it whole. Pinned both ways: a fragment over
    the SAME prior now refuses; the whole file still accepts."""
    context = _context(
        "todo.py",
        _TODO_PY_PRIOR,
        "add a complete_todo function to todo.py that marks a todo done",
    )
    tests_writer = (
        "```python\n"
        "from todo import add_todo, list_todos, complete_todo\n\n"
        "def test_complete_marks_done():\n"
        "    todos = []\n"
        "    add_todo(todos, 'a')\n"
        "    complete_todo(todos, 0)\n"
        "    assert list_todos(todos)[0]['done'] is True\n"
        "```\n"
    )
    fragment = (
        "```python\ndef complete_todo(todos, index):\n"
        "    todos[index]['done'] = True\n```\n"
    )
    whole_file = (
        "```python\n"
        "def add_todo(todos, item):\n"
        "    todos.append({'text': item, 'done': False})\n\n"
        "def list_todos(todos):\n"
        "    return todos\n\n"
        "def complete_todo(todos, index):\n"
        "    todos[index]['done'] = True\n"
        "```\n"
    )

    fragment_result = _build_gated_edit(context, fragment, tests_writer)
    assert fragment_result["accept_gate"]["accept"] is False
    assert "add_todo" in fragment_result["accept_gate"]["reason"]
    assert "list_todos" in fragment_result["accept_gate"]["reason"]

    whole_result = _build_gated_edit(context, whole_file, tests_writer)
    assert whole_result["executor"]["tests_pass"] is True, whole_result["executor"][
        "report"
    ]
    assert whole_result["accept_gate"]["accept"] is True, whole_result["accept_gate"][
        "reason"
    ]


# --- (f) the held round: the refusal reaches the next round's coder input -


def test_f_the_refusal_reason_reaches_the_next_rounds_coder_input() -> None:
    """build_gated_envelope composes diagnostics.retry_input from the
    accept-gate's OWN reason string; the outer loop carries it as the next
    round's input. Asserted on the rendered string the next round's coder
    actually receives, not a boolean flag."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = "```python\ndef remove(self, todo_id):\n    pass\n```\n"
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    assert result["accept_gate"]["accept"] is False
    retry_input = result["envelope"]["diagnostics"]["retry_input"]
    assert "no longer defines TodoStore" in retry_input
    assert "an edit must ship the whole updated file" in retry_input


# --- (h) reason hygiene: no absolute path, username, or test source -------


def test_h_the_refusal_reason_carries_no_path_or_username() -> None:
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = "```python\ndef remove(self, todo_id):\n    pass\n```\n"
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    reason = result["accept_gate"]["reason"]
    assert os.sep not in reason
    assert str(Path.home()) not in reason
    assert "tmp" not in reason.lower()
    # the reason names the DROPPED surface, never the test source that
    # happened to exercise it
    assert "assert" not in reason


# --- F1 rework: no fake fields; the real suite always runs ---------------

_HELD_MARKER = "[HELD TESTS: round 1 spec; regenerate ONLY the code]"


def test_f1_ledger_never_claims_tests_passed_for_a_suite_that_did_not_run() -> None:
    """Review F1: the short-circuit's tests_pass: true / n_tests: 0
    convention reached build_gated_envelope's diagnostics, which the #114
    ledger preserves verbatim — a turn whose suite never ran must never
    read as 'the tests passed'. The real run must happen regardless."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = (
        "```python\n"
        "def remove(self, todo_id):\n"
        "    todos = self._load()\n"
        "    todos = [t for t in todos if t['id'] != todo_id]\n"
        "    self._save(todos)\n"
        "```\n"
    )
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    # the real suite ran: a genuine NameError/AttributeError, not a faked pass
    assert result["executor"]["tests_pass"] is False, result["executor"]["report"]
    assert result["executor"]["n_tests"] >= 1
    assert result["envelope"]["diagnostics"]["tests_pass"] is False
    assert result["accept_gate"]["accept"] is False
    # the surface reason stands ALONE — never joined with "tests did not
    # pass", which would be true only incidentally here
    assert result["accept_gate"]["reason"] == (
        "the deliverable for storage.py no longer defines TodoStore; "
        "an edit must ship the whole updated file"
    )


def test_f1_the_retry_is_routed_to_the_held_round() -> None:
    """n_tests must reflect the REAL enumerated count so round 1's
    adequate tests are held as round 2's spec — the coder alone re-runs
    against them, with the surface reason in its input — instead of a
    plain retry that discards them (review F1(b))."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = "```python\ndef remove(self, todo_id):\n    pass\n```\n"
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    assert result["accept_gate"]["accept"] is False
    retry_input = result["envelope"]["diagnostics"]["retry_input"]
    assert _HELD_MARKER in retry_input
    assert "no longer defines TodoStore" in retry_input


def test_f1_no_dangling_executor_report_sentence() -> None:
    """F8: the short-circuit's empty report produced '... Executor
    report: . Regenerate...' in the retry prompt. The real run always
    carries a real report string."""
    context = _context("storage.py", _STORAGE_PY_PRIOR, _REMOVE_REQUIREMENT)
    fragment = "```python\ndef remove(self, todo_id):\n    pass\n```\n"
    result = _build_gated_edit(context, fragment, _REMOVE_TESTS)

    retry_input = result["envelope"]["diagnostics"].get("retry_input", "")
    assert "Executor report: ." not in retry_input


# --- the prompt half: the coder is told to ship the whole file on an edit -


def test_the_coder_prompt_asks_for_the_whole_file_on_a_visible_edit() -> None:
    """The guard above is what makes the turn honest when this is ignored
    (doctrine 2: structure, not a third prompt rule) — but the prompt
    should still ask for the right thing in the first place."""
    spec = yaml.safe_load(CODE_GENERATOR_YAML.read_text())
    coder = next(a for a in spec["agents"] if a["name"] == "coder")
    prompt = coder["system_prompt"]
    assert "whole" in prompt.lower() or "complete" in prompt.lower()
    assert "current content" in prompt.lower() or "shown" in prompt.lower()
