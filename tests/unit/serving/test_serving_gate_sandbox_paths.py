"""#184 (subsuming #182 slice A) — the gate sandbox mirrors the workspace.

Evidence: probe turn 1 (docs/plans/2026-09-11-daily-driver-probe/README.md
— tests written as ``from storage import`` for a ``todo/`` package,
destination flattened to ``test_storage.py``) and ladder turn 7 in 3/3
think-on runs (docs/plans/2026-09-11-ladder-runs-171-thinkoff/README.md —
"code failed to load: No module named 'storage'" on a correctly-shipped
fix, because the re-fix sandbox held no workspace files).

Two defects, one root cause: ``accept_gather._workspace`` keyed files by
BASENAME, so ``todo/storage.py`` and ``lib/storage.py`` conflated, and
``accept_executor._materialize`` wrote every workspace file and the
deliverable shadow at a bare basename in a single flat directory — a real
package's ``import todo.storage`` could never resolve there. This file
pins the fix: the workspace map is keyed by RELATIVE PATH (the client's
own absolute prefix stripped when present), and the executor materializes
every file at its real relative path (creating directories; refusing —
never sanitizing — an absolute or escaping path).

Driven via subprocess exactly as the L0 engine runs a script node, matching
every sibling harness in this directory; the ``_helpers``/``_materialize``
unit-level tests import the module directly (mirrors
test_serving_accept_gather.py's direct import of ``_workspace``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / ".llm-orc" / "scripts" / "agentic_serving"

sys.path.insert(0, str(SCRIPTS))
from _helpers import (  # type: ignore[import-not-found]  # noqa: E402
    resolve_workspace_entries,
    workspace,
    workspace_entries,
)
from accept_executor import (  # type: ignore[import-not-found]  # noqa: E402
    _materialize,
    _unplaceable_workspace_files,
)

_workspace = workspace
_workspace_entries = workspace_entries
_resolve_workspace_entries = resolve_workspace_entries

GATHER = SCRIPTS / "accept_gather.py"
EXECUTOR = SCRIPTS / "accept_executor.py"
REFIX_GATHER = SCRIPTS / "refix_gather.py"
REFIX_SELECT = SCRIPTS / "refix_select.py"
REFIX_ENVELOPE = SCRIPTS / "refix_envelope.py"
PRIOR_CODE_MARKER = "[PRIOR CODE: this turn's write, before the re-fix]"


def _node(script: Path, deps: dict[str, Any], input_data: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"dependencies": deps}
    if input_data:
        payload["input_data"] = input_data
    out = subprocess.run(
        [sys.executable, str(script)],
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


def _executor_from_gather(gathered: dict[str, Any]) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        input=json.dumps(gathered),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Any] = json.loads(out)
    return result


# --- mechanism 1: one workspace reader, keyed by relative path -------------


def test_two_files_sharing_a_basename_in_different_directories_stay_distinct() -> None:
    context = (
        "assistant: [read todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
        "assistant: [read lib/storage.py]\n"
        "  def helper():\n"
        "      return 1\n"
    )
    ws = _workspace(context)
    assert ws == {
        "todo/storage.py": "class TodoStore:\n    pass",
        "lib/storage.py": "def helper():\n    return 1",
    }


def test_single_absolute_header_strips_to_its_bare_basename() -> None:
    """The flat-workspace subcase (#184 invariant): one file at the
    client's repo root, absolute path, strips to its bare basename —
    byte-identical to the pre-#184 basename keying."""
    context = (
        "assistant: [read /Users/nate/proj/storage.py]\n"
        "  def put(k, v):\n"
        "      return (k, v)"
    )
    ws = _workspace(context)
    assert ws == {"storage.py": "def put(k, v):\n    return (k, v)"}


# --- A1 rework: a single absolute header recovers its real directory ------


def test_a1_single_absolute_header_matches_the_named_destination_as_a_suffix() -> None:
    """The reviewer's A1 repro: ONE absolute header, no glob listing — the
    ask's own named destination (todo/storage.py) matches as a suffix of
    the absolute path, recovering the true root. Mutant: reverting to
    dirname-only (this branch's ORIGINAL single-header rule) loses the
    todo/ directory -> red."""
    context = (
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass"
    )
    entries, rule = _resolve_workspace_entries(context, target_path="todo/storage.py")
    assert dict(entries) == {"todo/storage.py": "class TodoStore:\n    pass"}
    assert rule == "suffix-match"


def test_a1_an_unrelated_relative_sibling_is_not_a_coincidental_suffix_match() -> None:
    """The suffix candidates include every already-relative header in the
    same render, not only the ask's own target — but an unrelated sibling
    must not coincidentally match, and this stays an honest basename
    fallback rather than a false positive."""
    context = (
        "assistant: [read todo/other.py]\n"
        "  OTHER = 1\n"
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
    )
    entries, rule = _resolve_workspace_entries(context, target_path="")
    resolved = dict(entries)
    assert resolved["todo/other.py"] == "OTHER = 1"
    assert "storage.py" in resolved
    assert rule == "basename"


def test_a1_glob_listing_recovers_the_root_even_with_a_single_read_header() -> None:
    """Rule (a): a visible glob listing outranks everything else — its
    paths routinely span several directories, giving a much more reliable
    root than one or two read/write headers alone."""
    context = (
        "assistant: [globbed py]\n"
        "  /Users/u/proj/todo/storage.py\n"
        "  /Users/u/proj/todo/cli.py\n"
        "  /Users/u/proj/tests/test_storage.py\n"
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
    )
    entries, rule = _resolve_workspace_entries(context, target_path="")
    assert dict(entries) == {"todo/storage.py": "class TodoStore:\n    pass"}
    assert rule == "glob"


def test_a1_no_glob_no_suffix_match_falls_back_to_basename_honestly() -> None:
    """(d): when nothing recovers the real root, the single header still
    resolves (never left unmaterialized) — but as a NAMED, reported
    fallback, not a silent guess."""
    context = (
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass"
    )
    entries, rule = _resolve_workspace_entries(context, target_path="")
    assert dict(entries) == {"storage.py": "class TodoStore:\n    pass"}
    assert rule == "basename"


def test_a1_two_absolute_headers_still_use_common_prefix_when_no_glob() -> None:
    context = (
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
        "assistant: [read /Users/u/proj/tests/test_storage.py]\n"
        "  def test_x():\n"
        "      pass\n"
    )
    entries, rule = _resolve_workspace_entries(context, target_path="")
    assert dict(entries) == {
        "todo/storage.py": "class TodoStore:\n    pass",
        "tests/test_storage.py": "def test_x():\n    pass",
    }
    assert rule == "common-prefix"


def test_multiple_absolute_headers_strip_the_shared_project_root() -> None:
    """Fresh-input hunt: two absolute headers under the same client root,
    in DIFFERENT subdirectories — the shared root strips to the client's
    project root, preserving each file's own subdirectory."""
    context = (
        "assistant: [read /Users/nate/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
        "assistant: [read /Users/nate/proj/tests/test_storage.py]\n"
        "  def test_x():\n"
        "      pass\n"
    )
    ws = _workspace(context)
    assert ws == {
        "todo/storage.py": "class TodoStore:\n    pass",
        "tests/test_storage.py": "def test_x():\n    pass",
    }


def test_absolute_headers_sharing_no_common_root_stay_absolute() -> None:
    """Fresh-input hunt: two clients' absolute trees in one render (no
    real shared project root beyond the filesystem root itself) — MEASURED
    outcome, recorded as a bound: neither path is guessed at, both stay
    exactly as rendered (still absolute), so materialization refuses them
    rather than inventing a relative path."""
    context = (
        "assistant: [read /Users/alice/proj/a.py]\n"
        "  A = 1\n"
        "assistant: [read /tmp/other/b.py]\n"
        "  B = 2\n"
    )
    ws = _workspace(context)
    assert ws == {
        "/Users/alice/proj/a.py": "A = 1",
        "/tmp/other/b.py": "B = 2",
    }


def test_one_relative_and_one_absolute_header_only_the_absolute_one_strips() -> None:
    """Fresh-input hunt: a mixed render (one header already relative, one
    absolute) — the relative header is untouched; the lone absolute
    header still strips to its own basename (the single-path rule)."""
    context = (
        "assistant: [read notes.md]\n"
        "  hello\n"
        "assistant: [read /Users/nate/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
    )
    ws = _workspace(context)
    assert ws == {
        "notes.md": "hello",
        "storage.py": "class TodoStore:\n    pass",
    }


def test_a_path_with_spaces_survives_the_strip() -> None:
    """Fresh-input hunt: a path with spaces in a directory component."""
    context = (
        "assistant: [read /Users/nate/my project/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
        "assistant: [read /Users/nate/my project/tests/test_storage.py]\n"
        "  def test_x():\n"
        "      pass\n"
    )
    ws = _workspace(context)
    assert "todo/storage.py" in ws
    assert "tests/test_storage.py" in ws


def test_a_trailing_truncated_variant_never_materializes_even_when_absolute() -> None:
    """Fresh-input hunt: a header with a trailing '(truncated)'/'(failed)'
    variant — never materialized, absolute or not (unchanged invariant)."""
    context = (
        "assistant: [read /Users/nate/proj/big.py (truncated)]\n"
        "  xxxx\n"
        "assistant: [read /Users/nate/proj/ok.py]\n"
        "  OK = 1\n"
    )
    ws = _workspace(context)
    assert ws == {"ok.py": "OK = 1"}


def test_a_dotdot_header_is_kept_as_a_plain_relative_string_by_the_reader() -> None:
    """The workspace READER never refuses anything (that's #184 mechanism
    2's job, at materialization) — a '../' header is simply a relative
    path like any other from this function's point of view."""
    context = "assistant: [read ../etc/passwd]\n  root:x:0:0\n"
    entries = _workspace_entries(context)
    assert entries == [("../etc/passwd", "root:x:0:0")]


# --- mechanism 2: materialize at real relative paths, refuse unsafe ones --


def _materialize_in_tmp(
    workspace: dict[str, str],
    target_file: str = "",
    target_path: str = "",
    code: str = "CODE = 1\n",
    tests: str = "TESTS = 1\n",
) -> Path:
    tmp = tempfile.mkdtemp()
    _materialize(tmp, code, tests, workspace, target_file, target_path)
    return Path(tmp)


def test_materialize_creates_nested_directories_for_a_workspace_file() -> None:
    tmp = _materialize_in_tmp({"todo/storage.py": "class TodoStore:\n    pass"})
    written = tmp / "todo" / "storage.py"
    assert written.read_text(encoding="utf-8") == "class TodoStore:\n    pass"


def test_materialize_refuses_an_absolute_workspace_path() -> None:
    # A harmless absolute path outside the sandbox root — never a real
    # system path: this must be REFUSED, not sanitized into a relative one.
    tmp = _materialize_in_tmp({"/nonexistent-184-probe/evil.py": "EVIL = True"})
    assert not any(p.name == "evil.py" for p in tmp.rglob("*"))


def test_materialize_refuses_a_path_that_escapes_the_sandbox_root() -> None:
    tmp = _materialize_in_tmp({"../evil.py": "EVIL = True"})
    assert not (tmp.parent / "evil.py").exists()
    assert not any(p.name == "evil.py" for p in tmp.rglob("*"))


def test_materialize_writes_the_deliverable_at_target_path_not_just_basename() -> None:
    tmp = _materialize_in_tmp(
        {},
        target_file="storage.py",
        target_path="todo/storage.py",
        code="RESULT = 42\n",
    )
    written = tmp / "todo" / "storage.py"
    assert written.read_text(encoding="utf-8") == "RESULT = 42\n"
    # never ALSO written flat at the bare basename
    assert not (tmp / "storage.py").exists()


def test_materialize_falls_back_to_target_file_when_target_path_is_empty() -> None:
    """Back-compat for the re-fix route (flat targets only, today):
    passing no target_path shadows at the bare target_file, unchanged."""
    tmp = _materialize_in_tmp({}, target_file="todo.py", target_path="", code="X = 1\n")
    assert (tmp / "todo.py").read_text(encoding="utf-8") == "X = 1\n"


def test_materialize_still_excludes_solution_and_tests_from_workspace_writes() -> None:
    tmp = _materialize_in_tmp(
        {"solution.py": "HACK = True", "tests.py": "HACK2 = True"},
        code="REAL_CODE = 1\n",
        tests="REAL_TESTS = 1\n",
    )
    assert (tmp / "solution.py").read_text(encoding="utf-8") == "REAL_CODE = 1\n"
    assert (tmp / "tests.py").read_text(encoding="utf-8") == "REAL_TESTS = 1\n"


def test_a3_a_nested_tests_dot_py_is_not_dropped_for_sharing_a_root_basename() -> None:
    """A3 (review, MEDIUM): the solution.py/tests.py skip compared only the
    BASENAME, so todo/tests.py — a real package module, unrelated to the
    runner's own root-level tests.py — vanished the same way. Nested
    namesakes must survive; only the bare root-level names are reserved."""
    tmp = _materialize_in_tmp(
        {"todo/tests.py": "FIXTURES = 1\n", "pkg/solution.py": "ANSWER = 42\n"}
    )
    assert (tmp / "todo" / "tests.py").read_text(encoding="utf-8") == "FIXTURES = 1\n"
    assert (tmp / "pkg" / "solution.py").read_text(encoding="utf-8") == "ANSWER = 42\n"


def test_a3_nested_tests_module_is_importable_end_to_end() -> None:
    """The reviewer's exact repro: tests import a nested todo/tests.py
    fixture module alongside todo/helpers.py — both must resolve."""
    context = (
        "assistant: [read todo/__init__.py]\n  \n"
        "assistant: [read todo/storage.py]\n"
        "  class TodoStore:\n      pass\n"
        "assistant: [read todo/helpers.py]\n"
        "  HELPER = 1\n"
        "assistant: [read todo/tests.py]\n"
        "  FIXTURE = 2\n"
        "\n\nCurrent request: add a remove method to TodoStore in todo/storage.py"
    )
    gathered = _node(
        GATHER,
        {
            "code_writer": _dep(
                _sub_ensemble_response("```python\nclass TodoStore:\n    pass\n```\n")
            ),
            "test_writer": _dep(
                _sub_ensemble_response(
                    "```python\nfrom todo.helpers import HELPER\n"
                    "from todo.tests import FIXTURE\n\n"
                    "def test_x():\n"
                    "    assert HELPER == 1\n"
                    "    assert FIXTURE == 2\n"
                    "```\n"
                )
            ),
        },
        input_data=context,
    )
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True, result["report"]


# --- A2: a workspace entry that can't be placed makes the executor refuse -


def test_unplaceable_workspace_files_names_absolute_and_escaping_entries() -> None:
    names = _unplaceable_workspace_files(
        {
            "todo/storage.py": "x",
            "/Users/u/proj/a.py": "y",
            "/home/v/other/lib.py": "z",
            "../escape.py": "w",
        }
    )
    assert names == ["a.py", "escape.py", "lib.py"]


def test_unplaceable_workspace_files_empty_when_everything_placed() -> None:
    assert _unplaceable_workspace_files({"todo/storage.py": "x", "b.py": "y"}) == []


def test_a2_two_clients_roots_refuses_instead_of_silently_shipping() -> None:
    """The reviewer's A2 repro: two absolute headers sharing no common
    root. Brief bound 4 promised 'refused, nothing materialized' — the
    materialization half held, but nothing turned the drop into a
    verdict, so the turn ran against an empty workspace and could ship a
    wrong-or-untested deliverable. Mutant: dropping the workspace_unplaced
    check from accept_gate's accept formula turns this red (accept: True)."""
    gathered = {
        "requirement": "add a remove method to TodoStore",
        "code": (
            "class TodoStore:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "    def remove(self, item):\n"
            "        self.items.remove(item)\n"
            "    def list(self):\n"
            "        return list(self.items)\n"
        ),
        "tests": (
            "def test_remove_clears_the_item():\n"
            "    store = TodoStore()\n"
            "    store.items = ['x']\n"
            "    store.remove('x')\n"
            "    assert store.list() == []\n"
        ),
        "workspace": {
            "/Users/u/proj/todo/storage.py": "class TodoStore:\n    pass\n",
            "/home/v/other/lib.py": "def helper():\n    return 1\n",
        },
        "target_file": "",
        "target_path": "",
        "prior_surface": [],
    }
    executor = _executor_from_gather(gathered)
    assert executor["workspace_unplaced"] == ["lib.py", "storage.py"]
    judge = _node(SCRIPTS / "adequacy_check.py", {"executor": _dep(executor)})
    accept_gate = _node(
        SCRIPTS / "accept_gate.py",
        {"executor": _dep(executor), "judge": _dep(judge)},
    )
    assert accept_gate["accept"] is False
    assert "could not be placed in the sandbox" in accept_gate["reason"]
    assert "/" not in accept_gate["reason"]


# --- mechanism 1+2+3 together: a real nested package resolves -------------

_TODO_STORE = (
    "class TodoStore:\n"
    "    def __init__(self):\n"
    "        self.items = []\n"
    "    def add(self, item):\n"
    "        self.items.append(item)\n"
    "    def list(self):\n"
    "        return list(self.items)\n"
)

_TODO_STORE_WITH_REMOVE = _TODO_STORE + (
    "    def remove(self, item):\n        self.items.remove(item)\n"
)

_REMOVE_CODE_WRITER = _sub_ensemble_response(
    f"```python\n{_TODO_STORE_WITH_REMOVE}\n```\n"
)

_REMOVE_TEST_WRITER = _sub_ensemble_response(
    "```python\nfrom todo.storage import TodoStore\n\n"
    "def test_remove():\n"
    "    store = TodoStore()\n"
    "    store.add('x')\n"
    "    store.remove('x')\n"
    "    assert store.list() == []\n"
    "```\n"
)


def _probe_turn_one_context(*, with_init: bool = True) -> str:
    blocks = ""
    if with_init:
        blocks += "assistant: [read todo/__init__.py]\n  \n"
    blocks += "assistant: [read todo/storage.py]\n" + "".join(
        f"  {line}\n" for line in _TODO_STORE.splitlines()
    )
    return blocks


def test_workspace_sibling_import_resolves_independent_of_the_target_shadow() -> None:
    """Isolates mechanism 1 (the workspace fold) from mechanism 2 (the
    deliverable's own shadow write): todo/helpers.py is a pure workspace
    READ, never the turn's target — only correct WORKSPACE keying can put
    it at todo/helpers.py in the sandbox. Mutant: basename keying restored
    in _helpers.fold_workspace turns this red (ModuleNotFoundError: no
    module named 'todo.helpers'), independent of the target's own shadow
    (which still lands todo/storage.py correctly either way)."""
    context = (
        "assistant: [read todo/helpers.py]\n"
        "  def format_item(item):\n"
        "      return str(item)\n"
        "assistant: [read todo/storage.py]\n"
        + "".join(f"  {line}\n" for line in _TODO_STORE.splitlines())
        + "\n\nCurrent request: add remove(item) to TodoStore in todo/storage.py"
    )
    gathered = _node(
        GATHER,
        {
            "code_writer": _dep(_REMOVE_CODE_WRITER),
            "test_writer": _dep(
                _sub_ensemble_response(
                    "```python\nfrom todo.storage import TodoStore\n"
                    "from todo.helpers import format_item\n\n"
                    "def test_remove():\n"
                    "    store = TodoStore()\n"
                    "    store.add('x')\n"
                    "    store.remove('x')\n"
                    "    assert store.list() == []\n"
                    "    assert format_item('x') == 'x'\n"
                    "```\n"
                )
            ),
        },
        input_data=context,
    )
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True, result["report"]


def test_nested_package_import_resolves_in_the_sandbox() -> None:
    """Instrument 1: the probe turn-1 shape. Mutant (report this in the
    commit body): reverting the workspace fold to basename keying, OR
    reverting _materialize to write every file at a bare basename, turns
    this red — ImportError: No module named 'todo'."""
    context = _probe_turn_one_context() + (
        "\n\nCurrent request: add remove(item) to TodoStore in todo/storage.py"
    )
    gathered = _node(
        GATHER,
        {
            "code_writer": _dep(_REMOVE_CODE_WRITER),
            "test_writer": _dep(_REMOVE_TEST_WRITER),
        },
        input_data=context,
    )
    assert gathered["target_path"] == "todo/storage.py"
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True, result["report"]


def test_namespace_package_import_without_init_still_resolves() -> None:
    """Mechanism 3's PEP 420 bound: no todo/__init__.py in the workspace at
    all — todo/ still imports as a namespace package."""
    context = _probe_turn_one_context(with_init=False) + (
        "\n\nCurrent request: add remove(item) to TodoStore in todo/storage.py"
    )
    gathered = _node(
        GATHER,
        {
            "code_writer": _dep(_REMOVE_CODE_WRITER),
            "test_writer": _dep(_REMOVE_TEST_WRITER),
        },
        input_data=context,
    )
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True, result["report"]


def test_edit_shadow_generalizes_to_a_nested_destination() -> None:
    """Nested sibling of test_edit_turn_deliverable_shadows_the_stale_
    workspace_copy (test_serving_accept_gather.py): the stale workspace
    copy at todo/storage.py must be shadowed by the deliverable at the
    SAME real path, not a flat basename copy that leaves the stale nested
    file untouched."""
    context = (
        "assistant: [read todo/storage.py]\n"
        + "".join(f"  {line}\n" for line in _TODO_STORE.splitlines())
        + "\n\nCurrent request: add remove(item) to TodoStore in todo/storage.py"
    )
    gathered = _node(
        GATHER,
        {
            "code_writer": _dep(_REMOVE_CODE_WRITER),
            "test_writer": _dep(_REMOVE_TEST_WRITER),
        },
        input_data=context,
    )
    result = _executor_from_gather(gathered)
    assert result["tests_pass"] is True, result["report"]


# --- mechanism 4: re-fix gets the workspace, end to end -------------------


def test_refix_candidate_imports_a_conversation_written_sibling() -> None:
    """Instrument 2: the ladder turn-7 shape through the REAL re-fix chain
    (refix_gather -> refix_select -> accept_executor -> refix_envelope),
    storage.py conversation-written. Mutant (report in the commit body):
    refix_gather emitting no workspace turns this red — the candidate's
    'from storage import save_todos' fails to load, and the envelope's
    accept_reason names a load failure instead of the real test result."""
    conversation = (
        "assistant: [wrote storage.py]\n"
        "  def save_todos(todos):\n"
        "      pass\n"
        "  def load_todos():\n"
        "      return []\n"
        "assistant: [ran pytest -q]\n"
        "  F.\n"
        "  E       assert 5 == 8\n"
        "  1 failed, 1 passed in 0.02s\n"
        "assistant: [read test_todo.py]\n"
        "  from todo import add_todo\n"
        "\n"
        "  def test_add_todo():\n"
        "      todos = []\n"
        "      add_todo(todos, 'x')\n"
        "      assert todos == ['x']"
    )
    prior_code = (
        "from storage import save_todos, load_todos\n\n"
        "def add_todo(todos, item):\n"
        "    todos.append(item)\n"
        "    save_todos(todos)\n"
    )
    dispatch_input = (
        f"Conversation so far:\n{conversation}\n\n"
        f"{PRIOR_CODE_MARKER}\n{prior_code}\n\n"
        "Current request: fix add_todo in todo.py"
    )
    gathered = _node(REFIX_GATHER, {}, input_data=dispatch_input)
    assert gathered["workspace"]["storage.py"] == (
        "def save_todos(todos):\n    pass\ndef load_todos():\n    return []"
    )
    assert gathered["needs_model_edit"] is True

    # the model_edit seat regenerates the same correct candidate, still
    # importing the sibling module the conversation wrote
    model_edit_response = f"```python\n{prior_code}\n```\n"
    selected = _node(
        REFIX_SELECT,
        {"gather": _dep(gathered), "model_edit": _dep(model_edit_response)},
    )
    executor = _node(EXECUTOR, {"select": _dep(selected)})
    envelope = _node(
        REFIX_ENVELOPE, {"select": _dep(selected), "executor": _dep(executor)}
    )

    assert (
        executor["report"]
        != "code failed to load: ModuleNotFoundError: No module named 'storage'"
    )
    diagnostics = envelope["diagnostics"]
    assert diagnostics["accept"] is True, diagnostics["accept_reason"]
    assert "failed to load" not in diagnostics["accept_reason"]


def test_a2_refix_refuses_when_a_workspace_file_could_not_be_placed() -> None:
    """A2's follow-up on the re-fix route: two absolute headers sharing no
    common root — the workspace stays absolute (nothing materializes), and
    refix_envelope must refuse rather than let a candidate that never
    touched the workspace ship anyway. Mutant: dropping the
    workspace_unplaced check from refix_envelope's accept formula turns
    this red (accept: True)."""
    conversation = (
        "assistant: [read /Users/u/proj/todo/storage.py]\n"
        "  class TodoStore:\n"
        "      pass\n"
        "assistant: [read /home/v/other/lib.py]\n"
        "  def helper():\n"
        "      return 1\n"
        "assistant: [ran pytest -q]\n"
        "  F.\n"
        "  E       assert 5 == 8\n"
        "  1 failed, 1 passed in 0.02s\n"
        "assistant: [read test_calc.py]\n"
        "  def test_add_one():\n"
        "      result = add_one(2)\n"
        "      assert result == 3"
    )
    prior_code = "def add_one(n):\n    return n\n"
    dispatch_input = (
        f"Conversation so far:\n{conversation}\n\n"
        f"{PRIOR_CODE_MARKER}\n{prior_code}\n\n"
        "Current request: fix add_one in calc.py"
    )
    gathered = _node(REFIX_GATHER, {}, input_data=dispatch_input)
    resolved = gathered["workspace"]
    assert resolved["/Users/u/proj/todo/storage.py"] == "class TodoStore:\n    pass"
    assert resolved["/home/v/other/lib.py"] == "def helper():\n    return 1"

    model_edit_response = "```python\ndef add_one(n):\n    return n + 1\n```\n"
    selected = _node(
        REFIX_SELECT,
        {"gather": _dep(gathered), "model_edit": _dep(model_edit_response)},
    )
    executor = _node(EXECUTOR, {"select": _dep(selected)})
    assert executor["workspace_unplaced"] == ["lib.py", "storage.py"]
    envelope = _node(
        REFIX_ENVELOPE, {"select": _dep(selected), "executor": _dep(executor)}
    )
    diagnostics = envelope["diagnostics"]
    assert diagnostics["accept"] is False
    assert "could not be placed in the sandbox" in diagnostics["accept_reason"]
    assert "/" not in diagnostics["accept_reason"]
