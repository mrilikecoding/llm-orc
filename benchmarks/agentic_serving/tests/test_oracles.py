"""Fixtures for the hidden build-turn correctness oracles (#131, WS-8 Arc D).

Methodology mirrors #84's adequacy-checker harness: every oracle is pinned
against BOTH accepting variants (FRR) and rejecting ones (FAR), so both error
rates are measured rather than assumed.

BOTH directions are dangerous here, and neither is the "safe" default:

- A FALSE ACCEPT hands a free pass to an arm that ships plausible-but-wrong
  code, restoring the §2 bias toward the comparator that the oracles exist to
  remove.
- A FALSE REJECT is WORSE, and this is the counter-intuitive part. Richer todo
  representations (dict, dataclass) correlate with design sophistication, so a
  representation-blind oracle rejects the frontier arm harder — and a frontier
  arm shipping GOOD code then scores "shipped, oracle-failed", which reads as
  exactly the plausible-but-wrong-code narrative the oracle was built to
  detect. An FRR that fabricates evidence for the hypothesis under test is
  worse than the bias it replaced.

Every case below was proven against the real CLI during the 2026-07-14
adversarial review before being encoded here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.agentic_serving import oracles


def _ws(tmp_path: Path, **modules: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, src in modules.items():
        (tmp_path / f"{name}.py").write_text(src)
    return tmp_path


# --------------------------------------------------------------------------
# Turn 1 — "write a function that adds a todo item to a list in todo.py"
# The prompt names NEITHER the function NOR the todo representation.
# --------------------------------------------------------------------------

T1_MUTATES = "def add_todo_item(todo_list, item):\n    todo_list.append(item)\n"
T1_OTHER_NAME = "def add_todo(items, item):\n    items.append(item)\n"
T1_RETURNS_NEW = "def add(items, item):\n    return items + [item]\n"
T1_SWAPPED_ARGS = "def add_item(item, items):\n    items.append(item)\n"
T1_GUARDS = (
    "def add_todo_item(todo_list, item):\n"
    "    if not isinstance(todo_list, list):\n"
    "        raise TypeError('todo_list must be a list')\n"
    "    todo_list.append(item)\n"
)
# The representation cases. The dict shape is the one turn 2 REQUIRES ("marks a
# todo done"), so rejecting it here would contradict the reason turn 2 is left
# un-oracled at all.
T1_DICT_WRAP = (
    "def add_todo(todos, item):\n    todos.append({'task': item, 'done': False})\n"
)
T1_DATACLASS_WRAP = (
    "from dataclasses import dataclass\n\n"
    "@dataclass\n"
    "class Todo:\n"
    "    text: str\n"
    "    done: bool = False\n\n"
    "def add_todo(todos, item):\n    todos.append(Todo(item))\n"
)
T1_KEYWORD_ONLY = "def add_todo(*, todos, item):\n    todos.append(item)\n"
T1_MODULE_LEVEL = "TODOS = []\n\ndef add_todo(item):\n    TODOS.append(item)\n"
# A PLAIN class (no custom __repr__) is at least as common in model output as a
# dataclass, and its default repr (`<todo.Todo object at 0x...>`) hides the
# item text. Recoverability must reach attribute VALUES, not just repr(seq) --
# the 2026-07-14 round-3 hunt showed the dataclass fixture passed only because
# dataclasses auto-generate a field-bearing repr.
T1_PLAIN_CLASS_WRAP = (
    "class Todo:\n"
    "    def __init__(self, text):\n"
    "        self.text = text\n"
    "        self.done = False\n\n"
    "def add_todo(todos, item):\n    todos.append(Todo(item))\n"
)
T1_SLOTS_CLASS_WRAP = (
    "class Todo:\n"
    "    __slots__ = ('text', 'done')\n"
    "    def __init__(self, text):\n"
    "        self.text = text\n"
    "        self.done = False\n\n"
    "def add_todo(todos, item):\n    todos.append(Todo(item))\n"
)
# Positional-only is keyword-only's mirror; the round that fixed keyword-only
# left it open.
T1_POSITIONAL_ONLY = "def add_todo(todos, item, /):\n    todos.append(item)\n"


@pytest.mark.parametrize(
    "src",
    [
        T1_MUTATES,
        T1_OTHER_NAME,
        T1_RETURNS_NEW,
        T1_SWAPPED_ARGS,
        T1_GUARDS,
        T1_DICT_WRAP,
        T1_DATACLASS_WRAP,
        T1_KEYWORD_ONLY,
        T1_MODULE_LEVEL,
        T1_PLAIN_CLASS_WRAP,
        T1_SLOTS_CLASS_WRAP,
        T1_POSITIONAL_ONLY,
    ],
    ids=[
        "mutates",
        "other_name",
        "returns_new",
        "swapped_args",
        "guards",
        "dict_wrap",
        "dataclass_wrap",
        "keyword_only",
        "module_level_list",
        "plain_class_wrap",
        "slots_class_wrap",
        "positional_only",
    ],
)
def test_turn1_accepts_correct_variants(tmp_path: Path, src: str) -> None:
    assert oracles.turn1_adds_todo(_ws(tmp_path, todo=src)).passed


T1_DROPS_ITEM = "def add_todo_item(todo_list, item):\n    pass\n"
T1_WRONG_ITEM = "def add_todo_item(todo_list, item):\n    todo_list.append('nope')\n"
T1_ALWAYS_RAISES = "def add_todo_item(todo_list, item):\n    raise ValueError('x')\n"
T1_NO_FUNCTION = "TODOS = []\n"
# An interpreter that does nothing exits 0. Exit code alone therefore cannot
# mean "the contract held" — a main() without an if-__name__ guard is ordinary
# model output, not an exotic attack.
T1_SYS_EXIT_AT_IMPORT = (
    "import sys\n\n"
    "def add_todo_item(todo_list, item):\n    pass\n\n"
    "def main():\n    sys.exit(0)\n\n"
    "main()\n"
)
T1_OS_EXIT_AT_IMPORT = (
    "import os\n\ndef add_todo_item(todo_list, item):\n    pass\n\nos._exit(0)\n"
)
# An empty seed cannot tell "adds" from "replaces", so these must be probed
# against a NON-empty list.
T1_DESTROYS_EXISTING = (
    "def add_todo(todos, item):\n    todos.clear()\n    todos.append(item)\n"
)
T1_REPLACES_LIST = "def add_todo(todos, item):\n    return [item]\n"
T1_ONLY_WORKS_ONCE = (
    "def add_todo(todos, item):\n"
    "    if todos:\n        return\n"
    "    todos.append(item)\n"
)
# `SENTINEL in items` calls __eq__ on the ELEMENT, so a permissive __eq__ passes
# while holding nothing.
T1_EQ_TRICK = (
    "class _Any:\n"
    "    def __eq__(self, other):\n        return True\n"
    "    def __hash__(self):\n        return 0\n\n"
    "def add_todo(todos, item):\n    todos.append(_Any())\n"
)
# The attribute-value recoverability that accepts a plain class must not become
# a blanket accept: a wrapper that DROPS the item's text still fails.
T1_CLASS_DROPS_TEXT = (
    "class Todo:\n"
    "    def __init__(self, text):\n"
    "        self.text = 'nope'\n\n"
    "def add_todo(todos, item):\n    todos.append(Todo(item))\n"
)


@pytest.mark.parametrize(
    "src",
    [
        T1_DROPS_ITEM,
        T1_WRONG_ITEM,
        T1_ALWAYS_RAISES,
        T1_NO_FUNCTION,
        T1_SYS_EXIT_AT_IMPORT,
        T1_OS_EXIT_AT_IMPORT,
        T1_DESTROYS_EXISTING,
        T1_REPLACES_LIST,
        T1_ONLY_WORKS_ONCE,
        T1_EQ_TRICK,
        T1_CLASS_DROPS_TEXT,
    ],
    ids=[
        "drops_item",
        "wrong_item",
        "always_raises",
        "no_function",
        "sys_exit_at_import",
        "os_exit_at_import",
        "destroys_existing",
        "replaces_whole_list",
        "only_works_once",
        "permissive_eq",
        "class_drops_text",
    ],
)
def test_turn1_rejects_broken_variants(tmp_path: Path, src: str) -> None:
    assert not oracles.turn1_adds_todo(_ws(tmp_path, todo=src)).passed


def test_turn1_missing_module_fails_closed(tmp_path: Path) -> None:
    assert not oracles.turn1_adds_todo(tmp_path).passed


def test_probe_instrument_failure_raises_instead_of_fabricating_a_verdict(
    tmp_path: Path,
) -> None:
    # An OSError starting the probe (missing workspace, disk full, bad path) is
    # a failure of the INSTRUMENT, not of the arm's code. Returning
    # passed=False would score a real shipped turn as shipped-broken -- the
    # thesis-fabricating direction -- with exit 0 and empty stderr, making the
    # battery's crash channel (oracle: null + nonzero in oracle-exits.tsv)
    # unreachable. The error must escape instead.
    with pytest.raises(OSError):
        oracles.turn1_adds_todo(tmp_path / "does-not-exist")


def test_turn1_syntax_error_fails_closed(tmp_path: Path) -> None:
    assert not oracles.turn1_adds_todo(_ws(tmp_path, todo="def add(:\n")).passed


# --------------------------------------------------------------------------
# Turn 6 — "create storage.py with save_todos and load_todos using json"
# NAMES the functions. Leaves signature order, arity and path type free.
# --------------------------------------------------------------------------

T6_TODOS_FIRST = (
    "import json\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)
T6_PATH_FIRST = (
    "import json\n"
    "def save_todos(path, todos):\n"
    "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)
# The prompt never mentions a path parameter, so a module-constant filename is a
# legitimate reading of it.
T6_NO_PATH = (
    "import json\n"
    "TODO_FILE = 'todos.json'\n"
    "def save_todos(todos):\n"
    "    with open(TODO_FILE, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos():\n"
    "    with open(TODO_FILE) as f:\n        return json.load(f)\n"
)
T6_PATHLIB = (
    "import json\n"
    "from pathlib import Path\n"
    "def save_todos(todos, path):\n    Path(path).write_text(json.dumps(todos))\n"
    "def load_todos(path):\n    return json.loads(Path(path).read_text())\n"
)
T6_GENERATOR_LOAD = (
    "import json\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n"
    "        for todo in json.load(f):\n            yield todo\n"
)


@pytest.mark.parametrize(
    "src",
    [T6_TODOS_FIRST, T6_PATH_FIRST, T6_NO_PATH, T6_PATHLIB, T6_GENERATOR_LOAD],
    ids=["todos_first", "path_first", "no_path_arg", "pathlib", "generator_load"],
)
def test_turn6_accepts_correct_variants(tmp_path: Path, src: str) -> None:
    assert oracles.turn6_storage_roundtrip(_ws(tmp_path, storage=src)).passed


T6_LOSES_DATA = (
    "import json\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'w') as f:\n        json.dump([], f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)
T6_LOAD_ONLY = "def load_todos(path):\n    return []\n"
# save/load run in ONE process, so a module-level cache round-trips perfectly
# while persisting nothing at all.
T6_MEMORY_CACHE = (
    "_CACHE = None\n"
    "def save_todos(todos, path):\n"
    "    global _CACHE\n    _CACHE = list(todos)\n"
    "def load_todos(path):\n"
    "    return _CACHE if _CACHE is not None else []\n"
)
# The prompt says "using json", so a pickle round-trip is a spec violation even
# though it round-trips.
T6_PICKLE = (
    "import pickle\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'wb') as f:\n        pickle.dump(todos, f)\n"
    "def load_todos(path):\n"
    "    with open(path, 'rb') as f:\n        return pickle.load(f)\n"
)
T6_EXIT_ZERO = "import sys\n\ndef save_todos(todos, path):\n    pass\n\nsys.exit(0)\n"


@pytest.mark.parametrize(
    "src",
    [T6_LOSES_DATA, T6_LOAD_ONLY, T6_MEMORY_CACHE, T6_PICKLE, T6_EXIT_ZERO],
    ids=["loses_data", "missing_save", "memory_cache", "pickle_not_json", "exit_zero"],
)
def test_turn6_rejects_broken_variants(tmp_path: Path, src: str) -> None:
    assert not oracles.turn6_storage_roundtrip(_ws(tmp_path, storage=src)).passed


def test_turn6_probe_does_not_mutate_the_scored_workspace(tmp_path: Path) -> None:
    # The probe CALLS the arm's save_todos. An implementation that hardcodes its
    # filename would otherwise overwrite the arm's real data file mid-run, and
    # later turns would be scored against a workspace the oracle corrupted.
    src = (
        "import json\n"
        "def save_todos(todos, path):\n"
        "    with open('todos.json', 'w') as f:\n        json.dump(todos, f)\n"
        "def load_todos(path):\n"
        "    with open('todos.json') as f:\n        return json.load(f)\n"
    )
    ws = _ws(tmp_path, storage=src)
    before = sorted(p.name for p in ws.iterdir())
    oracles.turn6_storage_roundtrip(ws)
    assert sorted(p.name for p in ws.iterdir()) == before


# --------------------------------------------------------------------------
# Turn 7 — "update todo.py to persist todos using storage.py"
# --------------------------------------------------------------------------

T7_IMPORTS_STORAGE = (
    "import storage\n\ndef save(todos, path):\n    storage.save_todos(todos, path)\n"
)
T7_FROM_IMPORT = (
    "from storage import save_todos\n\n"
    "def save(todos, path):\n    save_todos(todos, path)\n"
)


@pytest.mark.parametrize(
    "src", [T7_IMPORTS_STORAGE, T7_FROM_IMPORT], ids=["import_mod", "from_import"]
)
def test_turn7_accepts_real_storage_use(tmp_path: Path, src: str) -> None:
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert oracles.turn7_todo_persists(ws).passed


def test_turn7_rejects_todo_that_ignores_storage(tmp_path: Path) -> None:
    src = (
        "import json\n\n"
        "def save(todos, path):\n    open(path,'w').write(json.dumps(todos))\n"
    )
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


def test_turn7_rejects_import_in_a_comment(tmp_path: Path) -> None:
    src = "# uses storage.save_todos eventually\ndef save(todos, path):\n    pass\n"
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


def test_turn7_rejects_unused_import(tmp_path: Path) -> None:
    # A stray unused import is among the most common things a model ships, and a
    # namespace-only check credits it as composition. This is the exact case the
    # rung exists to catch.
    src = (
        "import json\n"
        "import storage  # noqa: F401\n\n"
        "def save(todos, path):\n"
        "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    )
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


def test_turn7_rejects_exit_zero_at_import(tmp_path: Path) -> None:
    src = "import sys\nsys.exit(0)\n"
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


# --- FRR found by the 2026-07-14 re-review ---------------------------------
# Every case below is CORRECT code that the oracle rejected. All three are
# design-SOPHISTICATION markers (a class, a deferred import, a versioned
# envelope), so they correlate with the frontier arm and would have scored
# "shipped, oracle-failed" -- fabricating exactly the plausible-but-wrong
# narrative these oracles exist to detect.

T7_CLASS_BASED = (
    "import storage\n\n"
    "class TodoList:\n"
    "    def __init__(self, items, path):\n"
    "        self.items = items\n        self.path = path\n"
    "    def save(self):\n        storage.save_todos(self.items, self.path)\n"
)
T7_FROM_IMPORT_IN_METHOD = (
    "from storage import save_todos\n\n"
    "class TodoList:\n"
    "    def save(self, items, path):\n        save_todos(items, path)\n"
)
T7_LOCAL_IMPORT = (
    "def save(todos, path):\n"
    "    import storage  # deferred to avoid a circular import\n"
    "    storage.save_todos(todos, path)\n"
)


@pytest.mark.parametrize(
    "src",
    [T7_CLASS_BASED, T7_FROM_IMPORT_IN_METHOD, T7_LOCAL_IMPORT],
    ids=["class_based", "from_import_in_method", "function_local_import"],
)
def test_turn7_accepts_sophisticated_composition(tmp_path: Path, src: str) -> None:
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert oracles.turn7_todo_persists(ws).passed


def test_turn7_still_rejects_unused_import_inside_a_class_only_file(
    tmp_path: Path,
) -> None:
    # The class fix must not become a blanket accept.
    src = (
        "import json\n"
        "import storage  # noqa: F401\n\n"
        "class TodoList:\n"
        "    def save(self, items, path):\n"
        "        with open(path, 'w') as f:\n            json.dump(items, f)\n"
    )
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


T6_ENVELOPE = (
    "import json\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'w') as f:\n"
    "        json.dump({'version': 1, 'todos': list(todos)}, f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n        return json.load(f)['todos']\n"
)
T6_INIT_ON_IMPORT = (
    "import json, os\n"
    "TODO_FILE = 'todos.json'\n"
    "if not os.path.exists(TODO_FILE):\n"
    "    with open(TODO_FILE, 'w') as f:\n        json.dump([], f)\n"
    "def save_todos(todos):\n"
    "    with open(TODO_FILE, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos():\n"
    "    with open(TODO_FILE) as f:\n        return json.load(f)\n"
)
T6_KEYWORD_ONLY = (
    "import json\n"
    "def save_todos(*, todos, path):\n"
    "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos(*, path):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)
T6_POSITIONAL_ONLY = (
    "import json\n"
    "def save_todos(todos, path, /):\n"
    "    with open(path, 'w') as f:\n        json.dump(todos, f)\n"
    "def load_todos(path, /):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)


@pytest.mark.parametrize(
    "src",
    [T6_ENVELOPE, T6_INIT_ON_IMPORT, T6_KEYWORD_ONLY, T6_POSITIONAL_ONLY],
    ids=[
        "versioned_envelope",
        "creates_file_on_import",
        "keyword_only",
        "positional_only",
    ],
)
def test_turn6_accepts_sophisticated_designs(tmp_path: Path, src: str) -> None:
    assert oracles.turn6_storage_roundtrip(_ws(tmp_path, storage=src)).passed


def test_turn6_accepts_when_a_json_file_already_exists(tmp_path: Path) -> None:
    # By turn 6 the workspace already holds files; a probe that only looks at
    # NEWLY created json misses a write to a pre-existing path.
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST)
    (ws / "todos_probe.json").write_text("[]")
    assert oracles.turn6_storage_roundtrip(ws).passed


def test_turn1_verdict_does_not_depend_on_declaration_order(tmp_path: Path) -> None:
    # The module-level branch seeds every module list. If it clobbers an
    # unrelated constant and never restores, the verdict depends on which name
    # getmembers reaches first -- non-deterministic scoring, the exact property
    # that disqualified name-keyed oracles in the first place.
    body = (
        "def add_todo(item):\n"
        "    if VALID_STATUSES != ['open', 'done']:\n"
        "        raise ValueError('bad status config: %r' % (VALID_STATUSES,))\n"
        "    TODOS.append(item)\n"
    )
    const_first = "VALID_STATUSES = ['open', 'done']\nTODOS = []\n\n" + body
    todos_first = "TODOS = []\nVALID_STATUSES = ['open', 'done']\n\n" + body
    a = oracles.turn1_adds_todo(_ws(tmp_path / "a", todo=const_first)).passed
    b = oracles.turn1_adds_todo(_ws(tmp_path / "b", todo=todos_first)).passed
    assert a == b, "verdict flipped on declaration order alone"
    assert a is True
