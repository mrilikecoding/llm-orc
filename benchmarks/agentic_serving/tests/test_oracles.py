"""Fixtures for the hidden build-turn correctness oracles (#131, WS-8 Arc D).

Methodology mirrors #84's adequacy-checker harness: every oracle is pinned
against BOTH accepting variants (styles a correct arm may legitimately ship)
and rejecting ones (plausible-but-wrong code), so FRR and FAR are measured, not
assumed. An oracle that accepts everything is worse than no oracle: it would
hand the comparator a free pass, which is the §2 bias the oracles exist to
remove.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.agentic_serving import oracles


def _ws(tmp_path: Path, **modules: str) -> Path:
    for name, src in modules.items():
        (tmp_path / f"{name}.py").write_text(src)
    return tmp_path


# --- turn 1: name-free. "write a function that adds a todo item to a list" ---
# Measured: the same seat shipped `add_todo` and `add_todo_item` for this exact
# prompt across two runs, so the oracle must not key on the name.

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


@pytest.mark.parametrize(
    "src",
    [T1_MUTATES, T1_OTHER_NAME, T1_RETURNS_NEW, T1_SWAPPED_ARGS, T1_GUARDS],
    ids=["mutates", "other_name", "returns_new", "swapped_args", "guards"],
)
def test_turn1_accepts_correct_variants(tmp_path: Path, src: str) -> None:
    # FRR: every legitimate style a correct arm may ship must pass.
    assert oracles.turn1_adds_todo(_ws(tmp_path, todo=src)).passed


T1_DROPS_ITEM = "def add_todo_item(todo_list, item):\n    pass\n"
T1_WRONG_ITEM = "def add_todo_item(todo_list, item):\n    todo_list.append('nope')\n"
T1_ALWAYS_RAISES = "def add_todo_item(todo_list, item):\n    raise ValueError('x')\n"
T1_NO_FUNCTION = "TODOS = []\n"


@pytest.mark.parametrize(
    "src",
    [T1_DROPS_ITEM, T1_WRONG_ITEM, T1_ALWAYS_RAISES, T1_NO_FUNCTION],
    ids=["drops_item", "wrong_item", "always_raises", "no_function"],
)
def test_turn1_rejects_broken_variants(tmp_path: Path, src: str) -> None:
    # FAR: tolerance must not decay into accepting anything that runs.
    assert not oracles.turn1_adds_todo(_ws(tmp_path, todo=src)).passed


def test_turn1_missing_module_fails_closed(tmp_path: Path) -> None:
    assert not oracles.turn1_adds_todo(tmp_path).passed


def test_turn1_syntax_error_fails_closed(tmp_path: Path) -> None:
    assert not oracles.turn1_adds_todo(_ws(tmp_path, todo="def add(:\n")).passed


# --- turn 6: names save_todos/load_todos; SIGNATURE order is free ---

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


@pytest.mark.parametrize(
    "src", [T6_TODOS_FIRST, T6_PATH_FIRST], ids=["todos_first", "path_first"]
)
def test_turn6_accepts_either_signature_order(tmp_path: Path, src: str) -> None:
    assert oracles.turn6_storage_roundtrip(_ws(tmp_path, storage=src)).passed


T6_LOSES_DATA = (
    "import json\n"
    "def save_todos(todos, path):\n"
    "    with open(path, 'w') as f:\n        json.dump([], f)\n"
    "def load_todos(path):\n"
    "    with open(path) as f:\n        return json.load(f)\n"
)
T6_LOAD_ONLY = "def load_todos(path):\n    return []\n"


@pytest.mark.parametrize(
    "src", [T6_LOSES_DATA, T6_LOAD_ONLY], ids=["loses_data", "missing_save"]
)
def test_turn6_rejects_broken_roundtrip(tmp_path: Path, src: str) -> None:
    assert not oracles.turn6_storage_roundtrip(_ws(tmp_path, storage=src)).passed


# --- turn 7: todo.py must actually persist VIA storage.py ---

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
    # The failure this rung exists to catch: a build that reimplements json
    # instead of composing with the module the turn named.
    src = (
        "import json\n\n"
        "def save(todos, path):\n    open(path,'w').write(json.dumps(todos))\n"
    )
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed


def test_turn7_rejects_import_in_a_comment(tmp_path: Path) -> None:
    # A substring check on the source would pass this; only a real import does.
    src = "# uses storage.save_todos eventually\ndef save(todos, path):\n    pass\n"
    ws = _ws(tmp_path, storage=T6_TODOS_FIRST, todo=src)
    assert not oracles.turn7_todo_persists(ws).passed
