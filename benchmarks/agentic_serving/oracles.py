"""Hidden build-turn correctness oracles for the WS-8 strict score (#131).

WHY THESE EXIST. The strict rule is that a turn passes only when its
deliverable ships AND IS CORRECT. On Arm 0 the accept gate stands between a
build and the workspace; a frontier arm behind OpenCode has no gate. Scoring
`wrote the file == pass` would hand a free pass to an arm that ships
plausible-but-wrong code, biasing the score toward the discretionary
verification WS-8 exists to test against. These oracles are the independent,
arm-blind correctness check that removes that bias. The frontier arm's
shipped-but-oracle-failed count is the headline number they produce: it tests
structural-vs-discretionary verification directly, needing no privileged
per-arm evidence channel (see
`docs/plans/2026-07-14-strict-per-turn-table-design.md`).

BOTH ERROR DIRECTIONS ARE HAZARDS, and the false-reject one is subtler:

- A FALSE ACCEPT restores the bias above.
- A FALSE REJECT can FABRICATE the hypothesis. Richer todo representations
  (dict, dataclass) correlate with design sophistication, so a
  representation-blind oracle rejects the frontier arm harder — and a frontier
  arm shipping GOOD code then scores "shipped, oracle-failed", which reads as
  exactly the plausible-but-wrong narrative the oracle was built to detect. An
  oracle that manufactures evidence for its author's thesis is worse than no
  oracle. Turn 1 therefore probes representation-agnostically: it never
  requires the stored element to BE the item, only that the item is
  recoverable from it.

POSITIVE PROOF, NOT ABSENCE OF FAILURE. Exit 0 is what an interpreter that did
nothing returns, and `sys.exit(0)` at import (an ordinary `main()` without an
`if __name__` guard) would otherwise force a pass on every oracle. Each probe
receives a per-run nonce and must print `PROBE-OK-<nonce>` on its success path;
`_run_probe` requires that token. Model code cannot forge an unseen nonce, and
`os._exit` cannot print one. Import guards catch BaseException, since SystemExit
is not an Exception.

WHY A COPY. Probes CALL the arm's code, which has side effects: turn 6's
`save_todos` may write wherever it likes, and turn 1 invokes every public
callable. Probing a throwaway copy means the oracle can never corrupt the
workspace it is scoring, nor the state a later turn is judged against.

TOLERANCE IS BOUNDED, NOT OPEN. Every oracle is pinned by fixtures in both
directions in `tests/test_oracles.py` (the #84 methodology). Known FAR bound,
inherent to name-free search: a broken deliverable alongside an unrelated
2-argument appender (e.g. `log_event(events, message)`) passes turn 1.

NOT ORACLED: turn 2. "add a complete_todo function that marks a todo done"
leaves the representation of done-ness free; an oracle that cannot pin it either
accepts an empty body or rejects valid designs. Turn 2 stays hand-classified.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_SECONDS = 30
_IGNORED = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache")


@dataclass(frozen=True)
class OracleResult:
    """One oracle's verdict. ``detail`` carries the probe's own output so a
    verdict is auditable rather than a bare bool."""

    passed: bool
    detail: str = ""


def _run_probe(workspace: Path, program: str) -> OracleResult:
    """Run ``program`` against a throwaway COPY of ``workspace``.

    Passes a fresh nonce as ``argv[1]``. The verdict is PASS only when the probe
    exits 0 AND printed ``PROBE-OK-<nonce>``: requiring positive proof is what
    makes the oracle fail closed, since exit 0 is also what a module that killed
    the interpreter at import produces.
    """
    nonce = uuid.uuid4().hex
    token = f"PROBE-OK-{nonce}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp) / "ws"
            shutil.copytree(workspace, sandbox, ignore=_IGNORED, dirs_exist_ok=True)
            proc = subprocess.run(
                [sys.executable, "-c", program, nonce],
                cwd=str(sandbox),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        return OracleResult(False, "probe timed out")
    except (OSError, shutil.Error) as exc:
        return OracleResult(False, f"probe could not start: {exc}")
    detail = (proc.stdout + proc.stderr).strip()[-300:]
    passed = proc.returncode == 0 and token in proc.stdout
    return OracleResult(passed, detail.replace(token, "ok:"))


# Turn 1 is NAME-FREE and REPRESENTATION-FREE. The probe seeds a NON-EMPTY list
# (an empty seed cannot tell "adds" from "replaces"), requires the list to grow
# by exactly one with the seed intact, and looks for the nonce in repr() rather
# than by equality — repr containment accepts dict/dataclass wrapping while
# defeating a permissive __eq__, and the nonce cannot be hardcoded.
_TURN1_PROBE = """
import inspect, sys
NONCE = sys.argv[1]
TOKEN = "PROBE-OK-" + NONCE
SEED = ["seed-a-" + NONCE, "seed-b-" + NONCE]
ITEM = "item-" + NONCE

def ok(how):
    print(TOKEN, how)
    raise SystemExit(0)

try:
    import todo
except BaseException as exc:
    print("import failed:", exc)
    raise SystemExit(1)

def grew(after):
    try:
        seq = list(after)
    except BaseException:
        return False
    if len(seq) != len(SEED) + 1:
        return False
    text = repr(seq)
    return ITEM in text and all(s in text for s in SEED)

def required(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [
        p for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is p.empty
    ]

fns = [
    fn for name, fn in inspect.getmembers(todo, inspect.isfunction)
    if not name.startswith("_") and getattr(fn, "__module__", None) == "todo"
]

for fn in fns:
    params = required(fn)
    if len(params) == 2:
        for swapped in (False, True):
            items = list(SEED)
            try:
                out = fn(ITEM, items) if swapped else fn(items, ITEM)
            except BaseException:
                out = None
            if grew(items):
                ok("mutated in place via " + fn.__name__)
            if isinstance(out, (list, tuple)) and grew(out):
                ok("returned a new list via " + fn.__name__)
        for order in ((0, 1), (1, 0)):
            items = list(SEED)
            kwargs = {params[order[0]].name: items, params[order[1]].name: ITEM}
            try:
                out = fn(**kwargs)
            except BaseException:
                continue
            if grew(items):
                ok("mutated in place via keyword " + fn.__name__)
            if isinstance(out, (list, tuple)) and grew(out):
                ok("returned a new list via keyword " + fn.__name__)
    elif len(params) == 1:
        # Module-level-collection style: seed each module list, then call.
        for name, value in list(vars(todo).items()):
            if not isinstance(value, list):
                continue
            value[:] = list(SEED)
            try:
                out = fn(ITEM)
            except BaseException:
                continue
            if grew(value):
                ok("appended to module list " + name + " via " + fn.__name__)
            if isinstance(out, (list, tuple)) and grew(out):
                ok("returned a new list via " + fn.__name__)

print("no public callable added the item to a list of existing todos")
raise SystemExit(1)
"""


# Turn 6 NAMES save_todos/load_todos but never mentions a path parameter, so
# arity, order and path type are all free. It DOES say "using json", and it says
# save/load, so the probe requires real JSON to reach disk: same-process
# round-tripping proves memoization, not persistence.
_TURN6_PROBE = """
import inspect, json, os, sys
from pathlib import Path
NONCE = sys.argv[1]
TOKEN = "PROBE-OK-" + NONCE
DATA = ["milk-" + NONCE, "eggs-" + NONCE]

def ok(how):
    print(TOKEN, how)
    raise SystemExit(0)

try:
    import storage
except BaseException as exc:
    print("import failed:", exc)
    raise SystemExit(1)

save = getattr(storage, "save_todos", None)
load = getattr(storage, "load_todos", None)
if not callable(save) or not callable(load):
    print("save_todos/load_todos not both present and callable")
    raise SystemExit(1)

def json_files():
    return {f for f in os.listdir(".") if f.endswith(".json")}

def wrote_real_json(candidates):
    # The data must be readable back as JSON FROM DISK. This is what separates
    # persistence from an in-memory cache, and json from pickle/repr.
    for name in candidates:
        try:
            with open(name) as handle:
                if list(json.load(handle)) == DATA:
                    return True
        except BaseException:
            continue
    return False

def required(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [
        p for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is p.empty
    ]

n_save = len(required(save))

def attempt(label, do_save, do_load, path):
    before = json_files()
    try:
        do_save()
    except BaseException:
        return False
    candidates = list(json_files() - before)
    if path is not None:
        candidates.append(str(path))
    if not wrote_real_json(candidates):
        return False
    try:
        got = list(do_load())
    except BaseException:
        return False
    if got == DATA:
        ok(label)
    return False

for raw in ("todos_probe.json",):
    for path in (raw, Path(raw)):
        if n_save >= 2:
            attempt("saved (todos, path)", lambda: save(DATA, path),
                    lambda: load(path), path)
            attempt("saved (path, todos)", lambda: save(path, DATA),
                    lambda: load(path), path)
        if n_save == 1:
            attempt("saved (todos), module-constant path",
                    lambda: save(DATA), lambda: load(), None)

print("no call shape round-tripped the data through real JSON on disk")
raise SystemExit(1)
"""


# Turn 7 must COMPOSE with storage.py. A namespace check alone credits a stray
# unused import (one of the most common things a model ships), so the probe also
# requires some function body to reference the bound name. co_names is read from
# compiled code, so a comment can never satisfy it.
_TURN7_PROBE = """
import sys, types
NONCE = sys.argv[1]
TOKEN = "PROBE-OK-" + NONCE

try:
    import todo
except BaseException as exc:
    print("import failed:", exc)
    raise SystemExit(1)

bound = set()
for name, obj in vars(todo).items():
    if name.startswith("__"):
        continue
    if isinstance(obj, types.ModuleType) and obj.__name__ == "storage":
        bound.add(name)
    elif getattr(obj, "__module__", None) == "storage":
        bound.add(name)

if not bound:
    print("todo.py does not import storage")
    raise SystemExit(1)

referenced = set()
for obj in vars(todo).values():
    code = getattr(obj, "__code__", None)
    if code is not None:
        referenced |= set(code.co_names)

used = bound & referenced
if not used:
    print("todo.py imports storage but no function body references it:", sorted(bound))
    raise SystemExit(1)

print(TOKEN, "composes with storage via", sorted(used))
raise SystemExit(0)
"""


def turn1_adds_todo(workspace: Path) -> OracleResult:
    """Turn 1: some public callable adds an item to a list of existing todos,
    whatever the item's representation."""
    return _run_probe(workspace, _TURN1_PROBE)


def turn6_storage_roundtrip(workspace: Path) -> OracleResult:
    """Turn 6: save_todos/load_todos round-trip todos through real JSON on
    disk."""
    return _run_probe(workspace, _TURN6_PROBE)


def turn7_todo_persists(workspace: Path) -> OracleResult:
    """Turn 7: todo.py actually composes with storage.py rather than importing
    it decoratively or reimplementing persistence."""
    return _run_probe(workspace, _TURN7_PROBE)


# Turn 2 is absent by design (see module docstring).
ORACLES: dict[int, Callable[[Path], OracleResult]] = {
    1: turn1_adds_todo,
    6: turn6_storage_roundtrip,
    7: turn7_todo_persists,
}


def probe(turn: int, workspace: Path) -> dict[str, object] | None:
    """The oracle verdict for ``turn``, or None when that turn has no oracle.

    Called by the battery immediately after the turn runs. It MUST run then, not
    post-hoc over the final workspace: later turns mutate files (turn 13 rewrites
    buggy.py), so an end-of-run probe would score a turn against a workspace that
    turn never saw.
    """
    oracle = ORACLES.get(turn)
    if oracle is None:
        return None
    result = oracle(workspace)
    return {"passed": result.passed, "detail": result.detail}


if __name__ == "__main__":  # pragma: no cover - the battery's entry point
    import json

    print(json.dumps(probe(int(sys.argv[1]), Path(sys.argv[2]))))
