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
`_run_probe` requires that token. Import guards catch BaseException, since
SystemExit is not an Exception.

WHAT THE NONCE DOES AND DOES NOT GUARANTEE. It defends against ACCIDENTAL
exit-0, which is the real threat model here: a model shipping plausible-but-
wrong code. It is NOT unforgeable, and an earlier version of this docstring
wrongly claimed it was. The module under test is imported into the probe's own
process, so it can read the nonce from `sys.argv`, from `ps -o command=`, or by
walking `sys._getframe().f_back` to the probe's own globals — all three were
demonstrated. This is unfixable in-process (any nonce the probe holds is
reachable from the module's frames), so the honest statement is the narrow one:
the token proves the probe's success path ran, not that the code under test
declined to lie. Deliberate gaming is out of scope; a model that emits the
probe's private token is not a model shipping wrong code, it is an adversary,
and this instrument does not claim to stop one.

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

    An OSError/shutil.Error starting the probe ESCAPES rather than becoming a
    verdict: it is a failure of the instrument (missing workspace, disk full),
    not of the arm's code, and converting it into ``passed=False`` would score
    a real shipped turn as shipped-broken — the thesis-fabricating direction —
    while exiting 0, which makes the battery's crash channel (``oracle: null``
    plus a nonzero code in ``oracle-exits.tsv``) unreachable. A TIMEOUT stays a
    verdict, since a hanging probe is plausibly the subject's own infinite loop.
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
    # Recoverability must reach attribute VALUES, one level deep: a PLAIN class
    # has a default repr (`<todo.Todo object at 0x...>`) that hides the item
    # text, and the dataclass case passes only because dataclasses generate a
    # field-bearing repr. Known bound: a wrapper nested two levels deep
    # (Todo(Item(text))) stays opaque and false-rejects.
    parts = [repr(seq)]
    for el in seq:
        try:
            parts.append(repr(vars(el)))
        except TypeError:
            pass
        slots = getattr(type(el), "__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            try:
                parts.append(repr(getattr(el, slot)))
            except BaseException:
                pass
    text = " ".join(parts)
    return ITEM in text and all(s in text for s in SEED)

_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)

def required(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p for p in params if p.kind in _KINDS and p.default is p.empty]

def padded(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p for p in params if p.kind in _KINDS]

def pair_of(fn):
    # The two params to probe two-ways. Usually the two REQUIRED ones, but the
    # mutable-default-avoidance idiom (add_todo(item, todos=None)) has one
    # required plus a defaulted list and must still be probed both ways.
    req = required(fn)
    if len(req) == 2:
        return req
    pad = padded(fn)
    if len(req) < 2 and len(pad) >= 2:
        return pad[:2]
    return None

fns = [
    fn for name, fn in inspect.getmembers(todo, inspect.isfunction)
    if not name.startswith("_") and getattr(fn, "__module__", None) == "todo"
]

for fn in fns:
    params = required(fn)
    pair = pair_of(fn)
    if pair is not None:
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
            kwargs = {pair[order[0]].name: items, pair[order[1]].name: ITEM}
            try:
                out = fn(**kwargs)
            except BaseException:
                continue
            if grew(items):
                ok("mutated in place via keyword " + fn.__name__)
            if isinstance(out, (list, tuple)) and grew(out):
                ok("returned a new list via keyword " + fn.__name__)
    if len(params) == 1:
        # Module-level-collection style: seed each module list, then call.
        # Seeding MUTATES the module's own globals, so every list is snapshotted
        # and restored around each attempt. Without that, probing one list
        # clobbers unrelated module constants (a validated config, say), the
        # corruption persists into later candidates, and the verdict comes to
        # depend on getmembers() ordering — i.e. on declaration order in the
        # arm's source. Non-deterministic scoring is exactly the property that
        # disqualified name-keyed oracles in the first place.
        module_lists = [
            (name, value)
            for name, value in vars(todo).items()
            if isinstance(value, list)
        ]
        snapshot = {name: list(value) for name, value in module_lists}
        for name, value in module_lists:
            for other, original in snapshot.items():
                vars(todo)[other][:] = list(original)
            value[:] = list(SEED)
            try:
                out = fn(ITEM)
            except BaseException:
                out = None
            grew_here = grew(value)
            returned = isinstance(out, (list, tuple)) and grew(out)
            for other, original in snapshot.items():
                vars(todo)[other][:] = list(original)
            if grew_here:
                ok("appended to module list " + name + " via " + fn.__name__)
            if returned:
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
    # WALK, not listdir: a module-constant path of data/todos.json is a
    # legitimate reading of a prompt that never mentions a path, and a
    # top-level-only scan would false-reject it.
    found = set()
    for root, _dirs, names in os.walk("."):
        found |= {os.path.join(root, f) for f in names if f.endswith(".json")}
    return found

def wrote_real_json(candidates):
    # The data must be readable back as JSON FROM DISK: that is what separates
    # persistence from an in-memory cache, and json from pickle/repr.
    #
    # Recoverability, NOT equality. Requiring `json.load(...) == DATA` would pin
    # the ON-DISK shape to a bare array and false-reject a versioned envelope
    # ({"version": 1, "todos": [...]}), which is the same equality mistake turn 1
    # made against list[str] — and it fails in the direction that penalises the
    # more sophisticated design. The nonce is what makes this safe: it cannot
    # appear in a file the arm did not write from DATA.
    for name in candidates:
        try:
            with open(name) as handle:
                raw = handle.read()
            json.loads(raw)
        except BaseException:
            continue
        if all(item in raw for item in DATA):
            return True
    return False

_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)

def required(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p for p in params if p.kind in _KINDS and p.default is p.empty]

def padded(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p for p in params if p.kind in _KINDS]

def pair_of(fn):
    # Asymmetric defaults (save_todos(todos, path='todos.json') beside a
    # load_todos(path) that REQUIRES one) must still probe the explicit-path
    # shapes; selecting on the required count alone never tried them.
    req = required(fn)
    if len(req) == 2:
        return req
    pad = padded(fn)
    if len(req) < 2 and len(pad) >= 2:
        return pad[:2]
    return None

n_save = len(required(save))

def attempt(label, do_save, do_load, path):
    global storage, save, load
    try:
        do_save()
    except BaseException:
        return False
    # EVERY json file is a candidate, not just newly-created ones. A set
    # difference sampled after import misses a module that creates its file at
    # import time, and misses any write to a path that already existed — and by
    # turn 6 the workspace is not empty. The nonce in the payload is what makes
    # the wider net safe: no pre-existing file can satisfy it.
    candidates = list(json_files())
    if path is not None:
        candidates.append(str(path))
    if not wrote_real_json(candidates):
        return False
    # The LOAD leg must be proven against DISK. save and load run in one
    # process, so a load that returns a module-level cache populated by save
    # round-trips perfectly while its load leg is broken in any fresh process.
    # Reimporting gives load a fresh module: state survives only through the
    # file. (A module whose import-time init clobbers its own file fails here
    # too — faithfully, since a fresh process would see the same clobber.)
    # EVERY sandbox-local module is popped, not just storage: a cross-module
    # cache (storage stashing state in another workspace module) would survive
    # a storage-only reimport.
    cwd = os.getcwd()
    for name in [
        n for n, m in list(sys.modules.items())
        if (getattr(m, "__file__", None) or "").startswith(cwd)
    ]:
        sys.modules.pop(name, None)
    sys.modules.pop("storage", None)
    try:
        import storage
    except BaseException:
        return False
    save = getattr(storage, "save_todos", None)
    load = getattr(storage, "load_todos", None)
    if not callable(load):
        return False
    try:
        got = list(do_load())
    except BaseException:
        return False
    if got == DATA:
        ok(label)
    return False

def kwargs_for(fn, path):
    # Mirror turn 1: bind keyword-only signatures by parameter ORDER, since the
    # names are free. Without this, `def save_todos(*, todos, path)` counts as
    # two required params and is then only ever called positionally.
    params = pair_of(fn)
    if params is None:
        return None
    return [
        {params[0].name: DATA, params[1].name: path},
        {params[0].name: path, params[1].name: DATA},
    ]

for raw in ("todos_probe.json",):
    for path in (raw, Path(raw)):
        if pair_of(save) is not None:
            attempt("saved (todos, path)", lambda: save(DATA, path),
                    lambda: load(path), path)
            attempt("saved (path, todos)", lambda: save(path, DATA),
                    lambda: load(path), path)
            for kw in (kwargs_for(save, path) or []):
                load_kw = kwargs_for(load, path)
                def do_load(kw=kw):
                    try:
                        return load(path)
                    except TypeError:
                        params = required(load)
                        return load(**{params[0].name: path}) if params else load()
                attempt("saved by keyword", lambda kw=kw: save(**kw), do_load, path)
        if n_save == 1:
            attempt("saved (todos), module-constant path",
                    lambda: save(DATA), lambda: load(), None)
        if n_save == 0:
            attempt("saved (), module-constant path",
                    lambda: save(), lambda: load(), None)

print("no call shape round-tripped the data through real JSON on disk")
raise SystemExit(1)
"""


# Turn 7 must COMPOSE with storage.py. A namespace check alone credits a stray
# unused import (one of the most common things a model ships), so the probe
# requires compiled code to actually LOAD a name bound from a storage import.
# The analysis is a bytecode walk over the COMPILED SOURCE, not runtime
# objects, for three reasons found across rounds 3-4:
#
# - The module's own top-level code object must be scanned: composition living
#   only in module-level code (load at import, save under a main guard, an
#   atexit lambda) is a reasoned non-invasive design once turn-4's tests pin
#   the function signatures, and a vars(module) walk never sees it.
# - The compiled tree reaches EVERY code object in the file (methods, nested
#   functions, lambdas, decorated bodies) without chasing wrappers through
#   __wrapped__/__closure__ at runtime.
# - Opcode KINDS matter, not co_names membership: `self.storage` compiles to
#   LOAD_ATTR and a method named save_todos to LOAD_METHOD, which raw co_names
#   cannot tell apart from a global load of the import — so a decorative
#   import plus a same-named attribute self-certified (round 4's catch).
#   IMPORT_NAME/IMPORT_FROM likewise must not credit themselves.
#
# Known FAR bounds, documented not hidden: reachability is not analyzed (a
# storage call on a dead line after a bare return counts); a bare LOAD with no
# call (`def f(): storage`) counts as a reference; and an F811 shadow (`from
# storage import save_todos` then a same-named local reimplementation the code
# calls) credits the original binding. Known FRR bounds: a local import used
# only through a closure cell of a NESTED function (the binding and the LOAD
# live in different code objects); `from storage import *` (IMPORT_STAR binds
# no analyzable name); `importlib.import_module("storage")` (a static
# analysis cannot see a dynamic import).
_TURN7_PROBE = """
import dis, sys
NONCE = sys.argv[1]
TOKEN = "PROBE-OK-" + NONCE

try:
    import todo
except BaseException as exc:
    print("import failed:", exc)
    raise SystemExit(1)

try:
    with open("todo.py") as handle:
        root = compile(handle.read(), "todo.py", "exec")
except BaseException as exc:
    print("could not compile todo.py:", exc)
    raise SystemExit(1)

def code_objects(root):
    out, stack = [], [root]
    while stack:
        code = stack.pop()
        out.append(code)
        stack.extend(c for c in code.co_consts if hasattr(c, "co_names"))
    return out

STORES = ("STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF")
GLOBAL_LOADS = ("LOAD_NAME", "LOAD_GLOBAL")
LOCAL_LOADS = ("LOAD_FAST", "LOAD_NAME", "LOAD_DEREF")

codes = code_objects(root)

def storage_bound_names(code):
    bound_here, pending = set(), False
    for ins in dis.get_instructions(code):
        if ins.opname == "IMPORT_NAME":
            pending = ins.argval == "storage"
        elif pending and ins.opname == "IMPORT_FROM":
            bound_here.add(ins.argval)
        elif pending and ins.opname in STORES:
            bound_here.add(ins.argval)
        else:
            pending = False
    return bound_here

# Names bound at module level are visible everywhere via global loads; names
# bound inside a code object are visible only there, via local loads.
module_bound = storage_bound_names(root)
local_bound = {id(code): storage_bound_names(code) for code in codes[1:]}

used = set()
for code in codes:
    local = local_bound.get(id(code), set())
    for ins in dis.get_instructions(code):
        if ins.opname in GLOBAL_LOADS and ins.argval in module_bound:
            used.add(ins.argval)
        elif ins.opname in LOCAL_LOADS and ins.argval in local:
            used.add(ins.argval)

bound = module_bound | set().union(*local_bound.values(), set())
if not bound:
    print("todo.py does not import storage")
    raise SystemExit(1)

if not used:
    print("todo.py imports storage but no code references it:", sorted(bound))
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
